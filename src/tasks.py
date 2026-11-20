"""Tasks: everything the training loop needs to know about the problem.

The loop in src/train.py is the same for every model (DECISIONS D14). What
changes from one problem to another lives in a task object:

    loss(output, target)        the loss used by backward()
    n_terms(target)             how many terms that loss averages over
    start() / collect(...)      what to keep from each evaluation batch
    epoch_metrics(loss, state)  metrics printed and logged at every epoch
    selection_metric, mode      which metric decides early stopping, and
                                whether higher ("max") or lower ("min") is
                                better
    report(...)                 final metrics of the best model, per split

Two tasks:
    MultiLabelTask  5 diagnostic superclasses, BCEWithLogitsLoss (D2, D4)
    NextTokenTask   autoregressive generator, cross-entropy per token (D13)
"""
import numpy as np
import torch
import torch.nn.functional as F

from src import metrics


class Task:
    name = "task"
    selection_metric = "loss"
    mode = "min"

    def loss(self, output, target, reduction: str = "mean"):
        raise NotImplementedError

    def n_terms(self, target) -> int:
        """Number of terms the mean loss averages over. The loop multiplies
        the batch loss by this number and divides by the total at the end,
        so the epoch loss is the exact mean over all terms of the split,
        not a mean of batch means (the last batch is usually smaller)."""
        return target.numel()

    def start(self):
        return None

    def collect(self, state, output, target):
        return state

    def epoch_metrics(self, mean_loss: float, state) -> dict:
        return {"loss": mean_loss}

    def report(self, results: dict) -> tuple[dict, dict, dict]:
        """results: split -> (mean_loss, state) of the best model.
        Returns (details, scalars, arrays): details go to the JSON log,
        scalars to the flat `scalars` block, arrays to the predictions
        file."""
        details, scalars = {}, {}
        for split, (mean_loss, state) in results.items():
            m = self.epoch_metrics(mean_loss, state)
            details[split] = m
            scalars.update({f"{split}_{k}": v for k, v in m.items()})
        return details, scalars, {}


# ---------------------------------------------------------------------------
class MultiLabelTask(Task):
    """Multi-label classification of the diagnostic superclasses.

    The model returns one logit per class, without a final sigmoid:
    BCEWithLogitsLoss applies it inside, in a numerically stable way, and
    expects float targets (weekly summary 13/11). Each class is treated as
    an independent yes/no problem (multi-label = several binary classifiers,
    PRDL_MLLB_LogLikelihood_CrossEntropy_KLDivergence.pptx, slide 29).
    """
    name = "multilabel"
    selection_metric = "macro_auroc"
    mode = "max"

    def __init__(self, class_names: list[str], groups: dict,
                 record_ids: dict | None = None,
                 pos_weight: np.ndarray | None = None,
                 n_boot: int = 1000, seed: int = 42):
        """groups: split -> patient id of each record, in loader order
        (the validation and test loaders do not shuffle). record_ids:
        split -> ecg_id, stored with the predictions to check alignment."""
        self.class_names = list(class_names)
        self.groups = groups
        self.record_ids = record_ids or {}
        self.pos_weight = (None if pos_weight is None
                           else torch.as_tensor(pos_weight, dtype=torch.float32))
        self.n_boot, self.seed = n_boot, seed

    def loss(self, output, target, reduction: str = "mean"):
        pw = None if self.pos_weight is None else self.pos_weight.to(output.device)
        return F.binary_cross_entropy_with_logits(output, target,
                                                  pos_weight=pw,
                                                  reduction=reduction)

    def start(self):
        return {"prob": [], "true": []}

    def collect(self, state, output, target):
        state["prob"].append(torch.sigmoid(output).float().cpu())
        state["true"].append(target.float().cpu())
        return state

    @staticmethod
    def _arrays(state):
        return (torch.cat(state["true"]).numpy(),
                torch.cat(state["prob"]).numpy())

    def epoch_metrics(self, mean_loss, state) -> dict:
        y, p = self._arrays(state)
        m = metrics.classification_metrics(y, p)
        out = {"loss": mean_loss, "macro_auroc": m["macro_auroc"]}
        out.update({f"auroc_{c}": float(v)
                    for c, v in zip(self.class_names, m["auroc"])})
        return out

    def report(self, results: dict):
        """Thresholds from validation; metrics and patient-bootstrap
        intervals on validation and, for final runs, on test (D4, D14)."""
        y_val, p_val = self._arrays(results["val"][1])
        thresholds = metrics.best_thresholds(y_val, p_val)
        details = {"thresholds": dict(zip(self.class_names,
                                          thresholds.tolist()))}
        scalars = {f"threshold_{c}": float(t)
                   for c, t in zip(self.class_names, thresholds)}
        arrays = {}
        for split, (mean_loss, state) in results.items():
            y, p = self._arrays(state)
            m = metrics.classification_metrics(y, p, thresholds)
            ci = metrics.bootstrap_ci(y, p, self.groups[split], thresholds,
                                      n_boot=self.n_boot, seed=self.seed)
            d = {"loss": mean_loss, "n_records": int(len(y)),
                 "macro_auroc": m["macro_auroc"],
                 "macro_auroc_ci": list(map(float, ci["macro_auroc"])),
                 "macro_f1": m["macro_f1"],
                 "macro_f1_ci": list(map(float, ci["macro_f1"])),
                 "bootstrap": {k: ci[k] for k in ("n_boot", "n_skipped",
                                                  "alpha", "seed")}}
            s = {"loss": mean_loss,
                 "macro_auroc": m["macro_auroc"],
                 "macro_auroc_lo": float(ci["macro_auroc"][0]),
                 "macro_auroc_hi": float(ci["macro_auroc"][1]),
                 "macro_f1": m["macro_f1"],
                 "macro_f1_lo": float(ci["macro_f1"][0]),
                 "macro_f1_hi": float(ci["macro_f1"][1])}
            for j, c in enumerate(self.class_names):
                d[f"auroc_{c}"] = float(m["auroc"][j])
                d[f"auroc_{c}_ci"] = [float(ci["auroc"][0][j]),
                                      float(ci["auroc"][1][j])]
                d[f"f1_{c}"] = float(m["f1"][j])
                s[f"auroc_{c}"] = float(m["auroc"][j])
                s[f"auroc_{c}_lo"] = float(ci["auroc"][0][j])
                s[f"auroc_{c}_hi"] = float(ci["auroc"][1][j])
                s[f"f1_{c}"] = float(m["f1"][j])
            details[split] = d
            scalars.update({f"{split}_{k}": v for k, v in s.items()})
            arrays[f"{split}_prob"] = p.astype(np.float32)
            arrays[f"{split}_true"] = y.astype(np.float32)
            arrays[f"{split}_patient_id"] = np.asarray(self.groups[split])
            if split in self.record_ids:
                arrays[f"{split}_ecg_id"] = np.asarray(self.record_ids[split])
        return details, scalars, arrays


# ---------------------------------------------------------------------------
class NextTokenTask(Task):
    """Next-token prediction for the autoregressive generator (D13).

    output: (batch, time, vocab) logits; target: (batch, time) integer
    token indices (the input shifted by one step, teacher forcing).
    CrossEntropyLoss expects the classes in dimension 1, i.e. (N, vocab):
    passing (batch, time, vocab) directly is a classic silent bug, so both
    tensors are flattened to one row per token. CrossEntropyLoss contains
    the softmax: the model has no softmax in its last layer.
    Early stopping on the validation cross-entropy (lower is better);
    perplexity = exp(cross-entropy) is reported with it.
    """
    name = "next_token"
    selection_metric = "loss"
    mode = "min"

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def loss(self, output, target, reduction: str = "mean"):
        return F.cross_entropy(output.reshape(-1, self.vocab_size),
                               target.reshape(-1), reduction=reduction)

    def epoch_metrics(self, mean_loss, state) -> dict:
        return {"loss": mean_loss, "perplexity": metrics.perplexity(mean_loss)}
