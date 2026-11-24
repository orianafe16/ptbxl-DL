"""Tests of report/make_numbers.py and src/compare.py (T7).

Run from the repository root:  python -m pytest -q tests
"""
import importlib.util
import json

import numpy as np
import pytest

from src import config

_spec = importlib.util.spec_from_file_location(
    "make_numbers", config.REPORT_DIR / "make_numbers.py")
make_numbers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_numbers)


@pytest.mark.parametrize("key, name", [
    ("n_with_1_labels", "edaNWithOneLabels"),
    ("prev_train_MI", "edaPrevTrainMI"),
    ("lead_std_V1", "edaLeadStdVOne"),
    ("val_auroc_NORM_lo", "edaValAurocNORMLo"),
    ("n_records_all", "edaNRecordsAll"),
])
def test_macro_names(key, name):
    assert make_numbers.macro_name("eda", key) == name


def test_number_formats():
    f = make_numbers.format_number
    assert f(21799) == "21{,}799"
    assert f(7) == "7"
    assert f(0.87654) == "0.877"
    assert f(-0.0123) == "\\ensuremath{-}0.012"
    assert f(-0.0004) == "0.000"          # no "-0.000"
    assert f(float("nan")) is None
    assert f(True) is None
    assert make_numbers.format_pct(0.2563) == "25.6\\%"


def write_log(dir_, name, scalars, fast=False):
    with open(dir_ / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": name, "fast": fast, "scalars": scalars,
                   "meta": {"created": "now", "git_commit": "abc"}}, f)


def test_build_writes_macros_and_percentages(tmp_path):
    write_log(tmp_path, "eda", {"n_records_all": 21799,
                                "prev_train_MI": 0.2563})
    lines, warnings = make_numbers.build({"eda": "eda"}, tmp_path)
    text = "\n".join(lines)
    assert "\\newcommand{\\edaNRecordsAll}{21{,}799}" in text
    assert "\\newcommand{\\edaPrevTrainMIPct}{25.6\\%}" in text
    assert "% eda: prev_train_MI" in text
    assert warnings == []


def test_two_keys_with_the_same_macro_name_are_an_error(tmp_path):
    write_log(tmp_path, "x", {"n_1": 1, "n1": 2})
    with pytest.raises(ValueError, match="produced by both"):
        make_numbers.build({"x": "x"}, tmp_path)


def test_fast_runs_and_bad_aliases_are_refused(tmp_path):
    write_log(tmp_path, "smoke_fast", {"a": 1}, fast=True)
    with pytest.raises(ValueError, match="fast"):
        make_numbers.build({"smoke": "smoke_fast"}, tmp_path)
    make_numbers.build({"smoke": "smoke_fast"}, tmp_path, allow_fast=True)
    with pytest.raises(ValueError, match="letters only"):
        make_numbers.build({"run1": "smoke_fast"}, tmp_path, allow_fast=True)


def test_nan_becomes_a_visible_placeholder(tmp_path):
    write_log(tmp_path, "r", {"val_macro_auroc": float("nan")})
    lines, warnings = make_numbers.build({"r": "r"}, tmp_path)
    assert "\\newcommand{\\rValMacroAuroc}{\\textbf{??}}" in lines
    assert len(warnings) == 1


def test_eda_log_translates_without_collisions():
    path = config.LOGS_DIR / "eda.json"
    if not path.exists():
        pytest.skip("results/logs/eda.json not present")
    lines, _ = make_numbers.build({"eda": "eda"})
    assert sum(l.startswith("\\newcommand") for l in lines) > 0


# ---------------------------------------------------------------- compare
def test_compare_refuses_runs_on_different_records(tmp_path, monkeypatch):
    from src import compare, runlog
    monkeypatch.setattr(config, "ROOT", tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    rng = np.random.default_rng(0)
    y = (rng.random((40, 2)) > 0.5).astype(np.float32)
    y[0], y[1] = 1, 0
    for name, ids in (("a", np.arange(40)), ("b", np.arange(40) + 1)):
        np.savez(tmp_path / f"{name}.npz", val_prob=rng.random((40, 2)),
                 val_true=y, val_patient_id=np.arange(40), val_ecg_id=ids)
        with open(logs / f"{name}.json", "w") as f:
            json.dump({"predictions": f"{name}.npz"}, f)
    monkeypatch.setattr(runlog, "load_run",
                        lambda n: json.load(open(logs / f"{n}.json")))
    with pytest.raises(ValueError, match="not evaluated on the same records"):
        compare.compare("a", "b")
    d = compare.compare("a", "a")["scalars"]
    assert d["diff_macro_auroc"] == 0 == d["diff_macro_auroc_lo"]
