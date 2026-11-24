    def loss(self, output, target, reduction: str = "mean"):
        pw = None if self.pos_weight is None else self.pos_weight.to(output.device)
        return F.binary_cross_entropy_with_logits(output, target,
                                                  pos_weight=pw,
                                                  reduction=reduction)
