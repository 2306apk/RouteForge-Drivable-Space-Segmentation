import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.6, eps=1e-6, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.pos_weight = None
            self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss