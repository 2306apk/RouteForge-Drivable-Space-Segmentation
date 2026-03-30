import torch
import torch.nn.functional as F

def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred_sigmoid = torch.sigmoid(pred)

    smooth = 1e-6
    intersection = (pred_sigmoid * target).sum()
    dice = 1 - (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)

    return bce + dice