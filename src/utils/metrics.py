import torch


@torch.no_grad()
def batch_iou_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = (preds + targets - preds * targets).sum(dim=1)
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()