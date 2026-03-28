import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.drivable_dataset import DrivableDataset
from src.models.unet_tiny import TinyUNet
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import batch_iou_from_logits


def parse_args():
    p = argparse.ArgumentParser(description="Train tiny U-Net for drivable-space segmentation.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-h", type=int, default=256)
    p.add_argument("--img-w", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    model.train(train)
    total_loss = 0.0
    total_iou = 0.0
    n = 0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast('cuda',enabled=(scaler is not None)):
            logits = model(images)
            loss = criterion(logits, masks)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        iou = batch_iou_from_logits(logits, masks)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou += iou * bs
        n += bs

    return total_loss / n, total_iou / n


@torch.no_grad()
def estimate_fps(model, device, h=288, w=512, warmup=30, iters=100):
    model.eval()
    x = torch.randn(1, 3, h, w, device=device)

    for _ in range(warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return iters / dt


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = DrivableDataset(args.train_csv, image_size=(args.img_h, args.img_w), is_train=True)
    val_ds = DrivableDataset(args.val_csv, image_size=(args.img_h, args.img_w), is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TinyUNet(in_ch=3, out_ch=1, base=32).to(device)
    criterion = BCEDiceLoss(bce_weight=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler('cuda',enabled=(args.amp and device.type == "cuda"))

    best_iou = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        va_loss, va_iou = run_epoch(model, val_loader, criterion, optimizer, device, scaler=None, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_iou={tr_iou:.4f} | "
            f"val_loss={va_loss:.4f} val_iou={va_iou:.4f}"
        )

        torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": va_iou}, last_path)

        if va_iou > best_iou:
            best_iou = va_iou
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_iou": va_iou}, best_path)
            print(f"  Saved new best: {best_path} (val_iou={best_iou:.4f})")

    fps = estimate_fps(model, device, h=args.img_h, w=args.img_w)
    print(f"Estimated FPS @ {args.img_w}x{args.img_h}: {fps:.2f}")


if __name__ == "__main__":
    main()