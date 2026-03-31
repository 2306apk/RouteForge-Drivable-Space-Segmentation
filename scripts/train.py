import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.drivable_dataset import DrivableDataset
from src.models.unet_tiny import TinyUNet
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import batch_iou_from_logits, batch_iou_from_probs


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
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--val-threshold-sweep", action="store_true")
    p.add_argument("--sweep-start", type=float, default=0.3)
    p.add_argument("--sweep-end", type=float, default=0.7)
    p.add_argument("--sweep-step", type=float, default=0.05)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--pos-weight", type=float, default=None)
    return p.parse_args()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    train=True,
    iou_threshold=0.5,
    eval_thresholds=None,
):
    model.train(train)
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    threshold_totals = None
    if (not train) and eval_thresholds:
        threshold_totals = {float(t): 0.0 for t in eval_thresholds}

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        use_amp = scaler is not None and scaler.is_enabled()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits)
        iou = batch_iou_from_probs(probs, masks, threshold=iou_threshold)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_iou += iou * bs
        n += bs

        if threshold_totals is not None:
            for thr in threshold_totals:
                thr_iou = batch_iou_from_probs(probs, masks, threshold=thr)
                threshold_totals[thr] += thr_iou * bs

    threshold_ious = None
    if threshold_totals is not None and n > 0:
        threshold_ious = {thr: total / n for thr, total in threshold_totals.items()}

    return total_loss / n, total_iou / n, threshold_ious


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
    criterion = BCEDiceLoss(bce_weight=0.6, pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    best_iou = -1.0
    best_epoch = 0
    best_threshold = args.iou_threshold
    no_improve_epochs = 0
    best_path = os.path.join(args.out_dir, "best.pt")
    last_path = os.path.join(args.out_dir, "last.pt")

    val_thresholds = None
    if args.val_threshold_sweep:
        val_thresholds = []
        t = args.sweep_start
        while t <= args.sweep_end + 1e-9:
            val_thresholds.append(round(t, 4))
            t += args.sweep_step
        if args.iou_threshold not in val_thresholds:
            val_thresholds.append(args.iou_threshold)
            val_thresholds = sorted(set(val_thresholds))

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            train=True,
            iou_threshold=args.iou_threshold,
        )
        va_loss, va_iou_base, sweep_ious = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            scaler=None,
            train=False,
            iou_threshold=args.iou_threshold,
            eval_thresholds=val_thresholds,
        )
        scheduler.step()

        va_iou = va_iou_base
        epoch_best_threshold = args.iou_threshold
        if sweep_ious:
            epoch_best_threshold, va_iou = max(sweep_ious.items(), key=lambda kv: kv[1])

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_iou={tr_iou:.4f} | "
            f"val_loss={va_loss:.4f} val_iou={va_iou:.4f}"
        )
        if sweep_ious:
            print(
                f"  val_iou@{args.iou_threshold:.2f}={va_iou_base:.4f} "
                f"best_thr={epoch_best_threshold:.2f} best_thr_iou={va_iou:.4f}"
            )

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "val_iou": va_iou,
                "val_iou_base": va_iou_base,
                "best_threshold": epoch_best_threshold,
            },
            last_path,
        )

        if va_iou > best_iou + args.early_stop_min_delta:
            best_iou = va_iou
            best_epoch = epoch
            best_threshold = epoch_best_threshold
            no_improve_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "val_iou": va_iou,
                    "val_iou_base": va_iou_base,
                    "best_threshold": epoch_best_threshold,
                },
                best_path,
            )
            print(f"  Saved new best: {best_path} (val_iou={best_iou:.4f})")
        else:
            no_improve_epochs += 1

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch} (no val IoU improvement for {no_improve_epochs} epochs)."
            )
            break

    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(
            f"Restored best checkpoint from epoch {best_epoch} "
            f"(val_iou={best_iou:.4f}, threshold={best_threshold:.2f})."
        )

    fps = estimate_fps(model, device, h=args.img_h, w=args.img_w)
    print(f"Estimated FPS @ {args.img_w}x{args.img_h}: {fps:.2f}")


if __name__ == "__main__":
    main()