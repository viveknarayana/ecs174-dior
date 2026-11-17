"""Fine-tune torchvision Faster R-CNN on DIOR. Owner: Vivek Narayana."""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import CONFIG

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "DIOR_Data"))
from classes import CLASS_NAMES
from dataset import DIORDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=CONFIG.data_root)
    parser.add_argument("--backbone", choices=("resnet50", "mobilenet"), default=CONFIG.backbone)
    parser.add_argument("--train-split", choices=("train", "trainval"), default=CONFIG.train_split)
    parser.add_argument("--output-root", type=Path, default=CONFIG.output_root)
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--workers", type=int, default=CONFIG.workers)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tiny-test", action="store_true", default=CONFIG.tiny_test)
    parser.add_argument("--tiny-train-images", type=int, default=CONFIG.tiny_train_images)
    parser.add_argument("--tiny-validation-images", type=int, default=CONFIG.tiny_validation_images)
    parser.add_argument("--no-validation", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(backbone, image_size):
    """Load COCO Faster R-CNN, replace its class/box predictor, and resize internally."""
    # Preserve the original 640 resize when loading older checkpoints.
    resize_options = {} if image_size == 800 else {"min_size": image_size, "max_size": image_size}
    if backbone == "resnet50":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
            trainable_backbone_layers=5,
            **resize_options,
        )
    else:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
            trainable_backbone_layers=6,
            **resize_options,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    return model


def make_loader(dataset, batch_size, shuffle, workers):
    generator = torch.Generator()
    generator.seed()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )


def run_train_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    sums = {}
    batches = 0
    images_seen = 0
    started = time.perf_counter()
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for name, value in loss_dict.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach().item())
        sums["total_loss"] = sums.get("total_loss", 0.0) + float(loss.detach().item())
        batches += 1
        images_seen += len(images)
        if batches % 50 == 0 or batches == len(loader):
            elapsed = time.perf_counter() - started
            print(f"Epoch {epoch}/{total_epochs} | "
                  f"images {images_seen}/{len(loader.dataset)} | "
                  f"batches {batches}/{len(loader)} | "
                  f"elapsed {elapsed / 60:.1f} min | "
                  f"average loss {sums['total_loss'] / batches:.4f}", flush=True)
    return {name: value / max(batches, 1) for name, value in sums.items()}


@torch.no_grad()
def run_validation_loss(model, loader, device):
    # Detection models emit losses only in train mode. No gradients are retained.
    model.train()
    sums = {}
    batches = 0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        for name, value in loss_dict.items():
            sums[name] = sums.get(name, 0.0) + float(value.item())
        sums["total_loss"] = sums.get("total_loss", 0.0) + float(loss.item())
        batches += 1
    return {name: value / max(batches, 1) for name, value in sums.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "backbone": config.backbone,
        "class_names": CLASS_NAMES,
        "config": vars(config),
    }, path)


def save_loss_plot(log_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []
    with log_path.open(newline="") as log_file:
        for row in csv.DictReader(log_file):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_total_loss"]))
            if row["val_total_loss"]:
                val_epochs.append(int(row["epoch"]))
                val_losses.append(float(row["val_total_loss"]))

    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, marker="o", label="Training loss")
    if val_losses:
        ax.plot(val_epochs, val_losses, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss")
    ax.set_title("DIOR Faster R-CNN loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    config = CONFIG
    config.data_root = args.data_root
    config.output_root = args.output_root
    config.backbone = args.backbone
    config.train_split = args.train_split
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.workers = args.workers
    config.tiny_test = args.tiny_test
    config.tiny_train_images = args.tiny_train_images
    config.tiny_validation_images = args.tiny_validation_images
    config.use_validation = args.train_split == "train" and not args.no_validation

    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = config.output_root / config.backbone
    run_dir.mkdir(parents=True, exist_ok=True)

    # Previous development setup: DIORDataset(config.data_root, split="train")
    train_data = DIORDataset(config.data_root, split=config.train_split)
    val_data = DIORDataset(config.data_root, split="val") if config.use_validation else None
    print(f"Training split: {config.train_split} ({len(train_data)} images); "
          f"validation: {'val' if val_data is not None else 'disabled'}")
    if config.tiny_test:
        train_data = Subset(train_data, range(min(config.tiny_train_images, len(train_data))))
        if val_data is not None:
            val_data = Subset(val_data, range(min(config.tiny_validation_images, len(val_data))))

    train_loader = make_loader(train_data, config.batch_size, True, config.workers)
    val_loader = make_loader(val_data, config.batch_size, False, config.workers) if val_data is not None else None

    model = build_model(config.backbone, config.image_size).to(device)
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_split = checkpoint.get("config", {}).get("train_split", "train")
        if checkpoint_split != config.train_split:
            raise ValueError(f"Checkpoint uses {checkpoint_split}, but this run uses {config.train_split}")
        if checkpoint.get("config", {}).get("image_size") != config.image_size:
            raise ValueError("Checkpoint image size does not match this run")
        if checkpoint["backbone"] != config.backbone:
            raise ValueError("Resume checkpoint backbone does not match --backbone")
        if checkpoint["class_names"] != CLASS_NAMES:
            raise ValueError("Resume checkpoint class ordering does not match this code")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Resuming at epoch {start_epoch + 1}")

    log_path = run_dir / "training_log.csv"
    write_header = not log_path.exists() or start_epoch == 0
    with log_path.open("a" if not write_header else "w", newline="") as log_file:
        fieldnames = ["epoch", "lr", "epoch_seconds", "train_total_loss", "val_total_loss",
                      "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, config.epochs):
            started = time.perf_counter()
            train_losses = run_train_epoch(model, train_loader, optimizer, device,
                                           epoch + 1, config.epochs)
            val_losses = run_validation_loss(model, val_loader, device) if val_loader else {}
            elapsed = time.perf_counter() - started
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}/{config.epochs} | {elapsed:.1f}s | lr={lr:.6g} | "
                  f"train={train_losses['total_loss']:.4f}" +
                  (f" | val={val_losses['total_loss']:.4f}" if val_losses else ""))
            for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
                print(f"  {key}: train={train_losses.get(key, float('nan')):.4f}" +
                      (f" val={val_losses.get(key, float('nan')):.4f}" if val_losses else ""))

            row = {
                "epoch": epoch + 1, "lr": lr, "epoch_seconds": elapsed,
                "train_total_loss": train_losses["total_loss"],
                "val_total_loss": val_losses.get("total_loss", ""),
            }
            for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
                row[key] = train_losses.get(key, "")
            writer.writerow(row)
            log_file.flush()

            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch + 1:02d}.pth",
                            model, optimizer, scheduler, epoch + 1, best_val_loss, config)
            selection_loss = val_losses.get("total_loss", train_losses["total_loss"])
            if selection_loss < best_val_loss:
                best_val_loss = selection_loss
                save_checkpoint(run_dir / "best.pth", model, optimizer, scheduler,
                                epoch + 1, best_val_loss, config)
            save_loss_plot(log_path, run_dir / "loss_curves.png")
            print(f"EPOCH_FILES_READY {epoch + 1}", flush=True)

    print(f"Finished. Checkpoints, log, and loss plot: {run_dir}")


if __name__ == "__main__":
    main()
