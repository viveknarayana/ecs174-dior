"""Draw ground-truth boxes for a sample of DIOR images. Owner: Jason Peng."""

import argparse
import random
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "DIOR_Data"))

from dataset import DIORDataset
from classes import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="DIOR root containing Main/, JPEGImages-trainval/, and Annotations/")
    parser.add_argument("--output", type=Path, default=Path("outputs/ground_truth"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=174)
    args = parser.parse_args()

    dataset = DIORDataset(args.root, split="train")
    count = min(args.count, len(dataset))
    indices = random.Random(args.seed).sample(range(len(dataset)), count)
    args.output.mkdir(parents=True, exist_ok=True)

    for number, idx in enumerate(indices, start=1):
        image, target = dataset[idx]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image.permute(1, 2, 0))
        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           linewidth=2, edgecolor="red", facecolor="none"))
            ax.text(x1, y1, CLASS_NAMES[label.item() - 1], color="white",
                    bbox={"facecolor": "red", "alpha": 0.8})
        ax.axis("off")
        fig.savefig(args.output / f"example_{number:02}.png", bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {count} ground-truth previews to {args.output}")


if __name__ == "__main__":
    main()
