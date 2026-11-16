"""Count training objects by class and save a bar chart. Owner: Jason Peng."""

import argparse
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "DIOR_Data"))

from classes import CLASS_NAMES, normalize_class_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="DIOR root containing Main/ and Annotations/")
    parser.add_argument("--split", choices=("train", "val", "trainval"), default="train")
    parser.add_argument("--output", type=Path, default=Path("outputs/class_distribution.png"))
    args = parser.parse_args()

    split_dir = args.root / "Main"
    if args.split == "trainval":
        ids = (split_dir / "train.txt").read_text().splitlines() + (split_dir / "val.txt").read_text().splitlines()
        if len(ids) != len(set(ids)):
            raise ValueError("Train and validation splits overlap")
    else:
        ids = (split_dir / f"{args.split}.txt").read_text().splitlines()

    annotation_dir = args.root / "Annotations" / "Horizontal Bounding Boxes"
    counts = Counter()
    for image_id in ids:
        xml_path = annotation_dir / f"{image_id.strip().removesuffix('.jpg')}.xml"
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = normalize_class_name(obj.findtext("name", default=""))
            if name not in {normalize_class_name(label) for label in CLASS_NAMES}:
                raise ValueError(f"Unknown class {name!r} in {xml_path}")
            counts[name] += 1

    labels = [normalize_class_name(name) for name in CLASS_NAMES]
    values = [counts[label] for label in labels]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(CLASS_NAMES, values)
    ax.invert_yaxis()
    ax.set_xlabel("Annotated objects")
    ax.set_title(f"DIOR class counts ({args.split}; {sum(values):,} objects)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Images counted: {len(ids)}")
    print(f"Objects counted: {sum(values):,}")
    for name, value in zip(CLASS_NAMES, values):
        print(f"{name}: {value}")
    print(f"Chart saved to {args.output}")


if __name__ == "__main__":
    main()
