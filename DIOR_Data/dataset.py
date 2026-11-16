import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from classes import CLASS_NAMES, CLASS_TO_ID, normalize_class_name

class DIORDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)

        split_file = self.root / "Main" / f"{split}.txt"

        with open(split_file) as f:
            self.ids = [
                line.strip().removesuffix(".jpg")
                for line in f if line.strip()
            ]
          
        image_folder = (
            "JPEGImages-test"
            if split == "test"
            else "JPEGImages-trainval"
        )

        self.image_dir = self.root / image_folder
        self.ann_dir = (
            self.root /
            "Annotations/Horizontal Bounding Boxes"
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        image_path = self.image_dir / f"{image_id}.jpg"
        xml_path = self.ann_dir / f"{image_id}.xml"

        with Image.open(image_path) as img:
            image = img.convert("RGB")
            width, height = image.size
            image = to_tensor(image)

        root = ET.parse(xml_path).getroot()
        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = normalize_class_name(
                obj.findtext("name", default="")
            )

            if name not in CLASS_TO_ID:
                raise ValueError(
                    f"Unknown class: {name}"
                )

            bbox = obj.find("bndbox")

            xmin = float(bbox.findtext("xmin"))
            ymin = float(bbox.findtext("ymin"))
            xmax = float(bbox.findtext("xmax"))
            ymax = float(bbox.findtext("ymax"))

            if not (
                0 <= xmin < xmax <= width
                and 0 <= ymin < ymax <= height
            ):
                raise ValueError(
                    f"Invalid box in {xml_path}"
                )

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_TO_ID[name])

        boxes = torch.tensor(
            boxes, dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.tensor(
            labels, dtype=torch.int64
        )

        area = (
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1])
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": torch.zeros(
                len(labels), dtype=torch.int64
            ),
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))

