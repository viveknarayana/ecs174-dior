"""Shared training defaults for the DIOR Faster R-CNN experiments."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Point this at the extracted DIOR root containing ImageSets/, images, and Annotations/.
    data_root: Path = Path(".")
    output_root: Path = Path("outputs")
    backbone: str = "resnet50"  # resnet50 or mobilenet
    train_split: str = "trainval"

    # DIOR's 800×800 images match Torchvision's default minimum resize.
    image_size: int = 800
    batch_size: int = 2
    epochs: int = 10
    workers: int = 2
    seed: int = 174

    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 5
    lr_gamma: float = 0.1

    use_validation: bool = True
    tiny_test: bool = False
    tiny_train_images: int = 300
    tiny_validation_images: int = 64


CONFIG = Config()
