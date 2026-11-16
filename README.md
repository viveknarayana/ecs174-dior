# DIOR Satellite Object Detection

University computer vision project: fine-tune torchvision Faster R-CNN on DIOR satellite imagery. The full dataset is not stored in this repository; see `DIOR_Data/README.md` for the expected local/Colab layout.

## Current contents

- `DIOR_Data/dataset.py`: reads Pascal VOC XML annotations into torchvision detection targets.
- `DIOR_Data/Main/`: image identifiers for train, validation, and test splits.
- `DIOR_Data/DIOR_Data_Prep.ipynb`: Colab-oriented extraction, split-count checks, data-loader smoke check, ground-truth previews, and a class distribution chart.
- `DIOR_Data/ground_truth/` and `DIOR_Data/class_distribution.png`: generated visual checks and class-count figure.

This is the initial data-preparation contribution. Model configuration, training, evaluation, and prediction scripts are not included yet.

## What the dataset code currently does

`DIORDataset(root, split)` reads IDs from `root/Main/{split}.txt`, opens the matching image under `JPEGImages-trainval/` (or `JPEGImages-test/` for `split="test"`), parses `Annotations/Horizontal Bounding Boxes/{id}.xml`, converts the image to RGB, and returns an image tensor plus a target dictionary (`boxes`, `labels`, `image_id`, `area`, `iscrowd`). `collate_fn` batches variable-length targets for detection models.

The current split files contain 5,862 train IDs, 5,863 validation IDs, and 11,738 test IDs. Train plus validation is 11,725 IDs. The notebook asserts those totals and that the splits do not overlap. The intended project policy is to train/validate only on trainval and reserve test for final evaluation.

## Setup

Install PyTorch and torchvision, plus Pillow for image loading. For notebook plots, install matplotlib and run `DIOR_Data/DIOR_Data_Prep.ipynb` in Colab after placing the DIOR archives in Drive as described in `DIOR_Data/README.md`.

Example:

```python
from torch.utils.data import DataLoader
from DIOR_Data.dataset import DIORDataset, collate_fn

train = DIORDataset("/path/to/DIOR", split="train")
loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collate_fn)
```

If `DIOR_Data` is used as a package, add an `__init__.py` or import the module with the directory on `PYTHONPATH`.

## Known follow-up work

- The project specification requires clipping boxes to image bounds and dropping invalid/degenerate boxes; the current loader raises an error for these instead.
- The prompt specifies a shared canonical class list (including names such as `baseball field` and `harbour`); the loader currently stores normalized compact names (`baseballfield`, `harbor`). Keep one authoritative ordered mapping when the other modules are added.
- Current notebook visuals are useful checks, but the requested standalone box-check and class-count scripts have not been added.
- Verify the annotation count: the README/chart currently state 192,518 objects, while the project brief states 192,472.
- The training-size requirement is 640 pixels. This must be enforced by the model/training configuration, with boxes transformed consistently.
