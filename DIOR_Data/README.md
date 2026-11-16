# DIOR Dataset Preparation

ECS 174 Final Project: Satellite Object Detection with Faster R-CNN

## Files
- `dataset.py`: PyTorch dataset loader for DIOR images and XML annotations.
- `DIOR_Data_Prep.ipynb`: Dataset verification, loader testing, and visualizations.
- `Main/`: Official train (5,862), validation (5,863), and test (11,738) splits.
- `class_distribution.png`: Distribution of 192,518 annotated objects across 20 classes.
- `ground_truth/`: 20 images with bounding boxes drawn from XML annotations.

## Setup
1. Download and extract the DIOR dataset.
2. Place `dataset.py` and `Main/` inside the extracted DIOR folder.
3. Install PyTorch, torchvision, and Pillow.

## Usage

```python
from dataset import DIORDataset, collate_fn
from torch.utils.data import DataLoader

dataset = DIORDataset("/content/DIOR", split="train")
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
```
## Important Reminder
`dataset.py` expects the below structure when running:

DIOR/
├── dataset.py
├── Main/
├── JPEGImages-trainval/
├── JPEGImages-test/
└── Annotations/
    └── Horizontal Bounding Boxes/

> You must place your files alongside their extracted DIOR images and annotations, or adjust the paths accordingly

**Note:** The full DIOR dataset is not included in this ZIP. 

**Note** dataset.py must exist in Colab's Python import path. If you keep it in /content/DIOR/, add below before import.
```python
import sys
sys.path.append("/content/DIOR")
```

