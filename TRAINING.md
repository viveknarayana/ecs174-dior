# Training

`config.py` contains the shared defaults. `train.py` supports both required backbones so the ResNet-50 and MobileNet experiments use the same training implementation.

For a Google Colab training example, see [`notebooks/ResNet50_Colab_Training.ipynb`](notebooks/ResNet50_Colab_Training.ipynb). Copy the setup cells you need into your own model-specific notebook.

The default dataset root is the repository directory and expects the extracted folders `JPEGImages-trainval/`, `JPEGImages-test/`, `Annotations/`, and `ImageSets/Main/`. If the files live elsewhere, pass that directory with `--data-root`.

Run the small end-to-end development job first:

```bash
python3 train.py --data-root /path/to/DIOR --backbone resnet50 --tiny-test --tiny-train-images 2 --tiny-validation-images 1 --epochs 1
```

The Colab notebook smoke test uses this command with a temporary output directory. Training runs save `loss_curves.png` in the same run directory as their checkpoints and CSV log. The plot shows training and validation loss by epoch.

`train.py` uses Matplotlib to write the loss plot. Install it with `python3 -m pip install matplotlib` when training outside Colab.

Run the full experiments with:

```bash
python3 train.py --data-root /path/to/DIOR --backbone resnet50
python3 train.py --data-root /path/to/DIOR --backbone mobilenet
```

Both train on the combined train and validation splits by default, without a separate validation pass. Use `--train-split train` to train on train only and monitor validation loss. The tiny mode uses up to 300 training and 64 validation images. Results are written under `outputs/<backbone>/`, including per-epoch checkpoints, `best.pth`, and a CSV training log. Resume with `--resume outputs/resnet50/checkpoint_epoch_03.pth` (adjust the path as needed).

The model uses COCO-pretrained Faster R-CNN weights, replaces the ROI box predictor for 21 outputs (background plus the 20 ordered DIOR classes), and keeps all backbone layers trainable. Torchvision's default internal resize uses a minimum side of 800 and a maximum side of 1333 pixels. DIOR's 800×800 images remain 800×800; the dataset loader retains original image coordinates and boxes, which the model transforms together.
