# DIOR object detection

## Training in Colab

1. Create your own Google Colab notebook and name it for your model or experiment, such as `MobileNet_Baseline.ipynb`. Use [ResNet50_Colab_Training.ipynb](notebooks/ResNet50_Colab_Training.ipynb) as a reference: copy the Drive mount, repository setup, Git pull, dataset, and training helper cells you need. Follow the same cell format for your model, but use your own notebook.
2. Select a GPU runtime. Run the Drive mount, repository setup, Git pull, dataset download and extraction, and training helper cells in order. Authorize your own Google Drive account when prompted. Check the dataset folder link and use a copy you can access.
3. Create and run your own training cells for the model you are working on. For example, start a new MobileNet run with:

   ```python
   run_one_epoch('mobilenet')
   ```

   Use `resnet50` instead of `mobilenet` to start a new ResNet run. The helper trains one epoch and saves its checkpoint, cumulative training log, and loss plot under `MyDrive/DIOR_checkpoints/<backbone>/`.
4. To continue, create a cell pointing `resume` to the latest checkpoint in your Drive. The matching `training_log.csv` must be in the same model output folder. For example:

   ```python
   run_one_epoch(
       'mobilenet',
       resume='/content/drive/MyDrive/DIOR_checkpoints/mobilenet/checkpoint_epoch_01.pth',
   )
   ```

Create your own cells and change checkpoint and output paths for your model and Drive. The existing ResNet resume and evaluation cells refer to Vivek's checkpoints. The older cells calling `run_training(...)` are outdated; use `run_one_epoch(...)` instead. Dataset images, annotations, and model checkpoints are not included in the repository.
