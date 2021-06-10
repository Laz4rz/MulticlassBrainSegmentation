# Multiclass Brain Segmentation
My code snippets from the UNET based multiclass brain segmentation model made for the <a href="https://feta-2021.grand-challenge.org/">Fetal Brain Tissue Annotation and Segmentation Challenge (FeTA), MICCAI 2021</a>.

As the challenge is still ongoing the repository has only less "custom made" elements like data loader or the training script.

## Training Script
As stated before - the repository has only some parts of the code, but if you had them all you could run the training by:
```python
python train5.py LEARNING_RATE BATCH_SIZE EPOCHS TRAIN_AMOUNT VALID_AMOUNT SAVE_MODEL_NAME DATA_PATH
```
where BATCH_SIZE - amount of pictures to go into one training step, TRAIN_AMOUNT - number of training patients to load, VALID_AMOUNT - number of validation patients to load, SAVE_MODEL - where to save the trained model, DATA_PATH - what goes to the Data Loader object.

**Resolved problems**:
- RAM management
Made DataLoader object iterable so we are able to load the data patient by patient during training - resulted in significant RAM usage decrease.
- VRAM management
Metrics variables HABE TO be changed to float() so they do not drag gradient vectors with them - significant VRAM usage decrease.
----
## Data Loader
You can create an iterative data object by passing the data folder path to the MRIDataset class as:
```python
data = MRIDataset(path)
for patient in data:
  print(patient.shape)
```
By each iteration through data object you get next 3D MRI image as a numpy array of 256 x 256 x 256 shape.  
Object length is taken from the filtered length of folders in the path directory list.  
```python
def __len__(self):
    return len([dir for dir in next(os.walk(self.folderpath))[1] if dir.startswith('sub')])
```
## Utils
We use utils functions to prepare loaded data for the UNET train/valid pass.
