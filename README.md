# Multiclass Brain Segmentation
My code snippets from the UNET based multiclass brain segmentation model made for the <a href="https://feta-2021.grand-challenge.org/">Fetal Brain Tissue Annotation and Segmentation Challenge (FeTA), MICCAI 2021</a>.

As the challenge is still ongoing the repository has only less "custom made" elements like data loader or the training script.

## Training Script
As stated before - the repository has only some parts of the code, but if you had them all you could run the training by:
```python
python train5.py LEARNING_RATE BATCH_SIZE EPOCHS TRAIN_AMOUNT VALID_AMOUNT SAVE_MODEL_NAME DATA_PATH
```
where _BATCH_SIZE_ - amount of pictures to go into one training step, _TRAIN_AMOUNT_ - number of training patients to load, _VALID_AMOUNT_ - number of validation patients to load, _SAVE_MODEL_ - where to save the trained model, _DATA_PATH_ - what goes to the Data Loader object.

**Resolved problems**:
- RAM management
Made DataLoader object iterable so we are able to load the data patient by patient during training - resulted in significant RAM usage decrease.
- VRAM management
Metrics variables HABE TO be changed to float() so they do not drag gradient vectors with them - significant VRAM usage decrease.

**To do:**  
- Cross Validation
- Data Augmentation
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
----
## Utils
We use utils functions to prepare loaded data for the UNET train/valid pass.
```python
def prepare_one(x,y, batch):
    x = np.expand_dims(x, 1)
    x = np.moveaxis(x, 3, 0)
    x = np.moveaxis(x, 1, 2)
    y = np.expand_dims(y, 1)
    y = np.moveaxis(y, 3, 0)
    y = np.moveaxis(y, 1, 2)
    zipped = list(zip(x,y))
    ready = DataLoader(zipped, batch_size=batch)
    return ready
```
Takes a list of images as _x_, masks as _y_ and number of images and masks to pack in a training/valid batch. Gives torch.DataLoader objects in return.  
```python
def mask_dim(current):
    current = np.concatenate([np.where(current == i, 1, 0) for i in range(1,8)], 1)
    return current
```
Takes mask array with shape of 1x256x256 and extractes mask 1-7 values as new dimensions with value of 1, so return shape is 7x256x256.  
Example:  
```
input = [[1 2 1]] -> output = mask_dim(input) -> output = [[1 0 1], [0 1 0]]  
         [0 1 2]                                           [0 1 0]  [0 0 1]  
         [1 2 0]                                           [1 0 0]  [0 0 0]  
```
