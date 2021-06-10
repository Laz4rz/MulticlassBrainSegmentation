# Multiclass Brain Segmentation
My code snippets from the UNET based multiclass brain segmentation model made for the <a href="https://feta-2021.grand-challenge.org/">Fetal Brain Tissue Annotation and Segmentation Challenge (FeTA), MICCAI 2021</a>.

As the challenge is still ongoing the repository has only less "custom made" elements like data loader or the training script.

## Data Loader
You can create an iterative data object by passing the data folder path to the MRIDataset class as:
``` 
data = MRIDataset
for patient in data:
  print(patient.shape)
```
By each iteration through data object you get next 3D MRI image as a numpy array of 256x256x256 shape.
