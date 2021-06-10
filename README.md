# Multiclass Brain Segmentation
My code snippets from the UNET based multiclass brain segmentation model made for the <a href="https://feta-2021.grand-challenge.org/">Fetal Brain Tissue Annotation and Segmentation Challenge (FeTA), MICCAI 2021</a>.

As the challenge is still ongoing the repository has only less "custom made" elements like data loader or the training script.

## Data Loader
You can create an iterative data object by passing the data folder path to the MRIDataset class as:
``` 
data = MRIDataset(path)
for patient in data:
  print(patient.shape)
```
By each iteration through data object you get next 3D MRI image as a numpy array of 256x256x256 shape.  
Object length is taken from the filtered amount of folders in the path directory.
```len([dir for dir in next(os.walk(self.folderpath))[1] if dir.startswith('sub')])```

