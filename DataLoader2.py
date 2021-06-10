import numpy as np
import nibabel as nib
import os
from utils import prepare_one, mask_dim
import matplotlib.pyplot as plt

class MRIDataset():
    def __init__(self, folderpath):
        self.folderpath = folderpath

    def __len__(self):
        return len([dir for dir in next(os.walk(self.folderpath))[1] if dir.startswith('sub')])

    def __getitem__(self, index):
        index += 1
        if index < 10: index = str(f'00{index}')
        else: index = str(f'0{index}')
        filepath = os.path.join(self.folderpath, "sub-" + index, "anat", "sub-" + index + "_rec-mial_T2w.nii.gz")
        mask_filepath = os.path.join(self.folderpath, "sub-" + index, "anat", "sub-" + index + "_rec-mial_dseg.nii.gz")
        if os.path.exists(filepath):
            main_img = nib.load(filepath)
            main_img = np.asarray(main_img.dataobj)
            mask_img = nib.load(mask_filepath)
            mask_img = np.asarray(mask_img.dataobj)
        else:
            filepath = os.path.join(self.folderpath, "sub-" + index, "anat", "sub-" + index + "_rec-irtk_T2w.nii.gz")
            mask_filepath = os.path.join(self.folderpath, "sub-" + index, "anat", "sub-" + index + "_rec-irtk_dseg.nii.gz")
            if os.path.exists(filepath):
                main_img = nib.load(filepath)
                main_img = np.asarray(main_img.dataobj)
                mask_img = nib.load(mask_filepath)
                mask_img = np.asarray(mask_img.dataobj)
            else:
                raise StopIteration
        return main_img, mask_img



dane = MRIDataset(r'feta_2.1')
index = 1
x, y = dane[50]
plt.imshow(x[100,:,:])
plt.show()

# for x, y in dane:
#     print(f'Patient index: {index}, x: {x.shape}, y: {y.shape}')
#     loaded = prepare_one(x, y, 16)
#     for i, (x, y) in enumerate(loaded):
#         print(f'Batch index: {i+1}, x: {x.shape}, y: {y.shape}, mask_as_dim: {mask_dim(y).shape}')
#         fig = plt.figure(figsize=(5,5))
#         fig.suptitle('Batch '+str(i))
#         for i in range(0,7):
#             ax = fig.add_subplot(4,2,i+1)
#             ax.imshow(mask_dim(y)[0,i,:,:])
#         fig.show()
#     break