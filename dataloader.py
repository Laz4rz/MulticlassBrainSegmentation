import numpy as np
import nibabel as nib
import os

class MRIDataset():
    def __init__(self, folderpath,transform = None):
        self.folderpath = folderpath
        self.transform = transform

    def __len__(self):
        return len([dir for dir in next(os.walk(self.folderpath))[1] if dir.startswith('sub')])

    def __getitem__(self, index):

        index += 1
        if index < 10:
            index = str(f'00{index}')
        else:
            index = str(f'0{index}')
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

        temp_img, temp_mask = list(),list()
        if self.transform is not None:
            for (image, mask) in zip(main_img,mask_img):

                transformed = self.transform(image=image, mask=mask)
                temp_img.append(transformed["image"])
                temp_mask.append(transformed["mask"])

            return np.array(temp_img,dtype=np.float32), np.array(temp_mask,dtype=np.float32)

        else:
            return main_img,mask_img