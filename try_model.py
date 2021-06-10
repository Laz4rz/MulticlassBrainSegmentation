import torch
from model import UNET
from DataLoader import MRIDataset
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PATIENTS = 1
path = r'C:\Users\miki\Desktop\FetaLatest\feta_2.1'
data = MRIDataset(path,PATIENTS).load()

imgs = data[0,0,:,:,:]
masks = data[0,1,:,:,:]

# masks_path = r'C:\Users\miki\Desktop\FetaLatest\feta_2.1\sub-080\anat\sub-080_rec-irtk_dseg.json'
# imgs_path = r'C:\Users\miki\Desktop\FetaLatest\feta_2.1\sub-080\anat\sub-080_rec-irtk_T2w.json'
#
# filepath = os.path.join(imgs_path)
# mask_filepath = os.path.join(masks_path)
#
# main_img = nib.load(filepath)
# main_img = np.asarray(main_img.dataobj)
#
# mask_img = nib.load(mask_filepath)
# mask_img = np.asarray(mask_img.dataobj)


PATH = r'C:\Users\miki\Desktop\FetaLatest\model_colab2.pth'
model = UNET(1,7)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

current = masks[:,33,:]
current = np.expand_dims(current,0)
current = np.concatenate([np.where(current == i, 1, 0) for i in range(1,8)], 0)
print(current.shape)


check = imgs[:,33,:]
plt.imshow(check,cmap='bone')
print(check.shape)
check = np.expand_dims(check, 0)
check = np.expand_dims(check, 0)
print(check.shape)
check = torch.Tensor(check)
pred = model(check)
pred = pred.detach().numpy()
for i in range(0,7):
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1, 2, 1)
    ax.title.set_text('model')
    ax.imshow(pred[0,i,:,:])
    ax = fig.add_subplot(1, 2, 2)
    ax.title.set_text('label')
    ax.imshow(current[i,:,:])
    fig.show()


# current = np.copy(masks[100,:,:])
# print(current.shape)
#
# current = np.expand_dims(current, 0)
# print(current.shape)

# current = np.concatenate(
#     (np.where(current == 1, 1, 0),
#      np.where(current == 2, 1, 0),
#      np.where(current == 3, 1, 0),
#      np.where(current == 4, 1, 0),
#      np.where(current == 5, 1, 0),
#      np.where(current == 6, 1, 0),
#      np.where(current == 7, 1, 0)), 0
# )

# current = np.concatenate([np.where(current == i, 1, 0) for i in range(1,8)], 0)


# print(current.shape)
#
# plt.imshow(current[1,:,:])
# plt.show()
# for i in range(1,8):
#     temp = np.copy(masks[100,:,:])
#     temp[temp != i] = int(0)
#     temp[temp == i] = int(1)
#     plt.imshow(temp)
#     plt.show()


# plt.imshow(masks[100,:,:],cmap='Greys',alpha=0.6)
# plt.imshow(pred.detach().numpy()[0,0,:,:],alpha=0.3)
# plt.show()