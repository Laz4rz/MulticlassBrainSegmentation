import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


'''index	name
i01	Extra-axial CSF
i02	Gray Matter and developing cortical plate
i03	White matter and subplate
i04	Lateral ventricles
i05	Cerebellum
i06	Thalamus and putamen
i07	Brainstem	'''


# class MRIimages(MRIDataset):
#     def __init__(self, folderpath, idx: str, section: int):
#         super().__init__(folderpath)
#         self.idx = idx
#         self.section = section
#
#     def MRIimages(self):
#         main_img, masks = MRIDataset(self.folderpath)[self.idx]
#         fig = plt.figure(figsize=(20, 12))
#         gs = gridspec.GridSpec(nrows=1, ncols=2)
#
#         ax0 = fig.add_subplot(gs[0, 0])
#         ax0.imshow(main_img[:, :, self.section], cmap='bone')
#         ax0.set_title("MAIN", fontsize=22, weight='bold', y=-0.2)
#
#         ax2 = fig.add_subplot(gs[0, 1])
#         ax2.imshow(masks[:, :, self.section], cmap='rainbow')
#         ax2.set_title("MASKS", fontsize=22, weight='bold', y=-0.2)
#         labels = ['none', 'Extra-axial CSF', 'Gray Matter and developing cortical plate',
#                                         'White matter and subplate', 'Lateral ventricles', 'Cerebellum',
#                                         'Thalamus and putamen', 'Brainstem']
#         colors = [plt.cm.rainbow(x) for x in np.linspace(0, 1, len(labels))]
#         patches = [mpatches.Patch(color = colors[i], label = labels[i]) for i in range(len(labels))]
#         plt.legend(handles = patches, loc='lower right')
#
#
#         plt.show()

# class Rotate(MRIDataset):
#
#     def __init__(self, folderpath, idx: str, rot_prob, angle):
#         # rot_prob in [0,1]
#         super().__init__(folderpath)
#         self.angle = angle
#         self.idx = idx
#         self.rot_prob = rot_prob
#
#     def rot(self, section):
#         image, mask = MRIDataset(self.folderpath)[self.idx]
#         image = image[:, :, section]
#         mask = mask[:, :, section]
#         if np.random.rand() > self.rot_prob:
#             angle = np.random.uniform(low=-self.angle, high=self.angle)
#             image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")
#             mask = rotate(
#                 mask, angle, resize=False, order=0, preserve_range=True, mode="constant"
#             )
#             return image, mask
#         return image, mask


# class HorizontalFlip(MRIDataset):
#
#     def __init__(self, folderpath, idx, flip_prob):
#         # flip_prob in [0, 1]
#         super().__init__(folderpath)
#         self.idx = idx
#         self.flip_prob = flip_prob
#
#     def flip(self, section):
#         image, mask = MRIDataset(self.folderpath)[self.idx]
#         image = image[:, :, section]
#         mask = mask[:, :, section]
#         if np.random.rand() > self.flip_prob:
#             return image, mask
#
#         image = np.fliplr(image).copy()
#         mask = np.fliplr(mask).copy()
#
#         return image, mask


def dice_coef(pred,target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    dice = ((numerator + 1) / (denominator + 1))
    return dice.mean()


def dice_coef_multilabel(y_true, y_pred, n_class: int):
    dice = 0.0
    for index in range(n_class):
        dice += dice_coef(y_true[:,index,:,:], y_pred[:,index,:,:])
    return dice/n_class # taking average


def Jaccard_index(pred,target):
    intersection = abs(torch.sum(pred * target))
    union = abs(torch.sum(pred + target) - intersection)
    iou = intersection/union
    return iou.mean()


def Jaccard_index_multiclass(y_pred,y_true,n_class: int):
    iou = 0.0
    for index in range(n_class):
        iou += Jaccard_index(y_true[:,index,:,:], y_pred[:,index,:,:]) # TODO indexing
    return iou/n_class # taking average


def prepare_particular(data, train, valid=0, train_mode = True):
    if train_mode:
        imgs = data[:train,0,:,:,:]
        masks = data[:train,1,:,:,:]
        imgs = np.dstack([imgs[i,:,:,:] for i in range(train)])
        masks = np.dstack([masks[i,:,:,:] for i in range(train)])
    else:
        imgs = data[train:,0,:,:,:]
        masks = data[train:,1,:,:,:]
        imgs = np.dstack([imgs[i,:,:,:] for i in range(valid)])
        masks = np.dstack([masks[i,:,:,:] for i in range(valid)])
    imgs = np.expand_dims(imgs,1)
    imgs = np.moveaxis(imgs, 3, 0)
    imgs = np.moveaxis(imgs, 1, 2)
    masks = np.expand_dims(masks,1)
    masks = np.moveaxis(masks, 3, 0)
    masks = np.moveaxis(masks, 1, 2)
    return imgs, masks


def prepare_data(data,train,valid, batch):
    imgs, masks = prepare_particular(data, train, valid)
    imgs_valid, masks_valid = prepare_particular(data, train, valid, False)
    part_train = list(zip(imgs,masks))
    part_valid = list(zip(imgs_valid,masks_valid))
    train_dl = DataLoader(part_train, batch_size=batch, shuffle=True)
    valid_dl = DataLoader(part_valid, batch_size=batch, shuffle=True)
    return train_dl, valid_dl


def mask_dim(current):
    current = np.concatenate([np.where(current == i, 1, 0) for i in range(1,8)], 1)
    return current


def check_preparation(image_number, imgs, masks, imgs_valid, masks_valid):
    plt.imshow(imgs[image_number,0,:,:],cmap='bone')
    plt.imshow(masks[image_number,0,:,:], alpha = 0.6, cmap='rainbow')
    plt.show()
    plt.imshow(imgs_valid[image_number,0,:,:],cmap='bone')
    plt.imshow(masks_valid[image_number,0,:,:], alpha = 0.6, cmap='rainbow')
    plt.show()
    return True


def prepare_one(x,y, batch):
    x = np.expand_dims(x, 1)
    #print(x.shape)
    x = np.moveaxis(x, 3, 0)
    #print(x.shape)
    x = np.moveaxis(x, 1, 2)
    #print(x.shape)
    y = np.expand_dims(y, 1)
    y = np.moveaxis(y, 3, 0)
    y = np.moveaxis(y, 1, 2)
    zipped = list(zip(x,y))
    ready = DataLoader(zipped, batch_size=batch)
    return ready
    # return x, y

