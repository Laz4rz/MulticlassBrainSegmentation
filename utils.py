import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import torch.nn.functional as F
from dataloader import MRIDataset

'''index	name
i00 Background
i01	Extra-axial CSF
i02	Gray Matter and developing cortical plate
i03	White matter and subplate
i04	Lateral ventricles
i05	Cerebellum
i06	Thalamus and putamen
i07	Brainstem	'''
class MRIimages(MRIDataset):
    def __init__(self, folderpath, idx: int, section: int):
        super().__init__(folderpath)
        self.idx = idx
        self.section = section

    def plot(self):
        main_img, masks = MRIDataset(self.folderpath)[self.idx]
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(nrows=1, ncols=2)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(main_img[ :, :,self.section], cmap='bone')
        #ax0.imshow(main_img[self.section, :,: ], cmap='bone')
        ax0.set_title("MAIN", fontsize=22, weight='bold', y=-0.2)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(masks[:,: , self.section], cmap='rainbow')
        #ax2.imshow(masks[self.section, :, :], cmap='rainbow')
        ax2.set_title("MASKS", fontsize=22, weight='bold', y=-0.2)
        labels = ['none', 'Extra-axial CSF', 'Gray Matter and developing cortical plate',
                                        'White matter and subplate', 'Lateral ventricles', 'Cerebellum',
                                        'Thalamus and putamen', 'Brainstem']
        colors = [plt.cm.rainbow(x) for x in np.linspace(0, 1, len(labels))]
        patches = [mpatches.Patch(color = colors[i], label = labels[i]) for i in range(len(labels))]
        plt.legend(handles = patches, loc='lower right')
        plt.show()


def soft_jaccard_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:

    assert output.size() == target.size()

    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


def iou_m(y_true,y_pred,smooth=0, eps=1e-7):
    # y_true shape = [B,W,H] B-batch_size
    # y_pred shape = [B,C,W,H] B-batch_size, C- N_CLASSES

    bs = y_true.size(0) # Batch size
    num_classes = y_pred.size(1) #N classes
    classes = [i for i in range(0, num_classes)] # classes = [0,...,7] nb_classes = 8
    dims = (0, 2)

    y_true = y_true.view(bs, -1)
    y_pred = y_pred.view(bs, num_classes, -1)
    y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # N, C, H*W

    scores = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=smooth, eps=eps, dims=dims)

    mask = y_true.sum(dims) > 0
    scores *= mask.float()

    scores = scores[classes]
    return scores.mean()

def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7,dims = None
    ) -> torch.Tensor:

    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

    return dice_score

def dice_m(y_true, y_pred, smooth=0, eps=1e-7):
    # y_true shape = [B,W,H] B-batch_size
    # y_pred shape = [B,C,W,H] B-batch_size, C- N_CLASSES

    bs = y_true.size(0)  # Batch size
    num_classes = y_pred.size(1)  # N classes
    classes = [i for i in range(0, num_classes)] # classes = [0,...,nb_classes-1] nb_c = 8 for multiclass
    dims = (0, 2)

    y_true = y_true.view(bs, -1)
    y_pred = y_pred.view(bs, num_classes, -1)
    y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # N, C, H*W

    scores = soft_dice_score(y_pred, y_true.type(y_pred.dtype), smooth=smooth, eps=eps, dims=dims)

    mask = y_true.sum(dims) > 0
    scores *= mask.float()

    scores = scores[classes]
    return scores.mean()

def mask_dim(current,n_classes=7):
    print(type(current))
    if type(current) is not 'numpy.ndarray':
        current = current.detach().numpy()
    current = np.concatenate([np.where(current == i, 1, 0) for i in range(0,n_classes)], 2)
    return torch.tensor(current).long().to('cuda:0' if torch.cuda.is_available() else 'cpu')


def mask_dim_torch(mask):
    mask = torch.unsqueeze(mask,1)

    mask.detach().numpy()
    mask = np.concatenate([np.where(mask == i, 1, 0) for i in range(1,8)], 1)
    return torch.tensor(mask).long()