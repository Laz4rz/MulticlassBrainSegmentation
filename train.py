from torch.utils.tensorboard import SummaryWriter
from dataloader import MRIDataset
from torch.autograd import Variable
from utils import iou_m, dice_m
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import torch
import time
import datetime
import argparse
import numpy as np
import warnings
import segmentation_models_pytorch as smp
import albumentations as A

warnings.filterwarnings("ignore")


train_model = smp.FPN(
    encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights= 'imagenet',     # use `imagenet` pre-trained weights for encoder initialization
    decoder_merge_policy= 'add',
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=8,                      # model output channels (number of classes in your dataset)
)


WEIGHTS  = torch.tensor([ 0.14765405, 10.30508434,  7.33515858,  3.16833475, 21.9364291,  54.936772,
 27.94518702, 77.0925504 ]).float()

WEIGHTS = F.normalize(WEIGHTS,p=1,dim = 0)

gamma = np.reciprocal(WEIGHTS)

p1 = 0.95
p2 = 0.85
p3 = 0.75

transform = A.Compose([

    A.Crop(x_min=10, x_max=170, y_min=10, y_max=170, p=1),
    A.OneOf([
        A.HorizontalFlip(p=0.9),
        A.Flip(p=0.6),
        ], p=p2),
    A.RandomRotate90(p=p3),
    A.ElasticTransform(p=0.4),
    A.RandomGamma(gamma_limit=(50,150),p=p2),
    A.Normalize(mean = 0,std = 1,max_pixel_value=255.0,p = 1),

                    ])

kfold = KFold(n_splits=5, shuffle=False)
in_channels = 1
out_channels = 8
writer = SummaryWriter(f'runs/FeTA/')
PATH = f"C:/Users/Filip/Desktop/feta_model_1.pth"

def train(model, path, patients, model_name, batch=8, epochs=3, lr=1e-3,wd = 1e-3):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    folding_index = [i for i in range(patients)]
    data_train = MRIDataset(path,transform)
    data_val = MRIDataset(path)
    criterion = smp.losses.FocalLoss(mode='multiclass',alpha = 0.25, gamma = 2.0)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(folding_index)):

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
        scheduler = ReduceLROnPlateau(optimizer,mode='min',factor = 0.1,patience = 4)
        print(f'Fold number: {fold + 1}/{kfold.get_n_splits()}')
        val_patients = len(test_ids)
        train_patients = len(train_ids)

        for epoch in range(epochs):
            start_time_epoch = time.time()
            print(f'Starting epoch number: {epoch + 1}')
            running_loss = 0.0
            running_dice = 0.0
            running_jac = 0.0

            model.train()
            patient = 0
            for i in train_ids:
                x, y = data_train[i]
                train_loader = DataLoader(list(zip(x, y)), shuffle=True, batch_size=batch)
                patient += 1
                for batch_idx, (images, masks) in enumerate(train_loader):
                    optimizer.zero_grad()
                    images = torch.unsqueeze(images, 1)  # reshape  BxWxH -> B x N_INPUT_CH x W x H
                    images = images.float()

                    images = Variable(images).to(device)
                    masks = Variable(masks).to(device)

                    output_mask = model(images)
                    loss = criterion(output_mask, masks.long())

                    loss.backward()
                    optimizer.step()

                    jac = iou_m(masks.long(), torch.softmax(output_mask, dim = 1))
                    dice = dice_m(masks.long(), torch.softmax(output_mask,dim = 1))

                    running_dice += dice.item()
                    running_jac += jac.item()
                    running_loss += loss.item()

                    print(" ", end="")
                    print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
                          f" Loss: {loss.item():.5f}"
                          f" Jaccard: {jac.item():.5f}"
                          f" Dice: {dice.item():.5f}"
                          )
                print(f'Patient {patient} finished')
            print(f"Training process has finished in {datetime.timedelta(seconds= (time.time() - start_time_epoch))} Saving training model...")

            print("Starting evaluation")

            save_path = f"C:/Users/Filip/Desktop/{model_name}_{fold + 1}.pth"
            torch.save(model.state_dict(), save_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,

            }, save_path)

            val_running_jac = 0.0
            val_running_loss = 0.0
            val_running_dice = 0.0


            for i in test_ids:
                x, y = data_val[i]
                test_loader = DataLoader(list(zip(x, y)), shuffle=True, batch_size=batch)

                for batch_idx, (images, masks) in enumerate(test_loader):
                    images = torch.unsqueeze(images,1)  # reshape  BxWxH -> B x N_INPUT_CH x W x H
                    images = torch.tensor(images)

                    images = images.float()

                    masks = masks.long()
                    images = Variable(images).to(device)
                    masks = Variable(masks).to(device)
                    with torch.no_grad():
                        output_mask = model(images)
                        loss = criterion(output_mask.float(), masks)

                    jac = iou_m(masks, torch.softmax(output_mask, dim = 1))
                    dice = dice_m(masks, torch.softmax(output_mask, dim = 1))

                    val_running_dice += dice.item()
                    val_running_jac += jac.item()
                    val_running_loss += loss.item()

            train_loss = running_loss / (len(train_loader) * train_patients)
            test_loss = val_running_loss / (len(test_loader) * val_patients)

            train_jac = running_jac / (len(train_loader) * train_patients)
            test_jac = val_running_jac / (len(test_loader) * val_patients)

            train_dice = running_dice / (len(train_loader) * train_patients)
            test_dice = val_running_dice / (len(test_loader) * val_patients)

            writer.add_scalar('Train Loss', train_loss, global_step=epoch + 1)
            writer.add_scalar('Train mIoU', train_jac, global_step=epoch + 1)
            writer.add_scalar('Train Dice', train_dice, global_step=epoch + 1)

            writer.add_scalar('Valid Loss',test_loss ,global_step=epoch+1)
            writer.add_scalar('Valid mIoU', test_jac, global_step=epoch + 1)
            writer.add_scalar('Valid Dice', test_dice, global_step=epoch + 1)

            print('    ', end='')

            print(f"Train Loss: {train_loss:.4f}"
                  f" Train Jaccard: {train_jac:.4f}"
                  f" Train Dice: {train_dice:.4f}"
                  f" Test Loss: {test_loss:.4f}"
                  f" Test Jaccard: {test_jac:.4f}"
                  f" Test Dice: {test_dice:.4f}"
                  f" Learning rate: {optimizer.param_groups[0]['lr']}")
            scheduler.step(test_loss)


    print(f"Training is finished!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Segmentation Network on Fetal Dataset.')
    parser.add_argument('--path',
                        type=str,
                        default=r'C:\Users\Filip\Downloads\feta_2.1\feta_2.1',
                        help='Path to the data')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs')
    parser.add_argument('--patients',
                        type=int,
                        default=80,
                        help='Number of patients')

    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Number of learning rate')
    parser.add_argument('--model_name',
                        type=str,
                        default='feta_model',
                        help='Model name')
    parser.add_argument('--wd',
                        type=float,
                        default=1e-4,
                        help='Weight decay')
    args = parser.parse_args()
    train(train_model, args.path, args.patients, args.model_name, args.batch_size,
          args.epochs, args.lr,args.wd)

