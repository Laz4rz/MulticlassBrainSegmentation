import gc

import torch
import time
from model import UNET
from DataLoader2 import MRIDataset
from torch.autograd import Variable
from utils import prepare_one, dice_coef_multilabel, mask_dim, Jaccard_index_multiclass
from loss import dice_loss
import sys
import os
import numpy as np

if __name__ == '__main__':
    # TODO: Hyperparameters
    LEARNING_RATE = float(sys.argv[1])
    BATCH_SIZE = int(sys.argv[2])
    EPOCHS = int(sys.argv[3])
    TRAIN_AMOUNT = int(sys.argv[4])
    VALID_AMOUNT = int(sys.argv[5])
    MODEL_NAME = sys.argv[6]
    PATH = sys.argv[7]

    # TODO: setting UNET
    in_channels = 1
    out_channels = 7
    train_model = UNET(in_channels, out_channels)

    # TODO: training
    opt = torch.optim.Adam(train_model.parameters(), lr=LEARNING_RATE)
    def train(model, dane, loss_fn, optimizer, train, valid, model_name, batch=8 ,epochs=3):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        start = time.time()
        model.to(device)

        train_loss, valid_loss = [], []

        for epoch in range(epochs):
            model.train(True)
            swap_valid = True
            running_loss = 0.0
            running_dice = 0.0
            running_iou = 0.0
            step = 0
            patient = 0
            data = MRIDataset(PATH)
            length = len(dane)
            for x, y in data:
                patient += 1
                gc.collect()
                torch.cuda.empty_cache()
                loaded = prepare_one(x,y,batch)
                for x, y in loaded:
                    x = Variable(x.float()).to(device)
                    y = mask_dim(y)
                    y = torch.Tensor(y)
                    y = Variable(y).to(device)

                    step += 1

                    if patient <= train:
                        print(f'Batch: {batch}')
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = loss_fn(outputs, y, 7)
                        loss.backward()
                        optimizer.step()

                    elif patient < train+valid & patient > train:
                        if swap_valid:
                            epoch_loss = running_loss / length
                            epoch_dice = running_dice / length
                            epoch_iou = running_iou / length
                            train_loss.append(epoch_loss)
                            print('Train Loss: {:.4f}, Dice score: {}, Jaccard index:  {}'.format(epoch_loss, epoch_dice, epoch_iou))
                            model.train(False)
                            swap_valid = False
                            running_loss = 0.0
                            running_dice = 0.0
                            running_iou = 0.0
                        with torch.no_grad():
                            outputs = model(x)
                            loss = loss_fn(outputs, y, 7)

                    else:
                        epoch_loss = running_loss / length
                        epoch_dice = running_dice / length
                        epoch_iou = running_iou / length
                        valid_loss.append(epoch_loss)
                        print('Valid Loss: {:.4f}, Dice score: {}, Jaccard index:  {}'.format(epoch_loss, epoch_dice, epoch_iou))
                        swap_valid = True

                    dice_score = dice_coef_multilabel(y, outputs, 7)
                    iou = Jaccard_index_multiclass(y, outputs, 7)
                    running_loss += np.float16(loss*batch)
                    running_dice += np.float16(dice_score*batch)
                    running_iou  += np.float16(iou*batch)

                    if step % 10 == 0:
                        print('Current step: {},  Loss: {},  Dice score: {}, Jaccard index: {}'.format(step, loss, dice_score, iou))
            torch.save(model.state_dict(), os.path.join(model_name))
        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return train_loss, valid_loss

    train, valid = train(train_model, PATH, dice_loss, opt, TRAIN_AMOUNT, VALID_AMOUNT, MODEL_NAME,BATCH_SIZE, EPOCHS)