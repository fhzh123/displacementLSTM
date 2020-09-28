import os
import math
import time
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

# Import PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

from dataset import CustomDataset
from model import model
from optimizer import WarmupLinearSchedule

def main(args):
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Custom dataset & dataloader setting
    datasets = {
        'train': CustomDataset(args.input_path, args.look_back, isTrain=True),
        'valid': CustomDataset(args.input_path, args.look_back, isTrain=False)
    }
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=True),
        'valid': DataLoader(datasets['valid'], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers, drop_last=True),
    }

    # Model
    model_ = model(d_model=args.d_model, n_head=args.n_head,
                   dim_feedforward=args.dim_feedforward, n_layers=args.n_layers,
                   dropout=args.dropout)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_.parameters()), lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(dataloaders['train'])*3, 
                                     t_total=len(dataloaders['train'])*args.num_epochs)
    model_.to(device)

    ## Training
    # Initialize
    best_loss = 9999999999
    early_stop = False

    # Train
    for epoch in range(args.num_epochs):
        start_time_e = time.time()
        freq = -1
        print('#'*30)
        print('Epoch {}/{}'.format(epoch + 1, args.num_epochs))

        if early_stop:
            print('Early Stopping!!!')
            break

        for phase in ['train', 'valid']:
            if phase == 'train':
                model_.train()
            else:
                model_.eval()
            running_loss = 0.0
            running_corrects = 0
            start_time = time.time()

            # Iterate over data
            for i, (x, y) in enumerate(dataloaders[phase]):

                x = x.to(device)
                y = y.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_(x).squeeze(2)
                    loss = criterion(outputs, y)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch_utils.clip_grad_norm_(model_.parameters(), 
                                                    args.max_grad_norm)
                        optimizer.step() 
                        scheduler.step()
                        # Print loss value only training
                        freq += 1
                        if freq == args.print_freq:
                            total_loss = loss.item()
                            print("[Epoch:%d][%d/%d] train_loss:%5.3f | spend_time:%5.2fmin"
                                    % (epoch+1, i, len(dataloaders['train']), total_loss, (time.time() - start_time_e) / 60))
                            freq = 0

                # Statistics
                running_loss += loss.item() * x.size(0)
                # running_corrects += torch.sum(preds == y.data)

            # Epoch loss calculate
            epoch_loss = running_loss / len(datasets[phase])

            if phase == 'valid' and epoch_loss < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                print("[!] saving model...")
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                torch.save(model_.state_dict(),
                           os.path.join(args.save_path, f'epoch_{epoch+1}_model_testing.pt'))

            if phase == 'train' and epoch_loss < 0.001:
                early_stop = True
                
            spend_time = (time.time() - start_time) / 60
            print('{} Loss: {:.4f} Time: {:.3f}min'.format(phase, epoch_loss, spend_time))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='argparser')
    # Path Setting
    parser.add_argument('--save_path', type=str, default='./save')
    # Pre Setting
    parser.add_argument('--input_path', type=str, default='./ACCSTRAIN_DISPL_20200821.mat')
    parser.add_argument('--look_back', type=int, default=6000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    # Model Setting
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.2)
    # Training Setting
    parser.add_argument('--num_epochs', type=int, default=300, help='The number of epoch')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate setting')
    parser.add_argument('--lr_step_size', type=int, default=60, help='Learning rate scheduling step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='Learning rate decay scheduling per lr_step_size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_grad_norm', type=int, default=5, help='Gradient clipping max norm')
    parser.add_argument('--print_freq', type=int, default=100, help='Print train loss frequency; Default is 100')
    args = parser.parse_args()
    main(args)