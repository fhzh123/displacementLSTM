import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat

import torch
from torchvision import transforms
from torch.utils.data import Dataset

def create_dataset(dataset, look_back=1):
    dataX, dataY = list(), list()
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back), 0:2])
        dataY.append(dataset[i:(i+look_back), 2])
    return dataX, dataY

class CustomDataset(Dataset):
    def __init__(self, input_path, look_back, isTrain=True):
        df_temp = loadmat(input_path)
        self.df = df_temp['ACCSTRAIN_DISPL']
        self.train_size = int(len(self.df) * 0.7)
        self.train = self.df[0:self.train_size,:]
        self.valid = self.df[self.train_size:,:]

        if isTrain:
            self.X, self.Y = create_dataset(self.train, look_back)
            self.num_data = len(self.X)
        else:
            self.X, self.Y = create_dataset(self.valid, look_back)
            self.num_data = len(self.X)

    def __getitem__(self, index):
        x_out = self.X[index].astype(np.float32)
        y_out = self.Y[index].astype(np.float32)
        return x_out, y_out

    def __len__(self):
        return self.num_data