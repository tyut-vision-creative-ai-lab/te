from __future__ import print_function, division
import os
import scipy.io as scio
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import warnings
import scipy.io as scio
import numpy as np
import random
import torch.nn.functional as F
from utils.Functions import ExtractPatch,ResizeLF
warnings.filterwarnings("ignore")
plt.ion()
import mat73


# Loading data
class LoadTestData(Dataset):
    """Light Field test dataset."""

    def __init__(self, opt):
        super(LoadTestData, self).__init__()
        dataSet = mat73.loadmat(opt.dataPath)
        self.LFSet = dataSet['lf']  # [u, v, x, y,ind]
        self.noiLFSet = dataSet['noilf_{}'.format(opt.sigma)]  # [u, v, x, y,ind]
        self.lfNameSet = dataSet['LF_name']  # [index, 1]

    def __getitem__(self, idx):
        LF = self.LFSet[:, :, :, :, :, idx]  # [u, v, x, y]
        noiLF = self.noiLFSet[:, :, :, :, :, idx]
        lfNameSet = np.array(self.lfNameSet[idx], dtype=np.int8)
        lfName = ''.join([(chr(lfNameSet[0][c])) for c in range(lfNameSet[0].shape[0])])
        # lfName = str(idx)
        LF = torch.from_numpy(LF[:, :, :, :, :, np.newaxis].astype(np.float32) / 255.0)
        noiLF = torch.from_numpy(noiLF[:, :, :, :, :, np.newaxis].astype(np.float32) / 255.0)  # [u,v,x,y,c]
        sample = {'LF': LF, 'noiLF': noiLF, 'lfName': lfName}
        return sample

    def __len__(self):
        return self.LFSet.shape[5]


# Loading data
class LoadTrainData(Dataset):
    """Light Field train dataset."""

    def __init__(self, opt):
        super(LoadTrainData, self).__init__()
        dataSet = mat73.loadmat(opt.dataPath)
        self.LFSet = dataSet['lf']  # [u, v, x, y, ind]
        self.noiLFSet = dataSet['noilf_{}'.format(opt.sigma)]
        self.patchSize = opt.patchSize

    def __getitem__(self, idx):
        LF = self.LFSet[:, :, :, :, :, idx]  # [u, v, x, y]
        noiLF = self.noiLFSet[:, :, :, :, :, idx]
        LFPatch, noiLFPatch = ExtractPatch(LF, noiLF, self.patchSize)  # [u v c x y]
        LFPatch = torch.from_numpy(LFPatch.astype(np.float32) / 255)
        noiLFPatch = torch.from_numpy(noiLFPatch.astype(np.float32) / 255)

        sample = {'LFPatch': LFPatch, 'noiLFPatch': noiLFPatch}
        return sample

    def __len__(self):
        return self.LFSet.shape[5]

