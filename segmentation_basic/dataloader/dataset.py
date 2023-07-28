# -*- CODING: UTF-8 -*-
# @time 2023/3/7 21:47
# @Author tyqqj
# @File dataset.py

import numpy as np
import torch
from torch.utils import data
from torch import nn

# 这个一般不用

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

import SimpleITK as sitk


class MRADataset(data.Dataset):
    def __init__(self, root, isTrain=True):
        self.labelPath = self.get_data_Path(root, False)
        self.imgPath = root
        self.isTraining = isTrain
        self.name = ' '
        self.name = ' '

    def standardization_intensity_normalization(self, dataset, dtype):
        mean = dataset.mean()
        std = dataset.std()
        dataset = ((dataset - mean) / std).astype(dtype)

    def __getitem__(self, index):
        labelPath = self.labelPath[index]
        filename = labelPath.split('/')[-1]
        self.name = filename
        imgPath = self.root + '/test/' + '/images/' + filename[:-4] + '-MRA.mha'
        img = sitk.ReadImage(imgPath)
        img = sitk.GetArrayFromImage(img)

    def __len__(self):
        return len(self.img)
