# -*- CODING: UTF-8 -*-
# @time 2023/3/7 21:42
# @Author tyqqj
# @File main.py

import numpy as np
import torch
from torch.utils import data
from torch import nn

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using ', device)


def print_hi(name):
    print(f'Hi, {name}')


if __name__ == '__main__':
    print_hi('Python')
