import pickle
import os
import numpy as np

import torch

sparse_tensors = torch.load(r'C:\Users\Administrator\PycharmProjects\HIAM-main\data\suzhou\OD\train_repeated_sparse_tensors.pt')

with open(r'C:\Users\Administrator\PycharmProjects\HIAM-main\data\suzhou\OD\train.pkl','rb') as f:
    train = pickle.load(f, errors='ignore')


