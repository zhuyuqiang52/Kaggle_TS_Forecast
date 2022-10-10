import torch
import unicodedata
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence
import numpy as np
import os
import string
import pandas as pd

train_df = pd.read_pickle('./data/sale_forcast/festures_train.pkl')
test_df = pd.read_pickle('./data/sale_forcast/festures_test.pkl')
trainr_df = train_df.iloc[:,1:]
testr_df = test_df.iloc[:,1:]

class sale_dataset(Dataset):
    def __init__(self,train_bool):
        super().__init__()
        if train_bool:
            type_str = 'train'
        else:
            type_str = 'test'
        path_str = f'./data/sale_forcast/festures_{type_str}.pkl'
        self.dat_df = pd.read_pickle(path_str)
        self.dat_df = self.dat_df.iloc[:, 1:]

    def __len__(self):
        return len(self.dat_df)

    def __getitem__(self):

