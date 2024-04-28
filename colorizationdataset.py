from os import listdir
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import random

#potentially read in a dataset using pickle, may have to change depending on our dataset.

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# def load_dataset(filename, full=False):
#     A = unpickle(filename) # np.loadtxt('data_batch_1')
#     X = A[b'data']
#     Y = A[b'labels'].astype(int)
#     if full:
#         return X,Y
    
#     test_size = int(0.25 * len(X)) # set aside 25% for testing
#     X_test = X[:test_size]
#     Y_test = Y[:test_size]
#     X = X[test_size:]
#     Y = Y[test_size:]
#     return X,Y,X_test,Y_test

class ColorizationDataset(Dataset):
    def __init__(self, x, y):
        """
        Args:
            x [np.array]: input vector
            y [np.array]: output vector          
        """
        self.input = x
        self.output = y

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.input[idx,:]
        output = self.output[idx, :]
        sample = {'input': input,'output': output}
        return sample

