from os import listdir
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
import random
import os
from pillow import Image

#potentially read in a dataset using pickle, may have to change depending on our dataset.

def read_in_data(directory, im_size):
    lab_im_list = [] 
    target_size = (im_size, im_size)
    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            filepath = os.path.join(directory, filename)
            img = Image.open(filepath)
            lab_img = img.convert('LAB')
            lab_im_list.append(lab_img)
    lab_im_list = resize_images(lab_im_list, target_size, 'bilinear')
    np_array_list = [np.array(img) for img in lab_im_list]


    return np.array(np_array_list)

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

