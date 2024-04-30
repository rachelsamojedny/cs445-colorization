import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import colorizationdataset
from colorizationdataset import ColorizationDataset

def lab2gray(labimages):
    grayimages = labimages[:,0,:,:]
    return grayimages

def train(net, train_loader, criterion, optimizer, num_epochs, report_interval = 2000):
    net.train()
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        for i, data in enumerate(train_loader):
            colorimages, _ = data            

            grayimages = lab2gray(colorimages)
            
            optimizer.zero_grad()
            results = net(grayimages)
            loss = criterion(results, colorimages)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            if i % report_interval + 1 == report_interval:
                print('report loss here')
    return