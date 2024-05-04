import torch
import numpy as np
from utils import rgb2lab_batch, lab2rgb_skimage
import matplotlib.pyplot as plt

def train(net, optimizer, criterion, trainloader, num_epochs, device, report_interval = 0):
    net.train()
    if report_interval == 0:
        report_interval = np.max([num_epochs//10, 1])
    for epoch in range(num_epochs):
        print("EPOCH:",epoch+1)
        epoch_total_loss = 0
        for x, _ in trainloader:
            batchsize, input_channels, img_size, _ = x.shape
            batch_lab = rgb2lab_batch(x, False)
            L = batch_lab[:,:1].to(device)
            real_AB = batch_lab[:,1:].detach().to(device)
            result = net(L)

            optimizer.zero_grad()
            loss = criterion(result, real_AB)
            epoch_total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if epoch%report_interval == 0:
            print("Cumulative loss of epoch=",epoch_total_loss)
