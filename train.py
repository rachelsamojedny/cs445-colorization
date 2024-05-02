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

def train(net, train_loader, criterion, optimizer, num_epochs, report_interval = 10):
    net.train()
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        # for i, data in enumerate(train_loader):
            # colorimages, _ = data            

            # grayimages = lab2gray(colorimages)
            
            # optimizer.zero_grad()
            # results = net(grayimages)
            # loss = criterion(results, colorimages)
            # loss.backward()
            # optimizer.step()

            # running_loss = loss.item()
            # if i % report_interval + 1 == report_interval:
            #     print('report loss here')
        for i, batch in enumerate(train_loader):
            gray_images = batch['input'].float()
            color_images = batch['output'].float()
            optimizer.zero_grad()
            results = net(gray_images)

            l_channel = np.expand_dims(gray_images, axis=-1)
            print("l_shape")
            print(l_channel.shape)
            print("res_shape")
            print(results.shape)
            reshaped_results = results.view(results.size(0), 256, 256, 2)
            lab_output = np.concatenate((l_channel, reshaped_results.detach().numpy()), axis=3)
            print("lab_output shape:", lab_output.shape)
            print("color_images shape:", color_images.shape)
            print(criterion)
            loss = criterion(lab_output, color_images)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            if i % report_interval + 1 == report_interval:
                print('report loss here')

    return