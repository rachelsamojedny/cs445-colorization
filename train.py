import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import colorizationdataset
from colorizationdataset import ColorizationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lab2gray(labimages):
    grayimages = labimages[:,0,:,:]
    return grayimages
def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

def train(net, train_loader, criterion, optimizer, num_epochs, report_interval = 5, isDebug = True):
    net.train()
    running_loss = 0
    for epoch in range(num_epochs):
        print("Epoch", epoch+1)
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
            #print(len(train_loader))
            gray_images = batch['input'].float().to(device)
            color_images = batch['output'].float().to(device)
            optimizer.zero_grad()
            results = net(gray_images).to(device)
            #maybe 
            # optimizer.step()
            gray_images = gray_images.cpu()
            l_channel = torch.tensor(np.expand_dims(gray_images, axis=-1)).to(device)


            if isDebug:
                print("l_shape")
                print(l_channel.shape)
                print("res_shape")
                print(results.shape)
            reshaped_results = results.view(results.size(0), 256, 256, 2) * 255.0

            #lab_output = torch.from_numpy(np.concatenate((l_channel, reshaped_resultsd.detach().numpy()), axis=3))
            lab_output = torch.cat((l_channel, reshaped_results), dim = 3)
            
            if isDebug:
                print("lab_output shape:", lab_output.shape)
                print("color_images shape:", color_images.shape)
                print(criterion)


            loss = criterion(lab_output, color_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if epoch % report_interval + 1 == report_interval:
            if isDebug:
                print("chan_l")
                print(l_channel[0,:5,:5,0])
                print("chan_a")
                print(reshaped_results[0,:5,:5,0])
                print("chan_b")
                print(reshaped_results[0,:5,:5,1])
                print("cat_chan_l")
                print(lab_output[0,:5,:5,0])     
                print("cat_chan_a")
                print(lab_output[0,:5,:5,1])
                print("cat_chan_b")
                print(lab_output[0,:5,:5,2])
                print_model_parameters(net)
            print('avg loss here: ' + str(running_loss / report_interval))
            running_loss = 0

    return