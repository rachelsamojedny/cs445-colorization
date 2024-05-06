import torch
import numpy as np
from torch import nn
from utils import rgb2lab_batch, lab2rgb_skimage
import matplotlib.pyplot as plt

def train(net, optimizer, trainloader, num_epochs, device, report_interval = 0):
    net.train()
    if report_interval == 0:
        report_interval = np.max([num_epochs//10, 1])
    bestloss = np.inf
    for epoch in range(num_epochs):
        print("EPOCH:",epoch+1)
        epoch_total_loss = 0
        batchcount = 0
        for x, _ in trainloader:
            if len(trainloader)>10:
                if epoch==0 and (batchcount+1)%(len(trainloader)//10)==0: print(batchcount+1,"/",len(trainloader))
            batchsize, input_channels, img_size, _ = x.shape
            batch_lab = rgb2lab_batch(x, False).to(device)
            L = batch_lab[:,:1].to(device)
            real_AB = batch_lab[:,1:].detach().to(device)
            result = net(L)

            optimizer.zero_grad()
            loss = nn.MSELoss(reduction='mean')(result, real_AB)
            epoch_total_loss += loss.item()
            loss.backward()
            optimizer.step()
            batchcount+=1
        epoch_total_loss /= batchcount
        if epoch_total_loss < bestloss:
            bestloss = epoch_total_loss
            torch.save(net.state_dict(), "models/trainbest.pth")
        if epoch%report_interval == 0 or epoch == 0 or epoch+1==num_epochs:
            print("Loss=",epoch_total_loss)
