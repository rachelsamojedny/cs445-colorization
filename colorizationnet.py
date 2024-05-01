import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import colorizationdataset as cd




class ColorizationCNN(nn.Module):
    def __init__(self, lrate, loss_fn, num_channels_in, num_channels_out, im_size):
        super(ColorizationCNN, self).__init__()

        self.num_in = num_channels_in
        self.num_out = num_channels_out
        self.im_size = im_size

        self.loss_fn = loss_fn

        self.conv1 = nn.Conv2d(num_channels_in, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, num_channels_out, kernel_size=3, padding=1)
        
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(32)
        
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d

        self.opt = optim.SGD(self.parameters(), lr=lrate, momentum=.9)

    # evaluates the model for a given image batch x
    def forward(self, x):   
        ''' 
        takes in x, an (N, in_size im_size, im_size) Tensor
        returns y an (N, out_size im_size, im_size ) Tensor of x passed through the model

        '''

        #x may need to be reshaped here if the
        x = x.view(-1, self.num_in, self.im_size, self.im_size)



        x1 = F.relu(self.batchnorm1(self.conv1(x)))
        x2 = F.relu(self.batchnorm2(self.conv2(x1)))
        x3 = F.relu(self.batchnorm3(self.conv3(x2)))
        x4 = F.relu(self.batchnorm4(self.conv4(x3)))
        x5 = F.relu(self.batchnorm5(self.conv5(x4)))
        x6 = torch.sigmoid(self.conv6(x5))  
        
        # Upsampling
        x7 = F.relu(self.upconv1(x6))
        x8 = torch.sigmoid(self.upconv2(x7))

        y = x8.view(x8.size(0), -1)
        return y
    



    def step(self, x , y):
        '''
        preforms a gradient step through a batch x of data with labels y

        takes in x, an (N, in_size, im_size, im_size) Tensor
        and y an (N, out_size, im_size, im_size) Tensor of color images for scoring

        
        '''
        self.opt.zero_grad()
        output = self.forward(x)
        self.opt.step()
        L = self.loss_fn(output, y)
        L.backward()
        self.opt.step()


        #could try and compute a loss here if that is nessecary L.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.
    takes in train_set: an (N, in_size, im_size, im_size) Tensor
    takes in colorized_train_set: an (N, out_size, im_size, im_size) Tensor
    takes in dev_set an (N, in_size, im_size, im_size) Tensor for testing
    epochs: an int, the number of epochs of training
    batch_size: size of each batch to train on. (default 100)

    """
    dataset = cd.ColorizationDataset(train_set, train_labels)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    output_imgs = []
    net = ColorizationCNN(.01, torch.nn.CrossEntropyLoss(), 1, 2, 256)

    
    #load and normalize data loaders for training and dev set.

    for epoch in range(epochs): 
        running_loss = 0.0
        for batch in train_dataloader:
            running_loss += net.step(batch['input'], batch['output'])
            


    net.eval() 

    for sample in dev_set:
        y = torch.argmax(net.forward(sample))
        output_imgs.append(y.item())

    

    output_imgs = np.array(output_imgs)
    return output_imgs,net
