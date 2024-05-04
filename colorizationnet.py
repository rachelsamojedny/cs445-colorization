import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import colorizationdataset as cd
import torch.nn.init as init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ColorizationCNN(nn.Module):
    def __init__(self, lrate, loss_fn, num_channels_in, num_channels_out, im_size):
        super(ColorizationCNN, self).__init__()

        self.num_in = num_channels_in
        self.num_out = num_channels_out
        self.im_size = im_size

        self.loss_fn = loss_fn

        self.conv1 = nn.Conv2d(num_channels_in, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, num_channels_out, kernel_size=3, padding=1)
        
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(32)
        
        # self.upconv1 = nn.ConvTranspose2d(num_channels_out, 32, kernel_size=4, stride=2, padding=1)
        # self.upconv2 = nn.ConvTranspose2d(32, num_channels_out, kernel_size=4, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

        self.opt = optim.SGD(self.parameters(), lr=lrate, momentum=.9)

    # evaluates the model for a given image batch x
    def forward(self, x):   
        ''' 
        takes in x, an (N, in_size im_size, im_size) Tensor
        returns y an (N, out_size im_size, im_size ) Tensor of x passed through the model

        '''
        flag = False
        if len(x.shape) == 2:
            x = torch.tensor(x).float().to(device)
            x = x.unsqueeze(0)

            flag = True
        #x may need to be reshaped here if the
        # print("x_shape_orig")
        # print(x.shape)
        x = x.view(x.size(0), self.num_in, self.im_size, self.im_size)
        # print("x_reshape")
        # print(x.shape)


        x1 = F.relu(self.batchnorm1(self.conv1(x)))
        # print("x1_shape" + str(x1.shape))
        x2 = F.relu(self.batchnorm2(self.conv2(x1)))
        # print("x2_shape" + str(x2.shape))
        # x3 = F.relu(self.batchnorm3(self.conv3(x2)))
        # x4 = F.relu(self.batchnorm4(self.conv4(x3)))
        # x5 = F.relu(self.batchnorm5(self.conv5(x4)))
        x5 = F.relu(self.batchnorm5(self.conv5(x2)))
        # print("x5_shape" + str(x5.shape))
        x6 = torch.sigmoid(self.conv6(x5))  
        # print("x6_shape" + str(x6.shape))
        # Upsampling
        # x7 = F.relu(self.upconv1(x6))
        # print("x7_shape" + str(x7.shape))
        # x8 = torch.sigmoid(self.upconv2(x7))
        # print("x8_shape" + str(x8.shape))

        y = x6.permute(0, 2, 3, 1)
        if flag:
            y = y.squeeze(0) 

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


def fit(train_set,train_labels,dev_set,epochs,batch_size=10):
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
