import torch
from torch import nn

class ColorizationCNN(nn.Module):
    
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.num_in = 1
        self.num_out = 2
        self.imsize = 256
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(65536,1024)
        self.fc2 = nn.Linear(1024,256)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1)
        self.fc3 = nn.Linear(256,65536)
        self.batchnorm3 = nn.BatchNorm2d(2)

        ##########       END      ##########

    def forward(self, x):
        #1*256*256
        x = self.conv1(x)
        x = self.batchnorm1(x)
        #16*256*256
        x = x.view(x.shape[0], x.shape[1], -1)
        #16*65536
        x = self.fc1(x)
        #16*2048
        x = self.relu(x)
        x = self.fc2(x)
        #16*256
        x = x.view(x.shape[0], x.shape[1], 16, 16)
        x = self.batchnorm2(x)
        #16*16*16
        x = self.conv2(x)
        #2*16*16
        x = x.view(x.shape[0], x.shape[1], -1)
        #2*256
        x = self.fc3(x)
        #2*65536
        x = x.view(x.shape[0], x.shape[1], 256, 256)
        x = self.relu(x)

        return x