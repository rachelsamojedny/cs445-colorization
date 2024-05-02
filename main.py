import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import colorizationnet 
import colorizationdataset
from colorizationnet import ColorizationCNN
from colorizationdataset import ColorizationDataset, read_in_data
from train import train
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

# Define the main function
def main():
    # Define hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    num_channels_in = 1
    num_channels_out = 2
    im_size = 256
    
    #reads in data as lab images of size (256, 256) [image_num, y, x, channel]
    train_color_ims, test_color_ims = read_in_data("data/colorimages", im_size)

    train_dataset = ColorizationDataset(train_color_ims[:,:,:,0], train_color_ims)
    test_dataset = ColorizationDataset(test_color_ims[:,:,:,0], test_color_ims)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
    # Define loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
   

    model = ColorizationCNN(learning_rate, criterion, num_channels_in, num_channels_out, im_size)
    
    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs)
    
    # Evaluate the model
    model.eval() 
    output_imgs = []
    for i,sample in enumerate(test_dataset['input']):
        
        grey = test_dataset['input'][i]
        color = test_dataset['output'][i]
        y = model.forward(grey)

        color_im_out = np.stack((grey, color[:,:,0], color[:,:,1]), axis=-1)
        output_imgs.append(color_im_out)

    

    output_imgs = np.array(output_imgs)

    rgb_image = lab2rgb(output_imgs[0])

    plt.imshow(rgb_image)
    plt.axis('off') 
    plt.show()


if __name__ == "__main__":
    main()