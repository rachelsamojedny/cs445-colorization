import torch
import os
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
    print("main.py")
    # Define hyperparameters
    batch_size = 32
    learning_rate = .01
    num_epochs = 1

    num_channels_in = 1
    num_channels_out = 2
    im_size = 256
    #reads in data as lab images of size (256, 256) [image_num, y, x, channel]
    train_color_ims, test_color_ims = read_in_data("data/colorimages", im_size)

    train_dataset = ColorizationDataset(train_color_ims[:,:,:,0], train_color_ims)
    test_dataset = ColorizationDataset(test_color_ims[:,:,:,0], test_color_ims)

    # print("sample photo")
    # print("l_chan")
    # print(train_color_ims[0,:10,:10,0])
    # print("a_chan")
    # print(train_color_ims[0,:10,:10,1])
    # print("b_chan")
    print(train_color_ims[0,:10,:10,2])
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
    for i,sample in enumerate(test_dataset):
        
        grey = sample['input']
        color = sample['output']
        y = model.forward(grey)

      


        color_im_out = np.stack((grey, (y[:,:,0]).detach().numpy() * 255, (y[:,:,1]).detach().numpy() * 255), axis=-1)
        output_imgs.append(color_im_out)
    lab_image = output_imgs[0]
    l_c = lab_image[:,:,0]
    a_c = lab_image[:,:,1]
    b_c = lab_image[:,:,2]
    print("c1")
    print(l_c[:5,:5])
    print("c2")
    print(a_c[:5,:5])
    print("c3")
    print(b_c[:5:,:5])
    rgb_image = lab2rgb(lab_image)
    print("rgbc1")
    print(rgb_image[:,:,0])
    print("rgbc2")
    print(rgb_image[:,:,1])
    print("rgbc3")
    print(rgb_image[:,:,2])
    output_directory = 'data/outputimages'

    scaled_rgb_image = rgb_image * 255.0

# Clip the values to ensure they are within the valid range [0, 255]
    scaled_rgb_image = np.clip(scaled_rgb_image, 0, 255)
    output_file_path = os.path.join(output_directory, 'output_image.png')
    plt.imshow(rgb_image)
    plt.axis('off') 
    plt.savefig(output_file_path)

    # output_file_path = os.path.join(output_directory, 'expected_output_image.png')
    # plt.imshow(np.array(test_dataset['output'][0]))
    # plt.axis('off') 
    # plt.savefig(output_file_path)

    output_file_path = os.path.join(output_directory, 'scaled_output_image.png')
    plt.imshow(np.array(scaled_rgb_image))
    plt.axis('off') 
    plt.savefig(output_file_path)


if __name__ == "__main__":
    main()