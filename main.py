import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
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
    vgg = models.vgg19(pretrained=True).eval()
    current_correct = 0
    current_total = 0
    correct_fake = 0
    total = 0
    best_accuracy = 0
    best_idx = 0
    with torch.no_grad():
        for i,sample in enumerate(test_dataset):
            grey = sample['input']
            color = sample['output']
            colorized_images = model.forward(grey)

            # colorized_images = colorized_images.unsqueeze(0)
            # #print(colorized_images.shape)
            # colorized_images = colorized_images.permute(0, 3, 2, 1)
            # #print(colorized_images.shape)
            colorized_images = np.stack((grey, (colorized_images[:,:,0]).detach().numpy() * 255, (colorized_images[:,:,1]).detach().numpy() * 255), axis=-1)
            

            colorized_images = torch.tensor(colorized_images).float().unsqueeze(0)
            colorized_images = colorized_images.permute(0, 3, 2, 1)
            #print(colorized_images.shape)

            color = torch.tensor(color).float().unsqueeze(0)
            color = color.permute(0, 3, 2, 1)
            #print(color.shape)
            outputs_fake = vgg(colorized_images) #run our fake colorized images thru VGG and see the scores
            outputs_real = vgg(color) #get the real labels using the output

            _, predicted_fake = torch.max(outputs_fake.data, 1) #take the scores and get the prediction (which label has highest score)
            _, predicted_real = torch.max(outputs_real.data, 1)
            current_correct += (predicted_fake == predicted_real).sum().item() #how many predictions are correct
            correct_fake += current_correct
            current_total = len(sample['input']) 
            total += current_total

            colorized_images = colorized_images.permute(0, 2, 3, 1)
            colorized_images = colorized_images.squeeze(0)
            output_imgs.append(colorized_images)

            if (current_correct / current_total) > best_accuracy:
              best_accuracy = current_correct / current_total
              best_idx = i

    accuracy = 100 * correct_fake / total

    print(f'Accuracy on fake colorized images from model: {accuracy}%') #total accuracy of model

    lab_image = output_imgs[best_idx]
    #l_c = lab_image[:,:,0]
    #a_c = lab_image[:,:,1]
    #b_c = lab_image[:,:,2]
    #print("c1")

    #print(l_c[:5,:5])
    #print("c2")

    #print(a_c[:5,:5])
    # print("c3")
    # print(b_c[:5:,:5])
    rgb_image = lab2rgb(lab_image)
    # print("rgbc1")
    # print(rgb_image[:,:,0])
    # print("rgbc2")
    # print(rgb_image[:,:,1])
    # print("rgbc3")
    # print(rgb_image[:,:,2])
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