import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.nn.functional as F
import colorizationnet 
import colorizationdataset
from colorizationnet import ColorizationCNN
from colorizationdataset import ColorizationDataset, read_in_data
from train import train

# Define the main function
def main():
    # Define hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    


    # Create dataset instances
    image_list = read_in_data("../data/colorimages")

    train_dataset = ImageFolder(root = "./dataset/orig", transform = transforms.Compose([
        transforms.Resize((800, 500)), transforms.ToTensor()
    ]))
    test_dataset = None #
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_channels_in = None # grayscale: H*W*1
    num_channels_out = None # ab channel: H*W*2
    im_size = None #
    
    # Create model instance
    model = ColorizationCNN(learning_rate, num_channels_in, num_channels_out, im_size)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs)
    
    # Evaluate the model
    evaluate(model, test_loader, criterion)

if __name__ == "__main__":
    main()