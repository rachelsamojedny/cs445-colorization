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

# Define the main function
def main():
    # Define hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    
    # Define data transformations
    transform = transforms.Compose([
        # Implement image transformations (e.g., resizing, normalization)
    ])
    
    # Create dataset instances
    train_dataset = ColorizationDataset(root_dir='path/to/train/dataset', transform=transform)
    test_dataset = ColorizationDataset(root_dir='path/to/test/dataset', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model instance
    model = ColorizationCNN()
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs)
    
    # Evaluate the model
    evaluate(model, test_loader, criterion)

if __name__ == "__main__":
    main()