from dataset import VideoDataset
from model import ConvNet3D, ConvLSTM_FC
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import glob
from torch import nn

def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_file = opt.data
    epochs = opt.epochs
    test_size = opt.test_size
    patience = opt.patience
    learningmethod = opt.learnmethod
    # Create your transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    file_list = glob.glob(os.path.join(opt.data, '**', '*.pt'), recursive=True)
    train_files, test_files = train_test_split(file_list, test_size=opt.test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=opt.test_size, random_state=42)

    if learningmethod=='conv3d':
        # Create your datasets
        train_dataset = VideoDataset(train_files, transform=transform)
        val_dataset = VideoDataset(val_files, transform=transform)
        test_dataset = VideoDataset(test_files, transform=transform)

    elif learningmethod=='convlstm':
        # Create your datasets
        train_dataset = VideoDataset(train_files, transform=transform, isconvon=False)
        val_dataset = VideoDataset(val_files, transform=transform, isconvon=False)
        test_dataset = VideoDataset(test_files, transform=transform, isconvon=False)

    else:
        print('error: 入力が不適切です')
        return

    # Create your dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    if learningmethod=='conv3d':
        # Create the model
        model = ConvNet3D().to(device)

    elif learningmethod=='convlstm':
        model = ConvLSTM_FC(input_dim=5, hidden_dim=[64, 32, 16], kernel_size=(3, 3), num_layers=3).to(device)

    else:
        print('error: 入力が不適切です')
        return

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Use MSE for regression
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize variables for Early Stopping
    val_loss_min = None
    val_loss_min_epoch = 0

    # Initialize lists to monitor train and validation losses
    train_losses = []
    val_losses = []

    # Train the model
    for epoch in range(epochs):  # Number of epochs
        train_loss = 0
        val_loss = 0
        model.train()

        for i, (inputs, labels) in tqdm(enumerate(train_loader, 0)):

            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            if inputs.dtype != torch.float32:
                inputs = inputs.float()

            if learningmethod=='conv3d':
                outputs = model(inputs)

            elif learningmethod=='convlstm':
                outputs = model(inputs)
                print('outputs:', outputs.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(val_loader, 0)):
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                if learningmethod=='conv3d':
                    outputs = model(inputs)
                    
                elif learningmethod=='convlstm':
                    outputs= model(inputs)

                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Validation loss: {val_loss:.4f}')

            # Save the model if validation loss decreases
        if val_loss_min is None or val_loss < val_loss_min:
            torch.save(model.state_dict(), 'lstmnolimit_model.pt')
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
            
        # If the validation loss didn't decrease for 'patience' epochs, stop the training
        elif (epoch - val_loss_min_epoch) >= patience:
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            break

    # Plotting the training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lstm_training_validation_nolimitloss.png')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str, required=True, help='csv data')
    parser.add_argument('--epochs',type=int, required=True, help='epochs')
    parser.add_argument('--test_size', type=float, required=True, default=0.2, help='testdata_ratio')
    parser.add_argument('--patience', type=int, required=True, default=5, help='patience')
    parser.add_argument('--learnmethod', type=str, default='conv3d', help='conv3d or convlstm')
    opt = parser.parse_args()
    print(opt)
    print('-----biginning training-----')
    train(opt)
    print('-----completing training-----')
