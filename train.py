from dataset import VideoDataset
from model import ConvNet3D, ConvLSTM_FC, ViViT
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
import numpy as np
from torchviz import make_dot
import sys

def calculate_dataset_statistics(file_list):
    # Initialize sum and square sum
    sum_ = torch.zeros_like(torch.load(file_list[0]), dtype=torch.float64)
    sum_of_squares = torch.zeros_like(torch.load(file_list[0]), dtype=torch.float64)
    total_frames = 0

    # Calculate the sum and sum of squares for each pixel value
    for file in file_list:
        video_tensor = torch.load(file).permute(3, 0, 1, 2)  # Change to [C, D, H, W]
        sum_ += torch.sum(video_tensor, dim=(1, 2, 3))  # sum per channel
        sum_of_squares += torch.sum(video_tensor ** 2, dim=(1, 2, 3))  # sum of squares per channel
        total_frames += video_tensor.shape[1]

    mean = sum_ / total_frames  # calculate mean per channel
    std = torch.sqrt(sum_of_squares / total_frames - mean**2)  # calculate std per channel

    return mean.tolist(), std.tolist()

class Normalize3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Determine if the input tensor is 4D or 5D
        if tensor.dim() == 5:  # Case of [batch_size, num_channels, depth, height, width]
            for t, m, s in zip(tensor.permute(1, 0, 2, 3, 4), self.mean, self.std):
                t.sub_(m).div_(s)
        elif tensor.dim() == 4:  # Case of [batch_size, depth, num_channels, height, width]
            for t, m, s in zip(tensor.permute(2, 0, 1, 3, 4), self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor

def train(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get the option info
    data_file = opt.data
    epochs = opt.epochs
    test_size = opt.test_size
    patience = opt.patience
    learningmethod = opt.learnmethod
    learning_rate = opt.lr

    file_list = glob.glob(os.path.join(opt.data, '**', '*.pt'), recursive=True)
    mean, std = calculate_dataset_statistics(file_list)
    # Create your transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        Normalize3D(mean, std),
    ])

    # data split
    train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=test_size, random_state=42)

    # create datasets
    if learningmethod=='conv3d':
        train_dataset = VideoDataset(train_files, transform=transform)
        val_dataset = VideoDataset(val_files, transform=transform)
        test_dataset = VideoDataset(test_files, transform=transform)

    elif learningmethod=='convlstm':
        train_dataset = VideoDataset(train_files, transform=transform, isconvon=False)
        val_dataset = VideoDataset(val_files, transform=transform, isconvon=False)
        test_dataset = VideoDataset(test_files, transform=transform, isconvon=False)

    elif learningmethod=='vivit': 
        train_dataset = VideoDataset(train_files, transform=transform, isconvon=False)
        val_dataset = VideoDataset(val_files, transform=transform, isconvon=False)
        test_dataset = VideoDataset(test_files, transform=transform, isconvon=False)

    else:
        print('error: 入力が不適切です')
        return

    # Create your dataloaders
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

    if learningmethod=='conv3d':
        # create 3dcnn model
        model = ConvNet3D().to(device)
        criterion = nn.CrossEntropyLoss()  # Use crosentropy for bi-problem

    elif learningmethod=='convlstm':
        # create convlstm model
        model = ConvLSTM_FC(input_dim=5, hidden_dim=[64, 32, 16], kernel_size=(3, 3), num_layers=3).to(device)
        criterion = nn.CrossEntropyLoss()  # Use crosentropy for bi-problem

    elif learningmethod=='vivit':
        # create vivit model
        model = ViViT(image_size=224, patch_size=16, num_classes=2, num_frames=64, in_channels=5).to(device)
        criterion = nn.CrossEntropyLoss()  # Use crosentropy for bi-problem

    else:
        print('error: 入力が不適切です')
        return

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # Distribute models across multiple GPUs using nn.DataParallel
        model = nn.DataParallel(model)


    model.to(device)
    # Define a loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

            elif learningmethod=='vivit':
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            if i == 0:  # only for the first iteration
                dot = make_dot(loss, params=dict(model.named_parameters()))
                dot.format = 'png'
                dot.render(filename=f'model_{learningmethod}_structure')
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
                    
                outputs= model(inputs)

                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Validation loss: {val_loss:.4f}')

            # Save the model if validation loss decreases
        if val_loss_min is None or val_loss < val_loss_min:
            model_save_name = f'{learningmethod}_lr{learning_rate}_ep{epochs}_pa{patience}.pt'
            torch.save(model.state_dict(), model_save_name)
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
    plot_save_name = f'{learningmethod}_lr{learning_rate}_ep{epochs}_pa{patience}.png'
    plt.savefig(plot_save_name)

if __name__=='__main__':
    # setting parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str, required=True, help='csv data')
    parser.add_argument('--epochs',type=int, required=True, help='epochs')
    parser.add_argument('--lr',type=float, required=True, help='learning rate')
    parser.add_argument('--test_size', type=float, required=True, default=0.2, help='testdata_ratio')
    parser.add_argument('--patience', type=int, required=True, default=5, help='patience')
    parser.add_argument('--learnmethod', type=str, default='conv3d', help='conv3d or convlstm or vivit')
    opt = parser.parse_args()
    print(opt)
    sys.stdout.flush()
    print('-----biginning training-----')
    sys.stdout.flush()
    train(opt)
    print('-----completing training-----')
    sys.stdout.flush()
