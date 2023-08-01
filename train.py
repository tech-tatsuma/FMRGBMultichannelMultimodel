from dataset import VideoDataset
from model import ConvNet3D, ConvLSTM_FC, ViViT
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import glob
from torch import nn
import numpy as np
from torchinfo import summary
import sys
import datetime
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Calculate the mean and standard deviation of the dataset
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


# main train function
def train(opt):

    seed_everything(opt.seed)

    # setting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get the optional info
    data_file = opt.data
    epochs = opt.epochs
    test_size = opt.test_size
    patience = opt.patience
    learningmethod = opt.learnmethod
    learning_rate = opt.lr
    usescheduler = opt.usescheduler

    # get the all file list
    file_list = glob.glob(os.path.join(opt.data, '**', '*.pt'), recursive=True)

    # calculate mean and std value
    mean, std = calculate_dataset_statistics(file_list)

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        Normalize3D(mean, std),
    ])

    # data split
    # train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=42)
    # train_files, val_files = train_test_split(train_files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(file_list, test_size=test_size, random_state=42)

    # create datasets
    if learningmethod=='conv3d':
        train_dataset = VideoDataset(train_files, transform=transform)
        val_dataset = VideoDataset(val_files, transform=transform)
        # test_dataset = VideoDataset(test_files, transform=transform)

    elif learningmethod=='convlstm':
        train_dataset = VideoDataset(train_files, transform=transform, isconvon=False)
        val_dataset = VideoDataset(val_files, transform=transform, isconvon=False)
        # test_dataset = VideoDataset(test_files, transform=transform, isconvon=False)

    elif learningmethod=='vivit': 
        train_dataset = VideoDataset(train_files, transform=transform, isconvon=False)
        val_dataset = VideoDataset(val_files, transform=transform, isconvon=False)
        # test_dataset = VideoDataset(test_files, transform=transform, isconvon=False)

    else:
        print('error: inappropriate input(learning method)')
        return

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)

    # setting learning network
    if learningmethod=='conv3d':
        # create 3dcnn model
        model = ConvNet3D(batch_size=20, image_size=64).to(device)
        criterion = nn.CrossEntropyLoss()  # Use crosentropy for bi-problem

    elif learningmethod=='convlstm':
        # create convlstm model
        model = ConvLSTM_FC(input_dim=3, hidden_dim=[64, 32, 16], kernel_size=(3, 3), num_layers=3).to(device)
        criterion = nn.CrossEntropyLoss()  # Use crosentropy for bi-problem

    elif learningmethod=='vivit':
        # create vivit model
        # The image_size must be divisible by the PATCH size
        model = ViViT(image_size=64, patch_size=16, num_classes=2, num_frames=64, in_channels=3).to(device)
        criterion = nn.CrossEntropyLoss()  # Use crosentropy for bi-problem

    else:
        print('error: inappropriate input(learning method)')
        return

    # code to use multi GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # Distribute models across multiple GPUs using nn.DataParallel
        model = nn.DataParallel(model)

    # transfer model to device
    model.to(device)

    # output summary of used model
    # with open('model_summary_vivit.txt', 'w') as f:
    #     sys.stdout = f
    #     # setting the each input size
    #     summary(model, input_size=(20, 64, 3, 64, 64)) # convlstm & vivit
    #     # summary(model, input_size=(20, 3, 32, 56, 56)) # conv3d
    #     sys.stdout = sys.__stdout__

    # Define a optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler setting
    if usescheduler == 'true':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize variables for Early Stopping
    val_loss_min = None
    val_loss_min_epoch = 0

    # Initialize lists to monitor train and validation losses
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Train the model
    for epoch in tqdm(range(epochs)):  # Number of epochs
        train_loss = 0
        train_corrects = 0
        val_loss = 0
        val_corrects = 0
        # change the model mode to train mode
        model.train()
        # train loading
        for i, (inputs, labels) in enumerate(train_loader):
            # transfer inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # inputs data have to be float
            if inputs.dtype != torch.float32:
                inputs = inputs.float()

            # apply model to train
            outputs = model(inputs)

            # Forward + backward + optimize
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels).cpu().detach()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = train_corrects.double() / len(train_loader.dataset)
        train_accuracies.append(train_accuracy.item())
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                    
                outputs = model(inputs)

                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).cpu().detach()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = val_corrects.double() / len(val_loader.dataset)
        val_accuracies.append(val_accuracy.item())

        print(f'Epoch {epoch+1}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')
        sys.stdout.flush()

        # Memory Clear Here
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        if usescheduler == 'true':
            scheduler.step()

    # Plotting the training and validation progress
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plot_save_name = f'{learningmethod}_lr{learning_rate}_ep{epochs}_pa{patience}.png'
    plt.savefig(plot_save_name)

    return train_loss, val_loss_min

if __name__=='__main__':
    # get start time of program
    start_time = datetime.datetime.now()
    print('start time:',start_time)
    sys.stdout.flush()
    # setting parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str, required=True, help='csv data')
    parser.add_argument('--epochs',type=int, required=True, help='epochs')
    parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
    parser.add_argument('--test_size', type=float, default=0.2, help='testdata_ratio')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--learnmethod', type=str, default='conv3d', help='conv3d or convlstm or vivit')
    parser.add_argument('--islearnrate_search', type=str, default='false', help='is learningrate search ?')
    parser.add_argument('--usescheduler', type=str, default='false', help='use lr scheduler true or false')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    opt = parser.parse_args()
    # confirm the option
    print(opt)
    sys.stdout.flush()

    if opt.islearnrate_search == 'false':
        print('-----biginning training-----')
        sys.stdout.flush()
        train_loss, val_loss = train(opt)
        print('final validation loss: ', val_loss)
        sys.stdout.flush()

    elif opt.islearnrate_search == 'true':
        learning_rates = [0.001, 0.0007, 0.0005, 0.0003, 0.0001]
        best_loss = float('inf')
        best_lr = 0
        for lr in learning_rates:
            opt.lr = lr
            print(f"\nTraining with learning rate: {lr}")
            sys.stdout.flush()
            print('-----beginning training-----')
            sys.stdout.flush()
        
            train_loss, val_loss = train(opt)
        
            if val_loss < best_loss:
                best_loss = val_loss
                best_lr = lr
        print('best validation loss: ', best_loss)
        sys.stdout.flush()
        print(f"Best learning rate: {best_lr}")
        sys.stdout.flush()

    else:
        print('error: inappropriate input(islearnrate_search)')
    # get end time of program
    end_time = datetime.datetime.now()
    # calculate execution time with start and end time
    execution_time = end_time - start_time
    print('-----completing training-----')
    sys.stdout.flush()
    print('end time:',end_time)
    sys.stdout.flush()
    print('Execution time: ', execution_time)
    sys.stdout.flush()
