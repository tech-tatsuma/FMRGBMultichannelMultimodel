from model import ConvNet3D, ConvLSTM_FC, ViViT
import torch
from torchvision import transforms
import argparse
import os
import sys
from dataset import VideoDataset
from torch.utils.data import DataLoader

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

def predict(opt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # calculate mean and std value
    mean, std = calculate_dataset_statistics([opt.data])

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        Normalize3D(mean, std),
    ])

    # specify the model
    if opt.network=='conv3d':
        model = ConvNet3D(batch_size=1, image_size=56).to(device)
    elif opt.network=='convlstm':
        model = ConvLSTM_FC(input_dim=3, hidden_dim=[64, 32, 16], kernel_size=(3, 3), num_layers=3).to(device)
    elif opt.network=='vivit':
        model = ViViT(image_size=64, patch_size=16, num_classes=2, num_frames=64, in_channels=3).to(device)
    else:
        print('error: inappropriate input(network)')
        return

    # Load the model
    state_dict = torch.load(opt.model)
    new_state_dict = {k.replace('module.',''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # create single video dataset
    dataset = VideoDataset([opt.data], transform=transform, isconvon=(opt.network == 'conv3d'))
    data = DataLoader(dataset, batch_size=1, shuffle=False)

    # Make the prediction
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data):
            inputs, labels = inputs.to(device), labels.to(device)
            prediction = model(inputs)
            _, predicted_class = torch.max(prediction, 1)

    return predicted_class.item()

if __name__ == '__main__':
    # setup parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='input video .pt file')
    parser.add_argument('--model', type=str, required=True, help='path to model .pt file')
    parser.add_argument('--network', type=str, required=True, help='conv3d or convlstm or vivit')
    opt = parser.parse_args()
    print('---------predict start---------')
    # make prediction
    predicted_class = predict(opt)

    print(f'Predicted class: {predicted_class}')
    sys.stdout.flush()
    print('---------predict complete---------')
