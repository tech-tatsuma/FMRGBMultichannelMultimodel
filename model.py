from torch import nn
import torch
import torch

# 3D Convolutional Neural Network
class ConvNet3D(nn.Module):
    def __init__(self):
        super(ConvNet3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            # Continue with more layers as needed...
        )
        self.fc = None

    def forward_features(self, x):
        print("Input shape:", x.shape, "Type:", x.dtype)  # Add this line
        x = self.conv(x)
        print("After conv shape:", x.shape, "Type:", x.dtype)  # And this line
        return x.view(x.size(0), -1)
        
    def forward(self, x):
        if self.fc is None:
            # Get output shape of conv layers
            out = self.forward_features(x)
            out_shape = out.shape[-1]
            print("Flattened output shape:", out_shape)
            # Define fc layer with the obtained output shape
            self.fc = nn.Sequential(
                nn.Linear(out_shape, 128),
                nn.ReLU(),
                nn.Linear(128, 2)  # Single output for regression
            ).to(x.device)
        x = self.forward_features(x)
        print("After forward_features shape:", x.shape, "Type:", x.dtype)  # And this line
        x = self.fc(x)
        print("After fc shape:", x.shape, "Type:", x.dtype)  # And this line
        return x

import torch
from torch import nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_dim : int
            Number of channels of input tensor.
        hidden_dim : int
            Number of channels of hidden state.
        kernel_size : (int, int)
            Size of the convolutional kernel.
        bias : bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        # Your implementation here
        pass

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A final hidden state of the next LSTM cell and/or list containing the output features from the last LSTM cell of each layer, list containing the hidden state and cell state for all timesteps.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        # Your implementation here
        pass

    def forward(self, input_tensor, hidden_state=None):
        # Implement forward pass
        pass