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

class ConvLSTM(nn.Module):
    def __init__(self, sample_size, num_classes=2):
        super(ConvLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Add more layers as needed...
        )

        sample_output = self.cnn(torch.zeros(*sample_size))
        cnn_output_size = sample_output.view(sample_output.size(0), -1).size(1)

        self.rnn = nn.LSTM(
            input_size=cnn_output_size,  # This should match the output of your CNN
            hidden_size=64,  # You can specify your own number
            num_layers=1,  # Number of LSTM layer
            batch_first=True  # input & output will has batch size as 1s dimension
        )

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        cnn_out = torch.zeros(batch_size, timesteps, self.rnn.input_size).to(x.device)  # This should match the output of your CNN
        for i in range(timesteps):
            cnn_out[:, i, :] = self.cnn(x[:, i, :, :, :]).reshape(batch_size, -1)

        r_out, _ = self.rnn(cnn_out)
        r_out2 = self.fc(r_out[:, -1, :])
        
        return r_out2