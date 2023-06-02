from torch import nn


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
        self.fc = nn.Sequential(
            nn.Linear(16*32*32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for regression
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x