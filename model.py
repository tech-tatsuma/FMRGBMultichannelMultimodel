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
        self.fc = None

    def forward_features(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)
        
    def forward(self, x):
        if self.fc is None:
            # Get output shape of conv layers
            out = self.forward_features(x)
            out_shape = out.shape[-1]
            # Define fc layer with the obtained output shape
            self.fc = nn.Sequential(
                nn.Linear(out_shape, 128),
                nn.ReLU(),
                nn.Linear(128, 1)  # Single output for regression
            ).to(x.device)
        x = self.forward_features(x)
        x = self.fc(x)
        return x
