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
