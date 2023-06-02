import torch
from torch.utils.data import Dataset

# Dataset class
class VideoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tensor = torch.load(self.df.iloc[idx, 0])
        label = self.df.iloc[idx, 1]
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label
