import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import functional as F

class VideoDataset(Dataset):
    def __init__(self, df, transform=None, target_frames=64):
        self.df = df
        self.transform = transform
        self.target_frames = target_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tensor = torch.load(self.df.iloc[idx, 0])
        label = self.df.iloc[idx, 1]
        # Determine how many frames to skip
        total_frames = tensor.shape[1] # assuming tensor shape is (C, T, H, W)
        skip_frames = total_frames // self.target_frames
        # Trim the tensor by selecting every "skip_frames" frames
        tensor = tensor[:, ::skip_frames, :, :]
        # If the video is still longer than target_frames, trim it to the correct length
        if tensor.shape[1] > self.target_frames:
            tensor = tensor[:, :self.target_frames, :, :]
        # If the video is shorter than target_frames, zero pad it
        if tensor.shape[1] < self.target_frames:
            size_diff = self.target_frames - tensor.shape[1]
            padding = torch.zeros((tensor.shape[0], size_diff, tensor.shape[2], tensor.shape[3]))
            tensor = torch.cat([tensor, padding], dim=1)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, label
        
    def apply_transform(self, tensor):
        C, T, H, W = tensor.shape
        tensor_transformed = torch.zeros_like(tensor)
        for t in range(T):
            img = F.to_pil_image(tensor[:, t])
            img = self.transform(img)
            tensor_transformed[:, t] = F.to_tensor(img)
        return tensor_transformed
