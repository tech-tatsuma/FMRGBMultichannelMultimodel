import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import csv

class VideoDataset(Dataset):

    def __init__(self, csv_file, transform=None, target_frames=64, target_size=(224, 224)):
        self.transform = transform
        self.target_frames = target_frames
        self.target_size = target_size
        self.file_list, self.labels = self.load_csv(csv_file)
        self.num_channels = 3

    def load_csv(self, csv_file):
        file_list = []
        labels = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                file_list.append(row[0])
                labels.append(list(map(float, row[1:])))
        return file_list, labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        frames = []
        cap = cv2.VideoCapture(file_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(F.to_tensor(frame))

        cap.release()

        tensor = torch.stack(frames)
        total_frames = tensor.shape[0]
        
        if total_frames > self.target_frames:
            skip_frames = total_frames // self.target_frames
            tensor = tensor[::skip_frames, :, :, :]
            tensor = tensor[:self.target_frames, :, :, :]

        elif total_frames < self.target_frames:
            size_diff = self.target_frames - total_frames
            padding = torch.zeros((size_diff, tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            tensor = torch.cat([tensor, padding], dim=0)

        tensor = tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)

        return tensor, label
