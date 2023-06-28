import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import functional as F
import os
import torch
from PIL import Image
from torchvision.transforms import Resize

class VideoDataset(Dataset):
    # def __init__(self, df, transform=None, target_frames=64):
    def __init__(self, file_list, transform=None, target_frames=64, target_size=(224, 224), isconvon=True):
        self.file_list = file_list
        self.transform = transform
        self.target_frames = target_frames
        self.target_size = target_size
        self.label_mapping = {"NonViolence": 0, "Violence": 1}
        self.num_channels = 5
        self.isconvon = isconvon

    def __len__(self):
        # return len(self.df)
        return len(self.file_list)

    def __getitem__(self, idx):
        # ファイルの読み込み
        file_path = self.file_list[idx]
        tensor = torch.load(file_path)
        # ファイルに付与されたラベルの取得
        label_name = os.path.basename(os.path.dirname(file_path))
        #ラベルを0 or 1にマッピング
        label = self.label_mapping[label_name]
        total_frames = tensor.shape[0]

        if total_frames > self.target_frames:
            skip_frames = total_frames // self.target_frames
            tensor = tensor[::skip_frames, :, :, :]
            tensor = tensor[:self.target_frames, :, :, :]

        elif total_frames < self.target_frames:
            size_diff = self.target_frames - total_frames
            padding = torch.zeros((size_diff, tensor.shape[1], tensor.shape[2], tensor.shape[3]))
            tensor = torch.cat([tensor, padding], dim=0)
            
        if self.transform:
            tensor = self.apply_transform(tensor)
        if self.isconvon == True:
            tensor = tensor.permute(3, 0, 1, 2)
        if self.isconvon == False:
            tensor = tensor.permute(0, 3, 1, 2)
        print(tensor.shape)
        return tensor, label


    def apply_transform(self, tensor):
        T, H, W, C = tensor.shape
        tensor_transformed = torch.zeros(T, self.target_size[1], self.target_size[0], C)
        for t in range(T):
            for c in range(C):
                img = tensor[t, :, :, c].cpu().numpy()
                img = Image.fromarray((img * 255).astype(np.uint8))
                img = img.resize(self.target_size, Image.BICUBIC) # resize all channels
                if c < 3 and self.transform:  # RGB channels
                    img = self.transform(img)
                    print(img)
                    print(img.dtype)
                if isinstance(img, Image.Image):  # If the transform did not convert to tensor
                    img = F.to_tensor(img).float()
                elif isinstance(img, np.ndarray):  # If the transform output is a numpy array
                    img = torch.from_numpy(img)
                tensor_transformed[t, :, :, c] = img  # the size of img must be (self.target_size[1], self.target_size[0]) now
        return tensor_transformed


        return tensor_transformed

