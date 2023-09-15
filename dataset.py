import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import csv

# データセットを作成するクラス
class VideoDataset(Dataset):

    # 初期化関数
    def __init__(self, csv_file, transform=None, target_frames=100, target_size=(224, 224), mean=None, std=None, addpath=''):
        # それぞれの変数に値を設定
        self.transform = transform
        self.target_frames = target_frames
        self.target_size = target_size
        self.file_list, self.labels = self.load_csv(csv_file)
        self.num_channels = 3
        self.mean = mean if mean is not None else torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1)
        self.std = std if std is not None else torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1)
        self.addpath=addpath

    # csvをロードする関数
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
    
    # データセットの大きさを返す関数
    def __len__(self):
        return len(self.file_list)
    
    # データセットを取得する関数
    def __getitem__(self, idx):
        # 動画のパス
        file_path = self.addpath + '/' + self.file_list[idx]
        print(file_path)
        # フレームを格納する配列
        frames = []
        # ビデオキャプチャー
        cap = cv2.VideoCapture(file_path)
        
        # 動画データをフレームごとに取得
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            # frameが既にtorch.Tensorであればそのまま使用する
            if not isinstance(frame, torch.Tensor):
                frame = F.to_tensor(frame)
            # フレームごとにテンソルとして動画に格納
            frames.append(frame)

        cap.release()

        tensor = torch.stack(frames)
        total_frames = tensor.shape[0]

        # 標準化と正規化
        tensor = (tensor - self.mean) / self.std
        
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

        # print("Shape of tensor from dataset:", tensor.shape)

        return tensor, label
