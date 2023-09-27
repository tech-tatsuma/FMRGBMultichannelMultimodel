import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataset import random_split
from dataset import VideoDataset
from model import ConvNet3D, MixtureOfExperts
import random 
import numpy as np
from torch import nn
import os
from loss import validation_function
from setproctitle import setproctitle

# シードの設定を行う関数
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 正規化、標準化のための統計量をサブセットを使って算出する関数
def calculate_approximate_mean_and_std(dataset, num_samples=1000, num_frames=10):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    mean = 0.
    std = 0.
    num_pixels = 0
    
    for i, (video, _) in enumerate(dataloader):
        if i == num_samples:
            break
        # Randomly sample 'num_frames' frames from the video
        indices = np.random.choice(video.shape[1], num_frames)
        sample = video[:, indices, :, :, :]

        mean += torch.mean(sample, dim=[0, 1, 2, 3])
        std += torch.std(sample, dim=[0, 1, 2, 3])
        num_pixels += np.prod(sample.shape[1:])

    mean /= num_samples
    std /= num_samples
    return mean, std

setproctitle("spearman_test")
# model = ConvNet3D(in_channels=3, num_tasks=5, batch_size=20, depth=100, height=56, width=56)
model = MixtureOfExperts(3, 3, 20, 100, 56, 56, 5)
# モデルのロード
# model_pathには保存されたモデルのパスを指定してください。
model_path = "moeconv3d_lr1e-05_ep100_pa10ranklosstrueintweak.pt"
# CPUでモデルをロード
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model = nn.DataParallel(model)
model.eval()

# データの前処理
transform = transforms.Compose([
    transforms.Resize((56, 56)), 
    transforms.ToTensor()
])

# データセットのロード
# data_fileにはデータセットのパスを指定してください。
data_file = "/data2/furuya/gaodatasets/combined1-140_modified.csv"
full_dataset1 = VideoDataset(csv_file=data_file, transform=transform, addpath = os.path.dirname(data_file))

# 500個のデータを抽出
_, val_dataset_500 = random_split(full_dataset1, [len(full_dataset1) - 500, 500])

mean, std = calculate_approximate_mean_and_std(val_dataset_500)

full_dataset2 = VideoDataset(csv_file=data_file, transform=transform, mean=mean, std=std, addpath=os.path.dirname(data_file))

# データローダの作成
train_dataset, val_dataset = random_split(full_dataset2, [len(full_dataset2) - 500, 500])

val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)

# val_spearmanの計算
val_spearman = 0.0
with torch.no_grad():
    for inputs, labels in val_loader:
        
        # データはすでにCPU上にあるため、転送の処理は不要
        # inputs, labels = inputs.to(device), labels.to(device)
        
        if inputs.dtype != torch.float32:
            inputs = inputs.float()
        
        outputs = model(inputs)
        
        # val_spearmanにはスピアマンの相関順位係数が入る
        val_spearman = validation_function(outputs, labels)
        break  # 一度のみ実行

# val_spearmanの値をNumPy配列に変換
val_spearman = val_spearman.detach().numpy()

# 結果の出力
print("val_spearman using 500 data:", val_spearman)
