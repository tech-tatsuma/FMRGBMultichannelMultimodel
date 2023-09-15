from dataset import VideoDataset
from model import ConvNet3D, ConvLSTM_FC, MultiTaskViViT
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import glob
from torch import nn
import numpy as np
from torchinfo import summary
import sys
import datetime
import random

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

# メイン関数
def train(opt):

    # シードの設定
    seed_everything(opt.seed)

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # オプションで入力された値を取得
    data_file = opt.data
    epochs = opt.epochs
    train_size = opt.train_size
    patience = opt.patience
    learningmethod = opt.learnmethod
    learning_rate = opt.lr
    usescheduler = opt.usescheduler

    addpath = os.path.dirname(data_file)

    # データの前処理
    transform = transforms.Compose([
        transforms.Resize((56, 56)), 
        transforms.ToTensor()
    ])

    # 正規化、標準化に必要な計算を行うために一度データをロード
    initial_dataset = VideoDataset(csv_file=data_file, transform=transform, addpath=addpath)

    # 訓練用、評価用のデータサイズを設定
    train_size = int(train_size * len(initial_dataset))
    val_size = len(initial_dataset) - train_size

    # データの平均、標準偏差を計算
    _, initial_val_dataset = random_split(initial_dataset, [train_size, val_size])
    mean, std = calculate_approximate_mean_and_std(initial_val_dataset)

    # 正規化を行いながらデータセットを作成
    full_dataset = VideoDataset(csv_file=data_file, transform=transform, mean=mean, std=std, addpath=addpath)

    # データの分割
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # データローダの取得
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # モデルの選択
    if learningmethod=='conv3d':
        # conv3dモデルの設定
        model = ConvNet3D(in_channels=3, num_tasks=5, batch_size=4, depth=100, height=56, width=56).to(device)
        criterion = nn.MSELoss()

    elif learningmethod=='convlstm':
        # convlstmの設定
        model = ConvLSTM_FC(input_dim=3, hidden_dim=[64, 32, 16], kernel_size=(3, 3), num_layers=3).to(device)
        criterion = nn.MSELoss()

    elif learningmethod=='vivit':
        # vivitの設定
        model = MultiTaskViViT(image_size=64, patch_size=16, num_classes=2, num_frames=64, in_channels=3).to(device)
        criterion = nn.MSELoss()

    elif learningmethod=='convlstmwithdcn':
        # DCNConvLSTM_FCの設定
        # model =  DCNConvLSTM_FC(input_dim=3, hidden_dim=[64, 32, 16], kernel_size=(3,3), num_layers=3).to(device)
        # criterion = nn.MSELoss()
        print('まだ実装が完了していません。別のオプションで実行してください。')
    else:
        # 入力が不適切だった時の処理
        print('error: inappropriate input(learning method)')
        return

    # 複数GPUの並行処理
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # モデルをデバイスに転送
    model.to(device)

    # 最適化手法の設定と学習率の設定
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 学習率スケジューラの設定
    if usescheduler == 'true':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # 早期終了のための変数を定義
    val_loss_min = None
    val_loss_min_epoch = 0

    # 学習結果を監視するための配列を定義
    train_losses = []
    val_losses = []

    # モデルの訓練
    for epoch in tqdm(range(epochs)):

        # 各種パラメータの初期化
        train_loss = 0.0
        val_loss = 0.0

        # モデルをtrainモードに設定
        model.train()

        # trainデータのロード
        for i, (inputs, labels) in enumerate(train_loader):

            # ビデオとラベルをGPUへ転送
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # データはfloat型でなければならない
            if inputs.dtype != torch.float32:
                inputs = inputs.float()

            # モデルの適用
            outputs = model(inputs)

            # Forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # モデルを評価モードの設定
        model.eval()
        
        # 検証データでの評価
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                
                # データをGPUに転送
                inputs, labels = inputs.to(device), labels.to(device)
                
                if inputs.dtype != torch.float32:
                    inputs = inputs.float()
                    
                outputs = model(inputs)

                # ここで labels と outputs の shape を確認
                # print(f"Labels shape: {labels.shape}, Outputs shape: {outputs.shape}")

                val_loss += criterion(outputs, labels).item()
                

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Validation loss: {val_loss:.4f}')
        sys.stdout.flush()

        # メモリーを最適化する
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # バリデーションロスが下がった時は結果を保存する
        if val_loss_min is None or val_loss < val_loss_min:
            model_save_name = f'{learningmethod}_lr{learning_rate}_ep{epochs}_pa{patience}intweak.pt'
            torch.save(model.state_dict(), model_save_name)
            val_loss_min = val_loss
            val_loss_min_epoch = epoch
            
        # もしバリデーションロスが一定期間下がらなかったらその時点で学習を終わらせる
        elif (epoch - val_loss_min_epoch) >= patience:
            print('Early stopping due to validation loss not improving for {} epochs'.format(patience))
            break

        # 学習率スケジューラを適用する
        if usescheduler == 'true':
            scheduler.step()

    # Plotting the training and validation progress
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    return train_loss, val_loss_min

if __name__=='__main__':
    # get start time of program
    start_time = datetime.datetime.now()
    print('start time:',start_time)
    sys.stdout.flush()
    # setting parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str, required=True, help='csv data')
    parser.add_argument('--epochs',type=int, required=True, help='epochs')
    parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
    parser.add_argument('--train_size', type=float, default=0.2, help='testdata_ratio')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--learnmethod', type=str, default='conv3d', help='conv3d or convlstm or vivit')
    parser.add_argument('--islearnrate_search', type=str, default='false', help='is learningrate search ?')
    parser.add_argument('--usescheduler', type=str, default='false', help='use lr scheduler true or false')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generators')
    opt = parser.parse_args()
    # confirm the option
    print(opt)
    sys.stdout.flush()

    if opt.islearnrate_search == 'false':
        print('-----biginning training-----')
        sys.stdout.flush()
        train_loss, val_loss = train(opt)
        print('final validation loss: ', val_loss)
        sys.stdout.flush()

    elif opt.islearnrate_search == 'true':
        learning_rates = [0.001, 0.002, 0.003, 0.004]
        best_loss = float('inf')
        best_lr = 0
        for lr in learning_rates:
            opt.lr = lr
            print(f"\nTraining with learning rate: {lr}")
            sys.stdout.flush()
            print('-----beginning training-----')
            sys.stdout.flush()
        
            train_loss, val_loss = train(opt)
        
            if val_loss < best_loss:
                best_loss = val_loss
                best_lr = lr
        print('best validation loss: ', best_loss)
        sys.stdout.flush()
        print(f"Best learning rate: {best_lr}")
        sys.stdout.flush()

    else:
        print('error: inappropriate input(islearnrate_search)')
    # get end time of program
    end_time = datetime.datetime.now()
    # calculate execution time with start and end time
    execution_time = end_time - start_time
    print('-----completing training-----')
    sys.stdout.flush()
    print('end time:',end_time)
    sys.stdout.flush()
    print('Execution time: ', execution_time)
    sys.stdout.flush()