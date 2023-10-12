# シンプルな3DCNN
from torch import nn
import torch

# Simple 3DCNN
class ConvNet3D(nn.Module):
    def __init__(self, in_channels=3, num_tasks=5, batch_size=20, depth=1500, height=56, width=56):
        super(ConvNet3D, self).__init__()

        # convolution層
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        # 全結合層への入力次元を計算する
        self._to_linear = None

        # 入力と同じ形式のデータを作成する（次元の計算用）
        x = torch.randn(batch_size, in_channels, depth, height, width)
        
        self.convs(x)

        # 全結合層
        self.shared_fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # マルチタスクに対応させるための層
        self.task_fcs = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_tasks)])

    def convs(self, x):
        x = self.conv(x)
        # 次の全結合層に入力するための次元を計算する
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]*x[0].shape[3]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.shared_fc(x)
        outputs = [task_fc(x) for task_fc in self.task_fcs]

        # 最後はテンソルに直して関数の出力とする
        outputs = torch.cat(outputs, dim=1)
        
        return outputs