# slowfast network
from torch import nn
import torch

# SlowFast(3DCNN)
class SlowFastConvNet3D(nn.Module):
    def __init__(self, in_channels=3, num_tasks=5, batch_size=20, depth=1500, height=56, width=56):
        super(SlowFastConvNet3D, self).__init__()

        # Slow Pathway
        self.slow_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        # Fast Pathway
        self.fast_conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=(8, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        # 全結合層への入力次元を計算する
        self._to_linear_slow = None
        self._to_linear_fast = None

        # 入力と同じ形式のダミーデータを作成する（次元の計算用）
        # x_slow = torch.randn(batch_size, in_channels, depth, height, width)
        # x_fast = torch.randn(batch_size, in_channels, depth//8, height, width)
        # 元の入力データ
        x_original = torch.randn(batch_size, in_channels, depth, height, width)

        # 入力と同じ形式のダミーデータを作成する（次元の計算用）
        x_slow = x_original[:,:,::16,:,:]
        x_fast = x_original[:,:,::2,:,:]    
        self.convs(x_slow, x_fast)

        # 全結合層
        self.shared_fc = nn.Sequential(
            nn.Linear(self._to_linear_slow + self._to_linear_fast, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # マルチタスクに対応させるための層
        self.task_fcs = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_tasks)])

    def convs(self, x_slow, x_fast):
        x_slow = self.slow_conv(x_slow)
        x_fast = self.fast_conv(x_fast)

        # 次の全結合層に入力するための次元を計算する
        if self._to_linear_slow is None:
            self._to_linear_slow = torch.prod(torch.tensor(x_slow.shape[1:]))
        if self._to_linear_fast is None:
            self._to_linear_fast = torch.prod(torch.tensor(x_fast.shape[1:]))
        return x_slow, x_fast

    def forward(self, x):
        # Separate the input tensor into slow and fast components
        x_slow = x[:,:,::16,:,:]
        x_fast = x[:,:,::2,:,:]
        x_slow, x_fast = self.convs(x_slow, x_fast)

        # Concatenate Slow and Fast pathways
        x = torch.cat((x_slow.view(-1, self._to_linear_slow), x_fast.view(-1, self._to_linear_fast)), dim=1)

        x = self.shared_fc(x)

        # Compute task-specific outputs
        outputs = [task_fc(x) for task_fc in self.task_fcs]

        # 最後はテンソルに直して関数の出力とする
        outputs = torch.cat(outputs, dim=1)

        return outputs