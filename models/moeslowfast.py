# 混合エキスパートを用いたslowfastネットワーク
from torch import nn
import torch
import torch.nn.functional as f

class SlowFastMoEConvNet3D(nn.Module):
    def __init__(self, in_channels, batch_size, depth, height, width):
        super(SlowFastMoEConvNet3D, self).__init__()
        
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
        
        self._to_linear_slow = None
        self._to_linear_fast = None
        x_original = torch.randn(batch_size, in_channels, depth, height, width)

        # 入力と同じ形式のダミーデータを作成する（次元の計算用）
        x_slow = x_original[:,:,::16,:,:]
        x_fast = x_original[:,:,::2,:,:]    
        self.convs(x_slow, x_fast)
        
        self.fc = nn.Linear(self._to_linear_slow + self._to_linear_fast, 64)
    
    def convs(self, x_slow, x_fast):
        x_slow = self.slow_conv(x_slow)
        x_fast = self.fast_conv(x_fast)
        if self._to_linear_slow is None:
            self._to_linear_slow = torch.prod(torch.tensor(x_slow.shape[1:]))
        if self._to_linear_fast is None:
            self._to_linear_fast = torch.prod(torch.tensor(x_fast.shape[1:]))
        return x_slow, x_fast
    
    def forward(self, x):
        x_slow = x[:,:,::16,:,:]
        x_fast = x[:,:,::2,:,:]
        x_slow, x_fast = self.convs(x_slow, x_fast)
        x = torch.cat((x_slow.view(-1, self._to_linear_slow), x_fast.view(-1, self._to_linear_fast)), dim=1)
        x = self.fc(x)
        return x

# SlowFast(MoE)expert network
class SlowFastMixtureOfExperts(nn.Module):
    def __init__(self, in_channels, num_experts, batch_size, depth, height, width, num_tasks):
        super(SlowFastMixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([SlowFastMoEConvNet3D(in_channels, batch_size, depth, height, width) for _ in range(num_experts)])
        self.gate = SlowFastMoEConvNet3D(in_channels, batch_size, depth, height, width)

        # Task-specific layers
        self.task_fcs = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_tasks)])

    def forward(self, x):
        gate_output = f.softmax(self.gate(x), dim=1)
        expert_outputs = [expert(x) for expert in self.experts]

        final_output = 0
        for i, expert_output in enumerate(expert_outputs):
            weighted_output = gate_output[:, i].view(-1, 1) * expert_output
            final_output += weighted_output

        outputs = [task_fc(final_output) for task_fc in self.task_fcs]
        outputs = torch.cat(outputs, dim=1)
        return outputs