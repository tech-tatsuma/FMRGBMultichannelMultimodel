from torch import nn
import torch
import torch.nn.functional as f

# 混合エキスパートを導入したconv3dネットワーク
class MoEConvNet3D(nn.Module):
    def __init__(self, in_channels, batch_size, depth, height, width):
        super(MoEConvNet3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self._to_linear = None
        x = torch.randn(batch_size, in_channels, depth, height, width)
        self.convs(x)

        self.fc = nn.Linear(self._to_linear, 64)

    def convs(self, x):
        x = self.conv(x)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]*x[0].shape[3]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = self.fc(x)
        return x


class MixtureOfExperts(nn.Module):
    def __init__(self, in_channels, num_experts, batch_size, depth, height, width, num_tasks):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([MoEConvNet3D(in_channels, batch_size, depth, height, width) for _ in range(num_experts)])
        self.gate = MoEConvNet3D(in_channels, batch_size, depth, height, width)

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