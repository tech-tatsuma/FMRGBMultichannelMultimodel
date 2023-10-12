# convlstm
from torch import nn
import torch

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined_conv = self.conv(torch.cat([input_tensor, h_cur], dim=1))
        i, f, o, g = torch.sigmoid(combined_conv).chunk(4, dim=1)
        c_next = f * c_cur + i * torch.tanh(g)
        return o * torch.tanh(c_next), c_next

    def init_hidden(self, batch_size, image_size):
        return (torch.zeros(batch_size, self.hidden_dim, *image_size, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, *image_size, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.return_all_layers = return_all_layers

        if not isinstance(kernel_size, list): kernel_size = [kernel_size] * num_layers
        if not isinstance(hidden_dim, list): hidden_dim = [hidden_dim] * num_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim, hidden_dim=hidden_dim[i], kernel_size=kernel_size[i], bias=bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = [cell.init_hidden(b, (h, w)) for cell in self.cell_list]

        layer_output_list, last_state_list = [], []
        for layer_idx, cell in enumerate(self.cell_list):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = cell(input_tensor[:, t, :, :, :], (h, c))
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_list.append(layer_output.permute(0, 2, 1, 3, 4))
            last_state_list.append((h, c))
            input_tensor = layer_output

        if not self.return_all_layers:
            layer_output_list, last_state_list = layer_output_list[-1:], last_state_list[-1:]

        return layer_output_list, last_state_list


class ConvLSTM_FC(ConvLSTM):
    def __init__(self, num_tasks=5, *args, **kwargs):
        super(ConvLSTM_FC, self).__init__(*args, **kwargs)
        self.num_tasks = num_tasks

        _, last_hidden_dim, _, _ = self.cell_list[-1].conv.weight.shape
        self.attention = nn.Sequential(
            nn.Conv2d(last_hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(last_hidden_dim, 128),
            nn.ReLU(),
            self.dropout1,
            nn.Linear(128, 32),
            nn.ReLU(),
            self.dropout2
        )
        
        self.task_fcs = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_tasks)])

    def forward(self, input_tensor, hidden_state=None):
        layer_output_list, last_state_list = super(ConvLSTM_FC, self).forward(input_tensor, hidden_state=hidden_state)
        output = layer_output_list[-1][:, -1, :, :, :]
        
        attention_map = self.attention(output)
        output = (output * attention_map).sum(dim=[2, 3])
        
        output = output.reshape(output.size(0), -1)
        shared_output = self.shared_fc(output)
        
        # Compute task-specific outputs
        outputs = [task_fc(shared_output) for task_fc in self.task_fcs]

        outputs = torch.cat(outputs, dim=1)
        
        return outputs
        