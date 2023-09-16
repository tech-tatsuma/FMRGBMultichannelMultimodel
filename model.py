from torch import nn, einsum
import torch
import torch.nn.functional as f
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from previvitmodel import Attention, PreNorm, FeedForward
import torch.nn as nn
# from .ops_dcnv3.modules.dcnv3 import DCNv3

# 3dcnn
class ConvNet3D(nn.Module):
    def __init__(self, in_channels=3, num_tasks=5, batch_size=20, depth=1500, height=56, width=56):
        super(ConvNet3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        # Compute the size of Linear layer input
        self._to_linear = None
        x = torch.randn(batch_size, in_channels, depth, height, width)  # Now matching the dataset shape
        # print('shape of x: ', x.shape)
        self.convs(x)
        # print(f"Calculated _to_linear: {self._to_linear}")

        # Shared layers
        self.shared_fc = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # Task-specific layers
        self.task_fcs = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_tasks)])

    def convs(self, x):
        x = self.conv(x)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]*x[0].shape[3]
        return x

    def forward(self, x):
        # print("Shape of input to the model:", x.shape)
        x = self.convs(x)
        # print("Shape of tensor after conv layers:", x.shape)
        x = x.view(-1, self._to_linear)
        # print("Shape of tensor after view layers:", x.shape)
        x = self.shared_fc(x)
        # print("Shape of tensor after shared layers:", x.shape)
        # Compute task-specific outputs
        outputs = [task_fc(x) for task_fc in self.task_fcs]

        outputs = torch.cat(outputs, dim=1)
        
        return outputs
    
# convlstm
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
        self.attention = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        
        self.shared_fc = nn.Sequential(
            nn.Linear(32, 128),
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

# ViViT
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, 128),
            nn.Softmax(),
            nn.Dropout(0.25),
            nn.Linear(128, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        # x += self.pos_embedding[:, :, :(n + 1)]
        x += self.pos_embedding[:, :t, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

class MultiTaskViViT(ViViT):
    def __init__(self, num_tasks=5, *args, **kwargs):
        super(MultiTaskViViT, self).__init__(*args, **kwargs)
        self.num_tasks = num_tasks

        # Shared Layer
        self.shared_mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
        )

        # Individual fully connected layers for each task
        self.task_fcs = nn.ModuleList([nn.Linear(self.dim, 1) for _ in range(num_tasks)])

    def forward(self, x):
        x = super(ViViT, self).forward(x)  # Run the forward pass of the original ViViT model until the pooling layer
        
        shared_output = self.shared_mlp_head(x)  # Use the shared layers

        # Compute task-specific outputs
        outputs = [task_fc(shared_output) for task_fc in self.task_fcs]

        return outputs  # Return list of outputs for each task

# class DCNConvLSTMCell(nn.Module):
#     def __init__(self, input_dim, hidden_dim, kernel_size, bias, group=4, offset_scale=1.0):
#         super(DCNConvLSTMCell, self).__init__()
#         self.hidden_dim = hidden_dim

#         self.dcn = DCNv3(
#             channels=input_dim + hidden_dim,
#             kernel_size=kernel_size,
#             group=group,
#             offset_scale=offset_scale
#         )

#         self.conv = nn.Conv2d(in_channels=hidden_dim,
#                               out_channels=4 * hidden_dim,
#                               kernel_size=1,
#                               padding=0,
#                               bias=bias)

#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state
#         dcn_output = self.dcn(torch.cat([input_tensor, h_cur], dim=1))
#         combined_conv = self.conv(dcn_output)
#         i, f, o, g = torch.sigmoid(combined_conv).chunk(4, dim=1)
#         c_next = f * c_cur + i * torch.tanh(g)
#         return o * torch.tanh(c_next), c_next

#     def init_hidden(self, batch_size, image_size):
#         return (torch.zeros(batch_size, self.hidden_dim, *image_size, device=self.conv.weight.device),
#                 torch.zeros(batch_size, self.hidden_dim, *image_size, device=self.conv.weight.device))

# class DCNConvLSTM(ConvLSTM):
#     def __init__(self, *args, **kwargs):
#         super(DCNConvLSTM, self).__init__(*args, **kwargs)

#     def forward(self, input_tensor, hidden_state=None):
#         b, seq_len, _, h, w = input_tensor.size()

#         if hidden_state is None:
#             hidden_state = [cell.init_hidden(b, (h, w)) for cell in self.cell_list]

#         layer_output_list, last_state_list = [], []
#         for layer_idx, cell in enumerate(self.cell_list):
#             h, c = hidden_state[layer_idx]
#             output_inner = []
#             for t in range(seq_len):
#                 h, c = cell(input_tensor[:, t, :, :, :], (h, c))
#                 output_inner.append(h)
#             layer_output = torch.stack(output_inner, dim=1)
#             layer_output_list.append(layer_output.permute(0, 2, 1, 3, 4))
#             last_state_list.append((h, c))
#             input_tensor = layer_output

#         if not self.return_all_layers:
#             layer_output_list, last_state_list = layer_output_list[-1:], last_state_list[-1:]

#         return layer_output_list, last_state_list

# class DCNConvLSTM_FC(DCNConvLSTM):
#     def __init__(self, num_tasks=5, *args, **kwargs):
#         super(DCNConvLSTM_FC, self).__init__(*args, **kwargs)
#         self.num_tasks = num_tasks
#         self.attention = nn.Sequential(
#             nn.Conv2d(32, 1, kernel_size=1),
#             nn.Sigmoid()
#         )
        
#         self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.5)
        
#         self.shared_fc = nn.Sequential(
#             nn.Linear(32, 128),
#             nn.ReLU(),
#             self.dropout1,
#             nn.Linear(128, 32),
#             nn.ReLU(),
#             self.dropout2
#         )
        
#         self.task_fcs = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_tasks)])

#     def forward(self, input_tensor, hidden_state=None):
#         layer_output_list, last_state_list = super(DCNConvLSTM_FC, self).forward(input_tensor, hidden_state=hidden_state)
#         output = layer_output_list[-1][:, -1, :, :, :]
        
#         attention_map = self.attention(output)
#         output = (output * attention_map).sum(dim=[2, 3])
        
#         output = output.reshape(output.size(0), -1)
#         shared_output = self.shared_fc(output)
        
#         # Compute task-specific outputs
#         outputs = [task_fc(shared_output) for task_fc in self.task_fcs]
        
#         return outputs