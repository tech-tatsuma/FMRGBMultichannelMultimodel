from torch import nn, einsum
import torch
import torch.nn.functional as f
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from previvitmodel import Attention, PreNorm, FeedForward
import torch.nn as nn
# from .ops_dcnv3.modules.dcnv3 import DCNv3

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

# SlowFast(MoE)gate network
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

        return x

class MultiTaskViViT(ViViT):
    def __init__(self, num_tasks=5, *args, **kwargs):

        self.dim = kwargs.get('dim', 1)

        super(MultiTaskViViT, self).__init__(*args, **kwargs)
        self.num_tasks = num_tasks

        # Shared Layer
        self.shared_mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
        )

        # Individual fully connected layers for each task
        self.task_fcs = nn.ModuleList([nn.Linear(self.dim, 1) for _ in range(num_tasks)])

    def forward(self, x):
        x = super().forward(x)  # Run the forward pass of the original ViViT model until the pooling layer
        shared_output = self.shared_mlp_head(x)  # Use the shared layers

        # Compute task-specific outputs
        outputs = [task_fc(shared_output) for task_fc in self.task_fcs]

        outputs = torch.cat(outputs, dim=1)

        return outputs  # Return list of outputs for each task

class AttentionMulti(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMulti, self).__init__()
        self.input_dim = input_dim

        # クエリおよびキーのための線形変換を設定
        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        
    def forward(self, query, key, value):
        # クエリおよびキーの変換を実行
        q = self.linear_q(query)
        k = self.linear_k(key)

        assert q.dim() == 3, f"Expected q to be 3D tensor, but got shape: {q.shape}"
        assert k.dim() == 3, f"Expected k to be 3D tensor, but got shape: {k.shape}"

        # アテンションスコアの計算
        scores = torch.bmm(q, k.transpose(1, 2)) / self.input_dim**0.5
        # スコアをsoftmaxを用いて正規化
        attn_weights = f.softmax(scores, dim=2)

        # 出力を計算
        output = torch.bmm(attn_weights, value)
        return output, attn_weights

class SingleChannelModel(nn.Module):
    def __init__(self, depth, height, width, hidden_dim):
        super(SingleChannelModel, self).__init__()
        
        # ビデオフレームを処理する3D畳み込み
        self.conv3d = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # 空間サイズが半分に縮小されることを考慮
        reduced_depth = depth // 2
        reduced_height = height // 2
        reduced_width = width // 2

        # LSTMの入力サイズを計算
        self.lstm_input_size = reduced_height * reduced_width * 16
        
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # 3D畳み込みのための形状へ変換
        x = x.unsqueeze(1)
        x = f.relu(self.conv3d(x))
        x = self.pool3d(x)

        # この時のxの形状は[batchsize, num channels, depth, height, width](num channels=16)
        
        # LSTMのための形状へ変換
        batch_size, _, depth, height, width = x.size()
        # print('before view',x.shape)
        x = x.view(batch_size, depth, -1)

        # print("Shape before LSTM:", x.shape)
        
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out)
        return out

class MultiTaskMultiChannelModel(nn.Module):
    def __init__(self, channels, depth, height, width, hidden_dim):
        super(MultiTaskMultiChannelModel, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim

        # 各チャンネルのモデルを作成
        self.channel_models = nn.ModuleList([SingleChannelModel(depth, height, width, hidden_dim) for _ in range(channels)])
        # 第一チャンネルを基準としてアテンションメカニズムを使用して他のチャンネルを整列させる
        self.attentions = nn.ModuleList([AttentionMulti(hidden_dim) for _ in range(channels - 1)])

        # 異なるタスクのための複数の出力層を持つ
        self.task_outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(5)])
        
        # ダミーデータがすでに流れたかどうかを示す
        self.initialized = False

    def forward(self, x):
        # 初期化されていない場合、ダミーデータをモデルに流し、途中の次元を決定する
        if not self.initialized:
            dummy_data = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4)).to(x.device)
            for channel_model in self.channel_models:
                _ = channel_model(dummy_data)
            self.initialized = True
        # チャンネルごとにネットワークに入力し、特徴量を出す
        channel_outs = [channel_model(x[:, i, :, :, :]) for i, channel_model in enumerate(self.channel_models)]

        # アテンションを使用してチャンネルを整列させる
        ref_outs = channel_outs[0]
        aligned_outs = [ref_outs]
        for i, attention in enumerate(self.attentions):
            aligned_out, _ = attention(channel_outs[i + 1], ref_outs, channel_outs[i + 1])
            aligned_outs.append(aligned_out)

        # すべての整列された出力を組み合わせる
        combined_out = torch.mean(torch.stack(aligned_outs, dim=0), dim=0)

        last_combined_out = combined_out[:, -1, :]
        # 各タスクのための出力を生成
        task_outputs = [task_out(last_combined_out) for task_out in self.task_outputs]

        outputs = torch.cat(task_outputs, dim=1)
        
        return outputs

class AttentionSeq(nn.Module):
    def __init__(self, input_dim):
        super(AttentionSeq, self).__init__()
        self.linear_q = nn.Linear(input_dim, input_dim)
        self.linear_k = nn.Linear(input_dim, input_dim)
        self.linear_v = nn.Linear(input_dim, input_dim)
        
    def forward(self, query, key, value):
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.linear_q.out_features ** 0.5)
        attn_weights = f.softmax(scores, dim=2)
        output = torch.bmm(attn_weights, v)
        return output, attn_weights
    
class Encoder(nn.Module):
    def __init__(self, depth, height, width, hidden_dim):
        super(Encoder, self).__init__()
        self.conv3d = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.lstm_input_size = (depth // 2) * (height // 2) * (width // 2) * 16
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_dim, batch_first=True)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = f.relu(self.conv3d(x))
        x = self.pool3d(x)
        x = x.view(x.size(0), x.size(2), -1)
        out, (hn, _) = self.lstm(x)
        return hn

class DecoderWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DecoderWithAttention, self).__init__()
        self.attention = AttentionSeq(hidden_dim)
        self.lstm = nn.LSTM(input_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x, encoder_outputs):
        outputs = []
        hidden = None
        for t in range(x.size(1)):
            context, _ = self.attention(x[:, t, :], encoder_outputs, encoder_outputs)
            lstm_out, hidden = self.lstm(torch.cat([x[:, t, :].unsqueeze(1), context], dim=2), hidden)
            out = self.fc(lstm_out.squeeze(1))
            outputs.append(out)
        return torch.stack(outputs, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, channels, depth, height, width, hidden_dim, num_tasks=5):
        super(Seq2Seq, self).__init__()
        self.encoders = nn.ModuleList([Encoder(depth, height, width, hidden_dim) for _ in range(channels)])
        self.sync_linear = nn.Linear(hidden_dim * channels, hidden_dim * channels)
        self.decoder = DecoderWithAttention(height * width, hidden_dim * channels)
        self.task_outputs = nn.ModuleList([nn.Linear(hidden_dim * channels, 1) for _ in range(num_tasks)])

    def forward(self, x):
        device = x.device
        encoder_outputs = [encoder(x[:, i, :, :, :].to(device)) for i, encoder in enumerate(self.encoders)]
        encoder_outputs = torch.cat(encoder_outputs, dim=2).squeeze(0)
        synced_encoder_outputs = self.sync_linear(encoder_outputs)
        decoder_output = self.decoder(x[:, 0, :, :, :].to(device), synced_encoder_outputs)
        last_decoder_output = decoder_output[:, -1, :]
        task_outputs = [task_out(last_decoder_output) for task_out in self.task_outputs]
        outputs = torch.cat(task_outputs, dim=1)
        return outputs