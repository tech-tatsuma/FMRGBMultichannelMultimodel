from torch import nn
import torch
import torch

# 3D Convolutional Neural Network
class ConvNet3D(nn.Module):
    def __init__(self):
        super(ConvNet3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            # Continue with more layers as needed...
        )
        self.fc = None

    def forward_features(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)
        
    def forward(self, x):
        if self.fc is None:
            # Get output shape of conv layers
            out = self.forward_features(x)
            out_shape = out.shape[-1]
            # Define fc layer with the obtained output shape
            self.fc = nn.Sequential(
                nn.Linear(out_shape, 128),
                nn.ReLU(),
                nn.Linear(128, 2)  # Single output for regression
            ).to(x.device)
        x = self.forward_features(x)
        x = self.fc(x)
        return x

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim # 入力データのチャンネル数
        self.hidden_dim = hidden_dim # 隠れ状態のチャンネル数

        self.kernel_size = kernel_size # 畳み込みのカーネルのサイズ
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # 畳み込みのパディングサイズ
        self.bias = bias # バイアス項の有無

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) # 入力と隠れ状態を結合して４つのゲートを計算する畳み込み層

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state # 前のタイムステップの隠れ状態とセル状態

        combined = torch.cat([input_tensor, h_cur], dim=1) # 入力データと前のタイムステップの隠れ状態を結合

        combined_conv = self.conv(combined) # 畳み込み演算を実行

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) # 畳み込み出力を４つのゲートに分割
        i = torch.sigmoid(cc_i) # 入力ゲートの活性化関数
        f = torch.sigmoid(cc_f) # 忘却ゲートの活性化関数
        o = torch.sigmoid(cc_o) # 出力ゲートの活性化関数
        g = torch.tanh(cc_g) # セル状態の更新値を計算するための活性化関数(tanh)

        c_next = f * c_cur + i * g # 新しいセル状態の計算
        h_next = o * torch.tanh(c_next) # 新しい隠れ状態の計算

        return h_next, c_next # 新しい隠れ状態とセル状態を返す

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)) # 初期の隠れ状態とセル状態を生成

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size) # カーネルサイズの整合性を確認

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers) # カーネルサイズをレイヤー数に合わせて拡張
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers) # 隠れ状態の次元数をレイヤー数に合わせて拡張
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.') # カーネルサイズと隠れ状態の次元数のリストの長さが一致しない場合にエラーを出す

        self.input_dim  = input_dim # 入力データのチャンネル数
        self.hidden_dim = hidden_dim # 各レイヤーの隠れ状態のチャンネル数
        self.kernel_size = kernel_size # 各レイヤーの畳み込みカーネルのサイズ
        self.num_layers = num_layers # LSTMのレイヤー数
        self.batch_first = batch_first # 入力データのバッチサイズを最初の次元として扱うかどうか
        self.bias = bias # バイアス項の有無
        self.return_all_layers = return_all_layers # 全てのレイヤーの出力を返すかどうか

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1] # 最初のレイヤーの入力次元は入力データのチャンネル数で、それ以降のレイヤーの入力次元は前のレイヤーの隠れ状態のチャンネル数

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))# ConvLSTMCellをレイヤー数だけ追加していく

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        b, seq_len, _, h, w = input_tensor.size() # 入力テンソルのサイズを取得

        if hidden_state is not None:
            raise NotImplementedError()# 隠れ状態の入力は未実装
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))# 隠れ状態の初期化

        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)# シーケンスの長さを取得
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]# レイヤーごとの隠れ状態とセル状態を取得
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])# 各レイヤーのConvLSTMCellに入力データと前のタイムステップの隠れ状態とセル状態を渡して計算
                output_inner.append(h)# 出力をリストに追加

            layer_output = torch.stack(output_inner, dim=1)# リストの出力を結合してテンソルに変換
            cur_layer_input = layer_output# 次のレイヤーの入力として使用

            layer_output = layer_output.permute(0, 2, 1, 3, 4)# テンソルの次元を入れ替えてサイズを変更

            layer_output_list.append(layer_output)# レイヤーごとの出力をリストに追加
            last_state_list.append([h, c])# レイヤーごとの最終の隠れ状態とセル状態をリストに追加

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]# 最後のレイヤーの出力のみを残す
            last_state_list   = last_state_list[-1:]# 最後のレイヤーの隠れ状態とセル状態のみを残す

        return layer_output_list, last_state_list# レイヤーごとの出力と最終の隠れ状態とセル状態を返す

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):# 各レイヤーの隠れ状態とセル状態の初期化
            raise ValueError('`kernel_size` must be tuple or list of tuples')# カーネルサイズの形式が正しくない場合にエラーを出す

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param]*num_layers# パラメータがリストでない場合、リストに変換して各レイヤーに同じパラメータを適用する
        return param

class ConvLSTM_FC(ConvLSTM):
    def __init__(self, *args, **kwargs):
        super(ConvLSTM_FC, self).__init__(*args, **kwargs)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),  # 入力特徴量の次元を変更
            nn.ReLU(),
            nn.Linear(128, 2)  # 出力を2次元に変更
        )

    def forward(self, input_tensor, hidden_state=None):
        layer_output_list, last_state_list = super(ConvLSTM_FC, self).forward(input_tensor, hidden_state=hidden_state)
        output = layer_output_list[-1][:, -1, :, :, :]  # 最後のレイヤーの最後のタイムステップの出力のみを取得
        output = self.gap(output)
        output = output.reshape(output.size(0), -1)  # 出力をフラットな形に変形
        output = self.fc(output)# 変形後の出力を全結合層に入力
        return output # モデルの出力を返す
