from torch import nn
import torch
import torch.nn.functional as f

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