from torch import nn
import torch
import torch.nn.functional as f

# マルチヘッドアテンションのクラス
class MultiHeadAttentionSeq(nn.Module):
    # 線形変換用の層を初期化する
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadAttentionSeq, self).__init__()
        self.num_heads = num_heads # ヘッドの数
        self.head_dim = hidden_dim // num_heads # ヘッドごとの次元数
        
        # 次元数がヘッドの数で割り切れるかチェック
        assert (
            self.head_dim * num_heads == hidden_dim
        ), "Hidden dimension must be divisible by number of heads"
        
        # 線型変換のためのレイヤーを定義
        self.linear_q = nn.Linear(input_dim, hidden_dim)
        self.linear_k = nn.Linear(input_dim, hidden_dim)
        self.linear_v = nn.Linear(input_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim) # 出力用の全結合レイヤー
        
    # このメソッドでマルチヘッドアテンションの処理が行われる
    def forward(self, query, key, value):
        N = query.shape[0]
        q = self.linear_q(query).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # クエリー用の線形変換
        k = self.linear_k(key).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # キー用の線形変換
        v = self.linear_v(value).view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # 値用の線形変換 

        # スコアを計算
        scores = torch.einsum("nqhd,nkhd->nhqk", [q, k]) / (self.head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1) # アテンションの重みをsoftmaxで計算
        
        # アテンションを適用
        output = torch.einsum("nhql,nlhd->nqhd", [attn_weights, v])
        output = output.permute(0, 2, 1, 3).contiguous().view(N, -1, self.head_dim * self.num_heads)
        output = self.fc_out(output) # 最後に全結合
        return output, attn_weights # アテンションの重みも返す
    
class CrossChannelAttention(nn.Module):
    def __init__(self, hidden_dim, num_channels):
        super(CrossChannelAttention, self).__init__()
        self.num_channels = num_channels
        self.attention = MultiHeadAttentionSeq(hidden_dim, hidden_dim, num_heads=8)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):

        # テンソルの形状を整える
        x = x.permute(1, 0, 2)  # [batch, num_channels, hidden_dim]
        output, _ = self.attention(x, x, x)  # アテンションを適用
        output, _ = self.lstm(output) # LSTMを適用
        output = output.permute(1, 0, 2)  # テンソルの形状を戻す
        return output
    
class Encoder(nn.Module):
    def __init__(self, depth, height, width, hidden_dim):
        super(Encoder, self).__init__()

        # 畳み込み層とプーリング層を設定
        self.hidden_dim = hidden_dim
        self.conv3d = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.lstm_input_size = (height // 2) * (width // 2) * 16

        # LSTM層の設定
        self.lstm = nn.LSTM(self.lstm_input_size, hidden_dim, batch_first=True)
    
    def forward(self, x):

        # ３次元の畳み込みとプーリングを実行
        x = x.unsqueeze(1)
        x = f.relu(self.conv3d(x))
        x = self.pool3d(x)

        # LSTMへの入力に適した形に変形
        x = x.view(x.size(0), x.size(2), -1)
        # encoded_x = x + positional_encoding(x.size(1), x.size(2), batch_size=x.size(0)).to(x.device)
        encoded_x = x

        # LSTMを通してエンコード
        out, (hn, _) = self.lstm(encoded_x)
        return hn, self.hidden_dim

class DecoderWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, adjust):
        super(DecoderWithAttention, self).__init__()

        # アテンション機構とLSTMの初期化
        self.attention = MultiHeadAttentionSeq(hidden_dim, hidden_dim,8)
        self.lstm = nn.LSTM(input_dim+hidden_dim, hidden_dim, batch_first=True)

        # 全結合層
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.adjust_linear = nn.Linear(adjust, self.attention.head_dim * self.attention.num_heads)
    
    def forward(self, x, encoder_outputs):
        outputs = []
        hidden = None
        x = x.view(x.size(0), x.size(1), -1)
        encoded_x = x
        # encoded_x = x + positional_encoding(x.size(1), x.size(2), batch_size=x.size(0)).to(x.device)
        for t in range(x.size(1)):

            # タイムステップごとの入力を用意
            x_flat = encoded_x[:, t, :].view(x.size(0), -1).unsqueeze(1)

            # アテンション機構を使ってコンテキストを取得
            x_flat_adjusted = self.adjust_linear(x_flat.squeeze(1))
            context, _ = self.attention(x_flat_adjusted, encoder_outputs, encoder_outputs)

            # LSTMと全結合層で出力を生成
            lstm_out, hidden = self.lstm(torch.cat([x_flat, context], dim=2), hidden)
            out = self.fc(lstm_out.squeeze(1))
            outputs.append(out)
        return torch.stack(outputs, dim=1)

# Seq2Seqクラスの定義
class Seq2Seq(nn.Module):
    def __init__(self, channels, depth, height, width, hidden_dim, num_tasks=5):
        super(Seq2Seq, self).__init__()

        # 複数のエンコーダを用意
        self.encoders = nn.ModuleList([Encoder(depth, height, width, hidden_dim) for _ in range(channels)])
        self.channel_specific_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(channels)])

        # アテンション機構と全結合層
        self.sync_attention = MultiHeadAttentionSeq(hidden_dim, hidden_dim, num_heads=8)
        self.sync_linear = nn.Linear(hidden_dim * channels, hidden_dim * channels)
        self.encoder_output_adjust = nn.Linear(hidden_dim * channels, hidden_dim)

        # デコーダ
        self.decoder = DecoderWithAttention(height * width, hidden_dim, height*width)

        # 複数のタスクに対応する出力層
        self.task_outputs = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_tasks)])
        self.cross_channel_attention = CrossChannelAttention(hidden_dim, channels)
        self.channels = channels

    def forward(self, x):
        device = x.device
        encoder_outputs = []

        # エンコーダの出力を取得
        for i, (encoder, channel_linear) in enumerate(zip(self.encoders, self.channel_specific_linears)):
            output, hidden_dim = encoder(x[:, i, :, :, :].to(device))
            output = channel_linear(output)
            encoder_outputs.append(output)

        # 注意機構で各エンコーダの出力を調整
        adjusted_encoder_outputs = []
        for output in encoder_outputs:
            adjusted_output, _ = self.sync_attention(output, output, output)
            adjusted_encoder_outputs.append(adjusted_output)
        
        # エンコーダの出力を統合
        encoder_outputs = torch.cat(adjusted_encoder_outputs, dim=2).squeeze(0)
        encoder_outputs = encoder_outputs.view(self.encoders.__len__(), encoder_outputs.shape[0], -1)
        
        # 注意機構でさらに統合
        encoder_outputs = self.cross_channel_attention(encoder_outputs)
        encoder_outputs = encoder_outputs.reshape(encoder_outputs.size(1), -1)
        synced_encoder_outputs = self.sync_linear(encoder_outputs)

        # デコーダに通す
        adjusted_encoder_outputs = self.encoder_output_adjust(synced_encoder_outputs)
        decoder_output = self.decoder(x[:, 0, :, :, :].to(device), adjusted_encoder_outputs)

        # 最終的な出力を生成
        last_decoder_output = decoder_output[:, -1, :]
        task_outputs = [task_out(last_decoder_output) for task_out in self.task_outputs]
        outputs = torch.cat(task_outputs, dim=1)

        return outputs
        