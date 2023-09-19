import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F

# スピアマンの順位相関係数
def spearman_rank_correlation(truth, pred):
    # truthのソートを実行
    rank_truth = torch.argsort(truth).float()
    # predのソートを実行
    rank_pred = torch.argsort(pred).float()
    # nにはtruth(pred)のサイズを格納
    n = truth.size(0)
    # スピアマンの順位相関係数を計算
    sum_diff_sq = torch.sum((rank_truth - rank_pred)**2)
    spearman_corr = 1 - (6 * sum_diff_sq) / (n * (n**2 - 1))
    # スピアマンの順位相関係数を返す
    return spearman_corr

# スピアマンの順位相関係数を向上させた評価関数
def validation_function(y_pred, y_true):
    # 予測値と実値の二乗誤差を計算
    
    # スピアマンの順位相関係数を計算
    spearman_corr = spearman_rank_correlation(y_true, y_pred)
    rank_loss = 1 - spearman_corr
    
    return rank_loss


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes, device):
        super(OrdinalRegressionLoss, self).__init__()
        # 閾値を学習するためのパラメータ
        self.thresholds = nn.Parameter(torch.arange(0, num_classes - 1).float()).to(device)
        self.device = device
    
    def forward(self, y_pred, y_true):

        # 5つの値を足し合わせる(芸術点の総合点)
        y_pred_sum = torch.sum(y_pred, dim=1)
        y_true_sum = torch.sum(y_true, dim=1)
        
        # 累積確率を計算
        cum_probs = torch.sigmoid(y_pred_sum.unsqueeze(1) - self.thresholds.unsqueeze(0))
        
        # 実際の順序ラベルに基づいて累積確率を選択
        gt_probs = cum_probs[torch.arange(y_true_sum.shape[0]), y_true_sum.long()]
        
        # 累積リンク損失（Cumulative Link Loss）を計算
        # ラベルが0より大きい場合と0の場合で損失を分けて計算
        loss = -torch.log(gt_probs) - torch.log(1 - cum_probs[:, 0])
        index_tensor = torch.arange(y_true_sum.shape[0], device=self.device)[y_true_sum > 0]
        y_true_sum_device = y_true_sum[y_true_sum > 0].long().to(self.device) - 1
        loss[y_true_sum > 0] += -torch.log(1 - cum_probs[index_tensor, y_true_sum_device])
        
        return loss.mean()