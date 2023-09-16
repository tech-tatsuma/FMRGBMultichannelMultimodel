import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F

# スピアマンの順位相関係数を計算する関数
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

# カスタム損失関数
def custom_loss(y_pred, y_true):
    # 予測値と実値の二乗誤差を計算
    # mse_loss = F.mse_loss(y_pred, y_true)
    
    # スピアマンの順位相関係数を計算
    spearman_corr = spearman_rank_correlation(y_true, y_pred)
    rank_loss = 1 - spearman_corr
    
    # 最終的な損失を計算
    # final_loss = mse_loss + rank_loss
    return rank_loss
