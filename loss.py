import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
import sys

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

def soft_rank_loss(y_pred, y_true, tau=0.01):
    # 各データポイントに対して、各予測値および真の値の差を計算
    pred_diffs = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
    true_diffs = y_true.unsqueeze(2) - y_true.unsqueeze(1)
    
    # Sigmoid関数を使ってsoft順位を計算
    # pred_order_probs = torch.sigmoid(-pred_diffs / tau)
    pred_order_probs = torch.tanh(-pred_diffs / tau)
    # true_order_probs = torch.sigmoid(-true_diffs / tau)
    true_order_probs = torch.tanh(-true_diffs / tau)

    # 各データポイントごとに順位の差に基づく損失の計算
    losses = (pred_order_probs - true_order_probs).pow(2)
    
    # 各データポイントの損失の平均を計算し、その後の損失を計算
    return losses.mean(dim=[1,2]).mean()

def soft_rank(y):
    """
    Compute the soft rank of y using the softmax function.
    """
    return F.softmax(y, dim=-1) * torch.arange(1, y.size(-1) + 1, device=y.device)

def spearman_rank_loss(y_pred, y_true):
    # Compute soft ranks
    true_ranks = soft_rank(y_true)
    pred_ranks = soft_rank(y_pred)
    
    # Compute the rank differences squared
    rank_diffs_squared = (true_ranks - pred_ranks).pow(2)
    
    # Average the rank differences squared to get the loss
    loss = rank_diffs_squared.mean()
    
    return loss