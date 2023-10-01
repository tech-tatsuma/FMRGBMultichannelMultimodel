import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch.nn.functional as F
import sys
import scipy.stats

def spearman_rank_correlation(truth, pred):

    # truthとpredをNumPy配列に変換
    truth = truth.numpy()
    pred = pred.numpy()
    
    # truthとpredをNumPy配列に変換
    sum_truth = np.sum(truth, axis=1)
    sum_pred = np.sum(pred, axis=1)
    
    # 合計値に基づいて順位を計算
    rank_truth = scipy.stats.rankdata(sum_truth)
    rank_pred = scipy.stats.rankdata(sum_pred)
    
    # nにはtruth(pred)のサイズを格納
    n = len(truth)
    
    # nが2未満の場合、スピアマンの順位相関係数は定義されないため、適切な値（例えば0）を返す
    if n < 2:
        return torch.tensor(0.0)
    
    # スピアマンの順位相関係数を計算
    sum_diff_sq = torch.sum((torch.tensor(rank_truth) - torch.tensor(rank_pred))**2).float()
    spearman_corr = 1 - (6 * sum_diff_sq) / (n * (n**2 - 1))
    
    # スピアマンの順位相関係数を返す
    return spearman_corr

# スピアマンの順位相関係数を向上させた評価関数
def validation_function(y_pred, y_true):
    # スピアマンの順位相関係数を計算
    spearman_corr = spearman_rank_correlation(y_true, y_pred)
    rank_loss = 1 - spearman_corr
    
    return rank_loss

def soft_rank_loss(y_pred, y_true, tau=1):
    # 各データポイントに対して、各予測値および真の値の差を計算
    pred_diffs = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
    true_diffs = y_true.unsqueeze(2) - y_true.unsqueeze(1)
    
    # Sigmoid関数を使ってsoft順位を計算
    pred_order_probs = torch.sigmoid(-pred_diffs / tau)
    true_order_probs = torch.sigmoid(-true_diffs / tau)
    
    # 順位の差に基づく損失の計算
    return torch.mean((pred_order_probs - true_order_probs).pow(2))

