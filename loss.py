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
    truth = truth.cpu().numpy()
    pred = pred.cpu().numpy()
    
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

def soft_rank_loss(y_pred, y_true, tau=0.1):
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

def softargmax(scores, beta=10):
    """ソフトな順位を計算する関数"""
    prob = F.softmax(beta * scores, dim=-1)
    positions = torch.arange(scores.size(-1), dtype=torch.float32, device=scores.device)
    soft_positions = torch.sum(prob * positions, dim=-1)
    return soft_positions

def rank_loss(pred, true):
    # ソフトランク関数を利用する
    pred_rank = softargmax(pred)
    true_rank = softargmax(true)

    # ランクの違いを計算する
    rank_diff = pred_rank - true_rank
    loss = (rank_diff ** 2).mean()

    return loss

def pairwise_ranking_loss(pred, true):
    # 形を変化させる
    pred = pred.view(-1, 1)
    true = true.view(-1, 1)
    
    # Create a matrix of pairwise differences for pred and true
    pred_diffs = pred - pred.t()
    true_diffs = true - true.t()

    # Create a mask for pairs where true[i] > true[j]
    mask = true_diffs > 0

    loss = F.binary_cross_entropy_with_logits(pred_diffs, mask.float(), reduction='none')
    return loss.mean()