import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# スピアマンの順位相関係数を計算する関数
def spearman_rank_correlation(truth, pred):
    rank_truth = torch.argsort(truth).float()
    rank_pred = torch.argsort(pred).float()
    n = truth.size(0)
    sum_diff_sq = torch.sum((rank_truth - rank_pred)**2)
    spearman_corr = 1 - (6 * sum_diff_sq) / (n * (n**2 - 1))
    return spearman_corr

# カスタム損失関数
def custom_loss(y_pred, y_true, all_data):
    mse_loss = F.mse_loss(y_pred, y_true)
    random_samples = random.sample(all_data, 10)
    random_samples = torch.stack(random_samples)
    test_score = torch.sum(y_true)
    random_scores = torch.sum(random_samples, dim=1)
    all_scores = torch.cat([test_score.view(1), random_scores])
    pred_test_score = torch.sum(y_pred)
    all_pred_scores = torch.cat([pred_test_score.view(1), random_scores])
    spearman_corr = spearman_rank_correlation(all_scores, all_pred_scores)
    rank_loss = 1 - spearman_corr
    final_loss = mse_loss + rank_loss
    return final_loss
