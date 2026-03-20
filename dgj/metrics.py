import torch
import numpy as np
import pandas as pd  # 用于美化打印


def calculate_metrics_numpy(y_true, y_pred):
    """
    计算 RMSE, MAE, R2, VAF
    y_true, y_pred: numpy array (N,)
    """
    # 1. RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mse = np.mean(y_pred - y_true)

    # 2. MAE
    mae = np.mean(np.abs(y_pred - y_true))

    # 3. R2 Score
    # R2 = 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # 防止分母为0 (极小概率)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # 4. VAF (Variance Accounted For)
    # VAF = [1 - var(y - y_hat) / var(y)] * 100
    var_error = np.var(y_true - y_pred)
    var_true = np.var(y_true)
    vaf = (1 - var_error / (var_true + 1e-8)) * 100

    return rmse, mae, r2, vaf