import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 复用现有的模块
from src.data_preprocessing import load_and_process_data
from src.data_preprocessing_2 import load_and_process_data_2
from src.metrics import calculate_metrics_numpy


# ==========================================
# 1. 定义纯 DNN 模型 (移除物理参数)
# ==========================================
class PureDNN(nn.Module):
    def __init__(self, layers=[9, 64, 64, 64, 64, 2]):
        super(PureDNN, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3]),
            nn.ReLU(),
            nn.Linear(layers[3], layers[4]),
            nn.ReLU(),
            nn.Linear(layers[4], layers[5])
        )

    def forward(self, x):
        return self.dnn(x)


# ==========================================
# 2. 评估函数 (保持不变)
# ==========================================
def evaluate_model(model, data_loader, device, scaler_y):
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for bx, by, bp in data_loader:
            bx = bx.to(device)
            pred_norm = model(bx)
            preds_list.append(pred_norm.cpu().numpy())
            targets_list.append(by.numpy())

    preds_norm = np.concatenate(preds_list, axis=0)
    targets_norm = np.concatenate(targets_list, axis=0)
    preds_real = scaler_y.inverse_transform(preds_norm)
    targets_real = scaler_y.inverse_transform(targets_norm)

    metrics = {}
    names = ['y1 (Low)', 'y2 (Up)']
    for i, name in enumerate(names):
        rmse, mae, r2, vaf = calculate_metrics_numpy(targets_real[:, i], preds_real[:, i])
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'VAF': vaf}
    return metrics, targets_real, preds_real


# ==========================================
# 3. 训练主函数 (Pure DNN)
# ==========================================
def train_pure_dnn():
    print(f"{'=' * 20} 启动纯 DNN 对比实验 {'=' * 20}")

    # 1. 加载数据
    X_train, Y_train, P_train, scaler_x, scaler_y = load_and_process_data('data/train_dataset.csv')
    X_test, Y_test, P_test, _, _ = load_and_process_data_2('data/test_dataset.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64

    train_loader = DataLoader(TensorDataset(X_train, Y_train, P_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test, P_test), batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化模型
    model = PureDNN().to(device)

    # 优化器 (只优化网络权重)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                           verbose=False)
    loss_fn = nn.MSELoss()

    epochs = 2000
    best_rmse = float('inf')
    history_loss = []

    # 3. 训练循环
    main_loop = tqdm(range(1, epochs + 1), desc='DNN Training', ncols=100)

    for epoch in main_loop:
        model.train()
        train_loss = 0.0

        for bx, by, bp in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            y_pred = model(bx)

            # 【关键区别】只计算 Data Loss (MSE)，没有 Physics Loss
            loss = loss_fn(y_pred, by)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        history_loss.append(avg_loss)
        main_loop.set_postfix(loss=f"{avg_loss:.5f}")

        # 定期验证
        if epoch % 50 == 0:
            # 计算测试集 Loss 用于 Scheduler
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for bx, by, _ in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    test_loss += loss_fn(model(bx), by).item()
            test_loss /= len(test_loader)
            scheduler.step(test_loss)

            # 计算详细指标
            metrics, _, _ = evaluate_model(model, test_loader, device, scaler_y)
            curr_rmse = (metrics['y1 (Low)']['RMSE'] + metrics['y2 (Up)']['RMSE']) / 2

            if curr_rmse < best_rmse:
                best_rmse = curr_rmse
                torch.save(model.state_dict(), 'best_dnn_model.pth')

                # 打印日志
                msg = f"\n📅 Epoch {epoch} | Best RMSE: {best_rmse:.2f}"
                tqdm.write(msg)

    # 4. 最终评估与画图
    print(f"\n{'=' * 20} 训练结束. Best RMSE: {best_rmse:.4f} {'=' * 20}")

    model.load_state_dict(torch.load('./models/best_dnn_model.pth'))
    metrics, targets, preds = evaluate_model(model, test_loader, device, scaler_y)

    # 打印最终表格
    print("\n🏆 Pure DNN Final Results:")
    print("-" * 65)
    # 修改表头：增加 VAF(%)
    print(f"{'Target':<10} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8} | {'VAF(%)':<8}")
    print("-" * 65)

    for k, v in metrics.items():
        # 修改内容：增加 v['VAF'] 的打印
        print(f"{k:<10} | {v['RMSE']:<8.4f} | {v['MAE']:<8.4f} | {v['R2']:<8.4f} | {v['VAF']:<8.2f}")

    print("-" * 65)

    # # 画图对比
    # plt.figure(figsize=(12, 5))
    #
    # # Loss
    # plt.subplot(1, 2, 1)
    # plt.plot(history_loss, label='MSE Loss')
    # plt.title('DNN Training Loss')
    # plt.yscale('log')
    # plt.legend()
    #
    # # Pred vs True
    # plt.subplot(1, 2, 2)
    # limit = 300
    # plt.plot(targets[:limit, 0], label='Truth', color='gray', alpha=0.7)
    # plt.plot(preds[:limit, 0], label='DNN Pred', color='blue', linestyle='--')
    # plt.title('Pure DNN Prediction (Lower Cylinder)')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    train_pure_dnn()