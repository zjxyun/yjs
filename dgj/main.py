import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# 假设这些是你项目中的模块
from dgj.data_preprocessing import load_and_process_data
from dgj.data_preprocessing_2 import load_and_process_data_2
from dgj.module import PIDL_Model
from dgj.physics_loss import calculate_physics_loss

# 导入新写的 metrics 模块
from dgj.metrics import calculate_metrics_numpy

def calculate_test_loss(model, data_loader, device, scaler_x, scaler_y):
    model.eval()
    loss_mse = nn.MSELoss()
    epoch_loss_total = 0.0

    with torch.no_grad():
        for bx, by, bp in data_loader:
            bx, by, bp = bx.to(device), by.to(device), bp.to(device)

            # 模型预测 (归一化状态 [0,1])
            pred_norm = model(bx)

            loss_data = loss_mse(pred_norm, by)
            loss_physics = calculate_physics_loss(
                model, bx, pred_norm, bp, scaler_x, scaler_y
            )

            loss_total = loss_data + 0.1 * loss_physics

            epoch_loss_total += loss_total.item()

    return epoch_loss_total


def evaluate_model_detailed(model, data_loader, device, scaler_y):
    """
    在测试集上评估模型，并计算论文所需的 RMSE, MAE, R2, VAF
    """
    model.eval()


    preds_list = []
    targets_list = []


    # 1. 收集所有预测值和真实值
    with torch.no_grad():
        for bx, by, bp in data_loader:
            bx = bx.to(device)

            # 模型预测 (归一化状态 [0,1])
            pred_norm = model(bx)




            preds_list.append(pred_norm.cpu().numpy())
            targets_list.append(by.numpy())

        # 拼接所有批次
    preds_norm = np.concatenate(preds_list, axis=0)  # shape: (N, 2)
    targets_norm = np.concatenate(targets_list, axis=0)  # shape: (N, 2)

    # 2. 反归一化 (还原为 mm)
    # inverse_transform 期望输入是 (N, 2)
    preds_real = scaler_y.inverse_transform(preds_norm)
    targets_real = scaler_y.inverse_transform(targets_norm)

    metrics = {}
    targets_name = ['y1 (Lower)', 'y2 (Upper)']

    for i, name in enumerate(targets_name):
        y_true_i = targets_real[:, i]
        y_pred_i = preds_real[:, i]

        rmse, mae, r2, vaf = calculate_metrics_numpy(y_true_i, y_pred_i)

        metrics[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'VAF': vaf
        }

    return metrics

def train_pidl():
    # ==========================================
    # 1. 数据准备与加载
    # ==========================================
    print(f"{'=' * 20} 正在加载数据 {'=' * 20}")
    # 注意：通常训练集和测试集应该来自不同的切分，这里假设你已经处理好或者为了演示读了同一个
    X_train, Y_train, P_train, scaler_x_train, scaler_y_train = load_and_process_data('data/train_dataset.csv')
    X_test, Y_test, P_test, scaler_x_test, scaler_y_test = load_and_process_data_2('data/test_dataset.csv')  # 这里建议换成 test_dataset.csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 运行设备: {device}")

    BATCH_SIZE = 64

    train_dataset = TensorDataset(X_train, Y_train, P_train)
    test_dataset = TensorDataset(X_test, Y_test, P_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    print(f"📊 训练集样本数: {len(train_dataset)} | 批次数: {len(train_loader)}")
    print(f"📊 测试集样本数: {len(test_dataset)}  | 批次数: {len(test_loader)}")

    # ==========================================
    # 2. 模型与优化器初始化
    # ==========================================
    model = PIDL_Model().to(device)

    # # 打印初始物理参数
    # print(f"{'=' * 20} 初始物理参数 {'=' * 20}")
    # print(f"B_low (Damping): {model.B_low.item():.4f}")
    # print(f"B_up (Damping): {model.B_up.item():.4f}")
    # print(f"K_low (Stiffness): {model.K_low.item():.4f}")
    # print(f"K_up (Stiffness): {model.K_up.item():.4f}")

    optimizer = torch.optim.Adam([
        {'params': model.get_net_parameters(), 'lr': 0.001},
        {'params': model.get_phy_parameters(), 'lr': 0.01, 'weight_decay': 1e-4}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True, threshold=1e-3)
    # 你可以在训练前打印一下，确保 B 和 K 在里面
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"优化参数: {name}, Shape: {param.shape}")

    loss_mse = nn.MSELoss()

    # ==========================================
    # 3. 训练循环
    # ==========================================
    epochs = 2000


    print(f"\n{'=' * 20} 开始训练 (Total Epochs: {epochs}) {'=' * 20}")
    # 打印表头
    header = f"{'Epoch':<8} | {'Loss':<12} | {'Data Loss':<12} | {'Phy Loss':<12} | {'B_low':<12} | {'B_up':<12} | {'K_low':<12} | {'K_up':<12} "
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    best_val_rmse = 100.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_total = 0.0
        total_data_loss = 0
        total_phy_loss = 0

        # --- 批次训练 ---
        for bx, by, bp in train_loader:
            bx, by, bp = bx.to(device), by.to(device), bp.to(device)

            optimizer.zero_grad()
            y_pred = model(bx)

            # 计算损失
            # 使用 abs() 绝对值误差 (L1)，对应公式中没有平方项
            # 如果不加 abs，误差会正负抵消甚至导致负无穷，无法训练
            # diff_i = torch.abs(by[:, 0] - y_pred[:, 0])
            # diff_j = torch.abs(by[:, 1] - y_pred[:, 1])

            # 求和后再除以 N (Batch Size)
            loss_data = loss_mse(y_pred,by)
            loss_physics = calculate_physics_loss(
                model, bx, y_pred, bp, scaler_x_train, scaler_y_train
            )



            loss_total = loss_data + 0.1 * loss_physics

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_total += loss_total.item()
            total_data_loss += loss_data.item()
            total_phy_loss += loss_physics.item()

        # 计算平均 Loss
        avg_loss = epoch_loss_total / len(train_loader)
        avg_data = total_data_loss / len(train_loader)
        avg_phy = total_phy_loss / len(train_loader)


        # ==========================================
        # 4. 定期验证与日志 (每 10 轮)
        # ==========================================
        if (epoch + 1) % 10 == 0:
            # 使用新写的详细评估函数
            # metrics = evaluate_model_detailed(model, test_loader, device, scaler_y_test)


            # 获取当前物理参数值
            real_B, real_K = model.get_real_physics_params()
            curr_B_low = real_B[0].item()
            curr_K_low = real_K[0].item()
            curr_B_up = real_B[1].item()
            curr_K_up = real_K[1].item()
            # curr_F_bias = (model.F_bias_coeff * model.F_bias_scale).item()



            # 格式化打印一行日志
            log_line = (f"{epoch:<8} | {avg_loss:<12.6f} | {avg_data:<12.6f} | {avg_phy:<12.6f}"
                        f"{curr_B_low:<12.2f} | {curr_B_up:<12.2f} | {curr_K_low:<12.2f} | {curr_K_up:<12.2f}"
                        )
            print(log_line)

        # 每50轮进行一次详细评估
        if epoch % 50 == 0:
            metrics = evaluate_model_detailed(model, test_loader, device, scaler_y_test)
            test_loss = calculate_test_loss(model, test_loader, device, scaler_x_test, scaler_y_test)
            print("-" * 60)
            df_metrics = pd.DataFrame(metrics).T
            df_metrics = df_metrics[['RMSE', 'MAE', 'R2', 'VAF']]
            print(df_metrics.round(4))
            print("-" * 60)

            val_loss = test_loss / len(test_loader)
            scheduler.step(val_loss)
            val_rmse = df_metrics.RMSE.mean()
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_model.pth')  # 只保存预测最准的那个

            # # 使用 Pandas 打印漂亮的表格
            # print("-" * 50)
            # df_metrics = pd.DataFrame(metrics).T  # 转置，行是 y1/y2，列是指标
            # # 格式化列名顺序
            # df_metrics = df_metrics[['RMSE', 'MAE', 'R2', 'VAF']]
            #
            # print(df_metrics.round(4))  # 保留4位小数打印
            # print("-" * 50)



    print(f"\n{'=' * 20} 训练结束,epoch:{best_epoch},best_val_rmse:{best_val_rmse}{'=' * 20}")



    return model


if __name__ == '__main__':
    train_pidl()