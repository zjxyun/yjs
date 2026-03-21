
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt  # 【新增】导入画图库

# 假设这些是你项目中的模块
from dgj.src.data_preprocessing import load_and_process_data
from dgj.src.data_preprocessing_2 import load_and_process_data_2
from dgj.src.module import PIDL_Model
from dgj.src.physics_loss import calculate_physics_loss
from dgj.src.metrics import calculate_metrics_numpy


# ... (calculate_test_loss 函数保持不变) ...
def calculate_test_loss(model, data_loader, device, scaler_x, scaler_y):
    model.eval()
    loss_mse = nn.MSELoss()
    epoch_loss_total = 0.0
    count = 0
    with torch.no_grad():
        for bx, by, bp in data_loader:
            bx, by, bp = bx.to(device), by.to(device), bp.to(device)
            pred_norm = model(bx)
            loss_data = loss_mse(pred_norm, by)
            loss_physics = calculate_physics_loss(model, bx, pred_norm, bp, scaler_x, scaler_y)
            loss_total = loss_data + 0.1 * loss_physics
            epoch_loss_total += loss_total.item()
            count += 1
    return epoch_loss_total / count


# ... (evaluate_model_detailed 函数保持不变) ...
def evaluate_model_detailed(model, data_loader, device, scaler_y):
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
    targets_name = ['y1 (Lower)', 'y2 (Upper)']
    for i, name in enumerate(targets_name):
        y_true_i = targets_real[:, i]
        y_pred_i = preds_real[:, i]
        rmse, mae, r2, vaf = calculate_metrics_numpy(y_true_i, y_pred_i)
        metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'VAF': vaf}
    return metrics


# 【新增】画图函数
def plot_results(history, model, test_loader_sorted, device, scaler_y):
    # 1. 绘制 Loss 曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Total Train Loss')
    plt.plot(history['data_loss'], label='Data Loss', linestyle='--')
    plt.plot(history['phy_loss'], label='Physics Loss', linestyle=':')
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 绘制 预测 vs 真实 对比图 (取测试集前200个点)
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for bx, by, bp in test_loader_sorted:
            bx = bx.to(device)
            pred_norm = model(bx)
            preds_list.append(pred_norm.cpu().numpy())
            targets_list.append(by.numpy())

    preds_real = scaler_y.inverse_transform(np.concatenate(preds_list, axis=0))
    targets_real = scaler_y.inverse_transform(np.concatenate(targets_list, axis=0))

    # 只画前 300 个点，看得清楚一点
    limit = 300

    plt.subplot(1, 2, 2)
    # 画下缸 (y1)
    plt.plot(targets_real[:limit, 0], label='True Lower', color='grey', alpha=0.6)
    plt.plot(preds_real[:limit, 0], label='Pred Lower', color='red', linewidth=1.5)
    plt.title(f'Prediction vs Truth (First {limit} steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Stroke (mm)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def train_pidl():
    print(f"{'=' * 20} 正在加载数据 {'=' * 20}")
    X_train, Y_train, P_train, scaler_x_train, scaler_y_train = load_and_process_data('data/train_dataset.csv')
    X_test, Y_test, P_test, scaler_x_test, scaler_y_test = load_and_process_data_2('data/test_dataset.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 运行设备: {device}")

    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train, Y_train, P_train)
    test_dataset = TensorDataset(X_test, Y_test, P_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader_sorted = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = PIDL_Model().to(device)
    optimizer = torch.optim.Adam([
        {'params': model.get_net_parameters(), 'lr': 0.001},
        {'params': model.get_phy_parameters(), 'lr': 0.01, 'weight_decay': 1e-4}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                           verbose=False, threshold=1e-3)
    loss_mse = nn.MSELoss()

    epochs = 2000
    history = {'train_loss': [], 'data_loss': [], 'phy_loss': []}
    best_val_rmse = 100.0
    best_epoch = 0

    print(f"\n{'=' * 20} 开始训练 (Total Epochs: {epochs}) {'=' * 20}")

    # 设置总进度条
    # dynamic_ncols=True 让进度条自动适应窗口宽度
    main_loop = tqdm(range(1, epochs + 1), desc='Training', dynamic_ncols=True, position=0, leave=True)

    for epoch in main_loop:
        model.train()
        epoch_loss_total = 0.0
        total_data_loss = 0
        total_phy_loss = 0

        for bx, by, bp in train_loader:
            bx, by, bp = bx.to(device), by.to(device), bp.to(device)
            optimizer.zero_grad()
            y_pred = model(bx)
            loss_data = loss_mse(y_pred, by)
            loss_physics = calculate_physics_loss(model, bx, y_pred, bp, scaler_x_train, scaler_y_train)
            loss_total = loss_data + 0.1 * loss_physics
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_total += loss_total.item()
            total_data_loss += loss_data.item()
            total_phy_loss += loss_physics.item()

        avg_loss = epoch_loss_total / len(train_loader)
        avg_data = total_data_loss / len(train_loader)
        avg_phy = total_phy_loss / len(train_loader)

        history['train_loss'].append(avg_loss)
        history['data_loss'].append(avg_data)
        history['phy_loss'].append(avg_phy)

        # 进度条右侧只显示最简信息
        main_loop.set_postfix(loss=f"{avg_loss:.5f}")

        # ====================================================
        # 每 50 轮输出详细报表
        # ====================================================
        if epoch % 50 == 0:
            # 1. 计算测试集 Loss (用于 Scheduler)
            test_loss = calculate_test_loss(model, test_loader, device, scaler_x_test, scaler_y_test)
            val_loss = test_loss / len(test_loader)
            scheduler.step(val_loss)

            # 2. 计算详细指标 (RMSE, MAE, R2, VAF)
            metrics = evaluate_model_detailed(model, test_loader, device, scaler_y_test)
            low = metrics['y1 (Lower)']
            up = metrics['y2 (Upper)']

            # 3. 获取物理参数
            real_B, real_K = model.get_real_physics_params()
            k_L, k_U = real_K[0].item(), real_K[1].item()
            b_L, b_U = real_B[0].item(), real_B[1].item()

            # 4. 判断是否保存模型
            val_rmse = (low['RMSE'] + up['RMSE']) / 2
            save_tag = ""
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), './models/best_model.pth')
                save_tag = "🌟 [New Best]"

            # 5. 构建美观的日志字符串 (使用 f-string 表格对齐)
            msgs = []
            border = "-" * 80

            msgs.append(border)
            msgs.append(f"📅 Epoch {epoch:4d} | Avg RMSE: {val_rmse:.4f} {save_tag}")
            msgs.append(border)
            # 表头
            msgs.append(f" {'Target':<10} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10} | {'VAF(%)':<10}")
            msgs.append("-" * 65)
            # y1 数据
            msgs.append(
                f" {'y1 (Low)':<10} | {low['RMSE']:<10.4f} | {low['MAE']:<10.4f} | {low['R2']:<10.4f} | {low['VAF']:<10.2f}")
            # y2 数据
            msgs.append(
                f" {'y2 (Up)':<10} | {up['RMSE']:<10.4f} | {up['MAE']:<10.4f} | {up['R2']:<10.4f} | {up['VAF']:<10.2f}")
            msgs.append(border)
            # 物理参数
            msgs.append("⚛️  Physics Parameters:")
            msgs.append(f"   🔸 Stiffness (K): Low = {k_L:.3e}  |  Up = {k_U:.3e}")
            msgs.append(f"   🔹 Damping   (B): Low = {b_L:.3e}  |  Up = {b_U:.3e}")

            # 6. 使用 tqdm.write 打印 (关键步骤)
            tqdm.write("\n".join(msgs))

    # 训练结束
    print(f"\n{'=' * 20} 训练结束, Best Epoch:{best_epoch}, RMSE:{best_val_rmse:.4f} {'=' * 20}")

    model.load_state_dict(torch.load('./models/best_model.pth'))
    plot_results(history, model, test_loader_sorted, device, scaler_y_test)

    return model


if __name__ == '__main__':
    train_pidl()