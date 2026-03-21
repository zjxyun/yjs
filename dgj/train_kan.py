import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from module_kan import PIDL_Model

from data_preprocessing import load_and_process_data
from data_preprocessing_2 import load_and_process_data_2
from physics_loss import calculate_physics_loss
from metrics import calculate_metrics_numpy


# ------------------------------------------------------------------------------
# 辅助函数：测试集 Loss 计算
# ------------------------------------------------------------------------------
def calculate_test_loss(model, data_loader, device, scaler_x, scaler_y):
    model.eval()
    loss_mse = nn.MSELoss()
    epoch_loss_total = 0.0
    count = 0
    with torch.no_grad():
        for bx, by, bp in data_loader:
            bx, by, bp = bx.to(device), by.to(device), bp.to(device)
            pred_norm = model(bx)

            # 数据 Loss
            loss_data = loss_mse(pred_norm, by)
            # 物理 Loss
            loss_physics = calculate_physics_loss(model, bx, pred_norm, bp, scaler_x, scaler_y)

            # 测试时通常不加正则化项
            loss_total = loss_data + 0.1 * loss_physics
            epoch_loss_total += loss_total.item()
            count += 1
    return epoch_loss_total / count


# ------------------------------------------------------------------------------
# 辅助函数：详细评估
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# 辅助函数：绘图
# ------------------------------------------------------------------------------
def plot_results(history, model, test_loader_sorted, device, scaler_y):
    plt.figure(figsize=(15, 5))

    # 1. Loss 曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Total Train Loss')
    plt.plot(history['data_loss'], label='Data Loss', linestyle='--')
    plt.plot(history['phy_loss'], label='Physics Loss', linestyle=':')

    if 'reg_loss' in history and len(history['reg_loss']) > 0:
        plt.plot(history['reg_loss'], label='KAN Reg Loss', linestyle='-.', alpha=0.7)

    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # 2. 预测 vs 真实
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

    limit = 300  # 只画前300个点

    plt.subplot(1, 3, 2)
    plt.plot(targets_real[:limit, 0], label='True Lower', color='grey', alpha=0.6)
    plt.plot(preds_real[:limit, 0], label='Pred Lower', color='red', linewidth=1.5)
    plt.title(f'Lower Cylinder (First {limit})')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(targets_real[:limit, 1], label='True Upper', color='grey', alpha=0.6)
    plt.plot(preds_real[:limit, 1], label='Pred Upper', color='blue', linewidth=1.5)
    plt.title(f'Upper Cylinder (First {limit})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# 主训练函数
# ------------------------------------------------------------------------------
def train_kan_pidl():
    print(f"{'=' * 20} 正在加载数据 {'=' * 20}")
    X_train, Y_train, P_train, scaler_x_train, scaler_y_train = load_and_process_data('data/train_dataset.csv')
    X_test, Y_test, P_test, scaler_x_test, scaler_y_test = load_and_process_data_2('data/test_dataset.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 运行设备: {device}")

    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train, Y_train, P_train)
    test_dataset = TensorDataset(X_test, Y_test, P_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    # 用于更新 grid 的数据加载器
    grid_update_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader_sorted = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 初始化模型
    model = PIDL_Model().to(device)

    # 优化器
    optimizer = torch.optim.Adam([
        {'params': model.get_net_parameters(), 'lr': 0.001},
        {'params': model.get_phy_parameters(), 'lr': 0.01}
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                           verbose=False)
    loss_mse = nn.MSELoss()

    epochs = 1000
    history = {'train_loss': [], 'data_loss': [], 'phy_loss': [], 'reg_loss': []}
    best_val_rmse = 100.0
    best_epoch = 0

    print(f"\n{'=' * 20} 开始训练 KAN-PIDL (Total Epochs: {epochs}) {'=' * 20}")
    main_loop = tqdm(range(1, epochs + 1), desc='Training', dynamic_ncols=True)

    for epoch in main_loop:
        # ====================================================
        # 【修复 2】正确调用 KAN Grid Update
        # ====================================================
        if epoch < 500 and epoch % 50 == 0:
            model.eval()  # 必须在 eval 模式或 no_grad 下准备数据，但 update_grid 内部会修改 buffer
            try:
                x_sample, _, _ = next(iter(grid_update_loader))
                x_sample = x_sample.to(device)

                # 兼容性检查：module.py 中定义的属性名是 kan 还是 dnn
                kan_module = None
                if hasattr(model, 'kan'):
                    kan_module = model.kan
                elif hasattr(model, 'dnn'):
                    kan_module = model.dnn

                if kan_module is not None:
                    # 关键修改：调用 forward(update_grid=True) 而不是 .update_grid()
                    kan_module(x_sample, update_grid=True)

            except Exception as e:
                pass  # 数据不足或出错则跳过

        # ====================================================
        # 训练循环
        # ====================================================
        model.train()
        epoch_loss_total = 0.0
        total_data_loss = 0
        total_phy_loss = 0
        total_reg_loss = 0

        for bx, by, bp in train_loader:
            bx, by, bp = bx.to(device), by.to(device), bp.to(device)
            optimizer.zero_grad()

            y_pred = model(bx)

            # Loss 计算
            loss_data = loss_mse(y_pred, by)
            loss_physics = calculate_physics_loss(model, bx, y_pred, bp, scaler_x_train, scaler_y_train)

            # KAN 正则化
            loss_reg = model.get_kan_reg_loss()
            reg_lambda = 1e-4

            loss_total = loss_data + 0.1 * loss_physics + reg_lambda * loss_reg

            loss_total.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_total += loss_total.item()
            total_data_loss += loss_data.item()
            total_phy_loss += loss_physics.item()
            total_reg_loss += loss_reg.item()

        avg_loss = epoch_loss_total / len(train_loader)

        history['train_loss'].append(avg_loss)
        history['data_loss'].append(total_data_loss / len(train_loader))
        history['phy_loss'].append(total_phy_loss / len(train_loader))
        history['reg_loss'].append(total_reg_loss / len(train_loader))

        main_loop.set_postfix(loss=f"{avg_loss:.5f}")

        # ====================================================
        # 验证与保存
        # ====================================================
        if epoch % 50 == 0:
            test_loss = calculate_test_loss(model, test_loader, device, scaler_x_test, scaler_y_test)
            scheduler.step(test_loss)

            metrics = evaluate_model_detailed(model, test_loader, device, scaler_y_test)
            low = metrics['y1 (Lower)']
            up = metrics['y2 (Upper)']

            real_B, real_K = model.get_real_physics_params()
            k_L, k_U = real_K[0].item(), real_K[1].item()
            b_L, b_U = real_B[0].item(), real_B[1].item()

            val_rmse = (low['RMSE'] + up['RMSE']) / 2
            save_tag = ""
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), './best_kan_model.pth')
                save_tag = "🌟 [New Best]"

            msgs = []
            border = "-" * 80
            msgs.append(border)
            msgs.append(f"📅 Epoch {epoch:4d} | Avg RMSE: {val_rmse:.4f} {save_tag}")
            msgs.append(border)
            msgs.append(f" {'Target':<10} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10} | {'VAF(%)':<10}")
            msgs.append("-" * 65)
            msgs.append(
                f" {'y1 (Low)':<10} | {low['RMSE']:<10.4f} | {low['MAE']:<10.4f} | {low['R2']:<10.4f} | {low['VAF']:<10.2f}")
            msgs.append(
                f" {'y2 (Up)':<10} | {up['RMSE']:<10.4f} | {up['MAE']:<10.4f} | {up['R2']:<10.4f} | {up['VAF']:<10.2f}")
            msgs.append(border)
            msgs.append("⚛️  Physics Parameters:")
            msgs.append(f"   🔸 K: Low = {k_L:.3e} | Up = {k_U:.3e}")
            msgs.append(f"   🔹 B: Low = {b_L:.3e} | Up = {b_U:.3e}")

            tqdm.write("\n".join(msgs))

    print(f"\n{'=' * 20} 训练结束, Best Epoch:{best_epoch}, RMSE:{best_val_rmse:.4f} {'=' * 20}")

    model.load_state_dict(torch.load('./best_kan_model.pth'))
    plot_results(history, model, test_loader_sorted, device, scaler_y_test)

    return model


if __name__ == '__main__':
    train_kan_pidl()