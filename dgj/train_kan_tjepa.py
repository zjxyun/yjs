import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import joblib

# 导入数据处理
from src.data_preprocessing import load_and_process_data
from src.data_preprocessing_2 import load_and_process_data_2
# 【重点】导入集成了 T-JEPA 和 KAN 的模型
from src.module_tjepa import TJEPA_KAN_PIDL
from src.physics_loss import calculate_physics_loss
from src.metrics import calculate_metrics_numpy


# ------------------------------------------------------------------------------
# 辅助函数 (保持不变)
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
            loss_data = loss_mse(pred_norm, by)
            loss_physics = calculate_physics_loss(model, bx, pred_norm, bp, scaler_x, scaler_y)
            loss_total = loss_data + 0.1 * loss_physics
            epoch_loss_total += loss_total.item()
            count += 1
    return epoch_loss_total / count


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


def plot_results(history, model, test_loader_sorted, device, scaler_y):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Total Train Loss')
    plt.plot(history['data_loss'], label='Data Loss', linestyle='--')
    plt.plot(history['phy_loss'], label='Physics Loss', linestyle=':')
    if 'reg_loss' in history:
        plt.plot(history['reg_loss'], label='KAN Reg Loss', linestyle='-.', alpha=0.7)
    plt.title('Training Loss History (T-JEPA + KAN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

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
    limit = 300
    plt.subplot(1, 3, 2)
    plt.plot(targets_real[:limit, 0], label='True Lower', color='grey', alpha=0.6)
    plt.plot(preds_real[:limit, 0], label='Pred Lower', color='red', linewidth=1.5)
    plt.title(f'Lower Cylinder (T-JEPA+KAN)')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(targets_real[:limit, 1], label='True Upper', color='grey', alpha=0.6)
    plt.plot(preds_real[:limit, 1], label='Pred Upper', color='blue', linewidth=1.5)
    plt.title(f'Upper Cylinder (T-JEPA+KAN)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# 主训练流程
# ------------------------------------------------------------------------------
def train_tjepa_kan():
    print(f"{'=' * 20} Loading Data {'=' * 20}")
    X_train, Y_train, P_train, scaler_x_train, scaler_y_train = load_and_process_data('data/train_dataset.csv')
    X_test, Y_test, P_test, scaler_x_test, scaler_y_test = load_and_process_data_2('data/test_dataset.csv')
    # 替换掉函数生成的 scaler
    scaler_x_train = joblib.load('models/scaler_tjepa_56.joblib')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train, Y_train, P_train)
    test_dataset = TensorDataset(X_test, Y_test, P_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader_sorted = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 1. 初始化 T-JEPA + KAN 模型
    # 确保 'tjepa_pretrained.pth' 存在于当前目录
    try:
        model = TJEPA_KAN_PIDL().to(device)
    except FileNotFoundError:
        print("⚠️ Warning: Pretrained weights not found! Initializing randomly.")
        model = TJEPA_KAN_PIDL(pretrained_path=None).to(device)

    # 2. 优化器
    # 注意：get_net_parameters() 默认只返回 KAN 的参数（如果冻结了 T-JEPA）
    # 如果你在 module_kan_tjepa.py 里解冻了 T-JEPA，这里会自动包含它的参数
    optimizer = torch.optim.Adam([
        {'params': model.tjepa.parameters(), 'lr': 1e-5},  # 极小学习率微调预训练层
        {'params': model.get_net_parameters(), 'lr': 0.001},  # KAN 正常学习率
        {'params': model.get_phy_parameters(), 'lr': 0.01}  # 物理参数
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                           verbose=False)
    loss_mse = nn.MSELoss()

    epochs = 5000
    history = {'train_loss': [], 'data_loss': [], 'phy_loss': [], 'reg_loss': []}
    best_val_rmse = 100.0
    best_epoch = 0

    print(f"\n{'=' * 20} Start Training T-JEPA-KAN-PIDL {'=' * 20}")
    main_loop = tqdm(range(1, epochs + 1), desc='Training', dynamic_ncols=True)

    for epoch in main_loop:
        # 注意：这里不需要 update_grid，因为 KAN 的输入特征已经是 T-JEPA 提取过的 Latent 特征
        # 这些特征分布相对稳定，不更新 Grid 也可以

        model.train()
        epoch_loss_total = 0.0
        total_data_loss = 0
        total_phy_loss = 0
        total_reg_loss = 0

        for bx, by, bp in train_loader:
            bx, by, bp = bx.to(device), by.to(device), bp.to(device)
            optimizer.zero_grad()

            # Forward: X -> T-JEPA -> [REG] -> KAN -> Y_pred
            y_pred = model(bx)

            loss_data = loss_mse(y_pred, by)
            loss_physics = calculate_physics_loss(model, bx, y_pred, bp, scaler_x_train, scaler_y_train)
            loss_reg = model.get_kan_reg_loss()

            # Loss 组合
            loss_total = loss_data + 0.1 * loss_physics + 1e-4 * loss_reg

            loss_total.backward()
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
                torch.save(model.state_dict(), 'models/best_tjepa_kan.pth')
                save_tag = "🌟"

            msgs = []
            msgs.append(f"-" * 60)
            msgs.append(f"📅 Epoch {epoch} | RMSE: {val_rmse:.4f} {save_tag}")
            msgs.append(f" y1(Low): RMSE={low['RMSE']:.4f}, R2={low['R2']:.4f}")
            msgs.append(f" y2(Up) : RMSE={up['RMSE']:.4f}, R2={up['R2']:.4f}")
            msgs.append(f" Physics: K_low={k_L:.2e}, K_up={k_U:.2e}")
            msgs.append(f"          B_low={b_L:.2e}, B_up={b_U:.2e}")
            tqdm.write("\n".join(msgs))

    print(f"Training Finished. Best Epoch: {best_epoch}, Best RMSE: {best_val_rmse:.4f}")

    # 绘图
    model.load_state_dict(torch.load('models/best_tjepa_kan.pth'))
    plot_results(history, model, test_loader_sorted, device, scaler_y_test)


if __name__ == '__main__':
    train_tjepa_kan()