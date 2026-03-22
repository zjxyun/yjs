# pretrain_t-jepa.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import joblib  # 【新增】用于保存 scaler
import os

from src.tjepa import TJEPA

# =============================================================================
# 配置
# =============================================================================
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 100 # 建议 200-500
MASK_RATIO = 0.3  # 遮挡 30% 让模型去猜
SCALER_PATH = 'models/scaler_tjepa.joblib'  # Scaler 保存路径
BEST_MODEL_PATH = 'models/tjepa_pretrained_best.pth'



# -----------------------------------------------------------------------------
# 1. 数据加载与持久化 Scaler
# -----------------------------------------------------------------------------
def load_pretrain_data(csv_path):
    print(f"Loading pre-training data from: {csv_path}")
    df = pd.read_csv(csv_path)

    X = df.values.astype(np.float32)

    # 【修改 1 & 2】使用 -1 到 1 的范围，并保存 Scaler
    print("Fitting MinMaxScaler (range -1 to 1)...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存 Scaler，这对下游 PIDL 至关重要！
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to {SCALER_PATH}")

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    print(f"Data shape: {X_tensor.shape}")
    return X_tensor, X.shape[1]


def generate_random_mask(batch_size, num_features, mask_ratio=0.3):
    """
    生成掩码: True = Context (可见), False = Target (被遮挡/需要预测)
    """
    num_masked = int(num_features * mask_ratio)
    # 至少遮挡 1 个，至多保留 1 个
    num_masked = max(1, min(num_features - 1, num_masked))

    mask = torch.ones(batch_size, num_features, dtype=torch.bool)
    for i in range(batch_size):
        mask_idx = torch.randperm(num_features)[:num_masked]
        mask[i, mask_idx] = False  # 设置为 False 表示这是 Target
    return mask


def pretrain():
    # ==========================================
    # 1. 准备数据
    # ==========================================

    # 确保预处理脚本是最新的
    # 假设你使用的是 56 特征版本，如果是 9 特征版本请修改 import
    from src.data_preprocessing_tjepa_9 import preprocess_shield_data

    if not os.path.exists('data/train_pretrain.csv'):
        print("Generating pre-training dataset...")
        preprocess_shield_data('train_dataset.csv', 'data/train_pretrain.csv')
    else:
        print("Using existing train_pretrain.csv")

    X_train, num_features = load_pretrain_data('data/train_pretrain.csv')

    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Num Features: {num_features}")

    # ==========================================
    # 2. 初始化模型
    # ==========================================
    model = TJEPA(num_features=num_features, embed_dim=128).to(device)

    # 3. 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 【新增】Cosine 调度器，有助于 SSL 收敛
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 4. 训练循环
    best_loss = float('inf')

    print("Start Pre-training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in pbar:
            x = batch[0].to(device)  # (B, N)

            # 生成 Mask
            mask = generate_random_mask(x.shape[0], num_features, mask_ratio=MASK_RATIO).to(device)

            # --- Forward ---
            # 1. Target Encoder (Teacher): 看到全量数据 (或部分)，这里简化为全量
            with torch.no_grad():
                h_target = model.forward_target(x)

            # 2. Context Encoder (Student): 只能看到 mask==True 的部分
            h_context = model.forward_context(x, mask)

            # 3. Predictor: 尝试恢复被遮挡的部分
            h_pred = model.forward_predictor(h_context, mask)

            # 【关键修改】切掉最后一个 [REG] token，只保留前 9 个特征
            h_pred_feats = h_pred[:, :-1, :]  # (Batch, 9, Embed)
            h_target_feats = h_target[:, :-1, :]  # (Batch, 9, Embed)

            # --- 【修改 3】Loss 计算 (只算 Masked 区域) ---
            # h_pred 和 h_target 形状通常是 (B, N, Embed)
            # 我们只计算 mask == False (被遮挡) 部分的损失

            # 构造 loss mask: shape (B, N, 1)
            # mask 是 bool，True 是可见。我们需要 ~mask (即 False变True) 作为 loss 的权重
            loss_mask = (~mask).unsqueeze(-1).float()

            mse = (h_pred_feats - h_target_feats) ** 2
            # 只保留 masked 区域的 loss
            masked_loss = (mse * loss_mask).sum() / (loss_mask.sum() + 1e-6)

            loss = masked_loss

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Update Target Encoder (EMA) ---
            model.update_target_encoder(momentum=0.996)

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # End of Epoch
        avg_loss = total_loss / len(dataloader)
        scheduler.step()  # 更新学习率

        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.6f}")

        # 【新增】保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'num_features': num_features,
                'feature_range': (-1, 1),
                'loss': best_loss
            }
            torch.save(best_checkpoint, BEST_MODEL_PATH)
            # print(f"🌟 Best model saved to {BEST_MODEL_PATH}")

    # # 5. 保存最终模型
    # torch.save({
    #     'state_dict': model.state_dict(),
    #     'epoch': EPOCHS,
    #     'num_features': num_features,
    #     'loss': avg_loss
    # }, FINAL_MODEL_PATH)

    print(f"Training Complete. Best Loss: {best_loss:.6f}")
    print(f"Scaler saved at: {SCALER_PATH}")
    print(f"Best Model saved at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    pretrain()