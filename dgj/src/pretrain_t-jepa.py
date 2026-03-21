# pretrain_t-jepa.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
# 【修改 1】导入 MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

from dgj.src.tjepa import TJEPA


# -----------------------------------------------------------------------------
# 1. 简单的数据加载函数 (直接读取清洗好的宽表)
# -----------------------------------------------------------------------------
def load_pretrain_data(csv_path):
    print(f"Loading pre-training data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 获取所有数值特征 (假设 csv 已经只包含数值列)
    # 如果还有非数值列，在这里 drop 掉
    X = df.values.astype(np.float32)

    # 【修改 2】归一化改为 MinMaxScaler
    # 强烈建议使用 feature_range=(-1, 1)，这更适合后续接入 KAN
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 转为 Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    print(f"Data shape: {X_tensor.shape}")
    print(f"Data Range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]") # 打印确认范围
    return X_tensor, X.shape[1]  # 返回数据和特征数量


def generate_random_mask(batch_size, num_features, mask_ratio=0.4):
    # ... (保持原来的 mask 生成逻辑不变) ...
    num_masked = int(num_features * mask_ratio)
    mask = torch.ones(batch_size, num_features, dtype=torch.bool)
    for i in range(batch_size):
        mask_idx = torch.randperm(num_features)[:num_masked]
        mask[i, mask_idx] = False
    return mask


def pretrain():
    # ==========================================
    # 1. 准备数据
    # ==========================================

    # A. 运行预处理脚本生成 csv (如果还没生成)
    # 建议：强制重新生成一次，确保数据是最新的
    print("Generating/Refreshing pre-training dataset...")
    from data_preprocessing_tjepa_56 import preprocess_shield_data
    preprocess_shield_data('data/train_dataset.csv', 'train_pretrain.csv')

    # B. 加载宽表数据
    X_train, num_features = load_pretrain_data('train_pretrain.csv')

    # C. 创建 DataLoader
    dataset = TensorDataset(X_train)
    # 【建议】调大 Batch Size，自监督学习需要较大的 Batch 来学习特征分布
    # 如果显存不够，再改回 32
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, Num Features: {num_features}")

    # ==========================================
    # 2. 初始化模型
    # ==========================================
    # 注意：num_features 必须动态传入，不再是固定的 9
    model = TJEPA(num_features=num_features, embed_dim=128).to(device)

    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4. 训练循环
    epochs = 500  # 预训练可以多跑一些轮次
    loss_fn = nn.MSELoss()

    best_loss = float('inf')

    print("Start Pre-training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            x = batch[0].to(device)  # (B, N)

            # 生成 Mask (mask_ratio 可以设为 0.2~0.5)
            mask = generate_random_mask(x.shape[0], num_features, mask_ratio=0.3).to(device)

            # T-JEPA 核心流程
            h_target = model.forward_target(x)
            h_context = model.forward_context(x, mask)
            h_pred = model.forward_predictor(h_context, mask)

            loss = loss_fn(h_pred, h_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_target_encoder(momentum=0.996)

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch + 1} Loss: {avg_loss:.6f}")

        # # 【新增】检查并保存最佳模型
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #
        #     # 保存最佳权重
        #     best_checkpoint = {
        #         'state_dict': model.state_dict(),
        #         'epoch': epoch,
        #         'feature_names': num_features,
        #         'loss': best_loss
        #     }
        #     torch.save(best_checkpoint, "tjepa_pretrained_best.pth")
        #     # print(f"🌟 New Best Model Saved! (Loss: {best_loss:.6f})")

    # 5. 保存预训练模型 (包含元数据)
    checkpoint = {
        'state_dict': model.state_dict(),
        'epoch': epochs,
        'feature_names': num_features,  # 记录一下特征数量，防止以后对不上
        'loss': total_loss / len(dataloader)
    }
    torch.save(checkpoint, "tjepa_pretrained.pth")
    print("Pre-training Done! Model saved to tjepa_pretrained.pth")


if __name__ == "__main__":
    pretrain()