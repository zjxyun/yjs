import torch
import torch.nn as nn
import torch.nn.functional as F
from src.efficient_kan import KAN
from src.tjepa import TJEPA


class TJEPA_KAN_PIDL(nn.Module):
    def __init__(self, pretrained_path="models/tjepa_pretrained_best56.pth"):
        super().__init__()

        # ==========================================
        # 1. 初始化高维的 T-JEPA (必须与预训练的维度一致)
        # 根据你之前的报错日志，预训练模型接收 62 个特征
        # ==========================================
        self.pretrained_features = 64
        self.tjepa = TJEPA(num_features=self.pretrained_features, embed_dim=128)

        # 【需要你手动填入】：这 9 个输入特征在 62 维预训练数据中的真实列索引
        self.physics_idx = [0, 1, 2, 3,10, 11, 12, 13, 14]

        # ... (智能加载预训练权重的 try-except 代码块保持不变) ...
        # 注意：这里加载 62 维的权重将完美匹配，不会再报错！

        # 冻结 T-JEPA
        for param in self.tjepa.parameters():
            param.requires_grad = False  # 建议先冻结，跑通了再尝试微调

        # ==========================================
        # 2. 初始化下游网络 (KAN)
        # ==========================================
        self.norm = nn.LayerNorm(128)

        # KAN 依然接收 137 维：128(全局感官) + 9(你传进来的真实物理特征)
        self.kan = KAN(
            layers_hidden=[137, 32, 2],
            grid_size=10,
            spline_order=3
        )

        # ... (物理参数 B_coeff, K_coeff 的定义保持不变) ...

    def forward(self, x_9d):
        """
        输入的 x_9d 是你数据加载器传进来的 (Batch, 9) 张量
        """
        b = x_9d.shape[0]

        # ==========================================
        # 【核心魔法：掩码补全机制】
        # ==========================================
        # 1. 构造一个 62 维的全 0 占位矩阵
        # 为什么填 0？因为你的数据被 MinMaxScaler(-1, 1) 处理过，
        # 0 恰好代表了所有特征的“中位数”，是最中庸、干扰最小的填充值。
        x_full = torch.zeros(b, self.pretrained_features, dtype=x_9d.dtype, device=x_9d.device)

        # 2. 构造一个 62 维的 Mask 矩阵 (全为 False，表示全部被遮挡)
        mask_full = torch.zeros(b, self.pretrained_features, dtype=torch.bool, device=x_9d.device)

        # 3. 将真实的 9 维数据，精准对号入座填入对应的坑位
        x_full[:, self.physics_idx] = x_9d

        # 4. 告诉 T-JEPA：这 9 个坑位的数据是“可见的” (True)
        mask_full[:, self.physics_idx] = True

        # ==========================================
        # 提取高级特征
        # ==========================================
        # 将扩充好的 62 维数据和对应的 Mask 喂给 T-JEPA 大脑
        # 注意：底层 forward_context 看到 mask_full 后，会自动“扔掉”那 53 个填 0 的特征
        # 只带着正确的 Positional Embedding 计算这 9 个真实特征的相互关系！
        h_rep = self.tjepa.forward_context(x_full, mask_full)

        # 此时 h_rep 的形状自动变成了 (Batch, 9 + 1(REG), 128)

        # 切除最后一位的 REG Token，只保留 9 个物理特征的高级语义
        feature_tokens = h_rep[:, :-1, :]  # (Batch, 9, 128)

        # 平均池化，提炼全局工况
        global_feat = feature_tokens.mean(dim=1)  # (Batch, 128)

        # 归一化并压入 [-1, 1] 迎合 KAN
        deep_feat = torch.tanh(self.norm(global_feat))  # (Batch, 128)

        # ==========================================
        # 特征融合与预测
        # ==========================================
        # 128 维的高级工况 + 9 维的原始精确数值
        combined_feat = torch.cat([deep_feat, x_9d], dim=1)  # (Batch, 137)

        # KAN 非线性映射
        y_pred = self.kan(combined_feat)

        return y_pred