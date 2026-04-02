import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tjepa import TJEPA


class TJEPA_MLP_PIDL(nn.Module):
    def __init__(self, pretrained_path="models/tjepa_pretrained_best56.pth"):
        super(TJEPA_MLP_PIDL, self).__init__()

        # ==========================================
        # 1. 引入预训练的 T-JEPA
        # 注意：这里的 65 需要和你的预训练权重维度严格一致
        # ==========================================
        self.pretrained_features = 65
        self.tjepa = TJEPA(num_features=self.pretrained_features, embed_dim=128)

        # 你的 9 个物理特征在 65 维大表中的索引位置
        self.physics_idx = [0, 1, 2, 3, 10, 11, 12, 13, 14]

        # 智能加载权重
        if pretrained_path:
            try:
                checkpoint = torch.load(pretrained_path)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                self.tjepa.load_state_dict(state_dict, strict=False)
                print(f"✅ 成功加载 T-JEPA 预训练权重: {pretrained_path}")
            except Exception as e:
                print(f"⚠️ 加载权重失败: {e}")

        # 建议初期冻结 T-JEPA，仅作为特征提取器
        for param in self.tjepa.parameters():
            param.requires_grad = False

        self.norm = nn.LayerNorm(128)

        # ==========================================
        # 2. 全连接网络部分 (DNN)
        # ==========================================
        # 【核心修改】输入维度变为 128(TJEPA) + 9(Raw) = 137
        input_dim = 137
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出层依然是 2 维
        )

        # ==========================================
        # 3. 物理参数定义 (保持完全不变)
        # ==========================================
        self.B_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.K_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.register_buffer('B_scale', torch.tensor(1e7))
        self.register_buffer('K_scale', torch.tensor(1e7))
        self.register_buffer('mass', torch.tensor(2206.0))
        self.register_buffer('area_up', torch.tensor(3 * 3.14159 * (0.1) ** 2))
        self.register_buffer('area_low', torch.tensor(4 * 3.14159 * (0.1) ** 2))

    def forward(self, x_9d):
        b = x_9d.shape[0]

        # --- 轨道 A: T-JEPA 掩码补全提取特征 ---
        x_full = torch.zeros(b, self.pretrained_features, dtype=x_9d.dtype, device=x_9d.device)
        mask_full = torch.zeros(b, self.pretrained_features, dtype=torch.bool, device=x_9d.device)

        x_full[:, self.physics_idx] = x_9d
        mask_full[:, self.physics_idx] = True

        h_rep = self.tjepa.forward_context(x_full, mask_full)
        feature_tokens = h_rep[:, :-1, :]  # 切除 REG
        global_feat = feature_tokens.mean(dim=1)  # 平均池化
        deep_feat = torch.tanh(self.norm(global_feat))  # 归一化 (Batch, 128)

        # --- 轨道 B: 特征融合 ---
        combined_feat = torch.cat([deep_feat, x_9d], dim=1)  # (Batch, 137)

        # --- MLP 预测 ---
        y_pred = self.dnn(combined_feat)
        return y_pred

    def get_real_physics_params(self):
        real_B = F.softplus(self.B_coeff) * self.B_scale
        real_K = F.softplus(self.K_coeff) * self.K_scale
        return real_B, real_K

    def get_net_parameters(self):
        """返回全连接层的参数"""
        return self.dnn.parameters()

    def get_phy_parameters(self):
        """返回物理系数"""
        return [self.B_coeff, self.K_coeff]