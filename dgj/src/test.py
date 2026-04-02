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
        self.pretrained_features = 65
        self.tjepa = TJEPA(num_features=self.pretrained_features, embed_dim=128)

        # 【需要你手动填入】：这 9 个输入特征在 62 维预训练数据中的真实列索引
        self.physics_idx = [0, 1, 2, 3,10, 11, 12, 13, 14]

        # ... (智能加载预训练权重的 try-except 代码块保持不变) ...
        if pretrained_path:
            try:
                # 读取权重文件
                checkpoint = torch.load(pretrained_path)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

                # 【修正 1】严格检查 Embedding 层维度
                # 如果预训练是 50 维，这里是 9 维，必须报错，不能静默跳过
                current_emb_shape = self.tjepa.feature_embeds[0].weight.shape
                pretrained_emb_shape = state_dict['feature_embeds.0.weight'].shape

                if current_emb_shape != pretrained_emb_shape:
                    raise ValueError(
                        f"❌ 特征维度不匹配！预训练模型: {pretrained_emb_shape}, 当前模型: {current_emb_shape}")

                # 加载权重 (strict=False 是因为我们需要忽略 target_encoder 和 predictor)
                self.tjepa.load_state_dict(state_dict, strict=False)
                print(f"✅ 成功加载预训练权重: {pretrained_path}")

            except FileNotFoundError:
                print(f"⚠️ 未找到权重文件: {pretrained_path}，使用随机初始化！")
            except Exception as e:
                print(f"⚠️ 加载权重时出错: {e}")
                raise e  # 建议抛出异常，不要继续训练，否则是随机初始化
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
        self.B_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.K_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.register_buffer('B_scale', torch.tensor(1e7))
        self.register_buffer('K_scale', torch.tensor(1e7))
        self.register_buffer('mass', torch.tensor(2206.0))
        self.register_buffer('area_up', torch.tensor(3 * 3.14159 * (0.1) ** 2))
        self.register_buffer('area_low', torch.tensor(4 * 3.14159 * (0.1) ** 2))

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

    def get_real_physics_params(self):
        real_B = F.softplus(self.B_coeff) * self.B_scale
        real_K = F.softplus(self.K_coeff) * self.K_scale
        return real_B, real_K

    def get_net_parameters(self):
        # 返回 KAN 的参数 (T-JEPA已冻结)
        return self.kan.parameters()

    def get_phy_parameters(self):
        return [self.B_coeff, self.K_coeff]

    def get_kan_reg_loss(self):
        return self.kan.regularization_loss()