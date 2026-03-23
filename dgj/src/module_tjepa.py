import torch
import torch.nn as nn
import torch.nn.functional as F
from src.efficient_kan import KAN
from src.tjepa import TJEPA


class TJEPA_KAN_PIDL(nn.Module):
    def __init__(self, pretrained_path="models/tjepa_pretrained_best.pth"):
        super().__init__()

        # 1. 初始化 T-JEPA (使用下游任务的特征数量，例如 9)
        # 注意：这里我们初始化的是“接受9个特征”的 T-JEPA
        self.tjepa = TJEPA(num_features=9, embed_dim=128)

        # 2. 智能加载预训练权重
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

        # 3. 冻结 T-JEPA (Context Encoder) 的参数
        for param in self.tjepa.parameters():
            param.requires_grad = True

        # 4. KAN 网络
        self.norm = nn.LayerNorm(128)
        # 输入维度是 T-JEPA 的 embed_dim (128)
        self.kan = KAN(
            layers_hidden=[137, 32, 2],
            grid_size=10,
            spline_order=3
        )

        # 5. 物理参数
        self.B_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.K_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.register_buffer('B_scale', torch.tensor(1e7))
        self.register_buffer('K_scale', torch.tensor(1e7))
        self.register_buffer('mass', torch.tensor(2206.0))
        self.register_buffer('area_up', torch.tensor(3 * 3.14159 * (0.1) ** 2))
        self.register_buffer('area_low', torch.tensor(4 * 3.14159 * (0.1) ** 2))

    # def forward(self, x):
    #     # 1. 提取特征
    #     b, n = x.shape
    #     # 下游任务使用全量的 x，不进行 Mask (全为 True)
    #     full_mask = torch.ones(b, n, dtype=torch.bool, device=x.device)
    #
    #     # 使用 Context Encoder 提取特征
    #     # forward_context 返回 (B, N+1, E), 其中 N=9, +1是REG
    #     h_rep = self.tjepa.forward_context(x, full_mask)
    #
    #     # 2. 特征聚合 【核心修正】
    #     # ❌ 原代码：reg_feat = h_rep[:, -1, :] (使用了论文明确说要丢弃的 REG)
    #     # ✅ 新代码：切除 REG，对特征 Token 进行 Mean Pooling
    #
    #     feature_tokens = h_rep[:, :-1, :]  # (B, 9, 128) 丢弃最后一位
    #     global_feat = feature_tokens.mean(dim=1)  # (B, 128) 全局平均池化
    #
    #     # 3. 归一化适配 KAN 【核心修正】
    #     # LayerNorm 输出范围不固定，KAN 需要 [-1, 1]
    #     x_kan = self.norm(global_feat)
    #     x_kan = torch.tanh(x_kan)  # 强行压缩到 [-1, 1] 区间
    #
    #     # 4. 喂给 KAN
    #     y_pred = self.kan(x_kan)
    #
    #     return y_pred

    def forward(self, x):
        # 1. T-JEPA 提取高级特征
        b, n = x.shape
        full_mask = torch.ones(b, n, dtype=torch.bool, device=x.device)
        h_rep = self.tjepa.forward_context(x, full_mask)

        feature_tokens = h_rep[:, :-1, :]
        global_feat = feature_tokens.mean(dim=1)  # (B, 128)

        # 2. 归一化
        deep_feat = torch.tanh(self.norm(global_feat))

        # 3. 【核心改进】将预训练特征与原始 9 维输入拼接
        # 这样 KAN 既有全局感官，又有原始精确数值
        combined_feat = torch.cat([deep_feat, x], dim=1)  # (B, 128 + 9 = 137维)

        # 4. 更新 KAN 的输入维度 (在 __init__ 里把 KAN 的输入从 128 改为 137)
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