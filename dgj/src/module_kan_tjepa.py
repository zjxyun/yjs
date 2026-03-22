import torch
import torch.nn as nn
import torch.nn.functional as F
from src.efficient_kan import KAN
from src.tjepa import TJEPA


class TJEPA_KAN_PIDL(nn.Module):
    def __init__(self, pretrained_path="tjepa_pretrained.pth"):
        super().__init__()

        # 1. 初始化 T-JEPA (使用下游任务的特征数量，例如 9)
        # 注意：这里我们初始化的是“接受9个特征”的 T-JEPA
        self.tjepa = TJEPA(num_features=9, embed_dim=128)

        # 2. 智能加载预训练权重
        if pretrained_path:
            try:
                # 读取权重文件
                checkpoint = torch.load(pretrained_path)
                # 处理可能存在的 'state_dict' 嵌套
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

                # 获取当前模型的字典
                model_dict = self.tjepa.state_dict()

                # 【核心逻辑】筛选权重
                # 只有当 key 存在且 shape 完全一致时才加载
                # 这样可以自动过滤掉维度不匹配的 embedding 层，保留 Transformer 核心层
                pretrained_dict = {
                    k: v for k, v in state_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }

                # 更新权重
                model_dict.update(pretrained_dict)
                self.tjepa.load_state_dict(model_dict)

                # 打印加载情况
                loaded_keys = len(pretrained_dict)
                total_keys = len(state_dict)
                print(f"✅ 成功加载预训练权重: {loaded_keys}/{total_keys} 层匹配成功。")
                print("   (Embedding层因维度不同被跳过是正常的，Transformer核心层已加载)")

            except FileNotFoundError:
                print("⚠️ 未找到预训练权重文件，将使用随机初始化！")
            except Exception as e:
                print(f"⚠️ 加载权重时出错: {e}，将使用随机初始化！")

        # 3. 冻结 T-JEPA (Context Encoder) 的参数
        # 建议：如果只是做迁移学习，先冻结比较好
        for param in self.tjepa.parameters():
            param.requires_grad = False

        # 4. KAN 网络
        self.norm = nn.LayerNorm(128)
        # 输入维度是 T-JEPA 的 embed_dim (128)
        # 这里的 layers 设置为 [128, 32, 2]
        self.kan = KAN(
            layers_hidden=[128, 32, 2],
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

    def forward(self, x):
        # 1. 提取特征
        b, n = x.shape
        # 下游任务使用全量的 x，不进行 Mask (全为 True)
        full_mask = torch.ones(b, n, dtype=torch.bool, device=x.device)

        # 使用 Context Encoder 提取特征
        # forward_context 返回 (B, N+1, E)
        h_rep = self.tjepa.forward_context(x, full_mask)

        # 2. 特征聚合
        # 取出 [REG] Token 作为全局特征 (它是序列的最后一个)
        reg_feat = h_rep[:, -1, :]  # (B, 128)

        reg_feat = self.norm(reg_feat)

        # 3. 喂给 KAN
        y_pred = self.kan(reg_feat)

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