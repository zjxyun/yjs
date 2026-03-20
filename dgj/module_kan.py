import torch
import torch.nn as nn
import torch.nn.functional as F
# 【新增】引入 KAN
from dgj.TJEPA.efficient_kan import KAN


class PIDL_Model(nn.Module):
    def __init__(self, layers=[9, 32, 2]):
        # 建议 KAN 的宽度可以比 MLP 小一些，比如 32 或 16
        super(PIDL_Model, self).__init__()

        # 【修改】使用 KAN 替换 DNN
        # grid_size=5 是默认值，想提高精度可以设为 10 或 20
        self.kan = KAN(
            layers_hidden=layers,
            grid_size=10,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0
        )

        # 2. 可学习的物理参数 (保持不变)
        self.B_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)  # [low, up]
        self.K_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)  # [low, up]

        self.register_buffer('B_scale', torch.tensor(1e7))
        self.register_buffer('K_scale', torch.tensor(1e7))

        # 3. 固定物理常数 (保持不变)
        self.register_buffer('mass', torch.tensor(2206.0))
        self.register_buffer('area_up', torch.tensor(3 * 3.14159 * (0.1) ** 2))
        self.register_buffer('area_low', torch.tensor(4 * 3.14159 * (0.1) ** 2))

    def forward(self, x):
        # KAN 的 forward
        return self.kan(x)

    # ... (其余方法 get_real_physics_params 等保持不变) ...

    def get_real_physics_params(self):
        real_B = F.softplus(self.B_coeff) * self.B_scale
        real_K = F.softplus(self.K_coeff) * self.K_scale
        return real_B, real_K

    def get_net_parameters(self):
        return self.kan.parameters()

    def get_phy_parameters(self):
        return [self.B_coeff, self.K_coeff]

    # 【新增】暴露 KAN 的正则化 loss
    def get_kan_reg_loss(self):
        return self.kan.regularization_loss()