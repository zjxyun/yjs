import torch
import torch.nn as nn
import torch.nn.functional as F

class PIDL_Model(nn.Module):
    def __init__(self, layers=[9, 128, 128, 128, 2]):
        super(PIDL_Model, self).__init__()

        # 1. 深度神经网络部分 (DNN)
        self.dnn = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[2]),
            nn.ReLU(),
            nn.Linear(layers[2], layers[3]),
            nn.ReLU(),
            nn.Linear(layers[3], layers[4]),
            # nn.ReLU(),
            # nn.Linear(layers[4], layers[5])
        )

        # 2. 可学习的物理参数 (Physical Parameters)
        # 将它们单独定义，方便管理
        self.B_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)  # [low, up]
        self.K_coeff = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)  # [low, up]
        # self.F_bias_coeff = nn.Parameter(torch.tensor(2.0), requires_grad=True)

        self.register_buffer('B_scale', torch.tensor(1e7))
        self.register_buffer('K_scale', torch.tensor(1e7))
        # self.register_buffer('F_bias_scale', torch.tensor(1e5)

        # 3. 固定物理常数 (不参与梯度更新)
        # 建议注册为 buffer，这样它们会随模型 state_dict 保存，但不会被视为参数更新
        self.register_buffer('mass', torch.tensor(2206.0))
        self.register_buffer('area_up', torch.tensor(3 * 3.14159 * (0.1) ** 2))
        self.register_buffer('area_low', torch.tensor(4 * 3.14159 * (0.1) ** 2))

    def forward(self, x):
        return self.dnn(x)

        # 【新增方法】获取真实的、保证为正的物理参数

    def get_real_physics_params(self):
        # Softplus: log(1 + exp(x))，保证结果永远 > 0
        real_B = F.softplus(self.B_coeff) * self.B_scale
        real_K = F.softplus(self.K_coeff) * self.K_scale
        return real_B, real_K

    # --- 新增：专门获取网络参数的方法 ---
    def get_net_parameters(self):
        """只返回神经网络层的参数 (Weights & Biases)"""
        return self.dnn.parameters()

    # --- 新增：专门获取物理参数的方法 ---
    def get_phy_parameters(self):
        """只返回物理系数 B, K, F"""
        return [self.B_coeff,  self.K_coeff]