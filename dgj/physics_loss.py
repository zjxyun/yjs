import torch


def calculate_physics_loss(model, x_input, y_pred, phys_context, scaler_x, scaler_y):
    """
    计算基于物理方程的损失 (Eq. 7 & 8)
    """

    IDX_F_LOW = 3
    IDX_F_UP = 5
    IDX_P_DIFF = 6  # 推进压力差
    IDX_Y_PREV_LOW = 7  # 上一时刻下组行程
    IDX_Y_PREV_UP = 8  # 上一时刻上组行程


    # Phys Columns: [0:dt_curr, 1:dt_prev]
    IDX_P_DT_CURR = 0
    IDX_P_DT_PREV = 1
    IDX_P_DT_PREV2 = 2
    IDX_Y_PREV2_LOW = 3 # 上上一时刻下组行程
    IDX_Y_PREV2_UP = 4  # 上上一时刻上组行程
    IDX_Y_PREV3_LOW = 5
    IDX_Y_PREV3_UP = 6
    IDX_N_LOW = 8
    IDX_N_UP = 7

    # ==========================================
    # 1. 准备反归一化参数 (Scaler -> Tensor)
    # ==========================================
    device = x_input.device

    # 辅助函数：将 sklearn 的 scale_ 转为 tensor
    def to_tensor(val):
        return torch.tensor(val, dtype=torch.float32, device=device)

    # Y 的缩放参数 (y_pred 是归一化的，范围 0-1)
    y_min = to_tensor(scaler_y.data_min_)
    y_scale = to_tensor(scaler_y.scale_)  # scale = 1 / (max - min)

    # X 的缩放参数 (x_input 是归一化的)
    # 假设 x columns: [0:t, 1:F_low, 2:F_left, 3:F_right, 4:F_up, 5:P_diff, ...]
    x_min = to_tensor(scaler_x.data_min_)
    x_scale = to_tensor(scaler_x.scale_)

    # 2. 还原真实物理量 (关键步骤)
    # ==========================================

    # A. 当前预测行程 (y_s) [单位: m]
    # y_pred 是归一化的，假设原始单位是 mm，需转为 m
    y_pred_mm = y_pred / y_scale + y_min
    y_curr_low = y_pred_mm[:, 0:1] / 1000.0
    y_curr_up = y_pred_mm[:, 1:2] / 1000.0

    def denorm_x(idx):
        return (x_input[:, idx:idx + 1] / x_scale[idx] + x_min[idx])

    # 2. 上一时刻真实值 (y_t-1) - 从输入X中拿
    y_prev_low = denorm_x(IDX_Y_PREV_LOW) / 1000.0
    y_prev_up = denorm_x(IDX_Y_PREV_UP) / 1000.0

    # 3. 上上时刻真实值 (y_t-2)
    y_prev2_low  = phys_context[:, IDX_Y_PREV2_LOW:IDX_Y_PREV2_LOW + 1] / 1000.0
    y_prev2_up = phys_context[:, IDX_Y_PREV2_UP:IDX_Y_PREV2_UP + 1] / 1000.0

    y_prev3_low = phys_context[:, IDX_Y_PREV3_LOW:IDX_Y_PREV3_LOW + 1] / 1000.0
    y_prev3_up = phys_context[:, IDX_Y_PREV3_UP:IDX_Y_PREV3_UP + 1] / 1000.0



    # --- C. 时间间隔 (单位: 秒) ---
    dt_curr = phys_context[:, IDX_P_DT_CURR:IDX_P_DT_CURR + 1]
    dt_prev = phys_context[:, IDX_P_DT_PREV:IDX_P_DT_PREV + 1]
    dt_prev2 = phys_context[:, IDX_P_DT_PREV2:IDX_P_DT_PREV2 + 1]

    if torch.isnan(y_pred).any():
        print("!! Panic: Network output contains NaN")
        return torch.tensor(0.0, requires_grad=True).to(device)

    # ==========================================
    # 2. 差分法计算运动学参数
    # ==========================================

    # --- 速度 (v = dy/dt) ---
    # v_curr = (y_t - y_t-1) / dt1
    v_curr_low = (y_curr_low - y_prev_low) / dt_curr
    v_curr_up = (y_curr_up - y_prev_up) / dt_curr

    # v_prev = (y_t-1 - y_t-2) / dt2
    # 注意：这里的 y_prev_low 我们用了 Input 中的值，视为常数（detach），
    # 但如果要反向传播训练前面的层，可能需要考虑计算图。
    # 通常为了计算加速度，v_prev 视作常数即可，因为网络优化的是当前的 y_t 使得加速度合理
    v_prev_low = (y_prev_low - y_prev2_low) / dt_prev
    v_prev_up = (y_prev_up - y_prev2_up) / dt_prev

    v_prev2_low = (y_prev2_low - y_prev3_low) / dt_prev2
    v_prev2_up = (y_prev2_up - y_prev3_up) / dt_prev2

    # --- 加速度 (a = dv/dt) ---
    # 使用后向差分
    a_curr_low = (v_curr_low - v_prev_low) / dt_curr
    a_curr_up = (v_curr_up - v_prev_up) / dt_curr

    a_prev_low = (v_prev_low - v_prev2_low) / dt_prev
    a_prev_up = (v_prev_up - v_prev2_up) / dt_prev

    # --- 构建论文所需的差值项 ---
    # Eq. (3) / (7) 中所有的项都是 (下缸 - 上缸) 或者 (i - j)
    # 我们定义 delta_quantity = quantity_low - quantity_up

    d2y_low = a_curr_low - a_prev_low
    d2y_up = a_curr_up - a_prev_up
    dy_low= v_curr_low - v_prev_low  # (dy_i - dy_j)
    dy_up = v_curr_up - v_prev_up

    # K项: 论文中是 K*(y_s - y_{s-mu})，即 K * (y_t - y_{t-1})
    # 这实际上是位移增量
    y_increment_low = y_curr_low - y_prev_low
    y_increment_up = y_curr_up - y_prev_up


    # ==========================================
    # 3. 组装物理方程 (Eq. 7)
    # ==========================================

    # 恢复真实的物理参数值
    # model.B_coeff[0] 是 lower, [1] 是 upper
    # 只调用一次，拿到两个张量 (Tensor)
    real_B, real_K = model.get_real_physics_params()

    # 然后再按索引取值
    B_real_low = real_B[0]
    B_real_up = real_B[1]

    K_real_low = real_K[0]
    K_real_up = real_K[1]
    # F_real_bias = model.F_bias_coeff * model.F_bias_scale

    # --- (A) 驱动项: A' * (p_i - p_j) ---
    n_low = phys_context[:, IDX_N_LOW:IDX_N_LOW + 1]
    n_up = phys_context[:, IDX_N_UP:IDX_N_UP + 1]
    # p_diff_bar = denorm_x(IDX_P_DIFF)

    # 1 Bar = 1e5 Pa -> Force = Pa * m^2 = N
    term_drive = model.area_low * (n_low * 1e5) - model.area_up * (n_up * 1e5)
    # print("term_drive is {}".format(term_drive))

    # --- (B) 动力学项 ---
    # m' * (a_i - a_j)
    term_mass = model.mass * (d2y_low - d2y_up)

    # B^i * v_i - B^j * v_j
    # 假设上下缸阻尼系数相同 B_i = B_j = B (简化模型，或分别训练)
    # 论文公式 (7) 是分开写的: - B^i(...) + B^j(...)
    # 这里我们对应论文 Eq. 4 的简化版 (假设参数对称): m(...) + B(v_diff) + K(y_inc_diff)
    term_damping = B_real_low * dy_low - B_real_up * dy_up

    # K项
    term_stiffness = K_real_low * y_increment_low - K_real_up * y_increment_up

    # F项 (负载差)
    # 你的模型中包含 F_bias 参数，或者直接使用输入的土压
    # 如果使用输入的土压作为 F:
    # F_i = Pressure_low * Area, F_j = Pressure_up * Area

    f_low = denorm_x(IDX_F_LOW) * 1e5 * model.area_low  # 注意：如果是压强(Bar)转力(N)，必须乘面积
    f_up = denorm_x(IDX_F_UP) * 1e5 * model.area_up

    term_load = f_low - f_up # 加上一个可学习的偏差修正
    # print("term_drive_2 is {}".format(term_mass + term_damping + term_stiffness + term_load))

    # ==========================================
    # 4. 计算残差
    # ==========================================

    # Eq. (4) 变形: Drive = Mass + Damping + Stiffness + Load
    # Residual = Drive - (Mass + Damping + Stiffness + Load)

    residual = term_drive - (term_mass + term_damping + term_stiffness + term_load)

    # 缩放 Loss (防止梯度爆炸)
    # 力的数量级是 10^5 ~ 10^6，平方后是 10^10，会导致训练失败
    # 我们将其除以 10^5 (100 kN) 进行归一化
    scale_factor = 1e-7

    loss = torch.mean((residual * scale_factor) ** 2)

    # if torch.isnan(loss):
    #     print("\n=== NaN Detected in Physics Loss ===")
    #     print(f"dt_min: {dt_curr.min().item()}")
    #     print(f"Drive Mean: {term_drive.mean().item()}")
    #     print(f"Mass Mean: {term_mass.mean().item()} (Acc: {d2y_diff.mean().item()})")
    #     print(f"Damp Mean: {term_damping.mean().item()} (Vel: {dy_diff.mean().item()})")
    #     print(f"Stiff Mean: {term_stiffness.mean().item()}")
    #     print(f"Load Mean: {term_load.mean().item()}")
    #     print(f"Params: B={model.B.item()}, K={model.K.item()}")
    # 调试代码
    # print(f"Res Mean: {residual.mean().item()}, Drive Mean: {term_drive.mean().item()}")

    return loss