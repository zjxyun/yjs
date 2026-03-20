import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from scipy.signal import savgol_filter


def load_and_process_data_2(file_path):
    print(f"====== 开始处理数据: {file_path} ======")

    # 1. 读取数据 (增强版：支持不同编码和分隔符)
    try:
        # 尝试标准CSV (逗号分隔) + UTF-8
        df = pd.read_csv(file_path, sep=',', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            # 尝试 GBK (Excel/中文常用)
            df = pd.read_csv(file_path, sep=',', encoding='gbk')
        except:
            # 尝试制表符
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')

    print(f"Step 1 [原始读取]: {df.shape}")

    # 2. 关键修复：强制转换为数字类型
    # 这一步解决了 "TypeError: str - str" 的问题
    # errors='coerce' 会把所有非数字字符（如中文、符号、空字符）直接变成 NaN
    cols_to_convert = [
        '2#土压传感器压力', '3#土压传感器压力', '4#土压传感器压力',
        '5#土压传感器压力', '6#土压传感器压力',
        'C组推进压力', 'E组推进压力',
        'C组推进位移行程', 'E组推进位移行程',
        '推进速度'
    ]

    # 如果有'时间'列（字符串格式），需要先解析为时间戳
    if '时间' in df.columns:
        try:
            df['time_parsed'] = pd.to_datetime(df['时间'])
            # 转换为相对于开始的秒数 (float)
            start_time = df['time_parsed'].iloc[0]
            df['t_sec'] = (df['time_parsed'] - start_time).dt.total_seconds()
        except:
            print("警告：'时间'列解析失败，回退使用索引作为时间")
            df['t_sec'] = df.index.values.astype(float)
    else:
        df['t_sec'] = df.index.values.astype(float)




    # 检查列是否存在，防止报错
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"警告：数据中缺少列 {col}，将用 0 填充")

    # 删除转换后产生 NaN 的脏数据行
    original_len = len(df)
    df.dropna(subset=cols_to_convert + ['t_sec'], inplace=True)
    print(f"Step 2 [清洗脏数据]: {original_len} -> {len(df)}")

    # 设定阈值
    df = df[df['推进速度'] > 0].copy()
    # 计算当前行与上一行的时间差
    # 注意：这里是基于筛选后的数据计算 diff
    time_diff = df['t_sec'].diff()
    # 2. 定义断点阈值
    # 正常采样间隔是 1s。考虑到网络延迟，如果间隔 > 3s，认为中间断开了
    MAX_GAP = 420

    # 3. 标记断点：如果 time_diff > 3s 或者 time_diff 为 NaN (第一行)，则是一个新片段的开始
    # fillna(True) 确保第一行被标记为新片段
    is_new_segment = (time_diff > MAX_GAP).fillna(True)

    # 4. 生成片段 ID (累加布尔值)
    # 例如: [True, False, False, True, False] -> [1, 1, 1, 2, 2]
    df['segment_id'] = is_new_segment.cumsum()

    print(f"检测到 {df['segment_id'].max()} 个连续掘进片段")

    # 修正点 B: 过滤过短的片段
    # ====================================================
    # 统计每个片段的长度
    segment_counts = df['segment_id'].value_counts()
    # 我们需要至少 3 行数据才能计算 t-2 和 dt_prev
    valid_segments = segment_counts[segment_counts >= 4].index

    # 只保留有效片段
    df = df[df['segment_id'].isin(valid_segments)].copy()
    print(f"过滤短片段(len<3)后: 剩余 {len(df)} 行")

    if len(df) == 0:
        raise ValueError("数据不足：没有长度大于3的连续掘进片段！请检查原始数据的时间连续性。")

    print("正在进行零损耗数据平滑处理 (min_periods=1)...")
    #
    # # 1. 定义需要平滑的列 (压力和行程)
    # # 不要平滑时间(t_sec)和ID
    cols_to_smooth = [
        '2#土压传感器压力', '3#土压传感器压力', '4#土压传感器压力',
        '5#土压传感器压力', '6#土压传感器压力',
        'C组推进压力', 'E组推进压力'
    ]

    def smooth_func(x):
        # 窗口长度 (window_length): 必须是奇数。建议 5 或 7。
        # 多项式阶数 (polyorder): 建议 2 或 3。
        # 窗口越小，保留细节越多；窗口越大，越平滑。

        # 异常处理：如果片段太短(少于窗口长度)，Savgol会报错
        # 此时降级使用简单的 rolling mean 或原值
        if len(x) < 5:
            return x.rolling(3, min_periods=1).mean()

        try:
            # mode='interp' 处理边界效果较好
            return savgol_filter(x, window_length=5, polyorder=2, mode='interp')
        except:
            return x

        # 应用变换

    df[cols_to_smooth] = df.groupby('segment_id')[cols_to_smooth].transform(smooth_func)

    print(f"平滑处理完成 (S-G Filter). 数据行数: {len(df)}")

    # ====================================================
    # 步骤 C: 在片段内部计算滞后特征和差分
    # ====================================================

    # 定义需要构建历史特征的列
    target_cols = ['C组推进位移行程', 'E组推进位移行程']  # y_lower, y_upper

    # 使用 groupby 确保 shift 不会跨越片段
    # 比如：片段1的最后一行绝不会滑到片段2的第一行去
    grouped = df.groupby('segment_id')



    # 1. 构建 y(t-1) 和 y(t-2)
    for col in target_cols:
        df[f'{col}_t-1'] = grouped[col].shift(1)
        df[f'{col}_t-2'] = grouped[col].shift(2)
        df[f'{col}_t-3'] = grouped[col].shift(3)

    df['y_lower'] = df['C组推进位移行程']
    df['y_upper'] = df['E组推进位移行程']
    df['y_lower_t-1'] = df['C组推进位移行程_t-1']
    df['y_upper_t-1'] = df['E组推进位移行程_t-1']
    df['y_lower_t-2'] = df['C组推进位移行程_t-2']
    df['y_upper_t-2'] = df['E组推进位移行程_t-2']
    df['y_lower_t-3'] = df['C组推进位移行程_t-3']
    df['y_upper_t-3'] = df['E组推进位移行程_t-3']

    # 2. 计算 dt (时间差)
    # dt_current: t - (t-1)
    df['dt_current'] = grouped['t_sec'].diff()

    # dt_prev: (t-1) - (t-2)
    # 这里通过 shift dt_current 得到
    df['dt_prev'] = grouped['dt_current'].shift(1)
    df['dt_prev2'] = grouped['dt_current'].shift(2)


    # 3. 特征映射 (Mapping)

    # # 1. 计算不固定的时间间隔 (Delta t)
    # # dt_current: t 与 t-1 的差值
    # df['dt_current'] = df['t_sec'].diff()
    #
    # # dt_prev: t-1 与 t-2 的差值 (用于计算上一时刻的速度，进而求加速度)
    # df['dt_prev'] = df['dt_current'].shift(1)
    # df['dt_prev2'] = df['dt_current'].shift(2)

    # 2. 目标值 (y_t)
    # df['y_lower'] = df['D组推进位移行程']
    # df['y_upper'] = df['A组推进位移行程']

    # 3. 一阶滞后 (y_t-1) -> 作为网络输入 X
    # df['y_lower_t-1'] = df['y_lower'].shift(1)
    # df['y_upper_t-1'] = df['y_upper'].shift(1)
    #
    # # 4. 二阶滞后 (y_t-2) -> 仅用于物理Loss计算加速度 (辅助数据)
    # df['y_lower_t-2'] = df['y_lower'].shift(2)
    # df['y_upper_t-2'] = df['y_upper'].shift(2)
    #
    # df['y_lower_t-3'] = df['y_lower'].shift(3)
    # df['y_upper_t-3'] = df['y_upper'].shift(3)

    # x1~x5: 地层负载/反力
    df['x1_load'] = df['2#土压传感器压力']
    df['x2_load'] = df['3#土压传感器压力']
    df['x3_load'] = df['4#土压传感器压力']
    df['x4_load'] = df['5#土压传感器压力']
    df['x5_load'] = df['6#土压传感器压力']

    # x6: 压力差 (下 - 上)
    # 此时这两列已经是 float 类型，不会报错了
    df['x6_diff'] = df['C组推进压力'] - df['E组推进压力']



    # ... (前面的特征计算代码保持不变) ...


    # ================= 修改开始 =================

    # 1. 先定义我们真正需要的列 (输入 + 输出)
    # 这一步本来是在后面做的，现在提上来
    used_cols = [
        't_sec',
        'x1_load', 'x2_load', 'x3_load', 'x4_load', 'x5_load',
        'x6_diff', 'y_lower_t-1', 'y_upper_t-1','y_lower_t-2','y_upper_t-2', 'y_lower_t-3','y_upper_t-3',
        'dt_current', 'dt_prev', 'dt_prev2', 'y_lower', 'y_upper', 'C组推进压力', 'E组推进压力'
    ]

    # 2. 只保留这些有用的列 (把其他无关的、可能含空值的列扔掉)
    df = df[used_cols].copy()

    df.dropna(inplace=True)



    # ================= 修改结束 =================

    print(f"Step 3 [只保留有用列并清洗后]: {df.shape}")

    # if len(df) == 0:
    #     raise ValueError("错误：构建时序特征后数据为空！请检查 selected_cols 里的列是否本身全就是空的。")

    # 5. 整理最终输入输出矩阵
    feature_cols_X = [
        't_sec',
        'x1_load', 'x2_load', 'x3_load', 'x4_load', 'x5_load',
        'x6_diff',
        'y_lower_t-1', 'y_upper_t-1'
    ]

    # 目标输出 Y (2维)
    feature_cols_Y = ['y_lower', 'y_upper']

    # 物理上下文 P
    feature_cols_Phys = ['dt_current', 'dt_prev', 'dt_prev2', 'y_lower_t-2','y_upper_t-2', 'y_lower_t-3','y_upper_t-3','E组推进压力', 'C组推进压力']

    X = df[feature_cols_X].values
    Y = df[feature_cols_Y].values
    Phys_raw = df[feature_cols_Phys].values

    # 6. 归一化 (Min-Max Scaling)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # 这里如果 X 为空会报 ValueError，但前面已经加了检查
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)

    # 7. 转换为 Tensor
    # requires_grad=True 是为了 PINN 计算 PDE 导数
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)
    Phys_tensor = torch.tensor(Phys_raw, dtype=torch.float32)

    print("====== 数据处理成功！======")
    print(f"输入 X 形状: {X_tensor.shape}")
    print(f"输出 Y 形状: {Y_tensor.shape}")
    print(f"Phys (Context) shape: {Phys_tensor.shape}")

    # 1. 把最终的 DataFrame 保存出来
    # encoding='utf-8-sig' 可以防止 Excel 打开中文乱码
    # df.to_csv('final_processed_data.csv', index=False, encoding='utf-8-sig')
    # print("已保存清洗后的数据到 final_processed_data.csv，请在文件夹中查看。")

    # # 2. 如果你想看归一化后的数据，也可以拼起来保存
    # # 把 X 和 Y 拼回去
    # data_norm = np.hstack((X_scaled, Y_scaled))
    # # 保存为 numpy 文件或者 csv
    # np.savetxt("final_normalized_data.csv", data_norm, delimiter=",")

    return X_tensor, Y_tensor, Phys_tensor, scaler_x, scaler_y



# 运行测试
if __name__ == "__main__":
    # 确保文件名和路径正确
    try:
        X_data, Y_data,Phys_data, scaler_x, scaler_y = load_and_process_data_2('train_dataset.csv')
    except Exception as e:
        print(f"运行出错: {e}")