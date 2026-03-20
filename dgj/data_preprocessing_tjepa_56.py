import pandas as pd
import numpy as np
import os

# =============================================================================
# 配置部分
# =============================================================================
# 原始数据文件路径
RAW_DATA_PATH = 'train_dataset.csv'  # 请修改为你的原始大宽表路径
OUTPUT_PATH = 'train_pretrain.csv'

# 需要强制转换为数值的列（从你的参数.txt中提取的核心参数）
# 这些列如果包含乱码或空值，会被强制转为 NaN 并填充 0
NUMERIC_COLS = [
    # 行程
    'A组推进位移行程', 'C组推进位移行程', 'E组推进位移行程', 'F组推进位移行程', 'B组推进位移行程', 'D组推进位移行程',
    # 土压
    '2#土压传感器压力', '3#土压传感器压力', '4#土压传感器压力', '5#土压传感器压力', '6#土压传感器压力',
    # 推进压力
    'A组推进压力', 'B组推进压力', 'C组推进压力', 'D组推进压力', 'E组推进压力', 'F组推进压力', '推进泵压力'
    # 关键状态
    '推进速度', '推进总推力', '刀盘贯入度', '皮带机转速', '螺机转速', '螺机扭矩', '刀盘总扭矩', '刀盘总功率',
    # 注浆压力 (示例，可根据需要添加更多)
    '1#注浆A液压力', '2#注浆A液压力', '5#注浆A液压力', '6#注浆A液压力',
    # 泡沫系统 (示例)
    '1路泡沫压力', '2路泡沫压力', '3路泡沫压力', '4路泡沫压力', '5路泡沫压力', '6路泡沫压力', '7路泡沫压力', '8路泡沫压力', '9路泡沫压力',
    '1路泡沫空气流量', '2路泡沫空气流量', '3路泡沫空气流量', '4路泡沫空气流量', '5路泡沫空气流量', '6路泡沫空气流量', '7路泡沫空气流量', '8路泡沫空气流量', '9路泡沫空气流量',
    '1路泡沫混合液流量', '2路泡沫混合液流量', '3路泡沫混合液流量', '4路泡沫混合液流量', '5路泡沫混合液流量', '6路泡沫混合液流量', '7路泡沫混合液流量', '8路泡沫混合液流量', '9路泡沫混合液流量',
    '中心喷水1#压力检测', '中心喷水2#压力检测', '中心喷水3#压力检测', '中心喷水4#压力检测'
]

# 用于判断是否在掘进的关键列
SPEED_COL = '推进速度'


def preprocess_shield_data(file_path, output_path):
    print(f"🚀 开始处理数据: {file_path}")

    # 1. 读取数据
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except:
            print("❌ 读取失败，请检查文件编码")
            return

    print(f"原始数据形状: {df.shape}")

    # 2. 自动识别所有可能的数值列
    # 除了时间、ID等，只要是传感器读数都应该纳入特征
    # 策略：先尝试将 NUMERIC_COLS 转为数值，然后把 df 中所有能转为数值的列都留下来

    # 强制转换核心列
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 删除推进速度为空或为0的行 (非掘进状态)
    # if SPEED_COL in df.columns:
    #     original_len = len(df)
    #     df.dropna(subset=[SPEED_COL], inplace=True)
    #     df = df[df[SPEED_COL] > 0.5]  # 阈值设为0.5，过滤极其微小的蠕动
    #     print(f"过滤非掘进状态后: {original_len} -> {len(df)}")
    # else:
    #     print(f"⚠️ 警告：未找到 '{SPEED_COL}' 列，无法过滤非掘进状态")

    # 3. 处理时间戳 (用于切分片段)
    if '时间' in df.columns:
        try:
            df['time_parsed'] = pd.to_datetime(df['时间'])
            start_time = df['time_parsed'].iloc[0]
            df['t_sec'] = (df['time_parsed'] - start_time).dt.total_seconds()
        except:
            print("时间列解析失败，使用索引代替")
            df['t_sec'] = df.index.values.astype(float)
    else:
        df['t_sec'] = df.index.values.astype(float)

    # # 4. 切分连续掘进片段
    # # 如果两条数据间隔超过 180秒 (3分钟)，认为是一个新的掘进段
    # time_diff = df['t_sec'].diff()
    # is_new_segment = (time_diff > 180).fillna(True)
    # df['segment_id'] = is_new_segment.cumsum()
    #
    # # 过滤过短的片段 (小于10个点的片段没有学习意义)
    # segment_counts = df['segment_id'].value_counts()
    # valid_segments = segment_counts[segment_counts >= 10].index
    # df = df[df['segment_id'].isin(valid_segments)].copy()
    #
    # print(f"保留了 {len(valid_segments)} 个有效连续掘进段，共 {len(df)} 行数据")

    # 5. 最终特征选择
    # 我们希望 T-JEPA 学习所有传感器之间的关联
    # 所以：保留所有数值型列，排除 ID、时间字符串等

    # 自动选择数值列
    numeric_df = df.select_dtypes(include=[np.number])

    # 排除不必要的列
    drop_cols = ['segment_id', '环数', 'Column1']  # 't_sec' 可以保留，作为时间特征
    final_cols = [c for c in numeric_df.columns if c not in drop_cols]

    final_cols = sorted(final_cols)  # 排序保证顺序固定
    # 填充剩余的 NaN (用 0 填充，或者用前向填充)
    df_final = numeric_df[final_cols].copy()
    df_final.dropna(inplace=True)

    # # x1~x5: 地层负载/反力
    # df['x1_load'] = df['2#土压传感器压力']
    # df['x2_load'] = df['3#土压传感器压力']
    # df['x3_load'] = df['4#土压传感器压力']
    # df['x4_load'] = df['5#土压传感器压力']
    # df['x5_load'] = df['6#土压传感器压力']
    #
    # # x6: 压力差 (下 - 上)
    # # 此时这两列已经是 float 类型，不会报错了
    # df['x6_diff'] = df['C组推进压力'] - df['E组推进压力']
    #
    # df['y_lower'] = df['C组推进位移行程']
    # df['y_upper'] = df['E组推进位移行程']
    #
    # used_cols = [
    #     't_sec',
    #     'x1_load', 'x2_load', 'x3_load', 'x4_load', 'x5_load',
    #     'x6_diff',  'y_lower', 'y_upper'
    # ]
    # df_final = df[used_cols].copy()
    #
    # df_final.dropna(inplace=True)

    print(f"✅ 最终特征数量: {df_final.shape[1]}")
    print(f"特征列表 : {df_final.columns[:10].tolist()}")

    # 记录 9 个核心特征的索引
    physics_9_names = [
        '2#土压传感器压力', '3#土压传感器压力', '4#土压传感器压力', '5#土压传感器压力', '6#土压传感器压力',
        'C组推进压力', 'E组推进压力', 'C组推进位移行程', 'E组推进位移行程'
    ]
    # 获取这 9 个特征在 df_final 中的位置
    physics_idx = [df_final.columns.get_loc(c) for c in physics_9_names if c in df_final.columns]

    # 保存列名列表，这是预训练和下游任务对接的唯一凭证
    with open("feature_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(df_final.columns.tolist()))

    print(f"核心 9 特征索引已确认: {physics_idx}")

    # 6. 保存
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    print(f"💾 预处理数据已保存至: {output_path}")

    # 返回列名，供后续检查
    return df_final.columns.tolist()


if __name__ == "__main__":
    preprocess_shield_data(RAW_DATA_PATH, OUTPUT_PATH)