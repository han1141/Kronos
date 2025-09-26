import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os


def preprocess_btc_data(config):
    """
    BTC数据预处理主函数
    """
    # 1. 读取BTC CSV数据
    df = pd.read_csv(config.csv_data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    # 2. 生成时间特征
    df = generate_time_features(df)

    # 3. 数据标准化
    df = normalize_features(df, config.feature_list)

    # 4. 创建滑动窗口样本
    samples = create_sliding_windows(df, config)

    # 5. 划分数据集
    train_data, val_data, test_data = split_dataset(samples, config)

    # 6. 保存处理后的数据
    save_processed_data(train_data, val_data, test_data, config)

    return train_data, val_data, test_data


def generate_time_features(df):
    """生成时间特征"""
    df["minute"] = df["timestamps"].dt.minute
    df["hour"] = df["timestamps"].dt.hour
    df["weekday"] = df["timestamps"].dt.weekday
    df["day"] = df["timestamps"].dt.day
    df["month"] = df["timestamps"].dt.month
    return df


def normalize_features(df, feature_list):
    """特征标准化"""
    for feature in feature_list:
        if feature in df.columns:
            df[f"{feature}_norm"] = (df[feature] - df[feature].mean()) / df[
                feature
            ].std()
    return df


def create_sliding_windows(df, config):
    """创建滑动窗口样本"""
    samples = []
    lookback = config.lookback_window
    predict = config.predict_window

    for i in range(lookback, len(df) - predict + 1):
        input_data = df.iloc[i - lookback : i][config.feature_list].values
        target_data = df.iloc[i : i + predict][config.feature_list].values

        sample = {
            "input_data": input_data,
            "target_data": target_data,
            "input_timestamps": df.iloc[i - lookback : i]["timestamps"].tolist(),
            "target_timestamps": df.iloc[i : i + predict]["timestamps"].tolist(),
            "symbol": "BTCUSDT",
        }
        samples.append(sample)

    return samples


def split_dataset(samples, config):
    """划分数据集为训练集、验证集和测试集"""
    total_samples = len(samples)
    
    # 按时间顺序划分：70%训练，15%验证，15%测试
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.85)
    
    train_data = samples[:train_end]
    val_data = samples[train_end:val_end]
    test_data = samples[val_end:]
    
    print(f"数据集划分完成:")
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(val_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    return train_data, val_data, test_data


def save_processed_data(train_data, val_data, test_data, config):
    """保存处理后的数据"""
    # 确保输出目录存在
    output_dir = "data/processed_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据集
    with open(f"{output_dir}/train_data.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
    with open(f"{output_dir}/val_data.pkl", "wb") as f:
        pickle.dump(val_data, f)
    
    with open(f"{output_dir}/test_data.pkl", "wb") as f:
        pickle.dump(test_data, f)
    
    print(f"数据已保存到 {output_dir}/")
    print("- train_data.pkl")
    print("- val_data.pkl")
    print("- test_data.pkl")


if __name__ == "__main__":
    from btc_config import BTCConfig
    
    print("开始BTC数据预处理...")
    
    # 加载配置
    config = BTCConfig()
    
    # 检查CSV文件是否存在
    if not os.path.exists(config.csv_data_path):
        print(f"错误: CSV文件不存在 - {config.csv_data_path}")
        exit(1)
    
    print(f"读取CSV文件: {config.csv_data_path}")
    
    try:
        # 执行数据预处理
        train_data, val_data, test_data = preprocess_btc_data(config)
        print("BTC数据预处理完成!")
        
    except Exception as e:
        print(f"数据预处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
