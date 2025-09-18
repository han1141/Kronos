# Kronos 基础使用流程指南

## 🎯 快速开始

本指南将带您完成Kronos的基础预测功能，从数据准备到获得预测结果。

## 📊 数据准备

### 1. 数据格式要求

Kronos支持CSV和Feather格式的金融数据，必须包含以下列：

**必需列**:
- `open`: 开盘价
- `high`: 最高价  
- `low`: 最低价
- `close`: 收盘价

**可选列**:
- `volume`: 成交量
- `amount`: 成交额（仅用于显示，不参与预测）
- `timestamps`/`timestamp`/`date`: 时间戳

### 2. 数据示例

```csv
timestamps,open,high,low,close,volume,amount
2024-01-01 09:30:00,100.0,102.5,99.8,101.2,1000000,101200000
2024-01-01 09:35:00,101.2,103.0,100.5,102.8,1200000,122400000
2024-01-01 09:40:00,102.8,104.2,102.0,103.5,1100000,113850000
...
```

### 3. 数据质量检查

```python
import pandas as pd

# 加载数据
df = pd.read_csv("your_data.csv")

# 基础检查
print(f"数据行数: {len(df)}")
print(f"数据列: {list(df.columns)}")
print(f"缺失值: {df.isnull().sum()}")

# 检查必需列
required_cols = ['open', 'high', 'low', 'close']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ 缺少必需列: {missing_cols}")
else:
    print("✅ 数据格式正确")
```

## 🚀 基础预测流程

### 步骤1: 导入模块

```python
import pandas as pd
import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor
```

### 步骤2: 加载模型

```python
# 从Hugging Face加载预训练模型
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 创建预测器
predictor = KronosPredictor(
    model=model, 
    tokenizer=tokenizer, 
    device="cuda:0",  # 或 "cpu", "mps"
    max_context=512
)
```

### 步骤3: 准备数据

```python
# 加载数据
df = pd.read_csv("./data/your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 设置参数
lookback = 400    # 历史数据长度
pred_len = 120    # 预测长度

# 准备输入数据
x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']]
x_timestamp = df.iloc[:lookback]['timestamps']
y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
```

### 步骤4: 执行预测

```python
# 进行预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # 温度参数
    top_p=0.9,      # 核采样概率
    sample_count=1, # 采样次数
    verbose=True    # 显示进度
)

print("预测结果:")
print(pred_df.head())
```

### 步骤5: 结果可视化

```python
def plot_prediction(historical_df, pred_df, actual_df=None):
    """绘制预测结果"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 绘制价格
    ax1.plot(historical_df['timestamps'], historical_df['close'], 
             label='历史数据', color='blue', linewidth=1.5)
    
    # 预测数据时间戳
    pred_timestamps = pd.date_range(
        start=historical_df['timestamps'].iloc[-1], 
        periods=len(pred_df)+1, 
        freq='5T'
    )[1:]  # 排除第一个点避免重复
    
    ax1.plot(pred_timestamps, pred_df['close'], 
             label='预测数据', color='red', linewidth=1.5)
    
    if actual_df is not None:
        ax1.plot(pred_timestamps, actual_df['close'], 
                 label='实际数据', color='green', linewidth=1.5)
    
    ax1.set_ylabel('收盘价')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Kronos 价格预测结果')
    
    # 绘制成交量
    ax2.plot(historical_df['timestamps'], historical_df['volume'], 
             label='历史成交量', color='blue', alpha=0.7)
    ax2.plot(pred_timestamps, pred_df['volume'], 
             label='预测成交量', color='red', alpha=0.7)
    
    ax2.set_ylabel('成交量')
    ax2.set_xlabel('时间')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 绘制结果
plot_prediction(x_df, pred_df)
```

## 🔧 参数调优

### 预测质量参数

#### 1. Temperature (T)
- **范围**: 0.1 - 2.0
- **作用**: 控制预测的随机性
- **建议**: 
  - `T=0.8-1.2`: 平衡的预测
  - `T<0.8`: 更保守的预测
  - `T>1.2`: 更多样化的预测

#### 2. Nucleus Sampling (top_p)
- **范围**: 0.1 - 1.0
- **作用**: 控制预测的多样性
- **建议**:
  - `top_p=0.9-1.0`: 考虑更多可能性
  - `top_p=0.7-0.9`: 平衡质量和多样性
  - `top_p<0.7`: 更集中的预测

#### 3. Sample Count
- **范围**: 1 - 5
- **作用**: 生成多个样本并平均
- **建议**:
  - `sample_count=1`: 快速预测
  - `sample_count=3-5`: 更稳定的结果

### 示例参数组合

```python
# 保守预测（适合风险厌恶）
pred_df_conservative = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=pred_len, T=0.8, top_p=0.7, sample_count=3
)

# 平衡预测（推荐设置）
pred_df_balanced = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=pred_len, T=1.0, top_p=0.9, sample_count=2
)

# 探索性预测（适合研究分析）
pred_df_exploratory = predictor.predict(
    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
    pred_len=pred_len, T=1.5, top_p=1.0, sample_count=5
)
```

## 📈 批量预测

对于多个时间序列或多个时间段的预测：

```python
# 准备多个数据集
df_list = []
x_timestamp_list = []
y_timestamp_list = []

for i in range(3):  # 3个不同的时间段
    start_idx = i * 200
    x_df_i = df.iloc[start_idx:start_idx+lookback][['open', 'high', 'low', 'close', 'volume']]
    x_ts_i = df.iloc[start_idx:start_idx+lookback]['timestamps']
    y_ts_i = df.iloc[start_idx+lookback:start_idx+lookback+pred_len]['timestamps']
    
    df_list.append(x_df_i)
    x_timestamp_list.append(x_ts_i)
    y_timestamp_list.append(y_ts_i)

# 批量预测
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# 处理结果
for i, pred_df in enumerate(pred_df_list):
    print(f"预测结果 {i+1}:")
    print(pred_df.head())
```

## 🎯 无成交量预测

如果数据中没有成交量信息：

```python
# 只使用OHLC数据
x_df_no_vol = df.iloc[:lookback][['open', 'high', 'low', 'close']]

pred_df_no_vol = predictor.predict(
    df=x_df_no_vol,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1
)
```

## 📊 结果分析

### 1. 基础统计

```python
# 预测统计
print("预测结果统计:")
print(f"预测点数: {len(pred_df)}")
print(f"价格范围: {pred_df['close'].min():.2f} - {pred_df['close'].max():.2f}")
print(f"平均价格: {pred_df['close'].mean():.2f}")
print(f"价格变化: {((pred_df['close'].iloc[-1] / pred_df['close'].iloc[0]) - 1) * 100:.2f}%")
```

### 2. 趋势分析

```python
# 趋势判断
price_change = pred_df['close'].iloc[-1] - pred_df['close'].iloc[0]
if price_change > 0:
    trend = "上涨"
elif price_change < 0:
    trend = "下跌"
else:
    trend = "横盘"

print(f"预测趋势: {trend}")
print(f"预测涨跌幅: {(price_change / pred_df['close'].iloc[0]) * 100:.2f}%")
```

## ⚠️ 注意事项

1. **数据质量**: 确保输入数据无缺失值和异常值
2. **时间连续性**: 时间戳应该连续，无大的间隔
3. **数据长度**: 至少需要400个历史数据点
4. **设备选择**: GPU可显著提升预测速度
5. **内存管理**: 大批量预测时注意内存使用

## 🔗 下一步

- 学习使用 [Web UI界面] 进行可视化预测
- 了解 [微调训练] 适应特定市场
- 查看 [故障排除指南] 解决常见问题

---

**提示**: 这只是基础使用流程，实际应用中建议结合领域知识和风险管理策略。