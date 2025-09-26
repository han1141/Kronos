import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch
import os
# 预测btc - 使用微调后的模型
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

if torch.backends.mps.is_available():
    device = "mps"
    print("✅  MPS (Apple Silicon GPU).")
else:
    device = "cpu"

csv_name1 = "./csv/btc_usdt_1d_no_time.csv"
df = pd.read_csv(csv_name1)
df["timestamps"] = pd.to_datetime(df["timestamps"])

# =================================================================
# 使用微调后的模型
# =================================================================
print("🔄 正在加载微调后的模型...")

# 微调后的tokenizer路径
finetuned_tokenizer_path = "../outputs/models/btc_finetune_tokenizer/checkpoints/best_model"
# 微调后的predictor路径（如果存在）
finetuned_predictor_path = "../outputs/models/btc_finetune_predictor/checkpoints/best_model"

# 检查微调后的predictor是否存在
if os.path.exists(finetuned_predictor_path):
    print("✅ 找到微调后的predictor模型，正在加载...")
    tokenizer = KronosTokenizer.from_pretrained(finetuned_tokenizer_path)
    model = Kronos.from_pretrained(finetuned_predictor_path)
    print("✅ 成功加载微调后的tokenizer和predictor")
else:
    print("⚠️  未找到微调后的predictor模型，使用微调后的tokenizer + 预训练的predictor")
    tokenizer = KronosTokenizer.from_pretrained(finetuned_tokenizer_path)
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print("✅ 成功加载微调后的tokenizer和预训练的predictor")

predictor = KronosPredictor(
    model=model,
    tokenizer=tokenizer,
    device=device,
)

# =================================================================
# 预测参数设置
# =================================================================
lookback = 400
pred_len = 120

if len(df) < lookback:
    print(
        f"❌ 数据不足！需要至少 {lookback} 行历史数据，但文件只有 {len(df)} 行。"
    )
    sys.exit()

# =================================================================
# 数据准备和预测
# =================================================================
x_df_for_forecast = df.iloc[-lookback:].copy()
x_timestamp_for_forecast = x_df_for_forecast["timestamps"]
last_timestamp = x_timestamp_for_forecast.iloc[-1]
freq = pd.infer_freq(x_timestamp_for_forecast)
if freq is None:
    freq = "h"
    print(f"无法自动推断时间频率，已默认使用 '{freq}'。")

future_timestamps = pd.date_range(
    start=last_timestamp, periods=pred_len + 1, freq=freq
)[1:]
y_timestamp_for_forecast = pd.Series(future_timestamps)

print("🔮 开始进行BTC价格预测...")
forecast_df = predictor.predict(
    df=x_df_for_forecast[["open", "high", "low", "close", "volume"]],
    x_timestamp=x_timestamp_for_forecast,
    y_timestamp=y_timestamp_for_forecast,
    pred_len=pred_len,
    T=0.7,
    top_p=1.0,
    sample_count=3,
    verbose=True,
)
forecast_df["timestamps"] = y_timestamp_for_forecast.values

# =================================================================
# 结果导出
# =================================================================
x_df_for_forecast_export = x_df_for_forecast.copy()
x_df_for_forecast_export['data_type'] = 'historical'
forecast_df_export = forecast_df.copy()
forecast_df_export['data_type'] = 'prediction'

# 处理时区信息
if hasattr(x_df_for_forecast_export['timestamps'].dtype, 'tz') and x_df_for_forecast_export['timestamps'].dtype.tz is not None:
    x_df_for_forecast_export['timestamps'] = x_df_for_forecast_export['timestamps'].dt.tz_localize(None)

if hasattr(forecast_df_export['timestamps'].dtype, 'tz') and forecast_df_export['timestamps'].dtype.tz is not None:
    forecast_df_export['timestamps'] = forecast_df_export['timestamps'].dt.tz_localize(None)

combined_df = pd.concat([x_df_for_forecast_export, forecast_df_export], ignore_index=True)
combined_df = combined_df.sort_values('timestamps').reset_index(drop=True)
output_filename = "csv/kronos_btc_finetuned.csv"
combined_df.to_csv(output_filename, index=False)

print(f"\n✅ 已成功导出合并数据到: {output_filename}")
print(f"   - 历史数据: {len(x_df_for_forecast_export)} 条")
print(f"   - 预测数据: {len(forecast_df_export)} 条")
print(f"   - 总计: {len(combined_df)} 条")

# =================================================================
# 模型信息总结
# =================================================================
print("\n📊 使用的模型信息:")
if os.path.exists(finetuned_predictor_path):
    print("   - Tokenizer: 微调后的BTC专用tokenizer")
    print("   - Predictor: 微调后的BTC专用predictor")
    print("   - 模型状态: 完全微调")
else:
    print("   - Tokenizer: 微调后的BTC专用tokenizer")
    print("   - Predictor: 预训练的通用predictor")
    print("   - 模型状态: 部分微调（仅tokenizer）")
print(f"   - 设备: {device}")
print(f"   - 历史窗口: {lookback} 天")
print(f"   - 预测窗口: {pred_len} 天")