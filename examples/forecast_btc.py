import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch
# 预测btc
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

if torch.backends.mps.is_available():
    device = "mps"
    print("✅  MPS (Apple Silicon GPU).")
else:
    device = "cpu"

csv_name1 = "./btc_usdt_1d_no_time.csv"
df = pd.read_csv(csv_name1)
df["timestamps"] = pd.to_datetime(df["timestamps"])

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

predictor = KronosPredictor(
    model=model,
    tokenizer=tokenizer,
    device=device,
)

lookback = 400
pred_len = 120

if len(df) < lookback:
    print(
        f"❌ shu ju bu zu! xu yao zhi shao {lookback} hang li shi shu ju, dan wen jian zhi you {len(df)} hang."
    )
    sys.exit()

x_df_for_forecast = df.iloc[-lookback:].copy()
x_timestamp_for_forecast = x_df_for_forecast["timestamps"]
last_timestamp = x_timestamp_for_forecast.iloc[-1]
freq = pd.infer_freq(x_timestamp_for_forecast)
if freq is None:
    freq = "h"
    print(f"wu fa zi dong tui duan shi jian pin lv, yi mo ren shi yong '{freq}'.")
future_timestamps = pd.date_range(
    start=last_timestamp, periods=pred_len + 1, freq=freq
)[1:]
y_timestamp_for_forecast = pd.Series(future_timestamps)

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

x_df_for_forecast_export = x_df_for_forecast.copy()
x_df_for_forecast_export['data_type'] = 'historical'
forecast_df_export = forecast_df.copy()
forecast_df_export['data_type'] = 'prediction'
if hasattr(x_df_for_forecast_export['timestamps'].dtype, 'tz') and x_df_for_forecast_export['timestamps'].dtype.tz is not None:
    x_df_for_forecast_export['timestamps'] = x_df_for_forecast_export['timestamps'].dt.tz_localize(None)

if hasattr(forecast_df_export['timestamps'].dtype, 'tz') and forecast_df_export['timestamps'].dtype.tz is not None:
    forecast_df_export['timestamps'] = forecast_df_export['timestamps'].dt.tz_localize(None)
combined_df = pd.concat([x_df_for_forecast_export, forecast_df_export], ignore_index=True)
combined_df = combined_df.sort_values('timestamps').reset_index(drop=True)
output_filename = "kronos_btc.csv"
combined_df.to_csv(output_filename, index=False)

print(f"\n✅ 已成功导出合并数据到: {output_filename}")
print(f"   - 历史数据: {len(x_df_for_forecast_export)} 条")
print(f"   - 预测数据: {len(forecast_df_export)} 条")
print(f"   - 总计: {len(combined_df)} 条")
