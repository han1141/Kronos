import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# --- dong tai xuan ze she bei ---
if torch.backends.mps.is_available():
    device = "mps"
    print("✅ jian ce dao MPS (Apple Silicon GPU), jiang shi yong GPU.")
else:
    device = "cpu"
    print("⚠️ wei jian ce dao MPS, jiang shi yong CPU.")

# 1. jia zai shu ju he mo xing
csv_name = "./history.csv"
df = pd.read_csv(csv_name)
df["timestamps"] = pd.to_datetime(df["timestamps"])

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

predictor = KronosPredictor(
    model=model,
    tokenizer=tokenizer,
    device=device,
)

# 2. zhun bei shu ju (yong yu zhen shi yu ce)
lookback = 400
pred_len = 120

if len(df) < lookback:
    print(
        f"❌ shu ju bu zu! xu yao zhi shao {lookback} hang li shi shu ju, dan wen jian zhi you {len(df)} hang."
    )
    sys.exit()

x_df_for_forecast = df.iloc[-lookback:].copy()
x_timestamp_for_forecast = x_df_for_forecast["timestamps"]

print(f"shi yong zui hou {lookback} tiao shu ju jin xing zhen shi yu ce...")
print(
    f"li shi shu ju de zui hou yi ge shi jian dian shi: {x_timestamp_for_forecast.iloc[-1]}"
)

# shou dong chuang jian wei lai de shi jian chuo (zhe yi bu shi zheng que qie bi xu de)
last_timestamp = x_timestamp_for_forecast.iloc[-1]
freq = pd.infer_freq(x_timestamp_for_forecast)
if freq is None:
    freq = "h"
    print(f"wu fa zi dong tui duan shi jian pin lv, yi mo ren shi yong '{freq}'.")
future_timestamps = pd.date_range(
    start=last_timestamp, periods=pred_len + 1, freq=freq
)[1:]
y_timestamp_for_forecast = pd.Series(future_timestamps)

# 3. zhi xing zhen shi yu ce
forecast_df = predictor.predict(
    df=x_df_for_forecast[["open", "high", "low", "close", "volume"]],
    x_timestamp=x_timestamp_for_forecast,
    y_timestamp=y_timestamp_for_forecast,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=3,
    verbose=True,
)

# --- HE XIN XIU GAI: Jiang zheng que de shi jian chuo tian jia dao yu ce jie guo zhong ---
# predictor fan hui de forecast_df zhi you shu zhi, mei you shi jian. Wo men shou dong ba ta men pin jie qi lai.
# wei le que bao an quan pin jie, wo men hu lue liang zhe de yuan you suo yin
forecast_df["timestamps"] = y_timestamp_for_forecast.values

print("\nxiu zheng hou de yu ce jie guo (yi bao han zheng que de shi jian chuo):")
print(forecast_df.head())

# 5. 合并历史数据和预测数据一起导出
# 为历史数据添加标识列
x_df_for_forecast_export = x_df_for_forecast.copy()
x_df_for_forecast_export['data_type'] = 'historical'

# 为预测数据添加标识列
forecast_df_export = forecast_df.copy()
forecast_df_export['data_type'] = 'prediction'

# 修复时区问题：确保两个DataFrame的timestamps列都是相同的时区格式
# 将所有时间戳转换为无时区格式（tz-naive）
if hasattr(x_df_for_forecast_export['timestamps'].dtype, 'tz') and x_df_for_forecast_export['timestamps'].dtype.tz is not None:
    x_df_for_forecast_export['timestamps'] = x_df_for_forecast_export['timestamps'].dt.tz_localize(None)

if hasattr(forecast_df_export['timestamps'].dtype, 'tz') and forecast_df_export['timestamps'].dtype.tz is not None:
    forecast_df_export['timestamps'] = forecast_df_export['timestamps'].dt.tz_localize(None)

# 合并历史数据和预测数据
combined_df = pd.concat([x_df_for_forecast_export, forecast_df_export], ignore_index=True)

# 按时间排序
combined_df = combined_df.sort_values('timestamps').reset_index(drop=True)

# 导出合并后的数据
output_filename = "kronos_a_share.csv"
combined_df.to_csv(output_filename, index=False)

print(f"\n✅ 已成功导出合并数据到: {output_filename}")
print(f"   - 历史数据: {len(x_df_for_forecast_export)} 条")
print(f"   - 预测数据: {len(forecast_df_export)} 条")
print(f"   - 总计: {len(combined_df)} 条")

# 同时保留原来的单独预测数据导出
# forecast_only_filename = "kronos_forecast_only_btc.csv"
# forecast_df.to_csv(forecast_only_filename, index=False)
# print(f"✅ 同时导出纯预测数据到: {forecast_only_filename}")
