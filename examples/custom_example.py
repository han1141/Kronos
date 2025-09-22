import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
from datetime import datetime, timedelta

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


def is_chinese_stock_data(df):
    """
    检测是否为A股数据
    通过分析时间戳的模式来判断：
    1. 对于日级别数据：检查是否跳过周末且有交易日模式
    2. 对于小时级别数据：检查是否在A股交易时段内且跳过非交易时间
    """
    if 'timestamps' not in df.columns:
        return False
    
    timestamps = pd.to_datetime(df['timestamps'])
    
    if len(timestamps) < 10:
        return False
    weekdays = timestamps.dt.dayofweek
    has_weekend = any(weekdays >= 5)
    time_diffs = timestamps.diff().dropna()
    median_diff = time_diffs.median()
    is_hourly_data = pd.Timedelta(minutes=30) <= median_diff <= pd.Timedelta(hours=2)
    if not has_weekend:
        if is_hourly_data:
            hours = timestamps.dt.hour
            minutes = timestamps.dt.minute
            morning_session = ((hours == 9) & (minutes >= 30)) | (hours == 10) | ((hours == 11) & (minutes <= 30))
            afternoon_session = (hours >= 13) & (hours < 15)
            in_trading_hours = morning_session | afternoon_session
            
            trading_hours_ratio = in_trading_hours.sum() / len(timestamps)
            
            if trading_hours_ratio > 0.7:
                return True
        else:
            one_day_count = sum(time_diffs == pd.Timedelta(days=1))
            three_day_count = sum(time_diffs == pd.Timedelta(days=3))
            
            if one_day_count > 0 and three_day_count > 0:
                return True
    
    return False


def generate_chinese_trading_days(start_date, periods):
    """
    生成中国A股交易日
    跳过周末和主要法定节假日
    """
    holidays_2024 = [
        '2024-01-01',  # 元旦
        '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16', '2024-02-17',  # 春节
        '2024-04-04', '2024-04-05', '2024-04-06',  # 清明节
        '2024-05-01', '2024-05-02', '2024-05-03',  # 劳动节
        '2024-06-10',  # 端午节
        '2024-09-15', '2024-09-16', '2024-09-17',  # 中秋节
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06', '2024-10-07',  # 国庆节
    ]
    
    holidays_2025 = [
        '2025-01-01',  # 元旦
        '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',  # 春节
        '2025-04-05', '2025-04-06', '2025-04-07',  # 清明节
        '2025-05-01', '2025-05-02', '2025-05-03',  # 劳动节
        '2025-05-31',  # 端午节
        '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06', '2025-10-07',  # 国庆节
        '2025-10-06',  # 中秋节（与国庆重叠）
    ]
    
    all_holidays = holidays_2024 + holidays_2025
    holiday_dates = pd.to_datetime(all_holidays)
    
    trading_days = []
    current_date = pd.to_datetime(start_date)
    
    while len(trading_days) < periods:
        current_date += pd.Timedelta(days=1)
        
        if current_date.dayofweek >= 5:
            continue
            
        if current_date.normalize() in holiday_dates:
            continue
            
        trading_days.append(current_date)
    
    return pd.Series(trading_days)


def generate_chinese_trading_hours(start_timestamp, periods):
    """
    生成中国A股小时级别交易时间戳
    包含交易时段：9:30-11:30, 13:00-15:00
    跳过周末和主要法定节假日
    """
    holidays_2024 = [
        '2024-01-01',  # 元旦
        '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14', '2024-02-15', '2024-02-16', '2024-02-17',  # 春节
        '2024-04-04', '2024-04-05', '2024-04-06',  # 清明节
        '2024-05-01', '2024-05-02', '2024-05-03',  # 劳动节
        '2024-06-10',  # 端午节
        '2024-09-15', '2024-09-16', '2024-09-17',  # 中秋节
        '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06', '2024-10-07',  # 国庆节
    ]
    
    holidays_2025 = [
        '2025-01-01',  # 元旦
        '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31', '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',  # 春节
        '2025-04-05', '2025-04-06', '2025-04-07',  # 清明节
        '2025-05-01', '2025-05-02', '2025-05-03',  # 劳动节
        '2025-05-31',  # 端午节
        '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05', '2025-10-06', '2025-10-07',  # 国庆节
        '2025-10-06',  # 中秋节（与国庆重叠）
    ]
    
    all_holidays = holidays_2024 + holidays_2025
    holiday_dates = pd.to_datetime(all_holidays)
    
    morning_hours = ['09:30', '10:00', '10:30', '11:00', '11:30']
    afternoon_hours = ['13:00', '13:30', '14:00', '14:30', '15:00']
    trading_hours = morning_hours + afternoon_hours
    
    trading_timestamps = []
    current_date = pd.to_datetime(start_timestamp).normalize()
    
    while len(trading_timestamps) < periods:
        current_date += pd.Timedelta(days=1)
        
        if current_date.dayofweek >= 5:
            continue
            
        if current_date.normalize() in holiday_dates:
            continue
        
        for hour_str in trading_hours:
            if len(trading_timestamps) >= periods:
                break
            timestamp = pd.to_datetime(f"{current_date.date()} {hour_str}")
            trading_timestamps.append(timestamp)
    
    return pd.Series(trading_timestamps[:periods])


if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

csv_name = "./history.csv"
csv_name1 = "./btc_usdt_1d_no_time.csv"
df = pd.read_csv(csv_name)
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
    sys.exit()

x_df_for_forecast = df.iloc[-lookback:].copy()
x_timestamp_for_forecast = x_df_for_forecast["timestamps"]

last_timestamp = x_timestamp_for_forecast.iloc[-1]
freq = pd.infer_freq(x_timestamp_for_forecast)
if freq is None:
    freq = "h"

is_chinese_stock = is_chinese_stock_data(df)

if is_chinese_stock:
    if freq == 'h' or freq == 'H':
        future_timestamps = generate_chinese_trading_hours(last_timestamp, pred_len)
    else:
        future_timestamps = generate_chinese_trading_days(last_timestamp, pred_len)
else:
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

forecast_df["timestamps"] = y_timestamp_for_forecast.values
output_filename = "kronos_forecast_result.csv"
forecast_df.to_csv(output_filename, index=False)
def plot_prediction(historical_df, prediction_df, actual_df=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    ax1.plot(
        historical_df["timestamps"],
        historical_df["close"],
        label="li shi shu ju (Historical)",
        color="blue",
        linewidth=1.5,
    )

    ax1.plot(
        prediction_df["timestamps"],
        prediction_df["close"],
        label="yu ce shu ju (Prediction)",
        color="red",
        linewidth=1.5,
        linestyle="--",
    )

    if actual_df is not None and not actual_df.empty:
        ax1.plot(
            actual_df["timestamps"],
            actual_df["close"],
            label="zhen shi shu ju (Actual)",
            color="green",
            linewidth=1.5,
            alpha=0.8,
        )

    ax1.set_ylabel("shou pan jia (Close Price)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Kronos zhen shi yu ce jie guo (Forecasting)")

    ax2.plot(
        historical_df["timestamps"],
        historical_df["volume"],
        label="li shi cheng jiao liang",
        color="blue",
        alpha=0.7,
    )
    ax2.plot(
        prediction_df["timestamps"],
        prediction_df["volume"],
        label="yu ce cheng jiao liang",
        color="red",
        alpha=0.7,
        linestyle="--",
    )
    if actual_df is not None and not actual_df.empty:
        ax2.plot(
            actual_df["timestamps"],
            actual_df["volume"],
            label="zhen shi cheng jiao liang",
            color="green",
            alpha=0.6,
        )

    ax2.set_ylabel("cheng jiao liang (Volume)")
    ax2.set_xlabel("shi jian (Time)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


plot_prediction(x_df_for_forecast, forecast_df, actual_df=None)
