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
    T=0.5,
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

output_filename = "kronos_forecast_btc.csv"
forecast_df.to_csv(output_filename, index=False)


# 4. ding yi hui tu han shu (jin xing jian hua)
def plot_prediction(historical_df, prediction_df, actual_df=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # --- XIU GAI: hui tu han shu bu zai xu yao zi ji sheng cheng shi jian chuo ---
    # ta xian zai jia she shu ru de DataFrame zi shen jiu bao han 'timestamps' lie

    ax1.plot(
        historical_df["timestamps"],
        historical_df["close"],
        label="li shi shu ju (Historical)",
        color="blue",
        linewidth=1.5,
    )

    # zhi jie shi yong prediction_df zhong de shi jian chuo jin xing hui tu
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
    # zhi jie shi yong prediction_df zhong de shi jian chuo jin xing hui tu
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


# 5. hui zhi jie guo
# plot_prediction(x_df_for_forecast, forecast_df, actual_df=None)
