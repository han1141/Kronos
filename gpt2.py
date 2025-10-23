import numpy as np
import pandas as pd
import os
import logging
import pandas_ta as ta
import joblib
import time
import requests
import datetime
from tensorflow.keras.models import load_model
from backtesting import Strategy, Backtest
from backtesting.lib import FractionalBacktest

# --- 0. 设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
CONFIG = {"data_cache": "data_cache"}


# --- 彻底修正后的数据获取函数 ---
def fetch_binance_klines(symbol, interval, start_str, end_str=None):
    cache_dir = CONFIG["data_cache"]
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol.lower()}_{interval}.csv")
    start_dt = pd.to_datetime(start_str, utc=True)
    end_dt = pd.to_datetime(end_str, utc=True) if end_str else pd.Timestamp.utcnow()

    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            required_start = pd.to_datetime("2017-01-01", utc=True)
            if (
                not df.empty
                and df.index[0] <= required_start
                and df.index[-1] >= end_dt
            ):
                logger.info(
                    f"✅ 从有效缓存加载 {symbol} ({interval}) 数据: {cache_file}"
                )
                return df
        except Exception as e:
            logger.warning(f"读取缓存文件失败: {e}, 将重新获取数据。")

    logger.info(f"正在从币安获取 {symbol} ({interval}) 的数据...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start_ts = int(pd.to_datetime("2017-01-01", utc=True).timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    while current_start_ts < end_ts:
        # ** 之前的错误在这里，硬编码了 interval='1h'，现已修正 **
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current_start_ts = data[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.warning(f"获取 {symbol} 数据失败: {e}. 等待 3s 重试...")
            time.sleep(3)

    if not all_data:
        logger.error(f"未能获取 {symbol} 的任何数据。")
        return pd.DataFrame()
    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df = df.set_index("timestamp").sort_index()
    df.to_csv(cache_file)
    logger.info(f"✅ {symbol} ({interval}) 数据已获取并缓存到: {cache_file}")
    return df


# --- (其他函数和类保持不变) ---
def feature_engineering(df):
    df_feat = df.copy()
    df_feat.ta.rsi(length=14, append=True)
    df_feat.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_feat.ta.bbands(length=20, std=2, append=True)
    df_feat["volatility"] = (
        np.log(df_feat["Close"] / df_feat["Close"].shift(1)).rolling(window=20).std()
    )
    df_feat["volume_change_rate"] = df_feat["Volume"].pct_change()
    df_feat.dropna(inplace=True)
    return df_feat


def create_multivariate_sequences_for_prediction(data, look_back=60):
    X = []
    [X.append(data[i : (i + look_back), :]) for i in range(len(data) - look_back + 1)]
    return np.array(X)


def generate_signals(data, model, scaler, look_back):
    logger.info("开始批量生成预测信号...")
    featured_df = feature_engineering(data.copy())
    expected_features = scaler.feature_names_in_
    if not all(feat in featured_df.columns for feat in expected_features):
        missing = set(expected_features) - set(featured_df.columns)
        raise ValueError(f"预测数据中缺少必要的特征: {missing}")
    featured_df_ordered = featured_df[expected_features]
    scaled_features = scaler.transform(featured_df_ordered)
    X_pred = create_multivariate_sequences_for_prediction(scaled_features, look_back)
    pred_probs = model.predict(X_pred, verbose=1).flatten()
    signal_index = featured_df_ordered.index[look_back - 1 :]
    signals_df = pd.DataFrame(index=signal_index)
    signals_df["Probability"] = pred_probs
    logger.info(f"信号生成完毕，共 {len(signals_df)} 条。")
    return signals_df


class ContrarianSignalStrategy(Strategy):
    buy_threshold = 0.5353
    stop_loss_pct = 0.05
    take_profit_pct = 0.10
    signals = None

    def init(self):
        if self.signals is None:
            raise ValueError("Signals data not provided.")
            self.data.df["Signal"] = self.signals["Signal"]
            self.signal = self.I(lambda x: x, self.data.df["Signal"].fillna(0))

    def next(self):
        if not self.position and self.signal[-1] == 1:
            sl = self.data.Close[-1] * (1 + self.stop_loss_pct)
            tp = self.data.Close[-1] * (1 - self.take_profit_pct)
            self.sell(sl=sl, tp=tp)


# --- 主执行程序 ---
if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    backtest_start_date_str = "2024-01-01"
    backtest_end_date_str = "2024-05-01"
    model_path = "models/btc_lstm_model.keras"
    scaler_path = "models/btc_data_scaler.joblib"
    look_back = 60

    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        logger.info("✅ 模型和Scaler已成功加载。")
    except Exception as e:
        logger.error(f"加载模型或Scaler失败: {e}")
        exit()

    full_data = fetch_binance_klines(symbol, interval, "2017-01-01")
    if full_data.empty:
        exit()
    backtest_start_dt = pd.to_datetime(backtest_start_date_str, utc=True)
    backtest_end_dt = pd.to_datetime(backtest_end_date_str, utc=True)
    backtest_data = full_data.loc[backtest_start_dt:backtest_end_dt]
    pred_start_dt = backtest_start_dt - pd.DateOffset(days=look_back + 50)
    data_for_prediction = full_data.loc[pred_start_dt:backtest_end_dt]
    daily_probs = generate_signals(data_for_prediction, model, scaler, look_back)
    daily_probs["Signal"] = (
        daily_probs["Probability"] > ContrarianSignalStrategy.buy_threshold
    ).astype(int)
    logger.info(f"在回测期间内生成的信号统计: \n{daily_probs['Signal'].value_counts()}")
    bt = FractionalBacktest(
        backtest_data, ContrarianSignalStrategy, cash=100000, commission=0.001
    )
    stats = bt.run(signals=daily_probs)
    print("\n--- Contrarian LSTM Strategy (Short on 'Buy' Signal) ---")
    print(stats)
    print("\n--- Trades ---")
    print(stats["_trades"])
