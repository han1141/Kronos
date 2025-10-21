# trains/train.py
import logging
import warnings
import numpy as np  # V61.2 Hotfix: 增加缺少的導入
import pandas as pd
import joblib
import ta
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
import time
from datetime import datetime, timedelta
import requests

# 日志配置
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- 配置 ---
TRAIN_CONFIG = {
    "symbol": "ETHUSDT",
    "interval": "5m",
    "train_start_date": "2018-01-01",
    "train_end_date": "2022-12-31",
    "validation_start_date": "2023-01-01",
    "validation_end_date": "2023-12-31",
    "model_dir": "models_v61_trained",
    "data_cache": "data_cache_v61",
}

MODEL_PARAMS = {
    "label_look_forward_bars": 36,
    "label_risk_reward_ratio": 2.0,
    "label_sl_atr_multiplier": 1.5,
    "label_atr_period": 14,
}

# --- 辅助函数 ---


def ensure_dirs():
    os.makedirs(TRAIN_CONFIG["data_cache"], exist_ok=True)
    os.makedirs(TRAIN_CONFIG["model_dir"], exist_ok=True)


def fetch_binance_klines(
    symbol, interval, start_str, end_str=None, limit=1000, cache_dir=None
):
    cache_dir = cache_dir or TRAIN_CONFIG["data_cache"]
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}_full.csv")
    start_dt, end_dt = pd.to_datetime(start_str, utc=True), (
        pd.to_datetime(end_str, utc=True)
        if end_str
        else datetime.utcnow().astimezone(pd.Timestamp.utcnow().tz)
    )
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
            if not df.empty and df.index[0] <= start_dt and df.index[-1] >= end_dt:
                logger.info(f"✅ 從有效缓存載入 {symbol} 資料")
                return df.loc[start_dt:end_dt]
        except Exception as e:
            logger.warning(f"讀取缓存檔案失敗: {e}")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    while current_start_ts < end_ts:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start_ts,
            "endTime": end_ts,
            "limit": limit,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current_start_ts = data[-1][0] + 1
            logger.info(f"已獲取數據至 {pd.to_datetime(current_start_ts, unit='ms')}")
        except Exception as e:
            logger.warning(f"獲取資料失敗: {e}. 等待 3s 重試...")
            time.sleep(3)
    if not all_data:
        logger.error(f"未能獲取任何資料。")
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
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(cache_file)
    logger.info(f"✅ 資料已獲取並缓存")
    return df.loc[start_dt:end_dt]


def add_all_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    for p in [12, 26, 50, 100, 200]:
        df[f"ema_{p}"] = ta.trend.ema_indicator(df.Close, window=p)
    for p1, p2 in [(12, 26), (26, 50), (50, 200)]:
        df[f"ema_ratio_{p1}_{p2}"] = df[f"ema_{p1}"] / df[f"ema_{p2}"]
    adx = ta.trend.ADXIndicator(df.High, df.Low, df.Close, window=14)
    df["adx"], df["adx_pos"], df["adx_neg"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    for p in [7, 14, 21]:
        df[f"rsi_{p}"] = ta.momentum.rsi(df.Close, window=p)
    macd = ta.trend.MACD(df.Close)
    df["macd"], df["macd_signal"] = macd.macd(), macd.macd_signal()
    df["atr_pct"] = (
        ta.volatility.average_true_range(df.High, df.Low, df.Close, window=14)
        / df.Close
    )
    bb = ta.volatility.BollingerBands(df.Close)
    df["bb_width"] = bb.bollinger_wband()
    df["vol_ma_ratio"] = df.Volume / df.Volume.rolling(50).mean()
    df["obv"] = ta.volume.on_balance_volume(df.Close, df.Volume)
    df.drop(columns=[f"ema_{p}" for p in [12, 26, 50, 100, 200]], inplace=True)
    return df.dropna()


def make_multiclass_labels(df: pd.DataFrame) -> pd.Series:
    p = MODEL_PARAMS
    atr = ta.volatility.average_true_range(
        df.High, df.Low, df.Close, window=p["label_atr_period"]
    )
    labels = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(df) - p["label_look_forward_bars"]):
        entry_price = df.Close.iloc[i]
        sl_amount = atr.iloc[i] * p["label_sl_atr_multiplier"]
        if not np.isfinite(sl_amount) or sl_amount == 0:
            continue
        tp_long, sl_long = (
            entry_price + sl_amount * p["label_risk_reward_ratio"],
            entry_price - sl_amount,
        )
        path = df.iloc[i + 1 : i + 1 + p["label_look_forward_bars"]]
        long_tp_hit, long_sl_hit = (
            path[path.High >= tp_long].index.min(),
            path[path.Low <= sl_long].index.min(),
        )
        tp_short, sl_short = (
            entry_price - sl_amount * p["label_risk_reward_ratio"],
            entry_price + sl_amount,
        )
        short_tp_hit, short_sl_hit = (
            path[path.Low <= tp_short].index.min(),
            path[path.High >= sl_short].index.min(),
        )
        first_long_hit = (
            min(long_tp_hit, long_sl_hit)
            if pd.notna(long_tp_hit) or pd.notna(long_sl_hit)
            else pd.NaT
        )
        first_short_hit = (
            min(short_tp_hit, short_sl_hit)
            if pd.notna(short_tp_hit) or pd.notna(short_sl_hit)
            else pd.NaT
        )
        if pd.notna(first_long_hit) and (
            pd.isna(first_short_hit) or first_long_hit <= first_short_hit
        ):
            if first_long_hit == long_tp_hit:
                labels.iloc[i] = 1
        elif pd.notna(first_short_hit):
            if first_short_hit == short_tp_hit:
                labels.iloc[i] = 2
    return labels


# --- 训练主流程 ---
def main_training_pipeline():
    ensure_dirs()

    logger.info("正在加載全部歷史數據...")
    raw_data = fetch_binance_klines(
        TRAIN_CONFIG["symbol"],
        TRAIN_CONFIG["interval"],
        TRAIN_CONFIG["train_start_date"],
        TRAIN_CONFIG["validation_end_date"],
        cache_dir=TRAIN_CONFIG["data_cache"],
    )
    if raw_data.empty:
        logger.error("數據加載失敗，退出。")
        return

    logger.info("正在進行特徵工程...")
    featured_data = add_all_ml_features(raw_data)
    logger.info("正在生成標籤...")
    labels = make_multiclass_labels(featured_data)

    full_data = featured_data.join(labels.rename("target")).dropna()
    train_data = full_data.loc[
        TRAIN_CONFIG["train_start_date"] : TRAIN_CONFIG["train_end_date"]
    ]
    validation_data = full_data.loc[
        TRAIN_CONFIG["validation_start_date"] : TRAIN_CONFIG["validation_end_date"]
    ]

    X_train, y_train = train_data.drop("target", axis=1), train_data["target"]
    X_val, y_val = validation_data.drop("target", axis=1), validation_data["target"]

    logger.info(f"數據分割完成。訓練集: {len(X_train)} | 驗證集: {len(X_val)}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logger.info("開始最終模型訓練...")

    X_full_train = pd.concat([X_train, X_val])
    y_full_train = pd.concat([y_train, y_val])
    scaler_final = RobustScaler()
    X_full_train_scaled = scaler_final.fit_transform(X_full_train)

    final_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )
    final_model.fit(X_full_train_scaled, y_full_train)

    logger.info("在驗證集上評估最終模型:")
    y_pred_val = final_model.predict(X_val_scaled)
    print(
        classification_report(y_val, y_pred_val, target_names=["Hold", "Long", "Short"])
    )

    model_path = os.path.join(
        TRAIN_CONFIG["model_dir"], f"final_model_{TRAIN_CONFIG['symbol']}.joblib"
    )
    scaler_path = os.path.join(
        TRAIN_CONFIG["model_dir"], f"final_scaler_{TRAIN_CONFIG['symbol']}.joblib"
    )
    features_path = os.path.join(TRAIN_CONFIG["model_dir"], "feature_list.txt")

    joblib.dump(final_model, model_path)
    joblib.dump(scaler_final, scaler_path)
    with open(features_path, "w") as f:
        for feature in X_train.columns:
            f.write(f"{feature}\n")

    logger.info(f"✅ 最終模型、Scaler 及特徵列表已保存至 {TRAIN_CONFIG['model_dir']}")


if __name__ == "__main__":
    main_training_pipeline()
