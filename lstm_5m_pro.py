# ==============================================================================
# LSTM 趋势预测系统 - 生产级完整修复版（无绘图）
# 版本: v2.1 (2025-10-31)
# 特性: 统一EMA + 3小时趋势标签 + StandardScaler + 增强模型 + 无绘图干扰
# ==============================================================================

import numpy as np
import pandas as pd
import requests
import time
import logging
import pandas_ta as ta
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import joblib
from datetime import datetime

# ============================== 配置与日志 ==============================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------- 核心参数 -------------------
SYMBOL = "ETHUSDT"
INTERVAL = "5m"
DATA_START_DATE = "2021-01-01"
TRAIN_START_DATE = "2022-01-01"
TRAIN_END_DATE = "2025-09-30"
TEST_START_DATE = "2025-10-01"
TEST_END_DATE = "2025-10-29"
LOOK_BACK = 72  # 6小时历史

# 3小时趋势 + 动态阈值
TREND_CONFIG = {
    "look_forward_steps": 36,  # 36 × 5m = 3小时
    "ema_length": 34,  # 中期趋势
    "min_return_threshold": 0.003,  # 至少 0.3%
    "atr_length": 14,
    "atr_multiplier": 2.0,
}

# 保存路径
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
MODEL_SAVE_PATH = f"models/eth_trend_model_v2_{INTERVAL}_{TIMESTAMP}.keras"
SCALER_SAVE_PATH = f"models/eth_trend_scaler_v2_{INTERVAL}_{TIMESTAMP}.joblib"
FEATURE_COLS_PATH = f"models/feature_columns_v2_{INTERVAL}_{TIMESTAMP}.joblib"
THRESHOLD_PATH = f"models/best_f1_threshold_{INTERVAL}_{TIMESTAMP}.txt"
DATA_CACHE_PATH = f"data/{SYMBOL.lower()}_{INTERVAL}_data.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ============================== 数据获取 ==============================
def fetch_binance_klines(s, i, st, en=None, l=1000):
    url = "https://api.binance.com/api/v3/klines"
    cols = [
        "timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    sts = int(pd.to_datetime(st).timestamp() * 1000)
    ets = int(pd.to_datetime(en).timestamp() * 1000) if en else int(time.time() * 1000)
    all_d, retries = [], 5
    while sts < ets:
        p = {
            "symbol": s.upper(),
            "interval": i,
            "startTime": sts,
            "endTime": ets,
            "limit": l,
        }
        for attempt in range(retries):
            try:
                r = requests.get(url, params=p, timeout=15)
                r.raise_for_status()
                d = r.json()
                if not d:
                    sts = ets
                    break
                all_d.extend(d)
                sts = d[-1][0] + 1
                break
            except Exception as e:
                logger.warning(f"请求失败，重试 {attempt+1}/{retries}: {e}")
                time.sleep(2**attempt)
        else:
            logger.error("获取数据失败")
            return pd.DataFrame()
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d, columns=cols)[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col])
    logger.info(f"获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


# ============================== 特征工程 ==============================
def feature_engineering(df):
    logger.info("--- 开始计算特征指标 ---")
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df["volatility"] = np.log(df["Close"] / df["Close"].shift(1)).rolling(20).std()
    df["volume_change_rate"] = df["Volume"].pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ============================== 标签生成（核心） ==============================
def create_trend_labels(df, **config):
    logger.info(f"--- 创建3小时趋势标签 (阈值: {config['min_return_threshold']}) ---")
    df = df.copy()
    steps = config["look_forward_steps"]
    ema_len = config["ema_length"]
    thresh = config["min_return_threshold"]
    atr_len = config["atr_length"]
    mult = config["atr_multiplier"]

    df.ta.ema(length=ema_len, append=True)
    df.ta.atr(length=atr_len, append=True)

    ema_col = f"EMA_{ema_len}"
    atr_col = f"ATRr_{atr_len}"
    future_price = df["Close"].shift(-steps)
    return_rate = (future_price - df["Close"]) / df["Close"]
    dynamic_thresh = df[atr_col] / df["Close"] * mult
    threshold = np.maximum(thresh, dynamic_thresh)

    df["label"] = np.where(
        return_rate > threshold, 1, np.where(return_rate < -threshold, 0, -1)
    )
    df = df[df["label"] != -1].copy()
    df["label"] = df["label"].astype(int)
    df.dropna(inplace=True)
    logger.info(f"标签创建完成，过滤震荡样本后剩余: {len(df)} 条")
    return df


# ============================== 序列构建 ==============================
def create_sequences(data, labels, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(labels[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ============================== 训练与评估 ==============================
def train_and_evaluate(full_df, train_range, test_range, look_back, trend_config):
    logger.info("--- 统一计算特征与标签（避免分布漂移）---")
    full_featured = feature_engineering(full_df.copy())
    full_labeled = create_trend_labels(full_featured, **trend_config)

    train_df = full_labeled.loc[train_range[0] : train_range[1]]
    test_df = full_labeled.loc[test_range[0] : test_range[1]]

    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("训练或测试数据为空！")
        return None, None, None, None

    # 特征对齐
    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train_df = train_df[feature_cols]
    X_test_df = test_df[feature_cols]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # 保存特征列
    joblib.dump(feature_cols, FEATURE_COLS_PATH)

    # 构建序列
    X_train, y_train = create_sequences(
        X_train_scaled, train_df["label"].values, look_back
    )
    X_test, y_test = create_sequences(X_test_scaled, test_df["label"].values, look_back)

    logger.info(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
    logger.info(
        f"训练标签分布: 0={np.bincount(y_train)[0]}, 1={np.bincount(y_train)[1]}"
    )

    # 类别权重
    cw = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: cw[0], 1: cw[1]}

    # 模型
    model = Sequential(
        [
            Input(shape=(look_back, X_train.shape[2])),
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))),
            Dropout(0.3),
            Bidirectional(LSTM(64, kernel_regularizer=l2(1e-4))),
            Dropout(0.3),
            Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", "Precision", "Recall"],
    )
    model.summary()

    # 回调
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
        ),
    ]

    # 训练
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # 保存
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"模型已保存: {MODEL_SAVE_PATH}")

    # 评估
    y_pred_probs = model.predict(X_test).flatten()
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)
    f1s = 2 * recalls * precisions / (recalls + precisions + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    y_pred = (y_pred_probs > best_thresh).astype(int)

    logger.info("\n" + "=" * 50)
    logger.info("测试集最终评估 (最佳F1阈值)")
    logger.info(f"阈值: {best_thresh:.4f}")
    logger.info(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"精确率: {precision_score(y_test, y_pred):.4f}")
    logger.info(f"召回率: {recall_score(y_test, y_pred):.4f}")
    logger.info(f"F1: {f1_score(y_test, y_pred):.4f}")

    # 保存阈值
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(best_thresh))

    return model, scaler, best_thresh, history


# ============================== 预测演示 ==============================
def run_prediction_demo(raw_df, model_path, scaler_path, threshold_path, look_back):
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())

        latest_data = raw_df.iloc[-(look_back + 200) :].copy()
        latest_featured = feature_engineering(latest_data)
        latest_featured.ta.ema(length=TREND_CONFIG["ema_length"], append=True)
        latest_featured.dropna(inplace=True)
        latest_aligned = latest_featured.reindex(columns=feature_cols, fill_value=0)
        latest_scaled = scaler.transform(latest_aligned)
        seq = latest_scaled[-look_back:].reshape(1, look_back, -1)

        prob = model.predict(seq, verbose=0)[0][0]
        logger.info(f"\n最新预测: 上涨概率 = {prob:.4f}")
        logger.info(f"决策阈值: {threshold:.4f}")
        logger.info(f"信号: {'看涨' if prob > threshold else '观望/看跌'}")

    except Exception as e:
        logger.error(f"预测失败: {e}")


# ============================== 主流程 ==============================
if __name__ == "__main__":
    # 1. 加载数据
    if os.path.exists(DATA_CACHE_PATH):
        logger.info(f"加载缓存数据: {DATA_CACHE_PATH}")
        raw_df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        logger.info("从Binance获取数据...")
        raw_df = fetch_binance_klines(SYMBOL, INTERVAL, DATA_START_DATE, TEST_END_DATE)
        if not raw_df.empty:
            raw_df.to_csv(DATA_CACHE_PATH)
            logger.info(f"数据已缓存: {DATA_CACHE_PATH}")

    if raw_df.empty:
        exit()

    logger.info(f"训练周期: {TRAIN_START_DATE} ~ {TRAIN_END_DATE}")
    logger.info(f"测试周期: {TEST_START_DATE} ~ {TEST_END_DATE}")

    # 2. 训练
    model, scaler, threshold, history = train_and_evaluate(
        full_df=raw_df,
        train_range=(TRAIN_START_DATE, TRAIN_END_DATE),
        test_range=(TEST_START_DATE, TEST_END_DATE),
        look_back=LOOK_BACK,
        trend_config=TREND_CONFIG,
    )

    if model:
        # 3. 预测演示
        run_prediction_demo(
            raw_df, MODEL_SAVE_PATH, SCALER_SAVE_PATH, THRESHOLD_PATH, LOOK_BACK
        )
