# ==============================================================================
# LSTM 趋势预测系统 - Final Version (v4.6)
# 版本: v4.6 (2025-11-03) - 修复加载错误版
# 作者: Gemini
#
# 核心修正 (v4.6):
# 1. [加载逻辑修正] 在 `run_prediction_demo` 中，显式地重新创建在训练时使用的
#    特定分位数损失函数实例，并将其名称映射到函数对象，传递给 `load_model` 的
#    `custom_objects` 参数。这解决了因 Keras 无法找到动态命名的损失函数
#    `quantile_loss_95` 而导致的加载失败问题。
#
# 核心框架 (基于 v4.5):
# 1. [最终加载修复] 使用 Keras 官方推荐的 `@register_keras_serializable` 装饰器
#    来注册自定义损失函数，并通过 `custom_objects` 传递，彻底解决模型加载问题。
# 2. [代码定稿] 固化 v4.4 已验证的成功策略框架，包括分位数回归、ATR 特征、
#    以及专业的策略回报分析模块，形成一个稳定、可复现的研究基准。
# ==============================================================================

import numpy as np
import pandas as pd
import requests
import time
import logging
import pandas_ta as ta
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Input,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
import joblib
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# ============================== 配置与日志 ==============================
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------- 核心参数 -------------------
SYMBOL = "ETHUSDT"
INTERVAL = "15m"
DATA_FETCH_START, DATA_FETCH_END = "2018-01-01", "2025-11-02"
TRAIN_START, TRAIN_END = "2020-01-01", "2023-12-31"
VAL_START, VAL_END = "2024-01-01", "2024-12-31"
TEST_START, TEST_END = "2025-01-01", "2025-11-02"
LOOK_BACK = 72

TREND_CONFIG = {"look_forward_steps": 1, "min_return_threshold": 0.001}

REGRESSION_STRATEGY_CONFIG = {"candidate_quantiles": [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]}

# 用于夏普比率计算的无风险利率 (年化), 假设为0
RISK_FREE_RATE = 0.0
# 一年中的交易周期数 (15分钟K线: 365天 * 24小时 * 4个15分钟)
PERIODS_PER_YEAR = 365 * 24 * 4

MODEL_CONFIG = {
    "lstm_units_1": 48,
    "lstm_units_2": 24,
    "dropout_1": 0.4,
    "dropout_2": 0.4,
    "dense_units": 32,
    "dropout_3": 0.5,
    "l2_reg": 1e-4,
    "learning_rate": 3e-4,
}
TRAINING_CONFIG = {
    "batch_size": 256,
    "epochs": 25,
    "reduce_lr_patience": 3,
    "early_stopping_patience": 5,
}

# 路径配置
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
BASE_PATH = f"artifacts/final_v4.6_{TIMESTAMP}"
MODEL_SAVE_PATH = os.path.join(BASE_PATH, f"eth_final_model_{INTERVAL}.keras")
SCALER_SAVE_PATH = os.path.join(BASE_PATH, f"eth_final_scaler_{INTERVAL}.joblib")
FEATURE_COLS_PATH = os.path.join(BASE_PATH, f"feature_columns_final_{INTERVAL}.joblib")
THRESHOLD_PATH = os.path.join(
    BASE_PATH, f"best_quantile_threshold_value_{INTERVAL}.txt"
)
LOG_DIR = os.path.join("logs", f"final_v4.6_{TIMESTAMP}")
DATA_CACHE_PATH = f"data/{SYMBOL.lower()}_{INTERVAL}_data.csv"

os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
            except Exception as e:
                logger.warning(f"请求失败，重试 {attempt+1}/{retries}: {e}")
                time.sleep(2**attempt)
                continue
            if not d:
                sts = ets
                break
            all_d.extend(d)
            sts = d[-1][0] + 1
            break
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
    # 添加ATR作为波动率特征
    df.ta.atr(length=14, append=True)

    df.ta.rsi(length=14, append=True)
    df = pd.concat([df, df.ta.macd(fast=12, slow=26, signal=9)], axis=1)
    bbands = df.ta.bbands(length=20, std=2)
    df = pd.concat([df, bbands[["BBL_20_2.0", "BBU_20_2.0"]]], axis=1)
    df["bb_width"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / df["Close"]
    for lag in [1, 3, 6, 12, 24]:
        df[f"return_{lag}"] = df["Close"].pct_change(lag)
    df.reset_index(inplace=True)
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df.set_index("timestamp", inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_rows = len(df)
    df.dropna(inplace=True)
    logger.info(
        f"特征工程完成，共 {len(df.columns)} 个特征。清除NaN后剩余 {len(df)}/{initial_rows} 行。"
    )
    return df


# ============================== 标签生成 ==============================
def create_regression_labels(df, **config):
    logger.info(
        f"--- 创建回归标签 (未来 {config['look_forward_steps']} 步的收益率) ---"
    )
    df = df.copy()
    future_price = df["Close"].shift(-config["look_forward_steps"])
    return_rate = (future_price - df["Close"]) / df["Close"]
    df["label"] = return_rate
    df["future_return"] = return_rate
    initial_rows = len(df)
    df.dropna(inplace=True)
    logger.info(f"回归标签创建完成，清除NaN后剩余 {len(df)}/{initial_rows} 行。")
    return df


# ============================== 序列构建 ==============================
def create_sequences(data, labels, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(labels[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ============================== 分位数损失函数 (v4.5 最终修复) ==============================
@tf.keras.utils.register_keras_serializable()
def make_quantile_loss(q):
    """
    创建一个指定分位数的损失函数，并进行注册以便模型加载。
    """

    def quantile_loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.keras.backend.mean(
            tf.keras.backend.maximum(q * e, (q - 1) * e), axis=-1
        )

    quantile_loss.__name__ = f"quantile_loss_{int(q*100)}"
    return quantile_loss


# ============================== 模型构建 ==============================
def build_quantile_model(input_shape, config):
    logger.info("--- 构建分位数回归版 Stacked Bi-LSTM 模型 ---")
    model = Sequential(
        [
            Input(shape=input_shape),
            Bidirectional(
                LSTM(
                    config["lstm_units_1"],
                    return_sequences=True,
                    kernel_regularizer=l2(config["l2_reg"]),
                )
            ),
            BatchNormalization(),
            Dropout(config["dropout_1"]),
            Bidirectional(
                LSTM(config["lstm_units_2"], kernel_regularizer=l2(config["l2_reg"]))
            ),
            BatchNormalization(),
            Dropout(config["dropout_2"]),
            Dense(config["dense_units"], kernel_regularizer=l2(config["l2_reg"])),
            BatchNormalization(),
            Activation("relu"),
            Dropout(config["dropout_3"]),
            Dense(1, activation="linear"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"], clipnorm=1.0
    )

    # 训练时使用0.95分位数，这个值是硬编码的
    quantile_loss_fn = make_quantile_loss(0.95)

    model.compile(optimizer=optimizer, loss=quantile_loss_fn, metrics=["mae"])
    model.summary()
    return model


# ============================== 训练与评估 ==============================
def train_and_evaluate(
    full_df,
    train_range,
    val_range,
    test_range,
    look_back,
    trend_config,
    model_config,
    training_config,
):
    logger.info("--- v4.6: 最终定稿版 (修复加载错误) ---")

    # 1. 数据准备
    train_df_raw = full_df.loc[train_range[0] : train_range[1]]
    val_df_raw = full_df.loc[val_range[0] : val_range[1]]
    test_df_raw = full_df.loc[test_range[0] : test_range[1]]

    train_featured = feature_engineering(train_df_raw)
    val_featured = feature_engineering(val_df_raw)
    test_featured = feature_engineering(test_df_raw)

    train_df = create_regression_labels(train_featured, **trend_config)
    val_df = create_regression_labels(val_featured, **trend_config)
    test_df = create_regression_labels(test_featured, **trend_config)

    # 2. 特征工程与标准化
    feature_cols = [
        c
        for c in train_df.columns
        if c
        not in [
            "label",
            "timestamp",
            "future_return",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]
    joblib.dump(feature_cols, FEATURE_COLS_PATH)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_val_scaled = scaler.transform(val_df[feature_cols])
    X_test_scaled = scaler.transform(test_df[feature_cols])
    joblib.dump(scaler, SCALER_SAVE_PATH)

    # 3. 序列构建
    X_train, y_train = create_sequences(
        X_train_scaled, train_df["label"].values, look_back
    )
    X_val, y_val = create_sequences(X_val_scaled, val_df["label"].values, look_back)
    X_test, y_test = create_sequences(X_test_scaled, test_df["label"].values, look_back)
    logger.info(
        f"序列构建后 -> 训练集: {len(X_train)} | 验证集: {len(X_val)} | 测试集: {len(X_test)}"
    )

    # 4. 模型训练
    model = build_quantile_model(
        input_shape=(look_back, X_train.shape[2]), config=model_config
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=training_config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=training_config["reduce_lr_patience"],
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=training_config["batch_size"],
        epochs=training_config["epochs"],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("训练结束。")
    model.save(MODEL_SAVE_PATH)
    logger.info(f"最佳模型已保存: {MODEL_SAVE_PATH}")

    # 5. 在验证集上寻找最优策略
    logger.info("--- 在验证集上寻找最佳「排序」策略 (最优分位数) ---")
    y_val_preds = model.predict(X_val).flatten()
    val_returns = val_df["future_return"].iloc[look_back:].values
    best_sharpe = -np.inf
    best_quantile = 0.99
    best_threshold_value = np.inf
    for q in REGRESSION_STRATEGY_CONFIG["candidate_quantiles"]:
        threshold_value = np.quantile(y_val_preds, q)
        trade_signals = y_val_preds > threshold_value
        if np.sum(trade_signals) < 10:
            continue
        trade_returns = val_returns[trade_signals]
        sharpe_ratio = (trade_returns.mean() / (trade_returns.std() + 1e-9)) * np.sqrt(
            PERIODS_PER_YEAR
        )
        logger.info(
            f"测试分位数 {q*100:.0f}% (阈值={threshold_value:.6f}): "
            f"交易次数={len(trade_returns)}, 平均回报={trade_returns.mean()*100:.4f}%, 年化夏普={sharpe_ratio:.4f}"
        )
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_quantile = q
            best_threshold_value = threshold_value

    logger.info(
        f"\n最佳策略已确定: 选择预测回报率排名前 {(1-best_quantile)*100:.1f}% 的机会"
    )
    logger.info(f"  => 对应验证集上的预测值阈值: {best_threshold_value:.6f}")
    logger.info(f"  => 对应验证集上的年化夏普比率: {best_sharpe:.4f}")
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(best_threshold_value))

    # 6. 在测试集上进行最终评估
    logger.info(f"\n最终测试集评估报告 (使用优化阈值: {best_threshold_value:.6f})")
    y_test_preds = model.predict(X_test).flatten()
    y_pred_signals = y_test_preds > best_threshold_value
    actual_returns = test_df["future_return"].iloc[look_back:].values

    logger.info(
        "\n"
        + "=" * 50
        + "\n   专业回报分析 (Professional Return Analysis)\n"
        + "=" * 50
    )

    trade_indices = np.where(y_pred_signals == 1)[0]

    if len(trade_indices) > 0:
        trade_returns = actual_returns[trade_indices]

        total_trades = len(trade_returns)
        avg_return = trade_returns.mean()
        std_return = trade_returns.std()
        sharpe_ratio = (
            (avg_return - (RISK_FREE_RATE / PERIODS_PER_YEAR))
            / (std_return + 1e-9)
            * np.sqrt(PERIODS_PER_YEAR)
        )
        win_rate = np.sum(trade_returns > 0) / total_trades
        winning_trades = trade_returns[trade_returns > 0]
        losing_trades = trade_returns[trade_returns <= 0]
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
        profit_loss_ratio = avg_win / (avg_loss + 1e-9)

        cumulative_returns = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

        logger.info(f"模型总共发出了 {total_trades} 个'看涨'信号。")
        logger.info(f"  => 平均每笔交易回报率: {avg_return * 100:.4f}%")
        logger.info(f"  => 年化夏普比率: {sharpe_ratio:.4f}")
        logger.info(f"  => 胜率 (Win Rate): {win_rate:.2%}")
        logger.info(f"  => 盈亏比 (Profit/Loss Ratio): {profit_loss_ratio:.2f}")
        logger.info(f"  => 平均盈利回报: {avg_win * 100:.4f}%")
        logger.info(f"  => 平均亏损回报: {avg_loss * -100:.4f}%")
        logger.info(f"  => 最大回撤 (基于收益序列): {max_drawdown:.6f}")
    else:
        logger.info("模型在测试集上没有发出任何'看涨'信号。")

    return model, scaler, history


# ============================== 预测演示 ==============================
def run_prediction_demo(raw_df, model_path, scaler_path, threshold_path, look_back):
    try:
        # [FIX START] v4.6 修正: 解决模型加载失败问题
        # Keras 保存模型时，会将损失函数的名称（例如 'quantile_loss_95'）记录在配置中。
        # 加载时，它需要一个从这个名称到实际函数对象的映射。
        # 因此，我们必须重新创建在训练期间使用的完全相同的损失函数实例，
        # 并将其名称和函数对象一起传递给 `custom_objects`。

        # 1. 重新创建在 build_quantile_model 中使用的特定损失函数 (q=0.95)
        quantile_loss_fn = make_quantile_loss(0.95)

        # 2. 将其动态生成的名称 (quantile_loss_fn.__name__) 映射到函数本身
        model = load_model(
            model_path, custom_objects={quantile_loss_fn.__name__: quantile_loss_fn}
        )
        # [FIX END]

        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        with open(threshold_path, "r") as f:
            threshold = float(f.read().strip())
        latest_data = raw_df.iloc[-(look_back + 300) :].copy()
        latest_featured = feature_engineering(latest_data)
        latest_aligned = latest_featured.reindex(columns=feature_cols, fill_value=0)
        latest_scaled = scaler.transform(latest_aligned)
        seq = latest_scaled[-look_back:].reshape(1, look_back, -1)
        predicted_return = model.predict(seq, verbose=0)[0][0]
        signal = (
            "看涨 (Buy)" if predicted_return > threshold else "观望/看跌 (Hold/Sell)"
        )

        logger.info("\n" + "=" * 25 + " 预测演示 " + "=" * 25)
        logger.info(
            f"最新数据点预测: 预测未来回报率(95%分位数) = {predicted_return:.6f} ({predicted_return*100:.4f}%)"
        )
        logger.info(f"决策阈值 (来自验证集最优分位数): {threshold:.6f}")
        logger.info(f"最终信号: {signal}")
    except Exception as e:
        logger.error(f"预测演示失败: {e}")


# ============================== 主流程 ==============================
if __name__ == "__main__":
    if os.path.exists(DATA_CACHE_PATH):
        logger.info(f"加载缓存数据: {DATA_CACHE_PATH}")
        raw_df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        logger.info("从Binance获取数据...")
        raw_df = fetch_binance_klines(
            SYMBOL, INTERVAL, DATA_FETCH_START, DATA_FETCH_END
        )
        if not raw_df.empty:
            raw_df.to_csv(DATA_CACHE_PATH)
            logger.info(f"数据已缓存: {DATA_CACHE_PATH}")

    if raw_df.empty:
        exit()

    logger.info(f"训练周期: {TRAIN_START} ~ {TRAIN_END}")
    logger.info(f"验证周期: {VAL_START} ~ {VAL_END}")
    logger.info(f"测试周期: {TEST_START} ~ {TEST_END}")

    model, scaler, history = train_and_evaluate(
        full_df=raw_df,
        train_range=(TRAIN_START, TRAIN_END),
        val_range=(VAL_START, VAL_END),
        test_range=(TEST_START, TEST_END),
        look_back=LOOK_BACK,
        trend_config=TREND_CONFIG,
        model_config=MODEL_CONFIG,
        training_config=TRAINING_CONFIG,
    )

    if model:
        run_prediction_demo(
            raw_df, MODEL_SAVE_PATH, SCALER_SAVE_PATH, THRESHOLD_PATH, LOOK_BACK
        )
