import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import logging
import pandas_ta as ta
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib

# --- 0. 设置与数据获取 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_binance_klines(s, i, st, en=None, l=1000):
    url, cols = "https://api.binance.com/api/v3/klines", [
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
    sts, ets = int(pd.to_datetime(st).timestamp() * 1000), (
        int(pd.to_datetime(en).timestamp() * 1000) if en else int(time.time() * 1000)
    )
    all_d, retries, last_e = [], 5, None
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
            except requests.exceptions.RequestException as e:
                last_e = e
                logger.warning(f"请求失败，正在重试... ({attempt + 1}/{retries})")
                time.sleep(2**attempt)
        else:
            logger.error(f"获取 {s} 失败: {last_e}")
            return pd.DataFrame()
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d, columns=cols)[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


# --- 1. 特征工程与标签定义 ---
def feature_engineering(df):
    logger.info("--- 开始计算特征指标 ---")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df["volatility"] = (
        np.log(df["Close"] / df["Close"].shift(1)).rolling(window=20).std()
    )
    df["volume_change_rate"] = df["Volume"].pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logger.info(f"特征工程完成，剩余数据: {len(df)} 条")
    return df


def create_advanced_labels(df, look_forward_steps=5, price_rise_pct=0.01):
    future_max_high = (
        df["High"].shift(-look_forward_steps).rolling(window=look_forward_steps).max()
    )
    target_price = df["Close"] * (1 + price_rise_pct)
    df["label"] = (future_max_high > target_price).astype(int)
    df.dropna(inplace=True)
    return df


# --- 2. 数据准备 ---
def create_multivariate_sequences(data, labels, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : (i + look_back), :])
        y.append(labels[i + look_back])
    return np.array(X), np.array(y)


# --- 3. 核心训练与评估函数 ---
def train_and_evaluate(train_df, test_df, look_back, model_save_path, scaler_save_path):
    logger.info("--- 开始数据准备和预处理 ---")

    train_featured = feature_engineering(train_df.copy())
    train_labeled = create_advanced_labels(train_featured)

    test_featured = feature_engineering(test_df.copy())
    test_labeled = create_advanced_labels(test_featured)

    y_train_full = train_labeled["label"].values
    X_train_full_df = train_labeled.drop(columns=["label"])

    y_test_full = test_labeled["label"].values
    X_test_df = test_labeled.drop(columns=["label"])

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_full_df)
    X_test_scaled = scaler.transform(X_test_df)

    X_train, y_train = create_multivariate_sequences(
        X_train_scaled, y_train_full, look_back
    )
    X_test, y_test = create_multivariate_sequences(
        X_test_scaled, y_test_full, look_back
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logger.error(
            "数据不足，无法创建训练或测试序列。请检查数据范围和look_back设置。"
        )
        return None, None

    num_features = X_train.shape[2]

    logger.info(
        f"数据准备完成。特征数量: {num_features}, 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}"
    )

    train_label_counts = np.bincount(y_train)
    if len(train_label_counts) < 2 or 0 in train_label_counts:
        logger.error("训练数据中只有一个类别，无法继续。")
        return None, None
    logger.info(
        f"训练集标签分布: 0 = {train_label_counts[0]}, 1 = {train_label_counts[1]}"
    )
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    L2_REG = 0.0001
    model = Sequential(
        [
            Input(shape=(look_back, num_features)),
            Bidirectional(
                LSTM(48, return_sequences=True, kernel_regularizer=l2(L2_REG))
            ),
            Dropout(0.3),
            Bidirectional(LSTM(48, kernel_regularizer=l2(L2_REG))),
            Dropout(0.3),
            # --- 下面这行是修正过的，移除了行尾多余的括号 ---
            Dense(24, activation="relu", kernel_regularizer=l2(L2_REG)),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    logger.info("\n开始训练模型...")
    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        class_weight=class_weight_dict,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1,
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"模型已保存到: {model_save_path}")
    logger.info(f"Scaler已保存到: {scaler_save_path}")

    logger.info("\n--- 开始在测试集上评估模型 ---")
    y_pred_probs = model.predict(X_test).flatten()

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)
    f1_scores_numerator = 2 * recalls * precisions
    f1_scores_denominator = recalls + precisions
    f1_scores = np.divide(
        f1_scores_numerator,
        f1_scores_denominator,
        out=np.zeros_like(f1_scores_numerator),
        where=f1_scores_denominator != 0,
    )

    best_threshold_f1 = thresholds[np.argmax(f1_scores[:-1])]
    y_pred_labels_best_f1 = (y_pred_probs > best_threshold_f1).astype(int)

    print("\n--- 测试集评估结果 (使用最佳F1阈值) ---")
    print(f"找到的最佳F1阈值: {best_threshold_f1:.4f}")
    print(f"准确率: {accuracy_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"精确率: {precision_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"F1 分数: {f1_score(y_test, y_pred_labels_best_f1):.4f}")

    # plt.figure(figsize=(8, 6))
    # plt.plot(recalls, precisions, marker=".", label="BiLSTM-Model")
    # best_f1_idx = np.argmax(f1_scores[:-1])
    # plt.scatter(
    #     recalls[best_f1_idx],
    #     precisions[best_f1_idx],
    #     marker="o",
    #     color="red",
    #     label=f"Best F1 Threshold ({best_threshold_f1:.2f})",
    # )
    # plt.xlabel("召回率 (Recall)")
    # plt.ylabel("精确率 (Precision)")
    # plt.title("精确率-召回率曲线 (Precision-Recall Curve)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return model, scaler


# --- 主流程 ---
if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    INTERVAL = "1h"
    DATA_START_DATE = "2017-01-01"
    TRAIN_START_DATE = "2020-01-01"
    TRAIN_END_DATE = "2025-01-01"
    TEST_START_DATE = "2025-01-15"
    TEST_END_DATE = "2025-10-23"
    LOOK_BACK = 60
    MODEL_SAVE_PATH = "models/btc_lstm_model_v2.keras"
    SCALER_SAVE_PATH = "models/btc_data_scaler_v2.joblib"

    raw_df = fetch_binance_klines(
        s=SYMBOL, i=INTERVAL, st=DATA_START_DATE, en=TEST_END_DATE
    )
    if raw_df.empty:
        exit()

    train_df = raw_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    test_df = raw_df.loc[TEST_START_DATE:TEST_END_DATE]

    logger.info(f"训练数据周期: {train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"测试数据周期: {test_df.index.min()} to {test_df.index.max()}")

    trained_model, trained_scaler = train_and_evaluate(
        train_df, test_df, LOOK_BACK, MODEL_SAVE_PATH, SCALER_SAVE_PATH
    )

    if trained_model and trained_scaler:
        logger.info("\n" + "=" * 60)
        logger.info("--- 加载已保存的模型进行预测演示 ---")
        logger.info("=" * 60)

        try:
            loaded_model = load_model(MODEL_SAVE_PATH)
            loaded_scaler = joblib.load(SCALER_SAVE_PATH)
            logger.info("模型和Scaler加载成功。")

            prediction_start_date = pd.to_datetime(TEST_END_DATE) - pd.DateOffset(
                hours=LOOK_BACK + 100
            )
            latest_data = raw_df.loc[prediction_start_date:TEST_END_DATE]

            latest_featured = feature_engineering(latest_data.copy())

            scaler_features = joblib.load(SCALER_SAVE_PATH).feature_names_in_
            latest_featured_aligned = latest_featured[scaler_features]
            latest_scaled = loaded_scaler.transform(latest_featured_aligned)

            last_sequence = latest_scaled[-LOOK_BACK:]
            last_sequence = np.expand_dims(last_sequence, axis=0)

            prediction_prob = loaded_model.predict(last_sequence)[0][0]

            logger.info(f"对最新的数据序列进行预测...")
            logger.info(f"模型输出的上涨概率: {prediction_prob:.4f}")

            best_f1_threshold = 0.5
            logger.warning(f"演示预测使用默认阈值 0.5。实际应使用计算出的最佳阈值。")
            if prediction_prob > best_f1_threshold:
                logger.info(f"预测结果: 看涨 (概率 > {best_f1_threshold})")
            else:
                logger.info(f"预测结果: 不看涨 (概率 <= {best_f1_threshold})")

        except Exception as e:
            logger.error(f"加载模型或进行预测时出错: {e}")
    else:
        logger.info("模型训练失败，跳过预测演示。")
