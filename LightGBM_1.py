import numpy as np
import pandas as pd
import requests
import time
import logging
import pandas_ta as ta
import os
import joblib
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
)
from tqdm import tqdm
from scipy.signal import find_peaks

# --- 0. 设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 🚀 全局配置 ---
SYMBOL = "ETHUSDT"
INTERVAL = "4h"
DATA_START_DATE = "2017-01-01"
TRAIN_START = "2018-01-01"
VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = "2025-11-06"
LOOK_BACK = 60
TREND_CONFIG = {"look_forward_steps": 5, "ema_length": 8}

# --- 文件路径 ---
MODELS_DIR, DATA_DIR = "models", "data"
MODEL_SAVE_PATH = os.path.join(
    MODELS_DIR, f"eth_model_high_precision_v3_{INTERVAL}.joblib"
)
SCALER_SAVE_PATH = os.path.join(
    MODELS_DIR, f"eth_scaler_high_precision_v3_{INTERVAL}.joblib"
)
FEATURE_COLUMNS_PATH = os.path.join(
    MODELS_DIR, f"feature_columns_high_precision_v3_{INTERVAL}.joblib"
)
FLATTENED_COLUMNS_PATH = os.path.join(
    MODELS_DIR, f"flattened_columns_high_precision_v3_{INTERVAL}.joblib"
)
DATA_CACHE_PATH = os.path.join(DATA_DIR, f"{SYMBOL.lower()}_{INTERVAL}_data.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --- 数据获取函数 ---
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
    all_d, retries, max_retries = [], 0, 5
    while sts < ets:
        try:
            r = requests.get(
                url,
                params={
                    "symbol": s.upper(),
                    "interval": i,
                    "startTime": sts,
                    "endTime": ets,
                    "limit": l,
                },
                timeout=15,
            )
            r.raise_for_status()
            d = r.json()
            if not d:
                break
            all_d.extend(d)
            sts = d[-1][0] + 1
            retries = 0
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"获取数据失败超过最大重试次数: {e}")
                return pd.DataFrame()
            logger.warning(
                f"获取数据失败，正在重试 ({retries}/{max_retries})... Error: {e}"
            )
            time.sleep(retries * 2)
            continue
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


# --- 🚀 1. 升级版特征工程 ---
def get_market_structure_features(df, order=5):
    df = df.copy()
    high_peaks_idx, _ = find_peaks(
        df["High"], distance=order, prominence=df["High"].std() * 0.5
    )
    low_peaks_idx, _ = find_peaks(
        -df["Low"], distance=order, prominence=df["Low"].std() * 0.5
    )

    df["swing_high_price"] = np.nan
    df.iloc[high_peaks_idx, df.columns.get_loc("swing_high_price")] = df.iloc[
        high_peaks_idx
    ]["High"]
    df["swing_low_price"] = np.nan
    df.iloc[low_peaks_idx, df.columns.get_loc("swing_low_price")] = df.iloc[
        low_peaks_idx
    ]["Low"]

    df["swing_high_price"] = df["swing_high_price"].ffill()
    df["swing_low_price"] = df["swing_low_price"].ffill()

    df["is_uptrend"] = (
        (df["swing_high_price"] > df["swing_high_price"].shift(1))
        & (df["swing_low_price"] > df["swing_low_price"].shift(1))
    ).astype(int)
    df["is_downtrend"] = (
        (df["swing_high_price"] < df["swing_high_price"].shift(1))
        & (df["swing_low_price"] < df["swing_low_price"].shift(1))
    ).astype(int)
    df["market_structure"] = df["is_uptrend"] - df["is_downtrend"]

    return df[["market_structure"]]


def feature_engineering(df):
    """整合所有特征计算"""
    # 修复 SettingWithCopyWarning：显式创建副本
    df = df.copy()

    logger.info("--- 开始计算特征 (V3) ---")

    # 1. 基础技术指标
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.adx(length=14, append=True)

    # 2. 额外特征
    df["volatility"] = (
        (np.log(df["Close"] / df["Close"].shift(1))).rolling(window=20).std()
    )

    # 3. 🚀 新增：市场结构特征
    market_structure_df = get_market_structure_features(df)

    # 4. 🚀 新增：长周期MACD作为过滤器（特征也加入模型）
    df.ta.macd(
        fast=24,
        slow=52,
        signal=18,
        append=True,
        col_names=("MACD_long", "MACDh_long", "MACDs_long"),
    )

    all_features_df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])
    all_features_df = pd.concat([all_features_df, market_structure_df], axis=1)

    all_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return all_features_df


# --- 标签与序列函数 ---
def create_trend_labels(df, look_forward_steps=5, ema_length=8):
    df_copy = df.copy()
    ema_col = f"EMA_{ema_length}"
    df_copy.ta.ema(length=ema_length, close=df_copy["Close"], append=True)
    future_ema = df_copy[ema_col].shift(-look_forward_steps)
    df_copy["label"] = (future_ema > df_copy[ema_col]).astype(int)
    return df_copy


def create_flattened_sequences(data, labels, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : (i + look_back), :].flatten())
        y.append(labels[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# --- 🚀 3. 升级版训练与验证函数 ---
def train_and_validate(train_df, validation_df, look_back, trend_config):
    logger.info("--- 开始训练和验证流程 (V3) ---")

    X_train_full_df = feature_engineering(train_df).dropna()
    X_validation_full_df = feature_engineering(validation_df).dropna()

    train_labeled = create_trend_labels(train_df, **trend_config)
    validation_labeled = create_trend_labels(validation_df, **trend_config)

    # --- 修复 ValueError：在 align 调用中添加 axis=0 ---
    y_train_full = train_labeled["label"].align(X_train_full_df, join="inner", axis=0)[
        0
    ]
    X_train_full_df = X_train_full_df.align(
        train_labeled["label"], join="inner", axis=0
    )[0]

    y_validation_full = validation_labeled["label"].align(
        X_validation_full_df, join="inner", axis=0
    )[0]
    X_validation_full_df = X_validation_full_df.align(
        validation_labeled["label"], join="inner", axis=0
    )[0]

    X_validation_full_df = X_validation_full_df[X_train_full_df.columns]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_full_df)
    X_validation_scaled = scaler.transform(X_validation_full_df)

    original_columns = X_train_full_df.columns
    flattened_columns = [
        f"{col}_lag_{lag}"
        for lag in range(look_back - 1, -1, -1)
        for col in original_columns
    ]
    joblib.dump(original_columns, FEATURE_COLUMNS_PATH)
    joblib.dump(flattened_columns, FLATTENED_COLUMNS_PATH)

    X_train_np, y_train = create_flattened_sequences(
        X_train_scaled, y_train_full.values, look_back
    )
    X_validation_np, y_validation = create_flattened_sequences(
        X_validation_scaled, y_validation_full.values, look_back
    )

    X_train_df = pd.DataFrame(X_train_np, columns=flattened_columns)
    X_validation_df = pd.DataFrame(X_validation_np, columns=flattened_columns)

    logger.info(f"训练样本: {len(X_train_df)}, 验证样本: {len(X_validation_df)}")

    train_label_counts = np.bincount(y_train)
    if train_label_counts.size < 2 or train_label_counts[1] == 0:
        logger.error("训练数据中没有正样本(label=1)，无法继续。")
        return None, None, None

    precision_focus_ratio = 0.7
    scale_pos_weight = (
        train_label_counts[0] / train_label_counts[1]
    ) * precision_focus_ratio
    logger.info(f"调整后的 scale_pos_weight (追求高胜率): {scale_pos_weight:.2f}")

    lgb_params = {
        "objective": "binary",
        "metric": "logloss",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 20,
        "max_depth": 5,
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,
        "colsample_bytree": 0.7,
        "subsample": 0.7,
        "reg_alpha": 0.1,
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)

    logger.info("\n开始训练 LightGBM 模型 (V3)...")
    lgb_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_validation_df, y_validation)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    y_val_pred_probs = lgb_model.predict_proba(X_validation_df)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(
        y_validation, y_val_pred_probs
    )
    f1_scores = np.divide(
        2 * recalls * precisions,
        recalls + precisions,
        out=np.zeros_like(recalls),
        where=(recalls + precisions) != 0,
    )
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx] if len(thresholds) > 0 else 0.5
    logger.info(f"在验证集上找到的最佳F1阈值: {best_f1_threshold:.4f}")

    joblib.dump(lgb_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"模型已保存到: {MODEL_SAVE_PATH}")

    return lgb_model, scaler, best_f1_threshold


# --- 🚀 4. 升级版回测评估函数 ---
def run_backtest_and_evaluate(
    test_df, model, scaler, look_back, threshold, trend_config
):
    logger.info(
        "\n" + "=" * 60 + "\n--- 开始在测试集上进行严格的回测评估 (V3) ---\n" + "=" * 60
    )

    original_columns = joblib.load(FEATURE_COLUMNS_PATH)
    flattened_columns = joblib.load(FLATTENED_COLUMNS_PATH)

    test_features_df = feature_engineering(test_df).dropna()

    test_features_aligned = test_features_df.reindex(
        columns=original_columns, fill_value=0
    )
    test_scaled = scaler.transform(test_features_aligned)

    final_signals = []

    logger.info("逐根K线遍历测试集进行预测和过滤...")
    for i in tqdm(range(look_back, len(test_scaled))):
        current_timestamp = test_features_df.index[i]

        input_sequence = test_scaled[i - look_back : i, :]
        input_flattened_np = input_sequence.flatten().reshape(1, -1)
        input_df = pd.DataFrame(input_flattened_np, columns=flattened_columns)
        pred_prob = model.predict_proba(input_df)[0][1]
        model_signal = 1 if pred_prob > threshold else 0

        macd_long = test_features_df.loc[current_timestamp, "MACD_long"]
        macds_long = test_features_df.loc[current_timestamp, "MACDs_long"]
        is_trend_confirmed = (macd_long > macds_long) and (macd_long > 0)

        final_signal = 1 if (model_signal == 1 and is_trend_confirmed) else 0
        final_signals.append(final_signal)

    actual_labels_df = create_trend_labels(test_df, **trend_config).dropna()

    pred_index = test_features_df.index[look_back : look_back + len(final_signals)]
    pred_series = pd.Series(final_signals, index=pred_index)
    results_df = pd.DataFrame(actual_labels_df["label"]).join(
        pred_series.to_frame("final_signal"), how="inner"
    )

    if results_df.empty or np.sum(results_df["final_signal"]) == 0:
        logger.warning("回测期间没有产生任何交易信号，无法计算胜率。")
        return

    y_test_actual = results_df["label"].values
    y_pred_final = results_df["final_signal"].values

    print("\n--- [客观] 测试集回测评估结果 (应用MACD过滤器后) ---")
    print(f"总回测K线数: {len(y_pred_final)}")
    print(f"发出看涨信号总次数 (交易频率): {np.sum(y_pred_final)}")
    print(f"精确率 (胜率): {precision_score(y_test_actual, y_pred_final):.4f}")
    print(f"召回率 (盈利机会捕捉率): {recall_score(y_test_actual, y_pred_final):.4f}")
    print("\n混淆矩阵 (TN, FP / FN, TP):")
    print(confusion_matrix(y_test_actual, y_pred_final))


# --- 主流程 ---
if __name__ == "__main__":
    if os.path.exists(DATA_CACHE_PATH):
        logger.info(f"从缓存加载数据: {DATA_CACHE_PATH}")
        raw_df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        raw_df = fetch_binance_klines(
            s=SYMBOL, i=INTERVAL, st=DATA_START_DATE, en=TEST_END
        )
        if not raw_df.empty:
            raw_df.to_csv(DATA_CACHE_PATH)

    if raw_df.empty:
        logger.error("数据为空，程序退出。")
        exit()

    # --- [已修正] 严格的数据集划分 ---
    train_df = raw_df[(raw_df.index >= TRAIN_START) & (raw_df.index < VALIDATION_START)]
    validation_df = raw_df[
        (raw_df.index >= VALIDATION_START) & (raw_df.index < TEST_START)
    ]
    test_df = raw_df[(raw_df.index >= TEST_START) & (raw_df.index <= TEST_END)]

    logger.info(
        f"数据集划分完成: {len(train_df)} 训练, {len(validation_df)} 验证, {len(test_df)} 测试。"
    )

    trained_model, trained_scaler, best_threshold = train_and_validate(
        train_df, validation_df, LOOK_BACK, TREND_CONFIG
    )

    if trained_model and best_threshold is not None:
        run_backtest_and_evaluate(
            test_df,
            trained_model,
            trained_scaler,
            LOOK_BACK,
            best_threshold,
            TREND_CONFIG,
        )
    else:
        logger.error("模型训练失败，跳过回测。")
