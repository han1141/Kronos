import numpy as np
import pandas as pd
import requests
import time
import logging

# Temporary NumPy 2.0 compatibility for pandas_ta (expects numpy.NaN/Inf)
# pandas_ta<0.4 uses `from numpy import NaN`, which NumPy 2.0 removed.
# Ensure these aliases exist before importing pandas_ta.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # alias for backward compatibility
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # alias for backward compatibility

import pandas_ta as ta
# Prefer pandas_ta's pure-Python backend; disable optional TA-Lib to avoid
# NumPy 2.x binary-ABI issues with compiled talib wheels.
try:
    if hasattr(ta, "Imports"):
        ta.Imports["talib"] = False
except Exception:
    pass
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

# --- 0. 设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 🚀 全局配置 ---
SYMBOL = "ETHUSDT"
INTERVAL = "1h"
DATA_START_DATE = "2017-01-01"
TRAIN_START = "2018-01-01"
VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = "2025-11-17"
LOOK_BACK = 60
TREND_CONFIG = {"look_forward_steps": 5, "ema_length": 8}

# --- 实验/安全选项 ---
# 使用仅基于历史窗口的“市场结构”特征（在线、无前视）；
# 若设为 False 则完全禁用该特征。
USE_TRAILING_MARKET_STRUCTURE = True
# 回测时是否额外运行一次：关闭 MACD 过滤做对照
RUN_ABLATION_NO_MACD_FILTER = True
# 是否启用基于 LightGBM 的简单特征选择，降低高维风险
ENABLE_FEATURE_SELECTION = True
# 选保留的特征重要性分位（百分比），数值越小保留越多特征
FEATURE_IMPORTANCE_KEEP_PERCENTILE = 50  # 保留前 50% 重要性
# 是否在回测中严格使用“在线（expanding）计算”，完全避免一次性预计算
STRICT_ONLINE_EVAL = True

# MACD 过滤逻辑参数（更宽松且可配置）
MACD_FILTER_MODE = "hist_pos_or_rising"  # 可选："strict", "hist_pos", "hist_pos_or_rising", "recent_cross"
MACD_RECENT_CROSS_LOOKBACK = 3  # recent_cross 模式下允许最近多少根内上穿
USE_ADX_GATE = False  # 可选是否联动 ADX
ADX_THRESHOLD = 18.0

# --- 文件路径 ---
MODELS_DIR, DATA_DIR = "models_gbm2", "data"
# Using v6 to denote the final, warning-free version
MODEL_SAVE_PATH = os.path.join(
    MODELS_DIR, f"{SYMBOL.lower()}_model_high_precision_v6_{INTERVAL}.joblib"
)
SCALER_SAVE_PATH = os.path.join(
    MODELS_DIR, f"{SYMBOL.lower()}_scaler_high_precision_v6_{INTERVAL}.joblib"
)
FEATURE_COLUMNS_PATH = os.path.join(
    MODELS_DIR, f"{SYMBOL.lower()}_feature_columns_high_precision_v6_{INTERVAL}.joblib"
)
FLATTENED_COLUMNS_PATH = os.path.join(
    MODELS_DIR,
    f"{SYMBOL.lower()}_flattened_columns_high_precision_v6_{INTERVAL}.joblib",
)
# 若启用特征选择，保存选择后的列名
SELECTED_FLATTENED_COLUMNS_PATH = os.path.join(
    MODELS_DIR,
    f"{SYMBOL.lower()}_selected_flattened_columns_high_precision_v6_{INTERVAL}.joblib",
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
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d, columns=[*cols, "c1", "c2", "c3", "c4", "c5", "c6"])[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


# --- 🚀 1. 升级版特征工程 (彻底修复前视偏差 & 警告) ---


def get_market_structure_features_trailing(df, window=5):
    """在线、仅依赖过去数据的市场结构近似。
    使用滚动高点/低点变化来度量趋势强弱，不使用 find_peaks（避免前视）。
    """
    df_copy = df.copy()
    rolling_high = df_copy["High"].rolling(window=window, min_periods=1).max()
    rolling_low = df_copy["Low"].rolling(window=window, min_periods=1).min()
    is_higher_high = rolling_high > rolling_high.shift(1)
    is_higher_low = rolling_low > rolling_low.shift(1)
    is_lower_high = rolling_high < rolling_high.shift(1)
    is_lower_low = rolling_low < rolling_low.shift(1)
    df_copy["is_uptrend"] = (is_higher_high & is_higher_low).astype(int)
    df_copy["is_downtrend"] = (is_lower_high & is_lower_low).astype(int)
    df_copy["market_structure"] = df_copy["is_uptrend"] - df_copy["is_downtrend"]
    return df_copy[["market_structure"]]


def feature_engineering(df, verbose=True):
    df = df.copy()
    if verbose:
        logger.info("--- 开始计算特征 (V6 - Final Causal & Warning-Free) ---")

    # 指标计算 + 容错：确保关键列存在（即使前期窗口不足也创建为 NaN）
    # RSI
    if len(df) >= 14:
        df.ta.rsi(length=14, append=True)
    if "RSI_14" not in df.columns:
        df["RSI_14"] = np.nan

    # MACD(12,26,9)
    try:
        if len(df) >= (26 + 9):
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
    except Exception:
        pass
    for col in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
        if col not in df.columns:
            df[col] = np.nan

    df.ta.bbands(length=20, std=2, append=True)

    df.ta.adx(length=14, append=True)
    if "ADX_14" not in df.columns:
        df["ADX_14"] = np.nan

    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df["volatility_log_ret"] = (
        (np.log(df["Close"] / df["Close"].shift(1))).rolling(window=20).std()
    )
    # 在线、无前视的市场结构（可关闭）
    if USE_TRAILING_MARKET_STRUCTURE:
        market_structure_df = get_market_structure_features_trailing(df, window=5)
    else:
        market_structure_df = pd.DataFrame(index=df.index)
    if len(df) >= (52 + 18):
        try:
            df.ta.macd(
                fast=24,
                slow=52,
                signal=18,
                append=True,
                col_names=("MACD_long", "MACDh_long", "MACDs_long"),
            )
        except Exception:
            pass
    for col in ["MACD_long", "MACDh_long", "MACDs_long"]:
        if col not in df.columns:
            df[col] = np.nan
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_4"] = df["Close"].pct_change(4)
    df["ret_16"] = df["Close"].pct_change(16)
    df["rsi_delta_1"] = df["RSI_14"].diff(1)
    if "MACDh_12_26_9" not in df.columns:
        df["MACDh_12_26_9"] = np.nan
    df["macd_delta_1"] = df["MACDh_12_26_9"].diff(1)
    all_features_df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])
    all_features_df = pd.concat([all_features_df, market_structure_df], axis=1)
    all_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if verbose:
        logger.info(f"特征计算完成，共 {all_features_df.shape[1]} 个特征。")
    return all_features_df


# --- 标签与序列函数 ---
def create_trend_labels(df, look_forward_steps=5, ema_length=8):
    df_copy = df.copy()
    df_copy.ta.ema(length=ema_length, close=df_copy["Close"], append=True)
    future_ema = df_copy[f"EMA_{ema_length}"].shift(-look_forward_steps)
    df_copy["label"] = (future_ema > df_copy[f"EMA_{ema_length}"]).astype(int)
    return df_copy


def create_flattened_sequences(data, labels, look_back=60):
    X, y = [], []
    if len(data) <= look_back:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    for i in range(len(data) - look_back):
        X.append(data[i : (i + look_back), :].flatten())
        y.append(labels[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# --- 训练与验证函数 ---
def train_and_validate(train_df, validation_df, look_back, trend_config):
    logger.info("--- 开始训练和验证流程 (V6 - Final Causal & Warning-Free) ---")
    X_train_full_df = feature_engineering(train_df).dropna()
    X_validation_full_df_raw = feature_engineering(validation_df)
    train_labeled = create_trend_labels(
        train_df.loc[X_train_full_df.index], **trend_config
    )
    y_train_full = train_labeled["label"]
    common_cols = X_train_full_df.columns
    X_validation_full_df = X_validation_full_df_raw[common_cols].dropna()
    validation_labeled = create_trend_labels(
        validation_df.loc[X_validation_full_df.index], **trend_config
    )
    y_validation_full = validation_labeled["label"]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_full_df)
    X_validation_scaled = scaler.transform(X_validation_full_df)

    original_columns = X_train_full_df.columns.tolist()
    flattened_columns = [
        f"{col}_lag_{lag}"
        for lag in range(look_back - 1, -1, -1)
        for col in original_columns
    ]
    # 持久化基底列与展开后的全部列
    joblib.dump(original_columns, FEATURE_COLUMNS_PATH)
    joblib.dump(flattened_columns, FLATTENED_COLUMNS_PATH)

    X_train_np, y_train = create_flattened_sequences(
        X_train_scaled, y_train_full.values, look_back
    )
    X_validation_np, y_validation = create_flattened_sequences(
        X_validation_scaled, y_validation_full.values, look_back
    )
    if len(X_train_np) == 0:
        logger.error("创建序列后没有足够的训练数据。")
        return None, None, None

    X_train_df_seq = pd.DataFrame(X_train_np, columns=flattened_columns)
    X_validation_df_seq = pd.DataFrame(X_validation_np, columns=flattened_columns)

    train_label_counts = np.bincount(y_train)
    if len(train_label_counts) < 2 or train_label_counts[1] == 0:
        logger.error("训练数据中没有正样本(label=1)。")
        return None, None, None

    scale_pos_weight = (train_label_counts[0] / train_label_counts[1]) * 0.7
    logger.info(f"调整后的 scale_pos_weight: {scale_pos_weight:.2f}")

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
    logger.info("\n开始训练 LightGBM 模型 (全特征)...")
    lgb_model.fit(
        X_train_df_seq,
        y_train,
        eval_set=[(X_validation_df_seq, y_validation)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    def _best_threshold(y_true, y_prob):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = np.divide(
            2 * recalls * precisions,
            recalls + precisions,
            out=np.zeros_like(recalls),
            where=(recalls + precisions) != 0,
        )
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx] if len(thresholds) > best_idx else 0.5

    y_val_pred_probs = lgb_model.predict_proba(X_validation_df_seq)[:, 1]
    best_f1_threshold = _best_threshold(y_validation, y_val_pred_probs)
    logger.info(f"在验证集上(全特征)最佳F1阈值: {best_f1_threshold:.4f}")

    selected_columns = None
    if ENABLE_FEATURE_SELECTION:
        importances = lgb_model.feature_importances_
        # 保留前百分位的特征
        thr = np.percentile(importances, FEATURE_IMPORTANCE_KEEP_PERCENTILE)
        keep_mask = importances >= thr
        # 兜底：至少保留 200 或 20%（取较小），但不少于 64
        if keep_mask.sum() < 64:
            order = np.argsort(importances)[::-1]
            min_keep = max(64, int(0.2 * len(importances)))
            min_keep = min(min_keep, 200)
            keep_mask[:] = False
            keep_mask[order[:min_keep]] = True
        selected_columns = [c for c, k in zip(flattened_columns, keep_mask) if k]
        logger.info(
            f"特征选择: 从 {len(flattened_columns)} -> {len(selected_columns)} 列"
        )

        # 使用选择后的特征重新训练并基于验证集监控
        X_train_sel = X_train_df_seq[selected_columns]
        X_valid_sel = X_validation_df_seq[selected_columns]
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        logger.info("\n开始训练 LightGBM 模型 (选择后特征)...")
        lgb_model.fit(
            X_train_sel,
            y_train,
            eval_set=[(X_valid_sel, y_validation)],
            eval_metric="logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        y_val_pred_probs = lgb_model.predict_proba(X_valid_sel)[:, 1]
        best_f1_threshold = _best_threshold(y_validation, y_val_pred_probs)
        logger.info(
            f"在验证集上(选择后特征)最佳F1阈值: {best_f1_threshold:.4f}"
        )
        # 持久化选择结果
        joblib.dump(selected_columns, SELECTED_FLATTENED_COLUMNS_PATH)
    else:
        # 清理历史选择文件（若有）以避免误用
        try:
            if os.path.exists(SELECTED_FLATTENED_COLUMNS_PATH):
                os.remove(SELECTED_FLATTENED_COLUMNS_PATH)
        except Exception:
            pass

    joblib.dump(lgb_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"模型和Scaler已保存到: {MODELS_DIR}")
    return lgb_model, scaler, best_f1_threshold


# --- 回测评估函数 ---
def run_backtest_and_evaluate(
    test_df,
    model,
    scaler,
    look_back,
    threshold,
    trend_config,
    apply_macd_filter=True,
    strict_online=True,
):
    logger.info(
        "\n" + "=" * 60 + "\n--- 开始在测试集上进行严格的回测评估 ---\n" + "=" * 60
    )
    original_columns = joblib.load(FEATURE_COLUMNS_PATH)
    flattened_columns_full = joblib.load(FLATTENED_COLUMNS_PATH)
    # 若存在选择后的列，则按其进行推理；否则使用全量列
    if os.path.exists(SELECTED_FLATTENED_COLUMNS_PATH):
        selected_columns = joblib.load(SELECTED_FLATTENED_COLUMNS_PATH)
    else:
        selected_columns = flattened_columns_full

    final_signals = []
    signal_timestamps = []

    def macd_filter_ok(prefix_features: pd.DataFrame) -> bool:
        try:
            macd = prefix_features["MACD_long"].iloc[-1]
            macds = prefix_features["MACDs_long"].iloc[-1]
            macdh = prefix_features["MACDh_long"].iloc[-1]
            if np.isnan(macd) or np.isnan(macds) or np.isnan(macdh):
                return False
            ok = False
            if MACD_FILTER_MODE == "strict":
                ok = (macd > macds) and (macd > 0)
            elif MACD_FILTER_MODE == "hist_pos":
                ok = macdh > 0
            elif MACD_FILTER_MODE == "hist_pos_or_rising":
                prev_macdh = prefix_features["MACDh_long"].iloc[-2] if len(prefix_features) > 1 else np.nan
                slope = macd - (prefix_features["MACD_long"].iloc[-2] if len(prefix_features) > 1 else macd)
                ok = (macdh > 0) or ((macd > macds) and (slope > 0) and (not np.isnan(prev_macdh) and macdh >= prev_macdh))
            elif MACD_FILTER_MODE == "recent_cross":
                window = prefix_features.tail(MACD_RECENT_CROSS_LOOKBACK + 1)
                crossed = False
                if len(window) >= 2:
                    prev = window.iloc[:-1]
                    crossed = ((prev["MACD_long"] <= prev["MACDs_long"]).any()) and (macd > macds)
                ok = crossed
            else:
                ok = (macd > macds)
            if USE_ADX_GATE and "ADX_14" in prefix_features.columns:
                adx = prefix_features["ADX_14"].iloc[-1]
                if np.isnan(adx) or adx < ADX_THRESHOLD:
                    return False
            return bool(ok)
        except Exception:
            return False

    if strict_online:
        logger.info("逐根K线严格在线（expanding）评估...")
        from collections import deque
        scaled_buffer = deque(maxlen=look_back)

        # 遍历测试数据，逐步扩展窗口，仅使用历史信息
        for end in tqdm(range(1, len(test_df) + 1)):
            prefix_df = test_df.iloc[:end]
            prefix_features = feature_engineering(prefix_df, verbose=False)
            # 对齐并仅取最后一行作为当前时刻的特征
            prefix_aligned = prefix_features.reindex(columns=original_columns, fill_value=0)
            if prefix_aligned.empty:
                continue
            last_row = prefix_aligned.iloc[[-1]]
            # 以 DataFrame 形式传入，避免 sklearn 关于缺少列名的警告
            scaled_last = scaler.transform(last_row)
            scaled_buffer.append(scaled_last.reshape(-1))

            if len(scaled_buffer) < look_back:
                continue

            # 构建模型输入的展开序列（look_back x n_features -> 1 x (look_back*n_features)）
            seq_matrix = np.vstack(list(scaled_buffer))
            input_flat = seq_matrix.flatten().reshape(1, -1)
            input_df_full = pd.DataFrame(input_flat, columns=flattened_columns_full)
            input_df = input_df_full[selected_columns]

            pred_prob = model.predict_proba(input_df)[0][1]
            model_signal = 1 if pred_prob > threshold else 0

            if apply_macd_filter:
                is_trend_confirmed = macd_filter_ok(prefix_features)
                final_signal = 1 if (model_signal == 1 and is_trend_confirmed) else 0
            else:
                final_signal = model_signal

            final_signals.append(final_signal)
            signal_timestamps.append(prefix_df.index[-1])
    else:
        logger.info("批量预计算特征的快速评估（仍使用前缀过滤）...")
        test_features_df = feature_engineering(test_df, verbose=True).dropna()
        test_features_aligned = test_features_df.reindex(
            columns=original_columns, fill_value=0
        )
        test_scaled = scaler.transform(test_features_aligned)

        for i in tqdm(range(look_back, len(test_scaled))):
            current_timestamp = test_features_df.index[i]
            input_sequence = test_scaled[i - look_back : i, :].flatten().reshape(1, -1)
            input_df_full = pd.DataFrame(input_sequence, columns=flattened_columns_full)
            input_df = input_df_full[selected_columns]
            pred_prob = model.predict_proba(input_df)[0][1]
            model_signal = 1 if pred_prob > threshold else 0
            if apply_macd_filter:
                # 使用仅到 i 的前缀进行过滤判断
                prefix_features = test_features_df.iloc[: i + 1]
                is_trend_confirmed = macd_filter_ok(prefix_features)
                final_signal = 1 if (model_signal == 1 and is_trend_confirmed) else 0
            else:
                final_signal = model_signal
            final_signals.append(final_signal)
            signal_timestamps.append(current_timestamp)

    actual_labels_df = create_trend_labels(test_df, **trend_config).dropna()
    pred_series = pd.Series(final_signals, index=pd.Index(signal_timestamps))
    results_df = pd.DataFrame(actual_labels_df["label"]).join(
        pred_series.to_frame("final_signal"), how="inner"
    )

    if results_df.empty or np.sum(results_df["final_signal"]) == 0:
        logger.warning("回测期间没有产生任何交易信号。")
        return

    y_test_actual = results_df["label"].values
    y_pred_final = results_df["final_signal"].values
    title_suffix = "(应用MACD过滤器)" if apply_macd_filter else "(无MACD过滤)"
    print(f"\n--- [客观] 测试集回测评估结果 {title_suffix} ---")
    print(f"总回测K线数: {len(y_pred_final)}")
    print(f"发出看涨信号总次数: {np.sum(y_pred_final)}")
    if np.sum(y_pred_final) > 0:
        print(f"精确率 (胜率): {precision_score(y_test_actual, y_pred_final):.4f}")
        print(f"召回率: {recall_score(y_test_actual, y_pred_final):.4f}")
        print(
            "\n混淆矩阵 (TN, FP / FN, TP):\n",
            confusion_matrix(y_test_actual, y_pred_final),
        )


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
        # 基线：保留 MACD 过滤
        run_backtest_and_evaluate(
            test_df,
            trained_model,
            trained_scaler,
            LOOK_BACK,
            best_threshold,
            TREND_CONFIG,
            apply_macd_filter=True,
            strict_online=STRICT_ONLINE_EVAL,
        )
        # 对照：移除 MACD 过滤，评估评估偏差
        if RUN_ABLATION_NO_MACD_FILTER:
            run_backtest_and_evaluate(
                test_df,
                trained_model,
                trained_scaler,
                LOOK_BACK,
                best_threshold,
                TREND_CONFIG,
                apply_macd_filter=False,
                strict_online=STRICT_ONLINE_EVAL,
            )
    else:
        logger.error("模型训练失败，跳过回测。")
