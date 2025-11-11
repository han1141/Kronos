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
    precision_score,
    recall_score,
    precision_recall_curve,
    confusion_matrix,
)
from tqdm import tqdm
from scipy.signal import find_peaks

# <<< 新增导入 >>>
import numba

# --- 0. 设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 🚀 全局配置 ---
SYMBOL = "ETHUSDT"
INTERVAL = "15m"
DATA_START_DATE = "2017-01-01"
TRAIN_START = "2018-01-01"
VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = "2025-11-09"
LOOK_BACK = 60
TREND_CONFIG = {
    "look_forward_steps": 3,
    "target_return": 0.004,
    "max_drawdown_limit": 0.01,
}
logger.info(
    f"训练目标盈利：{TREND_CONFIG['target_return']*100}%，最大回撤限制：{TREND_CONFIG['max_drawdown_limit']*100}%"
)

# --- 文件路径 ---
MODELS_DIR, DATA_DIR = "models_gbm2", "data"
MODEL_SAVE_PATH = os.path.join(
    MODELS_DIR, f"eth_model_high_precision_v4_{INTERVAL}.joblib"
)
SCALER_SAVE_PATH = os.path.join(
    MODELS_DIR, f"eth_scaler_high_precision_v4_{INTERVAL}.joblib"
)
FEATURE_COLUMNS_PATH = os.path.join(
    MODELS_DIR, f"feature_columns_high_precision_v4_{INTERVAL}.joblib"
)
FLATTENED_COLUMNS_PATH = os.path.join(
    MODELS_DIR, f"flattened_columns_high_precision_v4_{INTERVAL}.joblib"
)
DATA_CACHE_PATH = os.path.join(DATA_DIR, f"{SYMBOL.lower()}_{INTERVAL}_data.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --- 数据获取与辅助函数 ---
def fetch_binance_klines(s, i, st, en=None, l=1000):
    # ... (此函数保持不变) ...
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


# <<< 已修改：Hurst函数使用Numba JIT进行加速，并手动实现线性回归 >>>
@numba.jit(nopython=True, cache=True)
def compute_hurst_numba(ts):
    if len(ts) < 100:
        return 0.5

    max_lag = 100
    lags = np.arange(2, max_lag)

    tau = np.empty(len(lags), dtype=np.float64)
    for i, lag in enumerate(lags):
        # Numba-friendly standard deviation calculation
        diff = ts[lag:] - ts[:-lag]
        if len(diff) > 0:
            # Manually calculate std dev: sqrt(E[X^2] - E[X]^2)
            tau[i] = np.sqrt(np.mean(diff**2) - (np.mean(diff)) ** 2)
        else:
            tau[i] = 0.0  # Should not happen if len(ts) is sufficient

    # Filter out zero values to avoid log(0) issues and ensure we have enough points
    valid_tau = tau[tau > 0]
    valid_lags = lags[tau > 0]
    if len(valid_tau) < 2:
        return 0.5

    # --- <<< 核心修改：手动实现 np.polyfit(deg=1) >>> ---
    # Convert to log scale
    log_lags = np.log(valid_lags)
    log_tau = np.log(valid_tau)

    # Calculate the slope (m) of the best-fit line y = mx + c
    # using the formula: m = ( (mean(x*y) - mean(x)*mean(y)) /
    #                          (mean(x^2) - mean(x)^2) )
    mean_log_lags = np.mean(log_lags)
    mean_log_tau = np.mean(log_tau)

    numerator = np.mean(log_lags * log_tau) - (mean_log_lags * mean_log_tau)
    denominator = np.mean(log_lags**2) - (mean_log_lags**2)

    if denominator == 0:
        return 0.5

    hurst_exponent = numerator / denominator

    return hurst_exponent * 2.0


def get_market_structure_features(df, order=5):
    # ... (此函数保持不变) ...
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
    """整合所有特征计算 - V4.2 增加特征交叉"""
    df_copy = df.copy()
    logger.info("--- 开始计算特征 (V4.2 增强版 - 含特征交叉) ---")

    # 1. 基础指标 (保持不变)
    df_copy.ta.rsi(length=14, append=True)
    df_copy.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_copy.ta.bbands(length=20, std=2, append=True)
    df_copy.ta.adx(length=14, append=True)
    df_copy.ta.atr(length=14, append=True, col_names=("ATR_14"))
    df_copy.ta.obv(append=True)

    # 2. 市场结构与长周期趋势 (保持不变)
    market_structure_df = get_market_structure_features(df_copy)
    df_copy.ta.macd(
        fast=24,
        slow=52,
        signal=18,
        append=True,
        col_names=("MACD_long", "MACDh_long", "MACDs_long"),
    )

    # 3. 市场状态与波动性 (保持不变)
    logger.info("正在计算Hurst指数 (此过程可能需要几分钟)...")
    df_copy["hurst"] = (
        df_copy["Close"].rolling(window=100).apply(compute_hurst_numba, raw=True)
    )
    logger.info("Hurst指数计算完成。")
    df_copy["volatility_log"] = (
        (np.log(df_copy["Close"] / df_copy["Close"].shift(1))).rolling(window=20).std()
    )

    # 4. "规则" 转化为 "特征" (保持不变)
    df_copy["macd_cross_signal"] = (
        df_copy["MACD_12_26_9"] > df_copy["MACDs_12_26_9"]
    ).astype(int)
    df_copy["macd_long_cross_signal"] = (
        df_copy["MACD_long"] > df_copy["MACDs_long"]
    ).astype(int)

    # 5. 多时间框架特征 (保持不变)
    close_4h = df_copy["Close"].resample("4h").last()
    ema_4h = ta.ema(close_4h, length=50)
    df_copy["ema_4h"] = ema_4h.reindex(df_copy.index, method="ffill")
    df_copy["price_above_ema_4h"] = (df_copy["Close"] > df_copy["ema_4h"]).astype(int)

    # 6. --- <<< 新增：特征交叉 (Feature Crossing) >>> ---
    logger.info("正在创建交叉特征...")
    # 示例1: 波动率与趋势强度的交互 (高ADX和高ATR可能意味着强力突破)
    df_copy["adx_x_atr_norm"] = (df_copy["ADX_14"] / 50) * (
        df_copy["ATR_14"] / df_copy["Close"]
    )
    # 示例2: RSI与市场状态的交互 (趋势市中的RSI vs 震荡市的RSI)
    df_copy["rsi_x_hurst"] = df_copy["RSI_14"] * df_copy["hurst"]
    # 示例3: 短期趋势与长周期趋势的确认 (两个MACD都看涨)
    df_copy["macd_cross_confirm"] = (
        df_copy["macd_cross_signal"] * df_copy["macd_long_cross_signal"]
    )
    logger.info("交叉特征创建完成。")

    # 7. 整合
    df_copy = pd.concat([df_copy, market_structure_df], axis=1)

    # 8. 选择特征列并进行清理
    feature_columns = [
        col
        for col in df_copy.columns
        if col
        not in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "swing_high_price",
            "swing_low_price",
            "ema_4h",
        ]
    ]

    all_features_df = (
        df_copy[feature_columns].replace([np.inf, -np.inf], np.nan).ffill()
    )

    return all_features_df


# --- (其他函数如 create_trend_labels, train_and_validate 等都使用我们上一轮讨论的最新版本) ---
# ...
# The rest of the script (create_trend_labels, train_and_validate, run_backtest_and_evaluate, __main__)
# remains IDENTICAL to the one provided in the previous response ("给出修改后的完整版代码").
# You only need to replace the `compute_hurst` and `feature_engineering` functions
# and add `import numba` at the top.
#
# For completeness, I'll paste the rest of the script again.
#
def create_trend_labels(df, look_forward_steps, target_return, max_drawdown_limit):
    df_copy = df.copy()
    df_copy["target_price"] = df_copy["Close"] * (1 + target_return)
    future_highs = (
        df_copy["High"]
        .rolling(window=look_forward_steps)
        .max()
        .shift(-look_forward_steps)
    )
    future_lows = (
        df_copy["Low"]
        .rolling(window=look_forward_steps)
        .min()
        .shift(-look_forward_steps)
    )
    drawdown_before_profit = (df_copy["Close"] - future_lows) / df_copy["Close"]
    profit_reached = future_highs >= df_copy["target_price"]
    risk_controlled = drawdown_before_profit < max_drawdown_limit
    df_copy["label"] = (profit_reached & risk_controlled).astype(int)
    return df_copy


def create_flattened_sequences(data, labels, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : (i + look_back), :].flatten())
        y.append(labels[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_and_validate(train_df, validation_df, look_back, trend_config):
    logger.info("--- 开始训练和验证流程 (V4) ---")
    X_train_full_df = feature_engineering(train_df).dropna()
    X_validation_full_df = feature_engineering(validation_df).dropna()
    train_labeled = create_trend_labels(train_df, **trend_config)
    validation_labeled = create_trend_labels(validation_df, **trend_config)
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
    precision_focus_ratio = 0.3
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
    logger.info("\n开始训练 LightGBM 模型 (V4)...")
    lgb_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_validation_df, y_validation)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    y_val_pred_probs = lgb_model.predict_proba(X_validation_df)[:, 1]
    MIN_PRECISION_TARGET = 0.55
    precisions, recalls, thresholds = precision_recall_curve(
        y_validation, y_val_pred_probs
    )
    valid_threshold_indices = np.where(precisions[:-1] >= MIN_PRECISION_TARGET)[0]
    best_threshold = 0.5
    if len(valid_threshold_indices) > 0:
        f1_scores = np.divide(
            2 * recalls * precisions,
            recalls + precisions,
            out=np.zeros_like(recalls),
            where=(recalls + precisions) != 0,
        )
        best_idx_within_valid = np.argmax(f1_scores[valid_threshold_indices])
        final_best_idx = valid_threshold_indices[best_idx_within_valid]
        best_threshold = thresholds[final_best_idx]
        logger.info(
            f"在满足胜率>{MIN_PRECISION_TARGET*100}%的条件下，找到最佳阈值: {best_threshold:.4f}"
        )
        logger.info(
            f"该阈值下的验证集表现: Precision={precisions[final_best_idx]:.4f}, Recall={recalls[final_best_idx]:.4f}"
        )
    else:
        logger.warning(
            f"未能找到任何阈值可以使验证集胜率达到 {MIN_PRECISION_TARGET*100}%。将使用最大化F1的阈值。"
        )
        f1_scores = np.divide(
            2 * recalls * precisions,
            recalls + precisions,
            out=np.zeros_like(recalls),
            where=(recalls + precisions) != 0,
        )
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = (
            thresholds[best_f1_idx] if len(thresholds) > best_f1_idx else 0.5
        )
        logger.info(f"在验证集上找到的最佳F1阈值: {best_threshold:.4f}")
    joblib.dump(lgb_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"模型已保存到: {MODEL_SAVE_PATH}")
    return lgb_model, scaler, best_threshold


# --- 🚀 4. 升级版回测评估函数 (V4.3 - 增加回撤分布分析) ---
def run_backtest_and_evaluate(
    test_df, model, scaler, look_back, threshold, trend_config
):
    logger.info(
        "\n" + "=" * 60 + "\n--- 开始在测试集上进行严格的回测评估 (V4) ---\n" + "=" * 60
    )
    original_columns = joblib.load(FEATURE_COLUMNS_PATH)
    flattened_columns = joblib.load(FLATTENED_COLUMNS_PATH)
    test_features_df = feature_engineering(test_df).dropna()
    test_features_aligned = test_features_df.reindex(
        columns=original_columns, fill_value=0
    )
    test_scaled = scaler.transform(test_features_aligned)
    final_signals = []
    logger.info("逐根K线遍历测试集进行预测...")
    for i in tqdm(range(look_back, len(test_scaled))):
        input_sequence = test_scaled[i - look_back : i, :]
        input_flattened_np = input_sequence.flatten().reshape(1, -1)
        input_df = pd.DataFrame(input_flattened_np, columns=flattened_columns)
        pred_prob = model.predict_proba(input_df)[0][1]
        model_signal = 1 if pred_prob > threshold else 0
        final_signals.append(model_signal)
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

    winning_trades_drawdown = []
    winning_signals_df = results_df[
        (results_df["final_signal"] == 1) & (results_df["label"] == 1)
    ]
    if not winning_signals_df.empty:
        logger.info("正在计算盈利信号在盈利前的最大回撤...")
        look_forward = trend_config["look_forward_steps"]
        test_df_copy = test_df.copy()
        test_df_copy["future_min_low"] = (
            test_df_copy["Low"].rolling(window=look_forward).min().shift(-look_forward)
        )
        winning_trades_details = winning_signals_df.join(
            test_df_copy[["Close", "future_min_low"]], how="inner"
        )
        winning_trades_details["drawdown_pct"] = (
            (winning_trades_details["Close"] - winning_trades_details["future_min_low"])
            / winning_trades_details["Close"]
        ) * 100
        winning_trades_drawdown = winning_trades_details["drawdown_pct"].tolist()

    print("\n--- [客观] 测试集回测评估结果 (仅ML模型信号) ---")
    print(f"总回测K线数: {len(y_pred_final)}")
    print(f"发出看涨信号总次数 (交易频率): {np.sum(y_pred_final)}")
    # 增加zero_division=0以防止在没有信号时报错
    print(
        f"精确率 (胜率): {precision_score(y_test_actual, y_pred_final, zero_division=0):.4f}"
    )
    print(
        f"召回率 (盈利机会捕捉率): {recall_score(y_test_actual, y_pred_final, zero_division=0):.4f}"
    )
    print("\n混淆矩阵 (TN, FP / FN, TP):")
    print(confusion_matrix(y_test_actual, y_pred_final))

    if winning_trades_drawdown:
        drawdown_array = np.array(winning_trades_drawdown)
        avg_drawdown = np.mean(drawdown_array)

        print("\n--- [新指标] 盈利交易在盈利前的最大回撤分析 ---")
        print(f"盈利的交易总数: {len(drawdown_array)}")
        print(f"平均回撤: {avg_drawdown:.4f}%")
        print(f"最大回撤 (最差情况): {np.max(drawdown_array):.4f}%")
        print(f"最小回撤 (最好情况): {np.min(drawdown_array):.4f}%")

        # --- <<< 新增：回撤分布分析 >>> ---
        # 计算回撤大于/小于平均回撤的交易百分比
        count_above_avg = np.sum(drawdown_array > avg_drawdown)
        count_below_or_equal_avg = len(drawdown_array) - count_above_avg

        percent_above_avg = (count_above_avg / len(drawdown_array)) * 100
        percent_below_or_equal_avg = (
            count_below_or_equal_avg / len(drawdown_array)
        ) * 100

        print(f"\n  回撤分布 (与平均值 {avg_drawdown:.4f}% 相比):")
        print(
            f"    - 大于平均回撤的交易占比: {percent_above_avg:.2f}% ({count_above_avg} 笔)"
        )
        print(
            f"    - 小于或等于平均回撤的交易占比: {percent_below_or_equal_avg:.2f}% ({count_below_or_equal_avg} 笔)"
        )

        # 计算不同回撤水平下的交易占比
        print("\n  回撤水平分布:")
        for threshold_pct in [0.1, 0.25, 0.5, 0.75]:
            count_below_threshold = np.sum(drawdown_array <= threshold_pct)
            percent_below_threshold = (
                count_below_threshold / len(drawdown_array)
            ) * 100
            print(
                f"    - 回撤 <= {threshold_pct:.2f}% 的交易占比: {percent_below_threshold:.2f}%"
            )

    else:
        print("\n--- [新指标] 盈利交易在盈利前的最大回撤分析 ---")
        print("本次回测没有产生任何盈利的交易，无法计算相关指标。")


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
