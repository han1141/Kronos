# -*- coding: utf-8 -*-
# V60.4-Dual-V3-Model-Filter (Final-Fix-7-Strategy-Upgrade)
# MODIFIED: Integrated a new high-precision 4h V3 model as a macro trend filter.

# --- 1. 导入库与配置 ---
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.font_manager
import joblib
import os
import glob
import warnings
from scipy.stats import linregress
from scipy.signal import find_peaks
import pandas_ta as pta

try:
    import numba

    jit = numba.jit(nopython=True, cache=True)
    NUMBA_INSTALLED = True
except ImportError:

    def jit(func):
        return func

    NUMBA_INSTALLED = False

try:
    import lightgbm as lgb

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False

try:
    import tensorflow as tf

    ADVANCED_ML_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_ML_LIBS_INSTALLED = False

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- 日志与字体配置 ---
# (此部分代码未变，保持原样)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def set_chinese_font():
    try:
        font_names = [
            "PingFang SC",
            "Microsoft YaHei",
            "SimHei",
            "Heiti TC",
            "sans-serif",
        ]
        for font in font_names:
            if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
                plt.rcParams["font.sans-serif"] = [font]
                plt.rcParams["axes.unicode_minus"] = False
                logger.info(f"成功设置中文字体: {font}")
                return
        logger.warning("未找到指定的中文字体")
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")


set_chinese_font()


# --- 核心配置 ---
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],
    "interval": "15m",
    "backtest_start_date": "2024-01-01",
    "backtest_end_date": "2024-12-31",
    "initial_cash": 500_000,
    "commission": 0.00005,
    "spread": 0.0005,
    "show_plots": False,
    "training_window_days": 365 * 1.5,
    "enable_ml_component": True,
}

# --- 模型路径配置 ---
# <-- 修改点 1: 更新模型路径配置 -->
# 15分钟 V3 模型 (用于精确入场)
V3_ML_MODEL_15M_PATH = "models/eth_model_high_precision_v3_15m.joblib"
V3_ML_SCALER_15M_PATH = "models/eth_scaler_high_precision_v3_15m.joblib"
V3_ML_FEATURE_COLUMNS_15M_PATH = "models/feature_columns_high_precision_v3_15m.joblib"
V3_ML_FLATTENED_COLUMNS_15M_PATH = (
    "models/flattened_columns_high_precision_v3_15m.joblib"
)
V3_ML_THRESHOLD_15M = 0.35
V3_ML_SEQUENCE_LENGTH_15M = 60

# 4小时 V3 模型 (用于宏观方向过滤)
V3_ML_MODEL_4H_PATH = "models/eth_model_high_precision_v3_4h.joblib"
V3_ML_SCALER_4H_PATH = "models/eth_scaler_high_precision_v3_4h.joblib"
V3_ML_FEATURE_COLUMNS_4H_PATH = "models/feature_columns_high_precision_v3_4h.joblib"
V3_ML_FLATTENED_COLUMNS_4H_PATH = "models/flattened_columns_high_precision_v3_4h.joblib"
V3_ML_THRESHOLD_4H = 0.3428  # 使用您训练日志中找到的最佳阈值
V3_ML_SEQUENCE_LENGTH_4H = 60
# <-- 修改结束 -->


# --- 策略参数 ---
STRATEGY_PARAMS = {
    # (此部分代码未变，保持原样)
    "kelly_trade_history": 20,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "regime_adx_period": 14,
    "regime_atr_period": 14,
    "regime_atr_slope_period": 5,
    "regime_rsi_period": 14,
    "regime_rsi_vol_period": 14,
    "regime_norm_period": 252,
    "regime_hurst_period": 100,
    "regime_score_weight_adx": 0.6,
    "regime_score_weight_atr": 0.3,
    "regime_score_weight_rsi": 0.05,
    "regime_score_weight_hurst": 0.05,
    "regime_score_threshold": 0.45,
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_adx_confirm_period": 14,
    "tf_adx_confirm_threshold": 18,
    "tf_chandelier_period": 22,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.5,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 25,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "counter_trend_suppression_factor": 0.1,
    "tf_long_entry_threshold": 0.6,
    "tf_short_entry_threshold": -0.4,
    "time_stop_bars": 0,
    "score_weights_tf": {
        "breakout": 0.25,
        "momentum": 0.20,
        "mtf": 0.15,
        "legacy_ml": 0.15,
        "advanced_ml": 0.0,
        "v3_ml": 0.25,
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETHUSDT": {
        "tf_long_entry_threshold": 0.60,
        "tf_short_entry_threshold": -0.35,
    }
}


# --- 函数定义 ---
# (fetch_binance_klines, get_hurst_exponent_numba, rolling_hurst_numba, compute_hurst, get_market_structure_features, feature_engineering_v3 函数保持不变)
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


@jit
def get_hurst_exponent_numba(ts, max_lag=100):
    lags = np.arange(2, max_lag)
    tau = np.empty(len(lags))
    for i, lag in enumerate(lags):
        tau[i] = np.sqrt(np.nanstd(ts[lag:] - ts[:-lag]))
    valid_indices = np.where(tau > 0)[0]
    if len(valid_indices) < 2:
        return 0.5
    x, y = np.log(lags[valid_indices]), np.log(tau[valid_indices])
    n = len(x)
    sum_x, sum_y, sum_xy, sum_x2 = np.sum(x), np.sum(y), np.sum(x * y), np.sum(x * x)
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.5
    return (n * sum_xy - sum_x * sum_y) / denominator * 2.0


@jit
def rolling_hurst_numba(ts_array, window):
    n = len(ts_array)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_slice = ts_array[i - window + 1 : i + 1]
        if not np.any(np.isnan(window_slice)):
            result[i] = get_hurst_exponent_numba(window_slice, max_lag=window)
    return result


def compute_hurst(ts, max_lag=100):
    if len(ts) < max_lag:
        return 0.5
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except (np.linalg.LinAlgError, ValueError):
        return 0.5


def get_market_structure_features(df, order=5):
    df = df.copy()
    high_peaks_idx, _ = find_peaks(
        df["High"], distance=order, prominence=df["High"].std() * 0.5
    )
    low_peaks_idx, _ = find_peaks(
        -df["Low"], distance=order, prominence=df["Low"].std() * 0.5
    )
    df["swing_high_price"], df["swing_low_price"] = np.nan, np.nan
    df.iloc[high_peaks_idx, df.columns.get_loc("swing_high_price")] = df.iloc[
        high_peaks_idx
    ]["High"]
    df.iloc[low_peaks_idx, df.columns.get_loc("swing_low_price")] = df.iloc[
        low_peaks_idx
    ]["Low"]
    df["swing_high_price"], df["swing_low_price"] = (
        df["swing_high_price"].ffill(),
        df["swing_low_price"].ffill(),
    )
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


def feature_engineering_v3(df_in):
    df = df_in.copy()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    macd_indicator = ta.trend.MACD(
        close=df["Close"], window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = (
        macd_indicator.macd(),
        macd_indicator.macd_signal(),
        macd_indicator.macd_diff(),
    )
    bb_indicator = ta.volatility.BollingerBands(
        close=df["Close"], window=20, window_dev=2
    )
    (
        df["BBL_20_2.0"],
        df["BBM_20_2.0"],
        df["BBU_20_2.0"],
        df["BBB_20_2.0"],
        df["BBP_20_2.0"],
    ) = (
        bb_indicator.bollinger_lband(),
        bb_indicator.bollinger_mavg(),
        bb_indicator.bollinger_hband(),
        bb_indicator.bollinger_wband(),
        bb_indicator.bollinger_pband(),
    )
    adx_indicator = ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = (
        adx_indicator.adx(),
        adx_indicator.adx_pos(),
        adx_indicator.adx_neg(),
    )
    df["volatility"] = (
        (np.log(df["Close"] / df["Close"].shift(1))).rolling(window=20).std()
    )
    market_structure_df = get_market_structure_features(df)
    macd_long_indicator = ta.trend.MACD(
        close=df["Close"], window_fast=24, window_slow=52, window_sign=18
    )
    df["MACD_long"], df["MACDh_long"], df["MACDs_long"] = (
        macd_long_indicator.macd(),
        macd_long_indicator.macd_diff(),
        macd_long_indicator.macd_signal(),
    )
    all_features_df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])
    all_features_df = pd.concat([all_features_df, market_structure_df], axis=1)
    all_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return all_features_df


def generate_v3_ml_predictions(
    df_with_ohlcv: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    orig_cols_path: str,
    flat_cols_path: str,
    seq_len: int,
    log_prefix: str = "[V3 MODEL]",
) -> tuple[pd.Series, pd.Series]:
    """通用V3模型预测函数，返回概率和二元信号"""
    logger.info(f"--- {log_prefix} 开始生成ML预测 ---")
    if not all(
        os.path.exists(p)
        for p in [model_path, scaler_path, orig_cols_path, flat_cols_path]
    ):
        logger.warning(f"{log_prefix} 缺少模型文件，ML预测将为0。")
        return pd.Series(0, index=df_with_ohlcv.index), pd.Series(
            0, index=df_with_ohlcv.index
        )
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        original_columns = joblib.load(orig_cols_path)
        flattened_columns = joblib.load(flat_cols_path)

        features_df = feature_engineering_v3(df_with_ohlcv).dropna()
        features_aligned = features_df.reindex(columns=original_columns, fill_value=0)
        scaled_features = scaler.transform(features_aligned)

        predictions = []
        for i in range(seq_len, len(scaled_features)):
            input_sequence = (
                scaled_features[i - seq_len : i, :].flatten().reshape(1, -1)
            )
            input_df = pd.DataFrame(input_sequence, columns=flattened_columns)
            pred_prob = model.predict_proba(input_df)[0][1]
            predictions.append(pred_prob)

        prediction_index = features_aligned.index[seq_len:]
        final_probs = pd.Series(predictions, index=prediction_index)

        logger.info(f"--- {log_prefix} ML预测生成完毕 ---")
        return final_probs.reindex(df_with_ohlcv.index, fill_value=0)
    except Exception as e:
        logger.error(f"{log_prefix} 生成ML预测时出错: {e}", exc_info=True)
        return pd.Series(0, index=df_with_ohlcv.index)


# (add_ml_features, add_market_regime_features, run_advanced_model_inference 等辅助函数保持不变)
def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    p = STRATEGY_PARAMS
    norm = lambda s: (
        (s - s.rolling(p["regime_norm_period"]).min())
        / (
            s.rolling(p["regime_norm_period"]).max()
            - s.rolling(p["regime_norm_period"]).min()
            + 1e-9
        )
    ).fillna(0.5)
    adx, atr, rsi = (
        ta.trend.ADXIndicator(df.High, df.Low, df.Close, p["regime_adx_period"]).adx(),
        ta.volatility.AverageTrueRange(
            df.High, df.Low, df.Close, p["regime_atr_period"]
        ).average_true_range(),
        ta.momentum.RSIIndicator(df.Close, p["regime_rsi_period"]).rsi(),
    )
    bb = ta.volatility.BollingerBands(
        df.Close, window=p["mr_bb_period"], window_dev=p["mr_bb_std"]
    )
    df["feature_adx_norm"], df["feature_atr_slope_norm"], df["feature_rsi_vol_norm"] = (
        norm(adx),
        norm(
            (atr - atr.shift(p["regime_atr_slope_period"]))
            / (atr.shift(p["regime_atr_slope_period"]) + 1e-9)
        ),
        1 - norm(rsi.rolling(p["regime_rsi_vol_period"]).std()),
    )
    if NUMBA_INSTALLED:
        df["feature_hurst"] = rolling_hurst_numba(
            df["Close"].to_numpy(), p["regime_hurst_period"]
        )
    else:
        df["feature_hurst"] = df.Close.rolling(p["regime_hurst_period"]).apply(
            compute_hurst, raw=False
        )
    df["feature_hurst"] = df["feature_hurst"].fillna(0.5)
    (
        df["feature_obv_norm"],
        df["feature_vol_pct_change_norm"],
        df["feature_bb_width_norm"],
        df["feature_atr_pct_change_norm"],
    ) = (
        norm(
            ta.volume.OnBalanceVolumeIndicator(df.Close, df.Volume).on_balance_volume()
        ),
        norm(df.Volume.pct_change(periods=1).abs()),
        norm(
            (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)
        ),
        norm(atr.pct_change(periods=1)),
    )
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + df["feature_hurst"] * p["regime_score_weight_hurst"]
    )
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["market_regime"] = np.where(
        df["feature_regime_score"] > STRATEGY_PARAMS["regime_score_threshold"], 1, -1
    )
    return df


def run_advanced_model_inference(df):
    if not ADVANCED_ML_LIBS_INSTALLED:
        df["advanced_ml_signal"] = 0.0
        return df
    df["advanced_ml_signal"] = np.random.choice(
        [-1, 0, 1], p=[0.2, 0.6, 0.2], size=len(df)
    )
    return df


# <-- 修改点 2: 更新数据预处理函数 -->
def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(
        f"[{symbol}] 开始数据预处理 (数据范围: {df.index.min()} to {df.index.max()})..."
    )

    # --- 1. 生成15分钟模型的预测 (用于精确入场) ---
    df["v3_ml_prob_15m"] = generate_v3_ml_predictions(
        df,
        V3_ML_MODEL_15M_PATH,
        V3_ML_SCALER_15M_PATH,
        V3_ML_FEATURE_COLUMNS_15M_PATH,
        V3_ML_FLATTENED_COLUMNS_15M_PATH,
        V3_ML_SEQUENCE_LENGTH_15M,
        log_prefix="[V3 MODEL 15M]",
    )
    df["v3_ml_signal_15m"] = (df["v3_ml_prob_15m"] > V3_ML_THRESHOLD_15M).astype(int)

    # --- 2. 生成4小时模型的预测 (用于方向过滤) ---
    df_4h = (
        df.resample("4h")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    probs_4h = generate_v3_ml_predictions(
        df_4h,
        V3_ML_MODEL_4H_PATH,
        V3_ML_SCALER_4H_PATH,
        V3_ML_FEATURE_COLUMNS_4H_PATH,
        V3_ML_FLATTENED_COLUMNS_4H_PATH,
        V3_ML_SEQUENCE_LENGTH_4H,
        log_prefix="[V3 MODEL 4H]",
    )
    signals_4h = (probs_4h > V3_ML_THRESHOLD_4H).astype(int)

    # 将4H信号填充回15M数据框。shift(1)是为了避免数据窥探
    df["v3_ml_signal_4h"] = (
        signals_4h.shift(1).reindex(df.index, method="ffill").fillna(0)
    )

    # --- 3. 添加其他特征 (用于分析或未来扩展) ---
    df = run_advanced_model_inference(df)
    df = add_ml_features(df)
    df = add_market_regime_features(df)

    d_start = df.index.min().normalize() - pd.Timedelta(
        days=STRATEGY_PARAMS["mtf_period"] + 5
    )
    d_end = df.index.max().normalize() + pd.Timedelta(days=1)
    data_1d = fetch_binance_klines(
        symbol, "1d", d_start.strftime("%Y-%m-%d"), d_end.strftime("%Y-%m-%d")
    )
    if not data_1d.empty:
        sma = ta.trend.SMAIndicator(
            data_1d["Close"], STRATEGY_PARAMS["mtf_period"]
        ).sma_indicator()
        df["mtf_signal"] = (
            pd.Series(np.where(data_1d["Close"] > sma, 1, -1), index=data_1d.index)
            .shift(1)
            .reindex(df.index, method="ffill")
            .fillna(0)
        )
    else:
        df["mtf_signal"] = 0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


# <-- 修改结束 -->


TREND_CONFIG = {"look_forward_steps": 5, "ema_length": 8}


# (analyze_v3_standalone_performance 函数保持不变, 但现在可能需要指定分析哪个信号)
def analyze_v3_standalone_performance(df: pd.DataFrame, signal_col="v3_ml_signal_15m"):
    print(f"\n{'-'*40}\n       V3 高精度模型独立表现分析 ({signal_col}) \n{'-'*40}")
    required_cols = [signal_col, "Close"]
    if not all(col in df.columns for col in required_cols):
        print(f"缺少必要列 {signal_col}，无法分析。")
        return
    # ... rest of function is the same, just using signal_col
    look_forward_steps = TREND_CONFIG["look_forward_steps"]
    ema_length = TREND_CONFIG["ema_length"]
    n = len(df)
    df_reset = df.reset_index(drop=True)
    df_reset[f"EMA_{ema_length}"] = pta.ema(close=df_reset["Close"], length=ema_length)
    macd_result = pta.macd(close=df_reset["Close"], fast=24, slow=52, signal=18)
    df_reset["MACD_long"] = macd_result["MACD_24_52_18"]
    df_reset["MACDs_long"] = macd_result["MACDs_24_52_18"]
    valid_mask = (
        (df_reset.index <= n - look_forward_steps - 1)
        & (df_reset[signal_col] == 1)
        & (df_reset["MACD_long"] > df_reset["MACDs_long"])
        & (df_reset["MACD_long"] > 0)
    )
    trade_signals = df_reset[valid_mask].copy()
    if trade_signals.empty:
        print("无有效信号（可能因MACD过滤或边界限制）。")
        return
    total_trades = len(trade_signals)
    current_ema = trade_signals[f"EMA_{ema_length}"]
    future_ema = df_reset.loc[
        trade_signals.index + look_forward_steps, f"EMA_{ema_length}"
    ].values
    wins = (future_ema > current_ema).sum()
    win_rate = (wins / total_trades) * 100
    entry_price = trade_signals["Close"]
    exit_price = df_reset.loc[trade_signals.index + look_forward_steps, "Close"].values
    price_returns = (exit_price - entry_price) / entry_price
    avg_price_return = price_returns.mean() * 100
    cum_price_return = price_returns.sum() * 100
    print(f"有效信号总数（含MACD过滤，可观测{look_forward_steps}步）: {total_trades}")
    print(f"✅ 胜率（EMA趋势上涨，与训练目标一致）: {win_rate:.2f}%")
    print(f"📊 平均价格回报率（实际盈亏参考）: {avg_price_return:.4f}%")
    print(f"📈 累计价格回报率: {cum_price_return:.2f}%")
    print(f"{'-'*40}")


# <-- 修改点 3: 更新策略以使用双模型信号 -->
class UltimateStrategy(Strategy):
    symbol = None

    def init(self):
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)
        c = pd.Series(self.data.Close)
        h = pd.Series(self.data.High)
        l = pd.Series(self.data.Low)

        # 15M V3 Model signals (for entry timing)
        self.v3_ml_signal_15m = self.I(lambda: self.data.v3_ml_signal_15m)
        self.v3_ml_prob_15m = self.I(lambda: self.data.v3_ml_prob_15m)

        # 4H V3 Model signal (for trend direction filter)
        self.v3_ml_signal_4h = self.I(lambda: self.data.v3_ml_signal_4h)

        # EMA for trend confirmation
        self.ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(c, self.tf_ema_fast_period).ema_indicator()
        )
        # ATR for stop-loss calculation
        self.atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                h, l, c, self.tf_atr_period
            ).average_true_range()
        )

    def next(self):
        price = self.data.Close[-1]

        # --- Define Entry and Exit Signals ---
        # 1. 宏观方向过滤器: 4小时模型必须看涨
        long_term_trend_is_up = self.v3_ml_signal_4h[-1] > 0

        # 2. 短期入场触发器: 15分钟模型信号为ON, 且价格在EMA之上
        short_term_entry_trigger = (
            self.v3_ml_signal_15m[-1] > 0 and price > self.ema_fast[-1]
        )

        # 最终入场信号: 必须同时满足两者
        entry_signal = long_term_trend_is_up and short_term_entry_trigger

        # 离场信号: 15分钟信号消失 或 价格跌破快速EMA
        # 离场逻辑保持灵敏，以快速止损
        exit_signal = self.v3_ml_signal_15m[-1] <= 0 or price < self.ema_fast[-1]

        # --- Trading Logic ---
        if not self.position:
            if entry_signal:
                self.open_dynamic_position(price)
        elif self.position.is_long:
            if exit_signal:
                self.position.close()

    def get_confidence_factor(self, probability: float) -> float:
        if probability > 0.65:
            return 2.0
        elif probability > 0.55:
            return 1.5
        elif probability > 0.45:
            return 1.0
        else:
            return 0.5

    def open_dynamic_position(self, price: float):
        # 仓位大小依然由15分钟模型的概率决定
        probability = self.v3_ml_prob_15m[-1]
        confidence_factor = self.get_confidence_factor(probability)
        dynamic_risk_pct = min(
            self.default_risk_pct * confidence_factor, self.max_risk_pct
        )
        risk_per_share = self.atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_per_share <= 0:
            return
        size = self._calculate_position_size(price, risk_per_share, dynamic_risk_pct)
        if size > 0:
            self.buy(size=size)

    def _calculate_position_size(self, price, risk_per_share, risk_pct):
        if risk_per_share <= 0 or price <= 0 or risk_pct <= 0:
            return 0
        return int((self.equity * risk_pct) / risk_per_share)


# <-- 修改结束 -->


# --- 主程序入口 ---
if __name__ == "__main__":
    logger.info(f"🚀 (V60.4-Dual-V3-Model-Filter) 开始运行...")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_fetch_start_date_str = (
        backtest_start_dt - pd.Timedelta(days=365 * 2)
    ).strftime("%Y-%m-%d")
    if CONFIG["enable_ml_component"]:
        training_window = timedelta(days=CONFIG["training_window_days"])
        data_fetch_start_date_str = (backtest_start_dt - training_window).strftime(
            "%Y-%m-%d"
        )

    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期 (包含训练窗口): {data_fetch_start_date_str}")
    raw_data = {
        s: fetch_binance_klines(
            s,
            CONFIG["interval"],
            data_fetch_start_date_str,
            CONFIG["backtest_end_date"],
        )
        for s in CONFIG["symbols_to_test"]
    }
    if not any(not df.empty for df in raw_data.values()):
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    logger.info("### 模式: 跳过动态训练，使用静态模型进行回测 ###")
    logger.info(f"### 准备完整回测数据 ###")
    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        if data.empty:
            continue
        logger.info(f"为 {symbol} 预处理完整时段数据...")
        full_processed_data = preprocess_data_for_strategy(data, symbol)
        backtest_slice = full_processed_data.loc[CONFIG["backtest_start_date"] :].copy()
        if not backtest_slice.empty:
            processed_backtest_data[symbol] = backtest_slice

    if not processed_backtest_data:
        logger.error("无回测数据，程序终止。")
        exit()

    logger.info(f"### 进入回测模式 ###")
    all_stats = {}
    for symbol, data in processed_backtest_data.items():
        print(f"\n{'='*80}\n正在回测品种: {symbol}\n{'='*80}")
        bt = Backtest(
            data,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            margin=CONFIG["spread"] / 2,
            finalize_trades=True,
        )
        stats = bt.run(symbol=symbol)
        all_stats[symbol] = stats
        print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
        print(stats)

        # 对15分钟模型的独立表现进行分析
        analyze_v3_standalone_performance(data, signal_col="v3_ml_signal_15m")

        if CONFIG["show_plots"]:
            bt.plot()

    if all_stats:
        initial_total = CONFIG["initial_cash"] * len(all_stats)
        total_equity = sum(stats["Equity Final [$]"] for stats in all_stats.values())
        ret = ((total_equity - initial_total) / initial_total) * 100
        print(f"\n{'#'*80}\n                 组合策略表现总览\n{'#'*80}")
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A')}"
            )
        print(
            f"\n--- 投资组合整体表现 ---\n总初始资金: ${initial_total:,.2f}\n总最终权益: ${total_equity:,.2f}\n组合总回报率: {ret:.2f}%"
        )
