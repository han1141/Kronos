# -*- coding: utf-8 -*-
# V60.3-Optimized-Fix-With-V3-Integration (Final-Fix-6-Performance-Tuning)

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
    "commission": 0.0002,
    "spread": 0.0005,
    "show_plots": False,
    "training_window_days": 365 * 1.5,
    "enable_ml_component": True,
}

# --- 模型路径配置 ---
LEGACY_ML_MODEL_PATH = "models/eth_trend_model_lgb_4h.joblib"
LEGACY_ML_SCALER_PATH = "models/eth_trend_scaler_lgb_4h.joblib"
LEGACY_ML_FEATURE_COLUMNS_PATH = "models/feature_columns_lgb_4h.joblib"
LEGACY_ML_THRESHOLD = 0.3159
LEGACY_ML_SEQUENCE_LENGTH = 60

V3_ML_MODEL_PATH = "models/eth_model_high_precision_v3_15m.joblib"
V3_ML_SCALER_PATH = "models/eth_scaler_high_precision_v3_15m.joblib"
V3_ML_FEATURE_COLUMNS_PATH = "models/feature_columns_high_precision_v3_15m.joblib"
V3_ML_FLATTENED_COLUMNS_PATH = "models/flattened_columns_high_precision_v3_15m.joblib"
V3_ML_THRESHOLD = 0.3204
V3_ML_SEQUENCE_LENGTH = 60

# --- 策略参数 ---
STRATEGY_PARAMS = {
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
    # --- [OPTIMIZED] ---
    "time_stop_bars": 0,
    "score_weights_tf": {
        "breakout": 0.25,      # 权重恢复
        "momentum": 0.20,      # 权重恢复
        "mtf": 0.15,           # 权重恢复
        "legacy_ml": 0.15,     # 权重恢复
        "advanced_ml": 0.0,    # 保持禁用
        "v3_ml": 0.25,         # V3模型权重调整为平衡值
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETHUSDT": {
        "tf_long_entry_threshold": 0.60,
        "tf_short_entry_threshold": -0.35,
    }
}


# --- 函数定义 ---
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


def generate_v3_ml_signal(df_with_ohlcv: pd.DataFrame) -> pd.Series:
    logger.info("--- [V3 MODEL] 开始生成高精度ML信号 ---")
    if not all(
        os.path.exists(p)
        for p in [
            V3_ML_MODEL_PATH,
            V3_ML_SCALER_PATH,
            V3_ML_FEATURE_COLUMNS_PATH,
            V3_ML_FLATTENED_COLUMNS_PATH,
        ]
    ):
        logger.warning("缺少V3模型文件，V3 ML信号将为0。")
        return pd.Series(0, index=df_with_ohlcv.index)
    try:
        model, scaler, original_columns, flattened_columns = (
            joblib.load(V3_ML_MODEL_PATH),
            joblib.load(V3_ML_SCALER_PATH),
            joblib.load(V3_ML_FEATURE_COLUMNS_PATH),
            joblib.load(V3_ML_FLATTENED_COLUMNS_PATH),
        )
        features_df = feature_engineering_v3(df_with_ohlcv).dropna()
        features_aligned = features_df.reindex(columns=original_columns, fill_value=0)
        scaled_features = scaler.transform(features_aligned)
        signals = []
        look_back = V3_ML_SEQUENCE_LENGTH
        for i in range(look_back, len(scaled_features)):
            input_sequence = (
                scaled_features[i - look_back : i, :].flatten().reshape(1, -1)
            )
            input_df = pd.DataFrame(input_sequence, columns=flattened_columns)
            pred_prob = model.predict_proba(input_df)[0][1]
            model_signal = 1 if pred_prob > V3_ML_THRESHOLD else 0
            current_timestamp = features_aligned.index[i]
            macd_long, macds_long = (
                features_df.loc[current_timestamp, "MACD_long"],
                features_df.loc[current_timestamp, "MACDs_long"],
            )
            is_trend_confirmed = (macd_long > macds_long) and (macd_long > 0)
            signals.append(1 if (model_signal == 1 and is_trend_confirmed) else 0)
        signal_index = features_aligned.index[look_back:]
        final_series = pd.Series(signals, index=signal_index).shift(1)
        logger.info("--- [V3 MODEL] 高精度ML信号生成完毕 ---")
        return final_series.reindex(df_with_ohlcv.index, fill_value=0)
    except Exception as e:
        logger.error(f"生成V3 ML信号时出错: {e}", exc_info=True)
        return pd.Series(0, index=df_with_ohlcv.index)


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


def add_legacy_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
    df["volatility"] = (np.log(c / c.shift(1))).rolling(window=20).std()
    df["EMA_8"] = ta.trend.EMAIndicator(c, 8).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(c, 14).rsi()
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    df["ATRr_14"] = (
        ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range() / c
    ) * 100
    bb = ta.volatility.BollingerBands(c, 20, 2.0)
    (
        df["BBU_20_2.0"],
        df["BBM_20_2.0"],
        df["BBL_20_2.0"],
        df["BBB_20_2.0"],
        df["BBP_20_2.0"],
    ) = (
        bb.bollinger_hband(),
        bb.bollinger_mavg(),
        bb.bollinger_lband(),
        bb.bollinger_wband(),
        bb.bollinger_pband(),
    )
    macd = ta.trend.MACD(c, 12, 26, 9)
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = (
        macd.macd(),
        macd.macd_signal(),
        macd.macd_diff(),
    )
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["volume_change_rate"] = v.pct_change()
    return df


def generate_legacy_ml_signal(df: pd.DataFrame) -> pd.Series:
    if not all(
        os.path.exists(p)
        for p in [
            LEGACY_ML_MODEL_PATH,
            LEGACY_ML_SCALER_PATH,
            LEGACY_ML_FEATURE_COLUMNS_PATH,
        ]
    ):
        logger.warning("缺少老版本模型文件，ML信号将为0。")
        return pd.Series(0, index=df.index)
    try:
        model, scaler, f_cols = (
            joblib.load(LEGACY_ML_MODEL_PATH),
            joblib.load(LEGACY_ML_SCALER_PATH),
            joblib.load(LEGACY_ML_FEATURE_COLUMNS_PATH),
        )
        df_copy = df.copy()
        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in f_cols:
            if col not in df_copy.columns:
                df_copy[col] = 0
        df_aligned = df_copy[f_cols].dropna()
        if df_aligned.empty:
            return pd.Series(0, index=df.index)
        scaled = scaler.transform(df_aligned)
        X = np.array(
            [
                scaled[i : i + LEGACY_ML_SEQUENCE_LENGTH].flatten()
                for i in range(len(scaled) - LEGACY_ML_SEQUENCE_LENGTH + 1)
            ],
            dtype=np.float32,
        )
        if X.shape[0] == 0:
            return pd.Series(0, index=df.index)
        probs = model.predict_proba(X)[:, 1]
        signals = np.where(probs > LEGACY_ML_THRESHOLD, 1, -1)
        return pd.Series(
            signals, index=df_aligned.index[LEGACY_ML_SEQUENCE_LENGTH - 1 :]
        ).reindex(df.index, fill_value=0)
    except Exception as e:
        logger.error(f"生成老版本ML信号时出错: {e}")
        return pd.Series(0, index=df.index)


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(
        f"[{symbol}] 开始数据预处理 (数据范围: {df.index.min()} to {df.index.max()})..."
    )
    df["v3_ml_signal"] = generate_v3_ml_signal(df)
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
    legacy_4h_f = add_legacy_ml_features(df_4h.copy())
    df["legacy_ml_signal"] = (
        generate_legacy_ml_signal(legacy_4h_f)
        .shift(1)
        .reindex(df.index, method="ffill")
        .fillna(0)
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


def analyze_v3_standalone_performance(df: pd.DataFrame):
    print(f"\n{'-'*40}\n       V3 高精度模型独立表现分析\n{'-'*40}")
    if "v3_ml_signal" not in df.columns or df["v3_ml_signal"].sum() == 0:
        print("未找到有效的V3模型信号，无法进行分析。")
        return
    look_forward_steps = 5
    ema_length = STRATEGY_PARAMS.get("tf_ema_fast_period", 20)
    df_analysis = df.copy()
    df_analysis["ema"] = ta.trend.EMAIndicator(
        close=df_analysis["Close"], window=ema_length
    ).ema_indicator()
    df_analysis["future_ema"] = df_analysis["ema"].shift(-look_forward_steps)
    trade_signals = df_analysis[df_analysis["v3_ml_signal"] == 1].dropna()
    if trade_signals.empty:
        print("V3模型在回测期间内未发出任何做多信号。")
        return
    wins = (trade_signals["future_ema"] > trade_signals["ema"]).sum()
    total_trades = len(trade_signals)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    returns = (
        trade_signals["Close"].shift(-look_forward_steps) - trade_signals["Close"]
    ) / trade_signals["Close"]
    simple_cumulative_return = returns.sum() * 100
    average_return_per_trade = returns.mean() * 100
    print(f"信号总数: {total_trades}")
    print(
        f"胜率 (使用EMA评估，周期={ema_length}, 向前={look_forward_steps}根K线): {win_rate:.2f}%"
    )
    print(f"平均每笔信号回报率 (基于Close价格): {average_return_per_trade:.4f}%")
    print(f"简单累加总回报率 (基于Close价格): {simple_cumulative_return:.2f}%")
    print(f"{'-'*40}")


class BaseAssetStrategy:
    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy

    def _calculate_entry_score(self) -> float:
        m = self.main
        w = m.score_weights_tf
        breakout = (
            1
            if m.data.Close[-1] > m.tf_donchian_h[-1]
            else -1 if m.data.Close[-1] < m.tf_donchian_l[-1] else 0
        )
        momentum = 1 if m.tf_ema_fast[-1] > m.tf_ema_slow[-1] else -1
        trend_filter, suppression = (
            m.legacy_ml_signal[-1],
            m.counter_trend_suppression_factor,
        )
        breakout_w, momentum_w = w.get("breakout", 0), w.get("momentum", 0)
        if trend_filter * breakout < 0:
            breakout_w *= suppression
        if trend_filter * momentum < 0:
            momentum_w *= suppression
        return (
            breakout * breakout_w
            + momentum * momentum_w
            + m.mtf_signal[-1] * w.get("mtf", 0)
            + m.legacy_ml_signal[-1] * w.get("legacy_ml", 0)
            + m.advanced_ml_signal[-1] * w.get("advanced_ml", 0)
            + m.v3_ml_signal[-1] * w.get("v3_ml", 0)
        )

    def _define_mr_entry_signal(self) -> int:
        m = self.main
        if crossover(m.data.Close, m.mr_bb_lower) and m.mr_rsi[-1] < m.mr_rsi_oversold:
            return 1
        if (
            crossover(m.mr_bb_upper, m.data.Close)
            and m.mr_rsi[-1] > m.mr_rsi_overbought
        ):
            return -1
        return 0


class BTCStrategy(BaseAssetStrategy):
    def _calculate_entry_score(self) -> float:
        return super()._calculate_entry_score() if self.main.tf_adx[-1] > 20 else 0


class ETHStrategy(BaseAssetStrategy):
    pass


STRATEGY_MAPPING = {
    "BaseAssetStrategy": BaseAssetStrategy,
    "BTCStrategy": BTCStrategy,
    "ETHStrategy": ETHStrategy,
}


class UltimateStrategy(Strategy):
    symbol = None
    vol_weight = 1.0
    strategy_class_override = None
    score_weights_tf_override = None
    ml_weights_override = None
    ml_weighted_threshold_override = None

    def init(self):
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)
        overrides = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {})
        self.tf_long_entry_threshold = overrides.get(
            "tf_long_entry_threshold", self.tf_long_entry_threshold
        )
        self.tf_short_entry_threshold = overrides.get(
            "tf_short_entry_threshold", self.tf_short_entry_threshold
        )
        class_name = self.strategy_class_override or overrides.get(
            "strategy_class", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(class_name, BaseAssetStrategy)(self)
        c, h, l = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()
        (
            self.market_regime,
            self.mtf_signal,
            self.advanced_ml_signal,
            self.legacy_ml_signal,
            self.v3_ml_signal,
        ) = (
            self.I(lambda: self.data.market_regime),
            self.I(lambda: self.data.mtf_signal),
            self.I(lambda: self.data.advanced_ml_signal),
            self.I(lambda: self.data.legacy_ml_signal),
            self.I(lambda: self.data.v3_ml_signal),
        )
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                h, l, c, self.tf_atr_period
            ).average_true_range()
        )
        self.tf_donchian_h, self.tf_donchian_l = self.I(
            lambda: h.rolling(self.tf_donchian_period).max().shift(1)
        ), self.I(lambda: l.rolling(self.tf_donchian_period).min().shift(1))
        self.tf_ema_fast, self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(c, self.tf_ema_fast_period).ema_indicator()
        ), self.I(
            lambda: ta.trend.EMAIndicator(c, self.tf_ema_slow_period).ema_indicator()
        )
        self.tf_adx = self.I(
            lambda: ta.trend.ADXIndicator(h, l, c, self.tf_adx_confirm_period).adx()
        )
        bb = ta.volatility.BollingerBands(c, self.mr_bb_period, self.mr_bb_std)
        self.mr_bb_upper, self.mr_bb_lower, self.mr_bb_mid = (
            self.I(lambda: bb.bollinger_hband()),
            self.I(lambda: bb.bollinger_lband()),
            self.I(lambda: bb.bollinger_mavg()),
        )
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(c, self.mr_rsi_period).rsi()
        )

    def next(self):
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            if "market_regime" in self.data.df.columns and self.market_regime[-1] == 1:
                self.run_scoring_system_entry(self.data.Close[-1])
            else:
                self.run_mean_reversion_entry(self.data.Close[-1])

    def run_scoring_system_entry(self, p):
        score = self.asset_strategy._calculate_entry_score()
        if score > self.tf_long_entry_threshold:
            self.open_tf_position(p, True, score)
        elif score < self.tf_short_entry_threshold:
            self.open_tf_position(p, False, abs(score))

    def run_mean_reversion_entry(self, p):
        signal = self.asset_strategy._define_mr_entry_signal()
        if signal != 0:
            self.open_mr_position(p, signal == 1)

    def reset_trade_state(self):
        (
            self.active_sub_strategy,
            self.chandelier_exit_level,
            self.highest_high_in_trade,
            self.lowest_low_in_trade,
            self.mr_stop_loss,
            self.tf_initial_stop_loss,
            self.trade_entry_bar,
        ) = (None, 0.0, 0, float("inf"), 0.0, 0.0, 0)

    def manage_open_position(self, p):
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(p)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(p)

    def open_tf_position(self, p, is_long, confidence):
        risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * confidence
        )
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        self.trade_entry_bar = len(self.data)
        if is_long:
            self.buy(size=size)
            self.tf_initial_stop_loss = p - risk_ps
        else:
            self.sell(size=size)
            self.tf_initial_stop_loss = p + risk_ps

    def manage_trend_following_exit(self, p):
        bars_in_trade = len(self.data) - self.trade_entry_bar
        if self.time_stop_bars > 0 and bars_in_trade > self.time_stop_bars:
            self.position.close()
            return
        if self.position.is_long:
            if p < self.tf_initial_stop_loss:
                self.position.close()
                return
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            exit_level = (
                self.highest_high_in_trade
                - self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
            if p < exit_level:
                self.position.close()
        elif self.position.is_short:
            if p > self.tf_initial_stop_loss:
                self.position.close()
                return
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            exit_level = (
                self.lowest_low_in_trade
                + self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
            if p > exit_level:
                self.position.close()

    def open_mr_position(self, p, is_long):
        risk_ps = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * self.mr_risk_multiplier
        )
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        if is_long:
            self.buy(size=size)
            self.mr_stop_loss = p - risk_ps
        else:
            self.sell(size=size)
            self.mr_stop_loss = p + risk_ps

    def manage_mean_reversion_exit(self, p):
        if (
            self.position.is_long
            and (p >= self.mr_bb_mid[-1] or p <= self.mr_stop_loss)
        ) or (
            self.position.is_short
            and (p <= self.mr_bb_mid[-1] or p >= self.mr_stop_loss)
        ):
            self.position.close()

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0 or risk_pct <= 0:
            return 0
        return int((self.equity * risk_pct) / rps)

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        returns = np.array(list(self.recent_trade_returns))
        if len(returns) < 2 or np.all(returns >= 0) or np.all(returns <= 0):
            return self.default_risk_pct * self.vol_weight
        win_rate, avg_win, avg_loss = (
            np.mean(returns > 0),
            np.mean(returns[returns > 0]),
            np.abs(np.mean(returns[returns < 0])),
        )
        if avg_loss == 0:
            return self.default_risk_pct * self.vol_weight
        reward_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


# --- 主程序入口 ---
if __name__ == "__main__":
    logger.info(
        f"🚀 (V60.3-Optimized-Fix with V3 Integration - Final Fix 6 Performance Tuning) 开始运行..."
    )
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
        analyze_v3_standalone_performance(data)
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
