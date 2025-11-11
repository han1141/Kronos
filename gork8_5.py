# -*- coding: utf-8 -*-

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

# 兼容性修复：部分三方库（如旧版 pandas_ta）会尝试 `from numpy import NaN`
# 在 NumPy 较新版本中仅保留 `numpy.nan`，因此这里确保模块属性 `NaN` 存在
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # 使 `from numpy import NaN` 可用

# 检查必要库；尽量避免对 pandas_ta 的硬依赖
ADVANCED_ML_LIBS_INSTALLED = True
TENSORFLOW_AVAILABLE = True
try:
    import numba
except ImportError as e:
    ADVANCED_ML_LIBS_INSTALLED = False
    # 提供一个空装饰器以保持函数可用
    class _NumbaShim:
        def jit(self, *args, **kwargs):
            def _decorator(f):
                return f
            return _decorator
    numba = _NumbaShim()
    print(f"警告: 缺少必要的库 ({e})，部分ML功能将不可用。")
try:
    from scipy.signal import find_peaks
except ImportError as e:
    ADVANCED_ML_LIBS_INSTALLED = False
    print(f"警告: 缺少必要的库 ({e})，部分ML功能将不可用。")
try:
    import tensorflow as tf
except Exception:
    TENSORFLOW_AVAILABLE = False


from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 日志与字体设置 (保持不变) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_filename = f"trading_log_integrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-11-10",
    "initial_cash": 500_000,
    "commission": 0.00085,
    "spread": 0.0002,
    "show_plots": False,
    "data_lookback_days": 100,
    # 仅保留“最新”的 LGBM 模型组件，停用旧的 Keras 组件
    "enable_keras_component": True,
    "enable_lgbm_component": True,
}

# 使用 LightGBM_4 训练的“震荡/趋势”分类模型
TREND_ARTIFACTS_PATH = "models/eth_trend_artifacts_15m.joblib"

# 尝试导入与训练完全一致的特征工程函数，保证一致性
try:
    from LightGBM_4 import feature_engineering as _trend_feat_eng
    from LightGBM_4 import LOOK_BACK as _TREND_LOOK_BACK
    _HAVE_TREND_FE = True
except Exception:
    _trend_feat_eng = None
    _TREND_LOOK_BACK = 60  # 训练脚本中为 60
    _HAVE_TREND_FE = False

# 缓存已加载的分类器工件
_TREND_ARTIFACTS_CACHE = None

def _load_trend_artifacts():
    global _TREND_ARTIFACTS_CACHE
    if _TREND_ARTIFACTS_CACHE is not None:
        return _TREND_ARTIFACTS_CACHE
    if not os.path.exists(TREND_ARTIFACTS_PATH):
        logger.warning(f"未找到趋势/震荡分类器工件: {TREND_ARTIFACTS_PATH}")
        _TREND_ARTIFACTS_CACHE = None
        return None
    try:
        arts = joblib.load(TREND_ARTIFACTS_PATH)
        # 关键组件: model, scaler, feature_columns, flattened_keep_indices, flattened_columns
        for k in ["model", "scaler", "feature_columns"]:
            if k not in arts:
                raise ValueError(f"artifacts 缺少关键键: {k}")
        _TREND_ARTIFACTS_CACHE = arts
        logger.info("✅ 已加载趋势/震荡分类模型工件。")
        return arts
    except Exception as e:
        logger.error(f"加载趋势/震荡分类器失败: {e}")
        _TREND_ARTIFACTS_CACHE = None
        return None

def _fallback_trend_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """在无法从 LightGBM_4 导入时，最小复刻训练时的特征工程。"""
    x = df.copy()
    try:
        x["RSI_14"] = ta.momentum.RSIIndicator(x["Close"], window=14).rsi()
        macd = ta.trend.MACD(x["Close"], window_fast=12, window_slow=26, window_sign=9)
        x["MACD_12_26_9"], x["MACDs_12_26_9"], x["MACDh_12_26_9"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        bb = ta.volatility.BollingerBands(x["Close"], window=20, window_dev=2)
        x["BBL_20_2"], x["BBM_20_2"], x["BBU_20_2"] = bb.bollinger_lband(), bb.bollinger_mavg(), bb.bollinger_hband()
        adx = ta.trend.ADXIndicator(x["High"], x["Low"], x["Close"], window=14)
        x["ADX_14"] = adx.adx()
    except Exception:
        # 保底：如果 ta 不可用，尽量回退到空特征，后续会用 0 填充
        pass
    x["volatility"] = (np.log(x["Close"] / x["Close"].shift(1))).rolling(20).std()
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.dropna(inplace=True)
    return x

def _predict_trend_ranging_series(df_ohlcv: pd.DataFrame) -> pd.Series:
    """
    使用保存的工件对给定 OHLCV 数据进行滑窗推断，输出标签序列：
      - 0 => Trending, 1 => Ranging
    返回的索引与 df_ohlcv 对齐（前 _TREND_LOOK_BACK 行无标签）。
    """
    arts = _load_trend_artifacts()
    if arts is None or df_ohlcv is None or df_ohlcv.empty:
        return pd.Series(dtype=float)

    model = arts["model"]
    scaler = arts["scaler"]
    feat_cols = arts["feature_columns"]
    keep_idx = arts.get("flattened_keep_indices")
    flat_cols = arts.get("flattened_columns")

    # 计算与训练一致的特征
    if _HAVE_TREND_FE and _trend_feat_eng is not None:
        Xdf = _trend_feat_eng(df_ohlcv)
    else:
        Xdf = _fallback_trend_feature_engineering(df_ohlcv)

    # 对齐列顺序；缺失列以0填充（通常不会发生）
    if set(Xdf.columns) != set(feat_cols):
        extra = [c for c in Xdf.columns if c not in feat_cols]
        missing = [c for c in feat_cols if c not in Xdf.columns]
        if extra:
            logger.warning(f"趋势特征包含训练未用列: {extra[:10]}{'...' if len(extra)>10 else ''}")
        if missing:
            logger.warning(f"趋势特征缺失训练所需列: {missing[:10]}{'...' if len(missing)>10 else ''}")
    Xdf = Xdf.reindex(columns=feat_cols, fill_value=0)

    # 归一化
    X_np = Xdf.values
    try:
        X_scaled = scaler.transform(X_np)
    except Exception as e:
        logger.error(f"趋势/震荡分类器缩放失败: {e}")
        return pd.Series(dtype=float)

    if len(X_scaled) <= _TREND_LOOK_BACK:
        return pd.Series(dtype=float)

    preds = []
    idx = Xdf.index[_TREND_LOOK_BACK:]
    for i in range(_TREND_LOOK_BACK, len(X_scaled)):
        seq = X_scaled[i - _TREND_LOOK_BACK : i].flatten().reshape(1, -1)
        if keep_idx is not None:
            seq = seq[:, keep_idx]
        # LightGBM 在训练时保存了扁平列名（可选），优先保持一致
        if flat_cols is not None:
            in_df = pd.DataFrame(seq, columns=flat_cols)
            yhat = model.predict(in_df)[0]
        else:
            yhat = model.predict(seq)[0]
        preds.append(int(yhat))

    return pd.Series(preds, index=idx, name="trend_range_label")

# --- Keras模型文件路径配置 ---
KERAS_MODEL_PATH = "models/eth_trend_model_v1_15m.keras"
KERAS_SCALER_PATH = "models/eth_trend_scaler_v1_15m.joblib"
KERAS_FEATURE_COLUMNS_PATH = "models/feature_columns_15m.joblib"
KERAS_SEQUENCE_LENGTH = 60

# --- LightGBM模型文件路径配置 ---
LGBM_MODELS_DIR = "models_gbm2"
LGBM_MODEL_PATH = os.path.join(
    LGBM_MODELS_DIR, "eth_model_high_precision_v4_15m.joblib"
)
LGBM_SCALER_PATH = os.path.join(
    LGBM_MODELS_DIR, "eth_scaler_high_precision_v4_15m.joblib"
)
LGBM_FEATURE_COLUMNS_PATH = os.path.join(
    LGBM_MODELS_DIR, "feature_columns_high_precision_v4_15m.joblib"
)
LGBM_FLATTENED_COLUMNS_PATH = os.path.join(
    LGBM_MODELS_DIR, "flattened_columns_high_precision_v4_15m.joblib"
)
LGBM_LOOK_BACK_PERIOD = 60

# --- 策略参数 ---
STRATEGY_PARAMS = {
    "tsl_enabled": True,
    "tsl_activation_profit_pct": 0.005,
    "tsl_activation_atr_mult": 1.5,
    "tsl_trailing_atr_mult": 2.0,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "kelly_trade_history": 20,
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
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mr_rsi_period": 14,
    "mr_rsi_buy": 35,
    "mr_rsi_sell": 65,
    "mr_rsi_exit": 50,
    # 每日交易频控
    "daily_max_entries": 1,
    "mtf_period": 50,
    "score_entry_threshold": 0.4,
    # 仅保留最新的 LGBM 信号参与打分，Keras/advanced 设为 0
    # 确保权重总和≈1，阈值尺度保持稳定
    "score_weights_tf": {
        "breakout": 0.25,
        "momentum": 0.15,
        "mtf": 0.05,
        "keras_ml": 0.0,
        "lgbm_ml": 0.55,
        "advanced_ml": 0.0,
    },
    # LGBM 概率阈值（来自验证集最佳F1阈值）
    "lgbm_proba_threshold": 0.45,
    "lgbm_use_symmetric": True,
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETHUSDT": {"strategy_class": "ETHStrategy", "score_entry_threshold": 0.45},
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


@numba.jit(nopython=True, cache=True)
def compute_hurst_numba(ts):
    # Numba-friendly Hurst estimator without np.polyfit
    n = len(ts)
    if n < 100:
        return 0.5
    max_lag = 100
    lags = np.arange(2, max_lag)
    m = len(lags)
    tau = np.empty(m, dtype=np.float64)
    for i in range(m):
        lag = lags[i]
        diff = ts[lag:] - ts[: n - lag]
        if diff.size > 1:
            mu = np.mean(diff)
            var = np.mean(diff * diff) - mu * mu
            if var > 0:
                tau[i] = np.sqrt(var)
            else:
                tau[i] = 0.0
        else:
            tau[i] = 0.0

    # Filter valid entries
    valid_count = 0
    for i in range(m):
        if tau[i] > 0:
            valid_count += 1
    if valid_count < 2:
        return 0.5

    log_lags = np.empty(valid_count, dtype=np.float64)
    log_tau = np.empty(valid_count, dtype=np.float64)
    j = 0
    for i in range(m):
        if tau[i] > 0:
            log_lags[j] = np.log(lags[i])
            log_tau[j] = np.log(tau[i])
            j += 1

    # Manual linear regression slope
    mean_x = np.mean(log_lags)
    mean_y = np.mean(log_tau)
    num = np.mean(log_lags * log_tau) - mean_x * mean_y
    den = np.mean(log_lags * log_lags) - mean_x * mean_x
    if den == 0.0:
        return 0.5
    return (num / den) * 2.0


def get_market_structure_features(df, order=5):
    df_copy = df.copy()
    high_peaks_idx, _ = find_peaks(df_copy["High"], distance=order)
    low_peaks_idx, _ = find_peaks(-df_copy["Low"], distance=order)
    df_copy["swing_high_price"] = np.nan
    df_copy.iloc[high_peaks_idx, df_copy.columns.get_loc("swing_high_price")] = (
        df_copy.iloc[high_peaks_idx]["High"]
    )
    df_copy["swing_low_price"] = np.nan
    df_copy.iloc[low_peaks_idx, df_copy.columns.get_loc("swing_low_price")] = (
        df_copy.iloc[low_peaks_idx]["Low"]
    )
    df_copy["swing_high_price"] = df_copy["swing_high_price"].ffill()
    df_copy["swing_low_price"] = df_copy["swing_low_price"].ffill()
    df_copy["market_structure"] = (
        (df_copy["swing_high_price"] > df_copy["swing_high_price"].shift(1))
        & (df_copy["swing_low_price"] > df_copy["swing_low_price"].shift(1))
    ).astype(int) - (
        (df_copy["swing_high_price"] < df_copy["swing_high_price"].shift(1))
        & (df_copy["swing_low_price"] < df_copy["swing_low_price"].shift(1))
    ).astype(
        int
    )
    return df_copy[["market_structure"]]


# --- 特征工程 (融合版) ---
def add_features_for_lgbm_model(df: pd.DataFrame) -> pd.DataFrame:
    # 在降级模式下，不返回原始 df 以避免后续 concat 产生重复列
    if not ADVANCED_ML_LIBS_INSTALLED:
        return pd.DataFrame(index=df.index)
    logger.info("正在为 LGBM 模型生成特定特征...")
    df_copy = df.copy()
    close = df_copy["Close"]
    high = df_copy["High"]
    low = df_copy["Low"]
    volume = df_copy["Volume"]

    # 基础指标（命名与训练阶段保持一致）
    try:
        df_copy["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    except Exception:
        pass
    try:
        macd_12_26_9 = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
        df_copy["MACD_12_26_9"] = macd_12_26_9.macd()
        df_copy["MACDs_12_26_9"] = macd_12_26_9.macd_signal()
        df_copy["MACDh_12_26_9"] = macd_12_26_9.macd_diff()
    except Exception:
        pass
    try:
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
        # 无后缀版本（保留兼容）
        df_copy["BBL_20_2"] = bb.bollinger_lband()
        df_copy["BBM_20_2"] = bb.bollinger_mavg()
        df_copy["BBU_20_2"] = bb.bollinger_hband()
        # 训练所用的带 .0 后缀版本
        df_copy["BBL_20_2.0"] = df_copy["BBL_20_2"]
        df_copy["BBM_20_2.0"] = df_copy["BBM_20_2"]
        df_copy["BBU_20_2.0"] = df_copy["BBU_20_2"]
        df_copy["BBB_20_2.0"] = bb.bollinger_wband()
        df_copy["BBP_20_2.0"] = bb.bollinger_pband()
    except Exception:
        pass
    try:
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        df_copy["ADX_14"] = adx.adx()
        # 训练特征包含 DMP_14/DMN_14（正/负方向指标）
        df_copy["DMP_14"] = adx.adx_pos()
        df_copy["DMN_14"] = adx.adx_neg()
    except Exception:
        pass
    try:
        atr_raw = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        df_copy["ATR_14"] = atr_raw
    except Exception:
        pass
    try:
        df_copy["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    except Exception:
        pass
    try:
        macd_long = ta.trend.MACD(close=close, window_fast=24, window_slow=52, window_sign=18)
        df_copy["MACD_long"] = macd_long.macd()
        df_copy["MACDs_long"] = macd_long.macd_signal()
        df_copy["MACDh_long"] = macd_long.macd_diff()
    except Exception:
        pass
    market_structure_df = get_market_structure_features(df_copy)
    # 避免未来信息：峰谷识别需要至少确认下一根，统一右移一根
    if "market_structure" in market_structure_df.columns:
        market_structure_df["market_structure"] = market_structure_df["market_structure"].shift(1)
    df_copy["hurst"] = (
        df_copy["Close"].rolling(window=100).apply(compute_hurst_numba, raw=True)
    )
    df_copy["volatility_log"] = (
        (np.log(df_copy["Close"] / df_copy["Close"].shift(1))).rolling(window=20).std()
    )
    close_4h = df_copy["Close"].resample("4h").last()
    try:
        ema_4h = ta.trend.EMAIndicator(close_4h, window=50).ema_indicator()
    except Exception:
        ema_4h = close_4h.ewm(span=50, adjust=False).mean()
    # 避免在当前4小时区间内提前看到该区间的EMA，整体右移一根4小时K
    ema_4h = ema_4h.shift(1)
    df_copy["ema_4h"] = ema_4h.reindex(df_copy.index, method="ffill")
    df_copy["price_above_ema_4h"] = (df_copy["Close"] > df_copy["ema_4h"]).astype(int)
    # Cross and interaction features expected by LGBM
    df_copy["macd_cross_signal"] = (
        df_copy["MACD_12_26_9"] > df_copy["MACDs_12_26_9"]
    ).astype(int)
    df_copy["macd_long_cross_signal"] = (
        df_copy["MACD_long"] > df_copy["MACDs_long"]
    ).astype(int)
    df_copy["macd_cross_confirm"] = (
        df_copy["macd_cross_signal"] * df_copy["macd_long_cross_signal"]
    )
    # Bearish confirmations for symmetric gating
    df_copy["macd_cross_signal_bear"] = (
        df_copy["MACD_12_26_9"] < df_copy["MACDs_12_26_9"]
    ).astype(int)
    df_copy["macd_long_cross_signal_bear"] = (
        df_copy["MACD_long"] < df_copy["MACDs_long"]
    ).astype(int)
    df_copy["macd_cross_confirm_bear"] = (
        df_copy["macd_cross_signal_bear"] * df_copy["macd_long_cross_signal_bear"]
    )
    df_copy["adx_x_atr_norm"] = (df_copy["ADX_14"] / 50) * (
        df_copy["ATR_14"] / df_copy["Close"]
    )
    df_copy["rsi_x_hurst"] = df_copy["RSI_14"] * df_copy["hurst"]
    df_copy = pd.concat([df_copy, market_structure_df], axis=1)
    new_cols = set(df_copy.columns) - set(df.columns)
    return df_copy[list(new_cols)]


def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("正在为 Keras 模型生成特定特征 (使用 'ta' 库)...")
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    df["EMA_8"] = ta.trend.EMAIndicator(close=close, window=8).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_14"] = adx_indicator.adx()
    df["DMP_14"] = adx_indicator.adx_pos()
    df["DMN_14"] = adx_indicator.adx_neg()
    atr_raw = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    df["ATRr_14"] = (atr_raw / close) * 100
    bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
    df["BBU_20_2.0"] = bb_indicator.bollinger_hband()
    df["BBM_20_2.0"] = bb_indicator.bollinger_mavg()
    df["BBL_20_2.0"] = bb_indicator.bollinger_lband()
    df["BBB_20_2.0"] = bb_indicator.bollinger_wband()
    df["BBP_20_2.0"] = bb_indicator.bollinger_pband()
    macd_indicator = ta.trend.MACD(
        close=close, window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD_12_26_9"] = macd_indicator.macd()
    df["MACDs_12_26_9"] = macd_indicator.macd_signal()
    df["MACDh_12_26_9"] = macd_indicator.macd_diff()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()
    df["volume_change_rate"] = volume.pct_change()
    return df

# 旧的启发式 regime 特征已移除；不再计算相关分数


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """仅计算与波动相关的派生特征（参考/日志用）。"""
    close_col = df["Close"]
    if isinstance(close_col, pd.DataFrame):
        # 若存在重复的 "Close" 列名，取第一列以避免赋值报错
        close_col = close_col.iloc[:, 0]
    vol = close_col.pct_change().rolling(24 * 7).std() * np.sqrt(24 * 365)
    df["volatility"] = vol
    low_vol, high_vol = df["volatility"].quantile(0.33), df["volatility"].quantile(0.67)
    df["volatility_regime"] = pd.cut(
        df["volatility"],
        bins=[0, low_vol, high_vol, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    return df


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(f"[{symbol}] 开始数据预处理 (集成版)...")

    rsi_filter = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["ai_filter_signal"] = (
        (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
        .clip(-1, 1)
        .fillna(0)
    )

    # 完全以训练好的模型输出为准（不再使用启发式 regime 分数）
    if CONFIG.get("enable_keras_component", False):
        df = add_features_for_keras_model(df)

    lgbm_features = add_features_for_lgbm_model(df)
    df = pd.concat([df, lgbm_features], axis=1)

    # 先计算基础的波动率等派生特征
    df = add_market_regime_features(df)

    # 使用训练好的“震荡/趋势”分类器替换当前 regime 识别
    try:
        pred_sr = _predict_trend_ranging_series(df[["Open", "High", "Low", "Close", "Volume"]])
        if not pred_sr.empty:
            # 为避免前视偏差，所有标签在使用前统一右移一根K线
            pred_safe = pred_sr.shift(1)
            df.loc[pred_safe.index, "trend_range_label"] = pred_safe
            df.loc[pred_safe.index, "trend_regime"] = np.where(pred_safe == 0, "Trending", "Mean-Reverting")
            df.loc[pred_safe.index, "market_regime"] = np.where(pred_safe == 0, 1, -1)
        else:
            # 保持列存在，取中性值，避免后续引用报错
            df["market_regime"] = 0
    except Exception as e:
        logger.error(f"[{symbol}] 应用趋势/震荡分类器失败: {e}")
        # 失败时也保证列存在
        df["market_regime"] = 0

    # 计算日线 MTF 信号（与 gork8_3 对齐）
    daily_start = df.index.min().normalize() - pd.Timedelta(
        days=STRATEGY_PARAMS["mtf_period"] + 5
    )
    daily_end = df.index.max().normalize()
    data_1d = fetch_binance_klines(
        symbol, "1d", daily_start.strftime("%Y-%m-%d"), daily_end.strftime("%Y-%m-%d")
    )
    if not data_1d.empty:
        sma_1d = ta.trend.SMAIndicator(
            data_1d["Close"], window=STRATEGY_PARAMS["mtf_period"]
        ).sma_indicator()
        mtf_signal_1d = pd.Series(
            np.where(data_1d["Close"] > sma_1d, 1, -1), index=data_1d.index
        )
        # 避免日线当日未收盘时使用当日信息：整体右移一天
        mtf_signal_1d = mtf_signal_1d.shift(1)
        df["mtf_signal"] = mtf_signal_1d.reindex(df.index, method="ffill").fillna(0)
    else:
        df["mtf_signal"] = 0

    df_4h = df["Close"].resample("4h").last().to_frame()
    df_4h["macro_ema"] = ta.trend.EMAIndicator(
        df_4h["Close"], window=50
    ).ema_indicator()
    df_4h["macro_trend"] = np.where(df_4h["Close"] > df_4h["macro_ema"], 1, -1)
    df["macro_trend_filter"] = (
        df_4h["macro_trend"].reindex(df.index, method="ffill").fillna(0)
    )

    df.dropna(inplace=True)
    # 统计市场状态占比（来自模型识别）
    if "trend_range_label" in df.columns:
        cnt = df.loc[:, "trend_range_label"].map({0: "Trending", 1: "Ranging"}).value_counts()
        tr = float(cnt.get("Trending", 0))
        rg = float(cnt.get("Ranging", 0))
        total = tr + rg if (tr + rg) > 0 else 1
        logger.info(f"[{symbol}] 模型市场识别: Trending={tr/total*100:.2f}% Ranging={rg/total*100:.2f}% | 行数={len(df)}")
    else:
        logger.info(f"[{symbol}] 未能生成模型市场识别标签。")
    logger.info(f"[{symbol}] 集成数据预处理完成。数据行数: {len(df)}")
    return df


def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    sequences = []
    data_len = len(data)
    for i in range(data_len - sequence_length + 1):
        sequences.append(data[i : i + sequence_length])
    return np.array(sequences)


class BaseAssetStrategy:
    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy

    def _calculate_entry_score(self) -> float:
        m, w = self.main, self.main.score_weights_tf
        b_s = (
            1
            if m.data.High[-1] > m.tf_donchian_h[-1]
            else -1 if m.data.Low[-1] < m.tf_donchian_l[-1] else 0
        )
        mo_s = 1 if m.tf_ema_fast[-1] > m.tf_ema_slow[-1] else -1
        keras_ml_signal = m.keras_signal[-1] if hasattr(m, "keras_signal") else 0.0
        lgbm_ml_signal = m.lgbm_signal[-1] if hasattr(m, "lgbm_signal") else 0.0
        adv_ml_signal = (
            m.ai_filter_signal[-1] if hasattr(m, "ai_filter_signal") else 0.0
        )

        score = (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + m.mtf_signal[-1] * w.get("mtf", 0)
            + keras_ml_signal * w.get("keras_ml", 0)
            + lgbm_ml_signal * w.get("lgbm_ml", 0)
            + adv_ml_signal * w.get("advanced_ml", 0)
        )
        return score

    def _define_mr_entry_signal(self) -> int:
        return 0


class ETHStrategy(BaseAssetStrategy):
    pass


STRATEGY_MAPPING = {"BaseAssetStrategy": BaseAssetStrategy, "ETHStrategy": ETHStrategy}


class UltimateStrategy(Strategy):
    symbol = None
    vol_weight = 1.0
    # ...

    def init(self):
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)

        strategy_class_name = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get(
            "strategy_class", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(
            strategy_class_name, BaseAssetStrategy
        )(self)

        # 仅在启用时加载 Keras
        if CONFIG.get("enable_keras_component", False):
            self.keras_model, self.keras_scaler, self.keras_feature_columns = (
                self._load_keras_model_and_dependencies()
            )
            self.keras_signal = self.I(self._calculate_keras_predictions)

        (
            self.lgbm_model,
            self.lgbm_scaler,
            self.lgbm_feature_columns,
            self.lgbm_flattened_columns,
        ) = self._load_lgbm_model_and_dependencies()
        self.lgbm_signal = self.I(self._calculate_lgbm_predictions)

        self.market_regime = self.I(lambda: self.data.market_regime)
        self.macro_trend = self.I(lambda: self.data.macro_trend_filter)
        self.ai_filter_signal = self.I(lambda: self.data.ai_filter_signal)
        # Provide MTF signal if present; else fallback to zeros aligned to index
        self.mtf_signal = self.I(
            lambda: (self.data.df["mtf_signal"]
                     if "mtf_signal" in self.data.df.columns
                     else pd.Series(0, index=self.data.df.index))
        )
        # Trend-following ATR
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                pd.Series(self.data.High),
                pd.Series(self.data.Low),
                pd.Series(self.data.Close),
                self.tf_atr_period,
            ).average_true_range()
        )

        self.tf_donchian_h = self.I(
            lambda: pd.Series(self.data.High)
            .rolling(self.tf_donchian_period)
            .max()
            .shift(1)
        )
        self.tf_donchian_l = self.I(
            lambda: pd.Series(self.data.Low)
            .rolling(self.tf_donchian_period)
            .min()
            .shift(1)
        )
        self.tf_ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(
                pd.Series(self.data.Close), self.tf_ema_fast_period
            ).ema_indicator()
        )
        self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(
                pd.Series(self.data.Close), self.tf_ema_slow_period
            ).ema_indicator()
        )

        # ADX confirm (used by some asset overrides)
        self.tf_adx = self.I(
            lambda: ta.trend.ADXIndicator(
                pd.Series(self.data.High),
                pd.Series(self.data.Low),
                pd.Series(self.data.Close),
                self.tf_adx_confirm_period,
            ).adx()
        )

        # Indicators for mean-reversion (震荡) regime
        def _bb_upper():
            bb = ta.volatility.BollingerBands(
                pd.Series(self.data.Close),
                window=self.mr_bb_period,
                window_dev=self.mr_bb_std,
            )
            return bb.bollinger_hband()

        def _bb_middle():
            bb = ta.volatility.BollingerBands(
                pd.Series(self.data.Close),
                window=self.mr_bb_period,
                window_dev=self.mr_bb_std,
            )
            return bb.bollinger_mavg()

        def _bb_lower():
            bb = ta.volatility.BollingerBands(
                pd.Series(self.data.Close),
                window=self.mr_bb_period,
                window_dev=self.mr_bb_std,
            )
            return bb.bollinger_lband()

        self.tf_bb_upper = self.I(_bb_upper)
        self.tf_bb_middle = self.I(_bb_middle)
        self.tf_bb_lower = self.I(_bb_lower)
        self.tf_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(
                pd.Series(self.data.Close), self.mr_rsi_period
            ).rsi()
        )

        # StochRSI for MR confirmations
        stoch_rsi = ta.momentum.StochRSIIndicator(
            pd.Series(self.data.Close), window=14, smooth1=3, smooth2=3
        )
        self.mr_stoch_rsi_k = self.I(lambda: stoch_rsi.stochrsi_k())
        self.mr_stoch_rsi_d = self.I(lambda: stoch_rsi.stochrsi_d())

        # Dynamic risk memory
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()

    # === 环境度量与自适应控制 ===
    def _compute_env_metrics(self):
        try:
            price = float(self.data.Close[-1])
        except Exception:
            price = 0.0
        adx = float(self.tf_adx[-1]) if len(self.tf_adx) else 0.0
        atr = float(self.tf_atr[-1]) if len(self.tf_atr) else 0.0
        vol_norm = (atr / price) if price > 0 else 0.0  # 归一化波动率 ~ ATR/Close
        regime = int(self.market_regime[-1]) if len(self.market_regime) else 0
        macro = int(self.macro_trend[-1]) if len(self.macro_trend) else 0
        return {"price": price, "adx": adx, "vol_norm": vol_norm, "regime": regime, "macro": macro}

    def _recent_perf(self, n=10):
        if not hasattr(self, "recent_trade_returns") or len(self.recent_trade_returns) == 0:
            return 0.0
        arr = list(self.recent_trade_returns)[-n:]
        return float(np.mean(arr)) if len(arr) else 0.0

    def _dynamic_controls(self):
        m = self._compute_env_metrics()
        base_thr = float(self.score_entry_threshold)
        delta = 0.0

        strong_trend = (m["regime"] == 1) and (m["adx"] >= (self.tf_adx_confirm_threshold + 5))
        weak_trend = (m["adx"] <= 15) or (m["regime"] == -1)
        high_vol = m["vol_norm"] > 0.02  # >2% ATR/Close 视为高波动

        if strong_trend:
            delta -= 0.10  # 趋势强，放宽入场
        if weak_trend:
            delta += 0.10  # 趋势弱/震荡，抬高门槛
        if high_vol:
            delta += 0.05  # 高波动，进一步抬高门槛以减少噪声交易

        # 基于近期表现进行自适应微调（避免连续亏损时过度交易）
        perf = self._recent_perf(n=10)
        if perf < -0.005:  # 近10笔均值 < -0.5%
            delta += 0.05
        elif perf > 0.005:  # 近10笔均值 > +0.5%
            delta -= 0.03

        score_thr = float(np.clip(base_thr + delta, 0.35, 0.8))

        # 动态 TSL 与风险比例
        if strong_trend:
            tsl_act_pct = 0.8  # 0.8% 才激活追踪
            tsl_tr_atr = 3.5   # 更宽的追踪距离
            tf_risk_scale = 1.25
        elif weak_trend:
            tsl_act_pct = 1.2
            tsl_tr_atr = 2.5
            tf_risk_scale = 0.75
        else:
            tsl_act_pct = 1.0
            tsl_tr_atr = 3.0
            tf_risk_scale = 1.0

        if high_vol:
            tsl_tr_atr = max(tsl_tr_atr, 4.0)  # 高波动下进一步放宽追踪
            tf_risk_scale *= 0.85              # 降低风险敞口

        # MR 在震荡下适度启动，趋势下降低其权重
        mr_risk_scale = 0.9 if weak_trend else 0.6

        # 极端震荡 + 高波动时，限制当日入场次数
        daily_entries = 0 if (weak_trend and high_vol) else self.daily_max_entries

        return {
            "score_entry_threshold": score_thr,
            "tsl_activation_profit_pct": float(tsl_act_pct),
            "tsl_trailing_atr_mult": float(tsl_tr_atr),
            "tf_risk_scale": float(tf_risk_scale),
            "mr_risk_scale": float(mr_risk_scale),
            "daily_max_entries": int(daily_entries),
        }

    def _load_keras_model_and_dependencies(self):
        if not CONFIG["enable_keras_component"] or not ADVANCED_ML_LIBS_INSTALLED:
            logger.warning(f"[{self.symbol}] Keras模型组件已禁用或TensorFlow未安装。")
            return None, None, None
        try:
            model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            scaler = joblib.load(KERAS_SCALER_PATH)
            feature_columns = joblib.load(KERAS_FEATURE_COLUMNS_PATH)
            logger.info(f"✅ [{self.symbol}] 成功加载Keras模型、缩放器和特征列。")
            return model, scaler, feature_columns
        except Exception as e:
            logger.error(f"[{self.symbol}] 加载Keras模型或依赖项失败: {e}")
            return None, None, None

    def _calculate_keras_predictions(self):
        if self.keras_model is None:
            return np.zeros(len(self.data.Close))
        features_df = self.data.df[self.keras_feature_columns].fillna(0)
        scaled_features_2d = self.keras_scaler.transform(features_df)
        scaled_features_3d = create_sequences(scaled_features_2d, KERAS_SEQUENCE_LENGTH)
        if scaled_features_3d.shape[0] == 0:
            return np.zeros(len(self.data.Close))
        predictions_proba = self.keras_model.predict(
            scaled_features_3d, verbose=0
        ).flatten()
        padding_size = len(self.data.Close) - len(predictions_proba)
        raw_signals = (predictions_proba - 0.5) * 2
        return np.pad(raw_signals, (padding_size, 0), "constant")

    def _load_lgbm_model_and_dependencies(self):
        if not CONFIG["enable_lgbm_component"]:
            logger.warning(f"[{self.symbol}] LightGBM模型组件已禁用。")
            return None, None, None, None
        try:
            model = joblib.load(LGBM_MODEL_PATH)
            scaler = joblib.load(LGBM_SCALER_PATH)
            feature_cols = joblib.load(LGBM_FEATURE_COLUMNS_PATH)
            flattened_cols = joblib.load(LGBM_FLATTENED_COLUMNS_PATH)
            logger.info(f"✅ [{self.symbol}] 成功加载LightGBM模型及依赖项。")
            return model, scaler, feature_cols, flattened_cols
        except Exception as e:
            logger.error(f"[{self.symbol}] 加载LightGBM模型失败: {e}")
            return None, None, None, None

    def _calculate_lgbm_predictions(self):
        if self.lgbm_model is None:
            return np.zeros(len(self.data.Close))
        predictions = np.full(len(self.data.Close), 0.0)
        probas = np.full(len(self.data.Close), np.nan)
        logger.info(f"[{self.symbol}] 开始计算LGBM预测信号...")
        # Ensure all expected columns exist; create placeholders if needed
        df_all = self.data.df.copy()
        missing = [c for c in self.lgbm_feature_columns if c not in df_all.columns]
        if missing:
            logger.warning(
                f"[{self.symbol}] LGBM缺失特征列 {missing}，将以0填充占位。"
            )
            for c in missing:
                df_all[c] = 0.0
        feature_df_full = df_all[self.lgbm_feature_columns].fillna(0)
        feature_data = feature_df_full.to_numpy()
        thr = float(STRATEGY_PARAMS.get("lgbm_proba_threshold", 0.5))
        use_sym = bool(STRATEGY_PARAMS.get("lgbm_use_symmetric", False))
        thr_short = 1.0 - thr
        # 可选: 结合训练阶段使用的 MACD 过滤器，减少噪声触发
        macd_confirm = (
            df_all["macd_cross_confirm"].to_numpy()
            if "macd_cross_confirm" in df_all.columns
            else np.zeros(len(df_all), dtype=float)
        )
        macd_confirm_bear = (
            df_all["macd_cross_confirm_bear"].to_numpy()
            if "macd_cross_confirm_bear" in df_all.columns
            else ((df_all.get("MACD_12_26_9", pd.Series(0, index=df_all.index)) < df_all.get("MACDs_12_26_9", pd.Series(0, index=df_all.index))).astype(int)
                  * (df_all.get("MACD_long", pd.Series(0, index=df_all.index)) < df_all.get("MACDs_long", pd.Series(0, index=df_all.index))).astype(int))
        )
        for i in range(LGBM_LOOK_BACK_PERIOD, len(self.data.Close)):
            features_np = feature_data[i - LGBM_LOOK_BACK_PERIOD : i]
            # Preserve feature names during scaling to avoid sklearn warnings
            df_window = pd.DataFrame(
                features_np, columns=self.lgbm_feature_columns
            )
            scaled_features = self.lgbm_scaler.transform(df_window)
            # Build a 1xK DataFrame with flattened feature names expected by the model
            flat = scaled_features.flatten(order="C").reshape(1, -1)
            df_flat = pd.DataFrame(flat, columns=self.lgbm_flattened_columns)
            pred_prob = self.lgbm_model.predict_proba(df_flat)[0][1]
            probas[i] = pred_prob
            # 对称信号：多头与空头
            if pred_prob >= thr and (macd_confirm[i] > 0):
                predictions[i] = (pred_prob - thr) / (1.0 - thr)
            elif use_sym and pred_prob <= thr_short and (macd_confirm_bear[i] > 0):
                predictions[i] = -((thr_short - pred_prob) / max(thr_short, 1e-6))
            else:
                predictions[i] = 0.0
        # 统计信息，帮助诊断“结果为何不变”
        active = np.isfinite(probas)
        pos_mask = (probas >= thr) & active
        neg_mask = (probas <= thr_short) & active
        macd_mask = (macd_confirm > 0)
        macd_bear_mask = (macd_confirm_bear > 0)
        gated_pos = pos_mask & macd_mask
        gated_neg = neg_mask & macd_bear_mask
        total = int(active.sum())
        if total > 0:
            logger.info(
                f"[{self.symbol}] LGBM概率统计: 样本={total}, 多>=阈值={int(pos_mask.sum())} ({pos_mask.sum()/total:.2%}), "
                f"空<=阈值={int(neg_mask.sum())} ({neg_mask.sum()/total:.2%}), 多且MACD={int(gated_pos.sum())} ({gated_pos.sum()/total:.2%}), 空且MACD={int(gated_neg.sum())} ({gated_neg.sum()/total:.2%}), "
                f"均值={np.nanmean(probas):.4f}, 中位数={np.nanmedian(probas):.4f}"
            )
        logger.info(f"[{self.symbol}] LGBM预测信号计算完成。")
        return predictions

    def _calculate_position_size(self, p, rps, risk_pct):
        """
        基于风险百分比的单位头寸计算（参考 gork8_3）。
        p: 当前价格; rps: 每单位风险(如 ATR*倍数); risk_pct: 承担的权益比例
        """
        if rps <= 0 or p <= 0:
            return 0
        risk_amount = self.equity * min(max(float(risk_pct), 0.0), self.max_risk_pct)
        units = risk_amount / rps
        # 资金约束
        if units * p > self.equity:
            units = (self.equity * 0.95) / p
        return max(1, int(units))

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        if avg_loss == 0:
            return self.default_risk_pct * self.vol_weight
        rr = avg_win / avg_loss if avg_loss else 0
        if rr == 0:
            return self.default_risk_pct * self.vol_weight
        kelly = win_rate - (1 - win_rate) / rr
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)

    # === 交易频率控制：每日至多 N 次入场 ===
    def _update_day_counter(self):
        try:
            cur_ts = self.data.df.index[-1]
        except Exception:
            return
        cur_day = cur_ts.date()
        if getattr(self, "_current_day", None) != cur_day:
            self._current_day = cur_day
            self._entries_today = 0

    def _can_enter_today(self) -> bool:
        self._update_day_counter()
        if not hasattr(self, "_entries_today"):
            self._entries_today = 0
        return self._entries_today < self.daily_max_entries

    def _mark_entered(self):
        self._update_day_counter()
        if not hasattr(self, "_entries_today"):
            self._entries_today = 0
        self._entries_today += 1

    def _manage_trailing_stop_loss(self):
        if not self.tsl_enabled or not self.position:
            return
        entry_price = self.trades[-1].entry_price
        current_price = self.data.Close[-1]
        # Activation by profit pct or ATR move
        if not getattr(self, "trailing_stop_active", False):
            profit_pct_met = (self.position.pl_pct * 100) > self.tsl_activation_profit_pct
            atr = self.tf_atr[-1]
            move_met = False
            if self.position.is_long and current_price >= entry_price + atr * self.tsl_activation_atr_mult:
                move_met = True
            elif self.position.is_short and current_price <= entry_price - atr * self.tsl_activation_atr_mult:
                move_met = True
            if profit_pct_met or move_met:
                self.trailing_stop_active = True
        if self.trailing_stop_active:
            atr = self.tf_atr[-1]
            distance = atr * self.tsl_trailing_atr_mult
            if self.position.is_long:
                new_sl = current_price - distance
                if new_sl > self.stop_loss_price:
                    self.stop_loss_price = new_sl
            else:
                new_sl = current_price + distance
                if new_sl < self.stop_loss_price:
                    self.stop_loss_price = new_sl

    def manage_trend_following_exit(self, p):
        if self.position.is_long:
            if not hasattr(self, "highest_high_in_trade"):
                self.highest_high_in_trade = 0.0
            self.highest_high_in_trade = max(self.highest_high_in_trade, self.data.High[-1])
            chandelier = self.highest_high_in_trade - self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            final_sl = max(self.stop_loss_price, chandelier)
            if p < final_sl:
                self.close_position("TF_Exit")
        else:
            if not hasattr(self, "lowest_low_in_trade"):
                self.lowest_low_in_trade = float("inf")
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            chandelier = self.lowest_low_in_trade + self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            final_sl = min(self.stop_loss_price, chandelier)
            if p > final_sl:
                self.close_position("TF_Exit")

    def manage_mean_reversion_exit(self, p):
        mid = float(self.tf_bb_middle[-1]) if np.isfinite(self.tf_bb_middle[-1]) else p
        if (self.position.is_long and (p >= mid or p <= self.stop_loss_price)) or (
            self.position.is_short and (p <= mid or p >= self.stop_loss_price)
        ):
            self.close_position("MR_Exit")

    def close_position(self, reason: str):
        eq_before = self.equity
        self.position.close()
        try:
            self.recent_trade_returns.append(self.equity / eq_before - 1)
        except Exception:
            pass
        self.reset_trade_state()

    def reset_trade_state(self):
        self.active_sub_strategy = None
        self.stop_loss_price = 0.0
        self.trailing_stop_active = False
        self.highest_high_in_trade = 0.0
        self.lowest_low_in_trade = float("inf")

    def next(self):
        price = self.data.Close[-1]
        # 每根K线根据环境自适应更新控制参数
        dyn = self._dynamic_controls()
        score_threshold = dyn["score_entry_threshold"]
        # 这些参数被退出/追踪逻辑即时读取，直接写入属性生效
        self.tsl_activation_profit_pct = dyn["tsl_activation_profit_pct"]
        self.tsl_trailing_atr_mult = dyn["tsl_trailing_atr_mult"]
        self.daily_max_entries = dyn["daily_max_entries"]
        # Manage existing position (TSL + regime-specific exit)
        if self.position:
            self._manage_trailing_stop_loss()
            if getattr(self, "active_sub_strategy", None) == "TF":
                self.manage_trend_following_exit(price)
            elif getattr(self, "active_sub_strategy", None) == "MR":
                self.manage_mean_reversion_exit(price)
            return

        macro = self.macro_trend[-1] if len(self.macro_trend) else 0
        is_trend_regime = self.market_regime[-1] == 1 if len(self.market_regime) else False
        score = self.asset_strategy._calculate_entry_score()

        # Trend-following entries (macro bull -> long; macro bear -> short)
        if is_trend_regime:
            if macro == 1 and score > score_threshold and self._can_enter_today():
                risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
                size = self._calculate_position_size(
                    price,
                    risk_ps,
                    self._calculate_dynamic_risk() * dyn["tf_risk_scale"] * max(score, 0),
                )
                if size > 0:
                    self.reset_trade_state()
                    self.active_sub_strategy = "TF"
                    self.buy(size=size)
                    self.stop_loss_price = price - risk_ps
                    self._mark_entered()
                return
            if macro == -1 and score < -score_threshold and self._can_enter_today():
                risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
                size = self._calculate_position_size(
                    price,
                    risk_ps,
                    self._calculate_dynamic_risk() * dyn["tf_risk_scale"] * max(-score, 0),
                )
                if size > 0:
                    self.reset_trade_state()
                    self.active_sub_strategy = "TF"
                    self.sell(size=size)
                    self.stop_loss_price = price + risk_ps
                    self._mark_entered()
                return

        # Mean-reversion entries (macro bull -> long re-entry; macro bear -> short re-entry)
        else:
            # BB re-entry with StochRSI confirmation
            can_check = len(self.data.Close) > 2 and len(self.mr_stoch_rsi_k) > 2
            if can_check:
                long_reentry = (
                    self.data.Close[-2] < self.tf_bb_lower[-2] and price > self.tf_bb_lower[-1]
                )
                stoch_long = (
                    self.mr_stoch_rsi_k[-1] > self.mr_stoch_rsi_d[-1]
                    and self.mr_stoch_rsi_k[-2] <= self.mr_stoch_rsi_d[-2]
                    and self.mr_stoch_rsi_k[-1] < 40
                )
                short_reentry = (
                    self.data.Close[-2] > self.tf_bb_upper[-2] and price < self.tf_bb_upper[-1]
                )
                stoch_short = (
                    self.mr_stoch_rsi_k[-1] < self.mr_stoch_rsi_d[-1]
                    and self.mr_stoch_rsi_k[-2] >= self.mr_stoch_rsi_d[-2]
                    and self.mr_stoch_rsi_k[-1] > 60
                )
            else:
                long_reentry = short_reentry = stoch_long = stoch_short = False

            # 仅在低ADX的震荡环境下启用MR入场，进一步减少顺势市场中的逆势交易
            weak_trend_now = (self.tf_adx[-1] < self.tf_adx_confirm_threshold)
            if macro == 1 and weak_trend_now and long_reentry and stoch_long and self._can_enter_today():
                risk_ps = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
                size = self._calculate_position_size(
                    price,
                    risk_ps,
                    self._calculate_dynamic_risk() * dyn["mr_risk_scale"],
                )
                if size > 0:
                    self.reset_trade_state()
                    self.active_sub_strategy = "MR"
                    self.buy(size=size)
                    self.stop_loss_price = price - risk_ps
                    self._mark_entered()
                return
            if macro == -1 and weak_trend_now and short_reentry and stoch_short and self._can_enter_today():
                risk_ps = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
                size = self._calculate_position_size(
                    price,
                    risk_ps,
                    self._calculate_dynamic_risk() * dyn["mr_risk_scale"],
                )
                if size > 0:
                    self.reset_trade_state()
                    self.active_sub_strategy = "MR"
                    self.sell(size=size)
                    self.stop_loss_price = price + risk_ps
                    self._mark_entered()
                return

    # ...


# --- 主流程 ---
if __name__ == "__main__":
    logger.info(f"🚀 (V5.1-Integrated-ML-FIXED) 开始运行...")
    if not ADVANCED_ML_LIBS_INSTALLED:
        logger.warning("检测到部分依赖缺失，将以降级模式运行（禁用部分高级特性）。")

    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_lookback = timedelta(days=CONFIG["data_lookback_days"])
    data_fetch_start_date = (backtest_start_dt - data_lookback).strftime("%Y-%m-%d")

    raw_data = {
        s: fetch_binance_klines(
            s, CONFIG["interval"], data_fetch_start_date, CONFIG["backtest_end_date"]
        )
        for s in CONFIG["symbols_to_test"]
    }
    raw_data = {s: d for s, d in raw_data.items() if not d.empty}
    if not raw_data:
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        logger.info(f"为 {symbol} 预处理完整时段数据...")
        full_processed_data = preprocess_data_for_strategy(data, symbol)
        backtest_period_slice = full_processed_data.loc[
            CONFIG["backtest_start_date"] :
        ].copy()
        if not backtest_period_slice.empty:
            processed_backtest_data[symbol] = backtest_period_slice

    if not processed_backtest_data:
        logger.error("无回测数据，程序终止。")
        exit()

    logger.info("### 进入回测模式 ###")
    for symbol, data in processed_backtest_data.items():
        print(f"\n{'='*80}\n正在回测品种: {symbol}\n{'='*80}")
        bt = Backtest(
            data,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            finalize_trades=True,
        )
        stats = bt.run(symbol=symbol)
        print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
        print(stats)
        if CONFIG["show_plots"]:
            bt.plot()
