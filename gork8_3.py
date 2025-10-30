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

# 检查TensorFlow是否已安装
try:
    import tensorflow as tf

    ADVANCED_ML_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_ML_LIBS_INSTALLED = False

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

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
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-10-29",
    "initial_cash": 500_000,
    "commission": 0.00075,
    "spread": 0.0002,
    "show_plots": False,
    "data_lookback_days": 60,
    "enable_ml_component": True,
}

# --- Keras模型文件路径配置 ---
KERAS_MODEL_PATH = "models/eth_trend_model_v1_15m.keras"
SCALER_PATH = "models/eth_trend_scaler_v1_15m.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns_15m.joblib"
KERAS_SEQUENCE_LENGTH = 60

# --- 策略参数 ---
STRATEGY_PARAMS = {
    "tsl_enabled": True,
    "tsl_activation_profit_pct": 0.005,
    "tsl_activation_atr_mult": 1.5,
    "tsl_trailing_atr_mult": 2.0,
    "kelly_trade_history": 20,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "dd_grace_period_bars": 240,
    "dd_initial_pct": 0.35,
    "dd_final_pct": 0.25,
    "dd_decay_bars": 4320,
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
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "score_entry_threshold": 0.4,
    "score_weights_tf": {
        "breakout": 0.25,
        "momentum": 0.18,
        "mtf": 0.10,
        "ml": 0.22,
        "advanced_ml": 0.25,
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETHUSDT": {"strategy_class": "ETHStrategy", "score_entry_threshold": 0.45},
}


# --- 函数定义 ---
class StrategyMemory:
    def __init__(self, filepath="strategy_memory.csv"):
        self.filepath, self.columns = filepath, [
            "timestamp",
            "symbol",
            "regime",
            "param_key",
            "param_value",
            "performance",
        ]
        self.memory_df = self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.filepath):
            return pd.read_csv(self.filepath, parse_dates=["timestamp"])
        return pd.DataFrame(columns=self.columns)

    def record_optimization(self, t, s, r, p, perf):
        new_df = pd.DataFrame(
            [
                {
                    "timestamp": t,
                    "symbol": s,
                    "regime": r,
                    "param_key": k,
                    "param_value": v,
                    "performance": perf,
                }
                for k, v in p.items()
            ]
        )
        self.memory_df = (
            pd.concat([self.memory_df, new_df], ignore_index=True)
            .sort_values(by="timestamp")
            .drop_duplicates(
                subset=["timestamp", "symbol", "regime", "param_key"], keep="last"
            )
        )
        self.memory_df.to_csv(self.filepath, index=False)

    def get_best_params(self, t, s, r):
        mem = self.memory_df[
            (self.memory_df["symbol"] == s)
            & (self.memory_df["regime"] == r)
            & (self.memory_df["timestamp"] <= t)
        ]
        if mem.empty:
            return None
        latest = mem[mem["timestamp"] == mem["timestamp"].max()]
        return pd.Series(latest.param_value.values, index=latest.param_key).to_dict()


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


def compute_hurst(ts, max_lag=100):
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        std = np.std(diff)
        if std > 0:
            tau.append(std)
    if len(tau) < 2:
        return 0.5
    try:
        return max(
            0.0, min(1.0, np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0])
        )
    except:
        return 0.5


def run_advanced_model_inference(df):
    logger.info("正在运行高级模型推理 (模拟)...")
    if not ADVANCED_ML_LIBS_INSTALLED:
        logger.warning("TensorFlow 未安装, 跳过高级模型推理。")
        df["advanced_ml_signal"] = 0.0
        return df
    df["advanced_ml_signal"] = (
        df.get("ai_filter_signal", pd.Series(0, index=df.index))
        .rolling(24)
        .mean()
        .fillna(0)
    )
    logger.info("高级模型推理 (模拟) 完成。")
    return df


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    p = STRATEGY_PARAMS
    norm = lambda s: (
        (s - s.rolling(p["regime_norm_period"]).min())
        / (
            s.rolling(p["regime_norm_period"]).max()
            - s.rolling(p["regime_norm_period"]).min()
        )
    ).fillna(0.5)
    adx = ta.trend.ADXIndicator(df.High, df.Low, df.Close, p["regime_adx_period"]).adx()
    atr = ta.volatility.AverageTrueRange(
        df.High, df.Low, df.Close, p["regime_atr_period"]
    ).average_true_range()
    rsi = ta.momentum.RSIIndicator(df.Close, p["regime_rsi_period"]).rsi()
    bb = ta.volatility.BollingerBands(
        df.Close, window=p["mr_bb_period"], window_dev=p["mr_bb_std"]
    )
    df["feature_adx_norm"] = norm(adx)
    df["feature_atr_slope_norm"] = norm(
        (atr - atr.shift(p["regime_atr_slope_period"]))
        / atr.shift(p["regime_atr_slope_period"])
    )
    df["feature_rsi_vol_norm"] = 1 - norm(rsi.rolling(p["regime_rsi_vol_period"]).std())
    df["feature_hurst"] = (
        df.Close.rolling(p["regime_hurst_period"])
        .apply(lambda x: compute_hurst(np.log(x + 1e-9)), raw=False)
        .fillna(0.5)
    )
    df["feature_obv_norm"] = norm(
        ta.volume.OnBalanceVolumeIndicator(df.Close, df.Volume).on_balance_volume()
    )
    df["feature_vol_pct_change_norm"] = norm(df.Volume.pct_change(periods=1).abs())
    df["feature_bb_width_norm"] = norm(
        (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    )
    df["feature_atr_pct_change_norm"] = norm(atr.pct_change(periods=1))
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["feature_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["regime_score"] = df["feature_regime_score"]
    df["trend_regime"] = np.where(
        df["regime_score"] > STRATEGY_PARAMS["regime_score_threshold"],
        "Trending",
        "Mean-Reverting",
    )
    df["volatility"] = df["Close"].pct_change().rolling(24 * 7).std() * np.sqrt(
        24 * 365
    )
    low_vol, high_vol = df["volatility"].quantile(0.33), df["volatility"].quantile(0.67)
    df["volatility_regime"] = pd.cut(
        df["volatility"],
        bins=[0, low_vol, high_vol, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    df["market_regime"] = np.where(df["trend_regime"] == "Trending", 1, -1)
    return df


def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    为 Keras 模型生成所有必需的特征。
    - 确保同时计算并添加所有5个布林带相关指标。
    - 修正了 MACD 列名的大小写以匹配模型期望。
    """
    logger.info("正在为 Kras 模型生成特定特征 (使用 'ta' 库)...")
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

    # --- 基础指标 ---
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

    # --- 修正: 计算并添加所有5个必需的布林带指标 ---
    bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
    # 1. 上轨
    df["BBU_20_2.0"] = bb_indicator.bollinger_hband()
    # 2. 中轨
    df["BBM_20_2.0"] = bb_indicator.bollinger_mavg()
    # 3. 下轨
    df["BBL_20_2.0"] = bb_indicator.bollinger_lband()
    # 4. 宽度 (Bandwidth)
    df["BBB_20_2.0"] = bb_indicator.bollinger_wband()
    # 5. %B 指标 (Percentage)
    df["BBP_20_2.0"] = bb_indicator.bollinger_pband()

    # --- 修正: 统一 MACD 列名的大小写 ---
    macd_indicator = ta.trend.MACD(
        close=close, window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD_12_26_9"] = macd_indicator.macd()
    df["MACDs_12_26_9"] = macd_indicator.macd_signal()  # 修正: s小写
    df["MACDh_12_26_9"] = macd_indicator.macd_diff()  # 修正: h小写

    # --- 其它指标 ---
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()
    df["volume_change_rate"] = volume.pct_change()

    logger.info("✅ 成功生成 Keras 模型所需的所有特定特征。")
    return df


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(
        f"[{symbol}] 开始数据预处理 (数据范围: {df.index.min()} to {df.index.max()})..."
    )
    rsi_filter = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["ai_filter_signal"] = (
        (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
        .clip(-1, 1)
        .fillna(0)
    )
    df = run_advanced_model_inference(df)
    df = add_ml_features(df)
    df = add_features_for_keras_model(df)
    df = add_market_regime_features(df)
    daily_start = df.index.min().normalize() - pd.Timedelta(
        days=STRATEGY_PARAMS["mtf_period"] + 5
    )
    daily_end = df.index.max().normalize()
    data_1d = fetch_binance_klines(
        symbol, "1d", daily_start.strftime("%Y-%m-%d"), daily_end.strftime("%Y-%m-%d")
    )
    if not data_1d.empty:
        sma = ta.trend.SMAIndicator(
            data_1d["Close"], window=STRATEGY_PARAMS["mtf_period"]
        ).sma_indicator()
        mtf_signal_1d = pd.Series(
            np.where(data_1d["Close"] > sma, 1, -1), index=data_1d.index
        )
        df["mtf_signal"] = mtf_signal_1d.reindex(df.index, method="ffill").fillna(0)
    else:
        df["mtf_signal"] = 0
    logger.info(f"[{symbol}] 正在计算 4h 宏观趋势过滤器...")
    df_4h = df["Close"].resample("4h").last().to_frame()
    df_4h["macro_ema"] = ta.trend.EMAIndicator(
        df_4h["Close"], window=50
    ).ema_indicator()
    df_4h["macro_trend"] = np.where(df_4h["Close"] > df_4h["macro_ema"], 1, -1)
    df["macro_trend_filter"] = (
        df_4h["macro_trend"].reindex(df.index, method="ffill").fillna(0)
    )
    logger.info(f"[{symbol}] 宏观趋势过滤器计算完成。")
    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    sequences = []
    data_len = len(data)
    for i in range(data_len - sequence_length + 1):
        sequences.append(data[i : i + sequence_length])
    return np.array(sequences)


# --- 策略类定义 ---
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
        return (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + m.mtf_signal[-1] * w.get("mtf", 0)
            + keras_ml_signal * w.get("ml", 0)
            + m.advanced_ml_signal[-1] * w.get("advanced_ml", 0)
        )

    def _define_mr_entry_signal(self) -> int:
        m = self.main
        long_reentry_condition = (
            m.data.Close[-2] < m.mr_bb_lower[-2]
            and m.data.Close[-1] > m.mr_bb_lower[-1]
        )
        stoch_long_confirmation = (
            m.mr_stoch_rsi_k[-1] > m.mr_stoch_rsi_d[-1]
            and m.mr_stoch_rsi_k[-2] <= m.mr_stoch_rsi_d[-2]
            and m.mr_stoch_rsi_k[-1] < 40
        )
        if long_reentry_condition and stoch_long_confirmation:
            return 1
        short_reentry_condition = (
            m.data.Close[-2] > m.mr_bb_upper[-2]
            and m.data.Close[-1] < m.mr_bb_upper[-1]
        )
        stoch_short_confirmation = (
            m.mr_stoch_rsi_k[-1] < m.mr_stoch_rsi_d[-1]
            and m.mr_stoch_rsi_k[-2] >= m.mr_stoch_rsi_d[-2]
            and m.mr_stoch_rsi_k[-1] > 60
        )
        if short_reentry_condition and stoch_short_confirmation:
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
    score_entry_threshold_override = None
    score_weights_tf_override = None
    ml_weights_override = None
    ml_weighted_threshold_override = None

    def init(self):
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)
        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {})
        self.score_entry_threshold = asset_overrides.get(
            "score_entry_threshold", self.score_entry_threshold
        )
        if self.score_weights_tf_override is not None:
            self.score_weights_tf = self.score_weights_tf_override
        strategy_class_name = self.strategy_class_override or asset_overrides.get(
            "strategy_class", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(
            strategy_class_name, BaseAssetStrategy
        )(self)
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()
        self.market_regime = self.I(lambda: self.data.market_regime)
        self.mtf_signal = self.I(lambda: self.data.mtf_signal)
        self.advanced_ml_signal = self.I(lambda: self.data.advanced_ml_signal)
        self.macro_trend = self.I(lambda: self.data.macro_trend_filter)
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.tf_atr_period
            ).average_true_range()
        )
        self.tf_donchian_h = self.I(
            lambda: high.rolling(self.tf_donchian_period).max().shift(1)
        )
        self.tf_donchian_l = self.I(
            lambda: low.rolling(self.tf_donchian_period).min().shift(1)
        )
        self.tf_ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(
                close, self.tf_ema_fast_period
            ).ema_indicator()
        )
        self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(
                close, self.tf_ema_slow_period
            ).ema_indicator()
        )
        self.tf_adx = self.I(
            lambda: ta.trend.ADXIndicator(
                high, low, close, self.tf_adx_confirm_period
            ).adx()
        )
        bb = ta.volatility.BollingerBands(close, self.mr_bb_period, self.mr_bb_std)
        self.mr_bb_upper = self.I(lambda: bb.bollinger_hband())
        self.mr_bb_lower = self.I(lambda: bb.bollinger_lband())
        self.mr_bb_mid = self.I(lambda: bb.bollinger_mavg())
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(
                close, STRATEGY_PARAMS["regime_rsi_period"]
            ).rsi()
        )
        stoch_rsi = ta.momentum.StochRSIIndicator(
            close, window=14, smooth1=3, smooth2=3
        )
        self.mr_stoch_rsi_k = self.I(lambda: stoch_rsi.stochrsi_k())
        self.mr_stoch_rsi_d = self.I(lambda: stoch_rsi.stochrsi_d())
        self.keras_model, self.scaler, self.feature_columns = (
            self._load_keras_model_and_dependencies()
        )
        self.keras_signal = self.I(self._calculate_keras_predictions)

    def _load_keras_model_and_dependencies(self):
        if not CONFIG["enable_ml_component"] or not ADVANCED_ML_LIBS_INSTALLED:
            logger.warning(f"[{self.symbol}] Keras模型组件已禁用或TensorFlow未安装。")
            return None, None, None
        try:
            model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            logger.info(f"✅ [{self.symbol}] 成功加载Keras模型、缩放器和特征列。")
            return model, scaler, feature_columns
        except Exception as e:
            logger.error(f"[{self.symbol}] 加载Keras模型或依赖项失败: {e}")
            return None, None, None

    def _calculate_keras_predictions(self):
        if self.keras_model is None or self.scaler is None:
            return np.zeros(len(self.data.Close))
        expected_features = set(self.feature_columns)
        actual_features = set(self.data.df.columns)
        if not expected_features.issubset(actual_features):
            missing = expected_features - actual_features
            logger.error(f"[{self.symbol}] 数据中缺少必要的特征列，无法进行模型预测。")
            logger.error(f"  - 缺失的特征 ({len(missing)}): {sorted(list(missing))}")
            return np.zeros(len(self.data.Close))
        features_df = self.data.df[self.feature_columns]
        if features_df.isnull().values.any():
            logger.warning(f"[{self.symbol}] 特征数据中存在NaN值，将用0填充。")
            features_df = features_df.fillna(0)
        scaled_features_2d = self.scaler.transform(features_df)
        logger.info(
            f"[{self.symbol}] 正在将2D特征数据转换为 {KERAS_SEQUENCE_LENGTH} 步长的3D序列..."
        )
        scaled_features_3d = create_sequences(scaled_features_2d, KERAS_SEQUENCE_LENGTH)
        logger.info(
            f"[{self.symbol}] 3D序列数据创建完成，形状为: {scaled_features_3d.shape}"
        )
        predictions_proba = self.keras_model.predict(
            scaled_features_3d, verbose=0
        ).flatten()
        num_predictions = len(predictions_proba)
        padding_size = len(self.data.Close) - num_predictions
        raw_signals = (predictions_proba - 0.5) * 2
        final_signals = np.zeros(len(self.data.Close))
        final_signals[padding_size:] = raw_signals
        logger.info(f"[{self.symbol}] Keras模型信号计算完成。")
        return final_signals

    # <<< 核心修改：实现做多/做空双向逻辑 >>>
    def next(self):
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            # 宏观牛市，只寻找做多机会
            if self.macro_trend[-1] == 1:
                if self.data.market_regime[-1] == 1:  # 趋势市
                    score = self.asset_strategy._calculate_entry_score()
                    if score > self.score_entry_threshold:
                        self.open_tf_position(
                            self.data.Close[-1], is_long=True, confidence_factor=score
                        )
                else:  # 震荡市
                    signal = self.asset_strategy._define_mr_entry_signal()
                    if signal == 1:
                        self.open_mr_position(self.data.Close[-1], is_long=True)

            # 宏观熊市，只寻找做空机会
            elif self.macro_trend[-1] == -1:
                if self.data.market_regime[-1] == 1:  # 趋势市
                    score = self.asset_strategy._calculate_entry_score()
                    if score < -self.score_entry_threshold:
                        self.open_tf_position(
                            self.data.Close[-1],
                            is_long=False,
                            confidence_factor=abs(score),
                        )
                else:  # 震荡市
                    signal = self.asset_strategy._define_mr_entry_signal()
                    if signal == -1:
                        self.open_mr_position(self.data.Close[-1], is_long=False)

    def reset_trade_state(self):
        self.active_sub_strategy = None
        self.stop_loss_price = 0.0
        self.trailing_stop_active = False
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")

    def manage_open_position(self, p):
        self._manage_trailing_stop_loss()
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(p)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(p)

    def _manage_trailing_stop_loss(self):
        if not self.tsl_enabled or not self.position:
            return
        is_active = self.trailing_stop_active
        entry_price = self.trades[-1].entry_price
        current_price = self.data.Close[-1]

        if not is_active:
            profit_pct_condition_met = (
                self.position.pl_pct * 100
            ) > self.tsl_activation_profit_pct
            atr = self.tf_atr[-1]
            activation_distance = atr * self.tsl_activation_atr_mult
            price_move_condition_met = False
            if (
                self.position.is_long
                and current_price >= entry_price + activation_distance
            ):
                price_move_condition_met = True
            elif (
                self.position.is_short
                and current_price <= entry_price - activation_distance
            ):
                price_move_condition_met = True
            if profit_pct_condition_met or price_move_condition_met:
                self.trailing_stop_active = True
                is_active = True

        if is_active:
            atr = self.tf_atr[-1]
            trailing_distance = atr * self.tsl_trailing_atr_mult
            new_stop_price = None
            if self.position.is_long:
                potential_stop = current_price - trailing_distance
                if potential_stop > self.stop_loss_price:
                    new_stop_price = potential_stop
            else:
                potential_stop = current_price + trailing_distance
                if potential_stop < self.stop_loss_price:
                    new_stop_price = potential_stop
            if new_stop_price is not None:
                self.stop_loss_price = new_stop_price

    def open_tf_position(self, p, is_long, confidence_factor, score=1.0):
        risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * score * confidence_factor
        )
        # <<< 修改: 移除错误的检查，替换为更合理的检查 >>>
        # 原来的检查 `if not 0 < size < 0.98:` 在计算单位数量时是错误的
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.stop_loss_price = p - risk_ps
        else:
            self.sell(size=size)
            self.stop_loss_price = p + risk_ps

    def manage_trend_following_exit(self, p):
        chandelier_exit_level = 0
        if self.position.is_long:
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            chandelier_exit_level = (
                self.highest_high_in_trade
                - self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
            final_sl = max(self.stop_loss_price, chandelier_exit_level)
            if p < final_sl:
                self.close_position("TF_Exit")
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            chandelier_exit_level = (
                self.lowest_low_in_trade
                + self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
            final_sl = min(self.stop_loss_price, chandelier_exit_level)
            if p > final_sl:
                self.close_position("TF_Exit")

    def open_mr_position(self, p, is_long):
        risk_ps = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * self.mr_risk_multiplier
        )
        # <<< 修改: 移除错误的检查，替换为更合理的检查 >>>
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        if is_long:
            self.buy(size=size)
            self.stop_loss_price = p - risk_ps
        else:
            self.sell(size=size)
            self.stop_loss_price = p + risk_ps

    def manage_mean_reversion_exit(self, p):
        if (
            self.position.is_long
            and (p >= self.mr_bb_mid[-1] or p <= self.stop_loss_price)
        ) or (
            self.position.is_short
            and (p <= self.mr_bb_mid[-1] or p >= self.stop_loss_price)
        ):
            self.close_position("MR_Exit")

    def close_position(self, reason: str):
        eq_before = self.equity
        self.position.close()
        self.recent_trade_returns.append(self.equity / eq_before - 1)
        self.reset_trade_state()

    # <<< 修改: 仓位管理核心逻辑修正 >>>
    def _calculate_position_size(self, p, rps, risk_pct):
        """
        根据风险百分比计算头寸的【单位数量】。
        这是一个更稳健的方法，避免了分数与单位数量之间的混淆。
        p: 当前价格
        rps: 每单位风险 (Risk Per Share/Unit)，即止损距离
        risk_pct: 愿意承担的风险百分比
        """
        if rps <= 0 or p <= 0:
            return 0

        # 1. 计算本次交易愿意承担的风险金额（美元）
        risk_amount_dollars = self.equity * risk_pct

        # 2. 计算可以购买多少单位的资产
        units = risk_amount_dollars / rps

        # 3. (安全检查) 确保购买这些单位的钱足够
        cash_needed = units * p
        if cash_needed > self.equity:
            # 如果所需现金超过全部资产，则用95%的资产来购买，以防万一
            units = (self.equity * 0.95) / p

        return int(units)

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins, losses = [r for r in self.recent_trade_returns if r > 0], [
            r for r in self.recent_trade_returns if r < 0
        ]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        if avg_loss == 0:
            return self.default_risk_pct * self.vol_weight
        reward_ratio = avg_win / avg_loss
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


if __name__ == "__main__":
    logger.info(f"🚀 (V41.05-MR-Enhanced with Keras Model) 开始运行...")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_lookback = timedelta(days=CONFIG["data_lookback_days"])
    data_fetch_start_date = (backtest_start_dt - data_lookback).strftime("%Y-%m-%d")
    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期 (包含指标计算所需历史数据): {data_fetch_start_date}")
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
    logger.info(f"### 准备回测数据 (开始日期: {CONFIG['backtest_start_date']}) ###")
    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        logger.info(f"为 {symbol} 预处理完整时段数据...")
        full_processed_data = preprocess_data_for_strategy(data, symbol)
        backtest_period_slice = full_processed_data.loc[
            CONFIG["backtest_start_date"] :
        ].copy()
        if not backtest_period_slice.empty:
            processed_backtest_data[symbol] = backtest_period_slice
    processed_backtest_data = {
        s: d for s, d in processed_backtest_data.items() if not d.empty
    }
    if not processed_backtest_data:
        logger.error("无回测数据，程序终止。")
        exit()
    logger.info(f"### 进入回测模式 ###")
    all_stats, total_equity = {}, 0
    vols = {
        s: d.Close.resample("D").last().pct_change().std() * np.sqrt(365)
        for s, d in processed_backtest_data.items()
    }
    inv_vols = {s: 1 / v for s, v in vols.items() if v > 0}
    vol_weights = {
        s: (iv / sum(inv_vols.values())) * len(inv_vols) for s, iv in inv_vols.items()
    }
    for symbol, data in processed_backtest_data.items():
        print(f"\n{'='*80}\n正在回测品种: {symbol}\n{'='*80}")
        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(symbol, {})
        bt_params = {"symbol": symbol, "vol_weight": vol_weights.get(symbol, 1.0)}
        override_keys = ["strategy_class", "score_entry_threshold", "score_weights_tf"]
        for key in override_keys:
            if key in asset_overrides:
                bt_params[f"{key}_override"] = asset_overrides[key]

        # <<< 修改: 正确添加滑点参数 margin >>>
        # margin 代表单边滑点，通常是 spread (买卖价差) 的一半。
        bt = Backtest(
            data,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            margin=CONFIG["spread"] / 2,
            finalize_trades=True,
        )
        stats = bt.run(**bt_params)
        all_stats[symbol] = stats
        total_equity += stats["Equity Final [$]"]
        print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
        print(stats)
        if CONFIG["show_plots"]:
            bt.plot()
    if all_stats:
        initial_total = CONFIG["initial_cash"] * len(all_stats)
        ret = ((total_equity - initial_total) / initial_total) * 100
        print(f"\n{'#'*80}\n                 组合策略表现总览\n{'#'*80}")
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A')}"
            )
        print(
            f"\n--- 投资组合整体表现 ---\n总初始资金: ${initial_total:,.2f}\n总最终权益: ${total_equity:,.2f}\n组合总回报率: {ret:.2f}%"
        )
