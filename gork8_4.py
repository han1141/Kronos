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

# 导入HMM库，并设置全局标志
try:
    from hmmlearn import hmm

    HMM_INSTALLED = True
except ImportError:
    HMM_INSTALLED = False

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 日志配置 ---
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


# --- 中文字体设置 ---
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
    "enable_hmm_regime_filter": False,
}

# --- Keras模型文件路径配置 ---
KERAS_MODEL_PATH = "models/eth_trend_model_v1_15m.keras"
SCALER_PATH = "models/eth_trend_scaler_v1_15m.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns_15m.joblib"
KERAS_SEQUENCE_LENGTH = 60
NEW_RANKED_MODEL_PATH = "artifacts/final_v4.6_20251103_1038/eth_final_model_15m.keras"

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
    "score_entry_threshold": 0.45,
    "score_weights_tf": {
        "breakout": 0.20,
        "momentum": 0.15,
        "mtf": 0.10,
        "ml": 0.15,
        "advanced_ml": 0.15,
        "ranked_ml": 0.0,
    },
    "grid_range_pct": 0.05,
    "grid_levels": 10,
    "grid_size_pct_equity": 0.01,
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETHUSDT": {"strategy_class": "ETHStrategy", "score_entry_threshold": 0.45},
}


# --- 辅助函数与特征工程 (保持不变) ---
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
    tau = [
        np.std(np.subtract(ts[lag:], ts[:-lag]))
        for lag in lags
        if np.std(np.subtract(ts[lag:], ts[:-lag])) > 0
    ]
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


def add_hmm_regime_features(
    df: pd.DataFrame, n_states: int = 3, vol_window: int = 24
) -> pd.DataFrame:
    if not HMM_INSTALLED:
        logger.warning("hmmlearn 库未安装，跳过HMM状态识别。")
        df["hmm_regime"] = 0
        return df
    logger.info(f"开始使用 HMM 进行市场状态识别 (隐藏状态数量: {n_states})...")
    df_hmm = df.copy()
    df_hmm["log_returns"] = np.log(df_hmm["Close"] / df_hmm["Close"].shift(1))
    df_hmm["volatility"] = df_hmm["log_returns"].rolling(vol_window).std()
    df_hmm.dropna(inplace=True)
    if df_hmm.empty or len(df_hmm) < n_states:
        logger.warning("准备HMM特征后数据不足，无法进行状态识别。")
        df["hmm_regime"] = 0
        return df
    hmm_features = df_hmm[["log_returns", "volatility"]].values
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42,
        tol=1e-3,
    )
    try:
        model.fit(hmm_features)
    except ValueError as e:
        logger.error(f"HMM 模型训练失败: {e}。")
        df["hmm_regime"] = 0
        return df
    hidden_states = model.predict(hmm_features)
    df_hmm["hmm_state_raw"] = hidden_states
    state_means = {
        i: df_hmm[df_hmm["hmm_state_raw"] == i]["log_returns"].mean()
        for i in range(n_states)
    }
    sorted_states = sorted(state_means.items(), key=lambda item: item[1])
    regime_map = {}
    if n_states >= 2:
        regime_map[sorted_states[-1][0]] = 1
        regime_map[sorted_states[0][0]] = -1
        for i in range(1, n_states - 1):
            regime_map[sorted_states[i][0]] = 0
    df_hmm["hmm_regime"] = df_hmm["hmm_state_raw"].map(regime_map)
    df["hmm_regime"] = df_hmm["hmm_regime"].reindex(df.index).ffill().fillna(0)
    logger.info("✅ HMM 市场状态识别完成。")
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
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["feature_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["regime_score"] = df.get("feature_regime_score", 0.5)
    df["market_regime"] = np.where(
        df["regime_score"] > STRATEGY_PARAMS["regime_score_threshold"], 1, -1
    )
    return df


def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("正在为 Kras 模型生成特定特征 (使用 'ta' 库)...")
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    df["EMA_8"] = ta.trend.EMAIndicator(close=close, window=8).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = (
        adx_indicator.adx(),
        adx_indicator.adx_pos(),
        adx_indicator.adx_neg(),
    )
    df["ATRr_14"] = (
        ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        / close
    ) * 100
    bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
    (
        df["BBU_20_2.0"],
        df["BBM_20_2.0"],
        df["BBL_20_2.0"],
        df["BBB_20_2.0"],
        df["BBP_20_2.0"],
    ) = (
        bb_indicator.bollinger_hband(),
        bb_indicator.bollinger_mavg(),
        bb_indicator.bollinger_lband(),
        bb_indicator.bollinger_wband(),
        bb_indicator.bollinger_pband(),
    )
    macd_indicator = ta.trend.MACD(
        close=close, window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = (
        macd_indicator.macd(),
        macd_indicator.macd_signal(),
        macd_indicator.macd_diff(),
    )
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()
    df["volume_change_rate"] = volume.pct_change()
    logger.info("✅ 成功生成 Keras 模型所需的所有特定特征。")
    return df


def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    sequences = []
    data_len = len(data)
    for i in range(data_len - sequence_length + 1):
        sequences.append(data[i : i + sequence_length])
    return np.array(sequences)


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
    df = add_features_for_keras_model(df)
    df = add_ml_features(df)
    df = add_market_regime_features(df)
    logger.info(f"✅ 战术市场状态 ('market_regime') 计算完成。")
    if CONFIG["enable_hmm_regime_filter"]:
        df = add_hmm_regime_features(df)
    else:
        logger.warning("HMM 已禁用...")
        df["hmm_regime"] = df["market_regime"]
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
        df["mtf_signal"] = (
            pd.Series(np.where(data_1d["Close"] > sma, 1, -1), index=data_1d.index)
            .reindex(df.index, method="ffill")
            .fillna(0)
        )
    else:
        df["mtf_signal"] = 0
    df_4h = df["Close"].resample("4h").last().to_frame()
    df_4h["macro_ema"] = ta.trend.EMAIndicator(
        df_4h["Close"], window=50
    ).ema_indicator()
    df["macro_trend_filter"] = (
        pd.Series(
            np.where(df_4h["Close"] > df_4h["macro_ema"], 1, -1), index=df_4h.index
        )
        .reindex(df.index, method="ffill")
        .fillna(0)
    )
    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


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
        keras_ml_signal = getattr(m, "keras_signal", [0])[-1]
        ranked_ml_signal = getattr(m, "ranked_ml_signal", [0])[-1]
        adv_ml_signal = getattr(m, "advanced_ml_signal", [0])[-1]
        mtf_signal = getattr(m, "mtf_signal", [0])[-1]
        return (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + mtf_signal * w.get("mtf", 0)
            + keras_ml_signal * w.get("ml", 0)
            + adv_ml_signal * w.get("advanced_ml", 0)
            + ranked_ml_signal * w.get("ranked_ml", 0)
        )

    def _define_mr_entry_signal(self) -> int:
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
    symbol, vol_weight = None, 1.0
    (
        strategy_class_override,
        score_entry_threshold_override,
        score_weights_tf_override,
    ) = (None, None, None)
    new_model_threshold = 0.010284

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
        self.reset_trade_state()
        self.market_regime = self.I(lambda: self.data.market_regime)
        close, high, low = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
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
        self.mr_bb_upper, self.mr_bb_lower, self.mr_bb_mid = (
            self.I(lambda: bb.bollinger_hband()),
            self.I(lambda: bb.bollinger_lband()),
            self.I(lambda: bb.bollinger_mavg()),
        )
        stoch_rsi = ta.momentum.StochRSIIndicator(
            close, window=14, smooth1=3, smooth2=3
        )
        self.mr_stoch_rsi_k, self.mr_stoch_rsi_d = self.I(
            lambda: stoch_rsi.stochrsi_k()
        ), self.I(lambda: stoch_rsi.stochrsi_d())
        self.keras_model, self.scaler, self.feature_columns = (
            self._load_keras_model_and_dependencies()
        )
        self.keras_signal = self.I(self._calculate_keras_predictions)
        self.ranked_ml_model = self._load_new_ranked_model()
        self.ranked_ml_signal = self.I(self._calculate_ranked_ml_predictions)
        self.grid_active = False
        self.grid_buy_orders = []
        self.grid_sell_orders = []
        self.grid_center_price = None
        # 【新增】状态缓冲区
        self.regime_buffer = deque(maxlen=3)  # 缓冲区长度为3，可调

    def _load_new_ranked_model(self):
        if not CONFIG["enable_ml_component"] or not ADVANCED_ML_LIBS_INSTALLED:
            return None
        try:
            logger.info(f"✅ [{self.symbol}] 成功加载（模拟的）新的排序ML模型。")
            return "DUMMY_RANKED_MODEL_LOADED"
        except Exception as e:
            logger.error(f"[{self.symbol}] 加载新的排序ML模型失败: {e}")
            return None

    def _calculate_ranked_ml_predictions(self):
        if self.ranked_ml_model is None:
            return np.zeros(len(self.data.Close))
        np.random.seed(42)
        signals = np.where(
            0.006
            + np.sin(np.linspace(0, 40, len(self.data.Close))) * 0.01
            + np.random.randn(len(self.data.Close)) * 0.005
            > self.new_model_threshold,
            1.0,
            0.0,
        )
        logger.info(f"[{self.symbol}] 新的排序ML模型信号计算完成。")
        return signals

    def _load_keras_model_and_dependencies(self):
        if not CONFIG["enable_ml_component"] or not ADVANCED_ML_LIBS_INSTALLED:
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
        if not set(self.feature_columns).issubset(self.data.df.columns):
            return np.zeros(len(self.data.Close))
        features_df = self.data.df[self.feature_columns].fillna(0)
        scaled_features_2d = self.scaler.transform(features_df)
        scaled_features_3d = create_sequences(scaled_features_2d, KERAS_SEQUENCE_LENGTH)
        predictions_proba = self.keras_model.predict(
            scaled_features_3d, verbose=0
        ).flatten()
        padding_size = len(self.data.Close) - len(predictions_proba)
        final_signals = np.zeros(len(self.data.Close))
        final_signals[padding_size:] = (predictions_proba - 0.5) * 2
        logger.info(f"[{self.symbol}] Keras模型信号计算完成。")
        return final_signals

    def _setup_grid(self):
        if self.grid_active:
            return
        self.grid_active = True
        self.grid_center_price = self.data.Close[-1]
        grid_range = self.grid_center_price * self.grid_range_pct
        upper_bound, lower_bound = (
            self.grid_center_price + grid_range,
            self.grid_center_price - grid_range,
        )
        cash_per_level = self.equity * self.grid_size_pct_equity
        buy_step = (self.grid_center_price - lower_bound) / self.grid_levels
        sell_step = (upper_bound - self.grid_center_price) / self.grid_levels
        if cash_per_level <= 0:
            logger.warning(
                f"计算出的网格下单金额为 {cash_per_level:.2f}，无法建立网格。"
            )
            self.grid_active = False
            return
        logger.info(
            f"市场进入震荡，建立网格: Center={self.grid_center_price:.2f}, Range=[{lower_bound:.2f}, {upper_bound:.2f}], Levels={self.grid_levels}"
        )
        for i in range(1, self.grid_levels + 1):
            buy_price = self.grid_center_price - i * buy_step
            sell_price = self.grid_center_price + i * sell_step
            if buy_price > 0:
                buy_size = int(cash_per_level / buy_price)
                if buy_size > 0:
                    self.grid_buy_orders.append(
                        self.buy(limit=buy_price, size=buy_size)
                    )
            if sell_price > 0:
                sell_size = int(cash_per_level / sell_price)
                if sell_size > 0:
                    self.grid_sell_orders.append(
                        self.sell(limit=sell_price, size=sell_size)
                    )
        if not self.grid_buy_orders and not self.grid_sell_orders:
            logger.warning("未能成功挂出任何网格订单，网格建立失败。")
            self.grid_active = False

    def _run_grid_logic(self):
        if not self.grid_active:
            return
        pass

    def _teardown_grid(self):
        if not self.grid_active:
            return
        logger.info(f"市场转为趋势，拆除网格，清仓所有网格头寸...")
        for order in self.orders:
            if order.is_contingent:
                order.cancel()
        if self.position:
            self.position.close()
        (
            self.grid_active,
            self.grid_buy_orders,
            self.grid_sell_orders,
            self.grid_center_price,
        ) = (False, [], [], None)

    def next(self):
        # 【修改】引入状态缓冲区实现状态粘滞性
        self.regime_buffer.append(self.market_regime[-1])
        if len(self.regime_buffer) < self.regime_buffer.maxlen:
            return  # 缓冲区未满，等待数据填充

        # 检查缓冲区内的状态是否一致
        is_consistent = all(s == self.regime_buffer[0] for s in self.regime_buffer)
        confirmed_state = self.regime_buffer[0] if is_consistent else None

        if confirmed_state is None:
            # 状态不一致，市场方向不明，可以考虑清仓或不做任何操作
            # 为保守起见，如果正在进行趋势交易，可以平仓；网格可以暂时维持
            if self.position and not self.grid_active:
                self.position.close()
            return

        # --- 状态转换管理 ---
        if confirmed_state == 1 and self.grid_active:
            self._teardown_grid()
        elif confirmed_state == -1 and not self.grid_active:
            if self.position and self.active_sub_strategy == "TF":
                self.position.close()
            self._setup_grid()

        # --- 根据已确认的状态执行相应逻辑 ---
        if confirmed_state == 1:
            if self.position:
                return
            macro_trend = self.macro_trend[-1]
            score = self.asset_strategy._calculate_entry_score()
            if macro_trend == 1 and score > self.score_entry_threshold:
                self.open_tf_position(
                    self.data.Close[-1], is_long=True, confidence_factor=score
                )
            elif macro_trend == -1 and score < -self.score_entry_threshold:
                self.open_tf_position(
                    self.data.Close[-1], is_long=False, confidence_factor=abs(score)
                )
        elif confirmed_state == -1:
            if self.grid_active:
                self._run_grid_logic()

    def reset_trade_state(self):
        self.active_sub_strategy, self.stop_loss_price = None, 0.0
        self.trailing_stop_active = False
        self.highest_high_in_trade, self.lowest_low_in_trade = 0, float("inf")

    def manage_open_position(self, p):
        if self.grid_active:
            return
        self._manage_trailing_stop_loss()
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(p)

    def _manage_trailing_stop_loss(self):
        if not self.tsl_enabled or not self.position:
            return
        is_active, entry_price, current_price = (
            self.trailing_stop_active,
            self.trades[-1].entry_price,
            self.data.Close[-1],
        )
        if not is_active:
            profit_cond = (self.position.pl_pct * 100) > self.tsl_activation_profit_pct
            activation_dist = self.tf_atr[-1] * self.tsl_activation_atr_mult
            price_cond = (
                self.position.is_long and current_price >= entry_price + activation_dist
            ) or (
                self.position.is_short
                and current_price <= entry_price - activation_dist
            )
            if profit_cond or price_cond:
                self.trailing_stop_active = True
        if self.trailing_stop_active:
            trailing_dist = self.tf_atr[-1] * self.tsl_trailing_atr_mult
            if self.position.is_long:
                self.stop_loss_price = max(
                    self.stop_loss_price, current_price - trailing_dist
                )
            else:
                self.stop_loss_price = min(
                    self.stop_loss_price, current_price + trailing_dist
                )

    def open_tf_position(self, p, is_long, confidence_factor):
        risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * confidence_factor
        )
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
        if self.position.is_long:
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            chandelier_exit = (
                self.highest_high_in_trade
                - self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
            if p < max(self.stop_loss_price, chandelier_exit):
                self.close_position("TF_Exit")
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            chandelier_exit = (
                self.lowest_low_in_trade
                + self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
            if p > min(self.stop_loss_price, chandelier_exit):
                self.close_position("TF_Exit")

    def close_position(self, reason: str):
        eq_before = self.equity
        self.position.close()
        self.recent_trade_returns.append(self.equity / eq_before - 1)
        self.reset_trade_state()

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0:
            return 0
        risk_amount_dollars = self.equity * risk_pct
        units = risk_amount_dollars / rps
        if units * p > self.equity:
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
        win_rate, avg_win, avg_loss = (
            len(wins) / len(self.recent_trade_returns),
            sum(wins) / len(wins),
            abs(sum(losses) / len(losses)),
        )
        if avg_loss == 0:
            return self.default_risk_pct * self.vol_weight
        reward_ratio = avg_win / avg_loss
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


# --- 主执行块 ---
if __name__ == "__main__":
    logger.info(f"🚀 (V45.2-Regime-Stickiness) 开始运行...")
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
