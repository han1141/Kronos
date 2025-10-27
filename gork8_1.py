# -*- coding: utf-8 -*-

# --- 1. 导入库与配置 ---
import pandas as pd
import warnings

# 彻底屏蔽 pkg_resources 警告
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import requests
import time
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import deque
import joblib
import os
import glob

# <<< 已移除 >>> matplotlib.pyplot 和 matplotlib.font_manager

try:
    import pandas_ta as pta
except ImportError:
    print("错误：请先安装 pandas_ta 库 (pip install pandas_ta)")
    exit()

try:
    import tensorflow as tf

    ADVANCED_ML_LIBS_INSTALLED = True
except Exception:
    ADVANCED_ML_LIBS_INSTALLED = False

from backtesting import Backtest, Strategy
import ta  # ta 库与绘图无关，予以保留

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


# --- 核心配置 ---
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],
    "interval": "30m",
    "backtest_start_date": "2025-06-01",
    "backtest_end_date": "2025-10-16",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,
}

# --- 参数与类定义 ---
STRATEGY_PARAMS = {
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
    # --- 根据ML模型评估结果调整后的参数 ---
    "score_weights_tf": {
        "breakout": 0.30,      # 恢复突破的主导地位
        "momentum": 0.25,      # 恢复动量的重要性
        "mtf": 0.15,           # 保持多时间框架的参考价值
        "ml": 0.15,            # <- [核心调整] 大幅降低ML权重，作为辅助信号
        "advanced_ml": 0.15,   # 调整权重以使总和为1
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETHUSDT": {
        "strategy_class": "ETHStrategy",
        "ml_weights": None,
        "ml_weighted_threshold": 0.2,
        # --- 根据ML模型评估结果调整后的参数 ---
        "score_entry_threshold": 0.35,
    },
}


# --- 函数定义 (无变动) ---
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
        logger.warning("TensorFlow/PyTorch 未安装。")
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


def generate_model_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("正在为 Keras 模型生成特定的技术指标特征...")
    model_ta_strategy = pta.Strategy(
        name="Pretrained Model Features",
        description="Calculates all features required by the pre-trained Keras model.",
        ta=[
            {"kind": "rsi", "length": 14},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
            {"kind": "bbands", "length": 20, "std": 2.0},
            {"kind": "adx", "length": 14},
            {"kind": "atr", "length": 14},
            {"kind": "obv"},
            {"kind": "ema", "length": 8},
        ],
    )
    df.ta.strategy(model_ta_strategy)
    df["volume_change_rate"] = df["Volume"].pct_change()
    logger.info("Keras 模型特征生成完毕。")
    return df


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(
        f"[{symbol}] 开始数据预处理 (数据范围: {df.index.min()} to {df.index.max()})..."
    )
    if symbol.upper() == "ETHUSDT":
        df = generate_model_specific_features(df)
    rsi_filter = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["ai_filter_signal"] = (
        (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
        .clip(-1, 1)
        .fillna(0)
    )
    df = run_advanced_model_inference(df)
    df = add_ml_features(df)
    df = add_market_regime_features(df)
    daily_start = df.index.min().normalize() - pd.Timedelta(
        days=STRATEGY_PARAMS["mtf_period"] + 5
    )
    daily_end = df.index.max().normalize()
    try:
        data_1d = fetch_binance_klines(
            symbol,
            "1d",
            daily_start.strftime("%Y-%m-%d"),
            daily_end.strftime("%Y-%m-%d"),
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
    except Exception as e:
        logger.warning(f"获取或处理1D数据时失败: {e}。mtf_signal将设为0。")
        df["mtf_signal"] = 0
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
        if w is None:
            w = STRATEGY_PARAMS["score_weights_tf"]
        return (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + m.mtf_signal[-1] * w.get("mtf", 0)
            + m.get_ml_confidence_score() * w.get("ml", 0)
            + m.advanced_ml_signal[-1] * w.get("advanced_ml", 0)
        )

    def _define_mr_entry_signal(self) -> int:
        m = self.main
        long_reentry = (
            m.data.Close[-2] < m.mr_bb_lower[-2]
            and m.data.Close[-1] > m.mr_bb_lower[-1]
        )
        stoch_long_confirm = (
            m.mr_stoch_rsi_k[-1] > m.mr_stoch_rsi_d[-1]
            and m.mr_stoch_rsi_k[-2] <= m.mr_stoch_rsi_d[-2]
            and m.mr_stoch_rsi_k[-1] < 40
        )
        if long_reentry and stoch_long_confirm:
            return 1
        short_reentry = (
            m.data.Close[-2] > m.mr_bb_upper[-2]
            and m.data.Close[-1] < m.mr_bb_upper[-1]
        )
        stoch_short_confirm = (
            m.mr_stoch_rsi_k[-1] < m.mr_stoch_rsi_d[-1]
            and m.mr_stoch_rsi_k[-2] >= m.mr_stoch_rsi_d[-2]
            and m.mr_stoch_rsi_k[-1] > 60
        )
        if short_reentry and stoch_short_confirm:
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
        if self.score_entry_threshold_override is not None:
            self.score_entry_threshold = self.score_entry_threshold_override
        if self.score_weights_tf_override is not None:
            self.score_weights_tf = self.score_weights_tf_override
        self.ml_weights_dict = self.ml_weights_override or asset_overrides.get(
            "ml_weights"
        )
        self.ml_weighted_threshold = (
            self.ml_weighted_threshold_override
            or asset_overrides.get("ml_weighted_threshold")
        )
        strategy_class_name = self.strategy_class_override or asset_overrides.get(
            "strategy_class", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(
            strategy_class_name, BaseAssetStrategy
        )(self)

        close, high, low = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()

        self.market_regime = self.I(lambda: self.data.market_regime)
        self.mtf_signal = self.I(lambda: self.data.mtf_signal)
        self.advanced_ml_signal = self.I(lambda: self.data.advanced_ml_signal)

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

        self.ml_model, self.ml_scaler, self.ml_feature_cols = None, None, None
        self._load_models()

    def _load_models(self):
        model_base_path = "models"
        if not self.symbol:
            return
        if self.symbol.upper() == "ETHUSDT":
            model_path = os.path.join(model_base_path, "eth_trend_model_v1_5m.keras")
            scaler_path = os.path.join(model_base_path, "eth_trend_scaler_v1_5m.joblib")
            feat_path = os.path.join(model_base_path, "feature_columns_5m.joblib")
            if not ADVANCED_ML_LIBS_INSTALLED:
                logger.warning(f"TensorFlow 未安装，无法加载 Keras 模型: {model_path}")
                return
            try:
                if os.path.exists(model_path):
                    self.ml_model = tf.keras.models.load_model(model_path)
                    logger.info(f"✅ 加载 Keras 模型: {model_path}")
                else:
                    logger.warning(f"未找到模型文件: {model_path}")
                if os.path.exists(scaler_path):
                    self.ml_scaler = joblib.load(scaler_path)
                    logger.info(f"✅ 加载 scaler: {scaler_path}")
                else:
                    logger.warning(f"未找到 scaler 文件: {scaler_path}")
                if os.path.exists(feat_path):
                    self.ml_feature_cols = joblib.load(feat_path)
                    logger.info(f"✅ 加载特征列定义: {feat_path}")
                else:
                    logger.warning(f"未找到特征列文件: {feat_path}")
            except Exception as e:
                logger.error(f"加载预训练模型失败: {e}")
        else:
            logger.info(f"[{self.symbol}] 当前仅为 ETHUSDT 支持预训练 Keras 模型。")

    def next(self):
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            if self.data.market_regime[-1] == 1:
                self.run_scoring_system_entry(self.data.Close[-1])
            else:
                self.run_mean_reversion_entry(self.data.Close[-1])

    def get_ml_confidence_score(self) -> float:
        REQUIRED_TIMESTEPS = 60
        if self.ml_model is None or self.ml_feature_cols is None:
            return 0.0
        try:
            if len(self.data.df) < REQUIRED_TIMESTEPS:
                return 0.0
            missing_cols = [
                c for c in self.ml_feature_cols if c not in self.data.df.columns
            ]
            if missing_cols:
                if not hasattr(self, "_warned_missing_cols"):
                    logger.warning(f"ML推理中止：缺少以下特征列: {missing_cols}")
                    self._warned_missing_cols = True
                return 0.0
            X_raw_df = (
                self.data.df[self.ml_feature_cols].iloc[-REQUIRED_TIMESTEPS:].fillna(0)
            )
            if len(X_raw_df) != REQUIRED_TIMESTEPS:
                return 0.0
            X = (
                self.ml_scaler.transform(X_raw_df)
                if self.ml_scaler is not None
                else X_raw_df.values
            )
            X = np.expand_dims(X, axis=0)
            pred = self.ml_model.predict(X, verbose=0)
            if pred is None:
                return 0.0
            pred = np.array(pred)
            if pred.ndim == 2 and pred.shape[1] == 2:
                p_up = float(pred[0, 1])
            elif pred.ndim == 2 and pred.shape[1] == 1:
                p_up = float(pred[0, 0])
            elif pred.ndim == 1:
                p_up = float(pred[0])
            else:
                p_up = float(np.ravel(pred)[0])
            return max(-1.0, min(1.0, p_up * 2.0 - 1.0))
        except Exception as e:
            if not hasattr(self, "_warned_ml_exception"):
                logger.error(f"计算 ML 置信度时出错: {e}", exc_info=False)
                self._warned_ml_exception = True
            return 0.0

    def run_scoring_system_entry(self, price):
        score = self.asset_strategy._calculate_entry_score()
        if abs(score) > self.score_entry_threshold:
            self.open_tf_position(
                price, is_long=(score > 0), confidence_factor=abs(score)
            )

    def run_mean_reversion_entry(self, price):
        signal = self.asset_strategy._define_mr_entry_signal()
        if signal != 0:
            self.open_mr_position(price, is_long=(signal == 1))

    def reset_trade_state(self):
        self.active_sub_strategy, self.chandelier_exit_level = None, 0.0
        self.highest_high_in_trade, self.lowest_low_in_trade = 0, float("inf")
        self.mr_stop_loss, self.tf_initial_stop_loss = 0.0, 0.0

    def manage_open_position(self, p):
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(p)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(p)

    def open_tf_position(self, p, is_long, confidence_factor, score=1.0):
        risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * score * confidence_factor
        )
        if not 0 < size < 0.98:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.tf_initial_stop_loss = p - risk_ps
            self.highest_high_in_trade = self.data.High[-1]
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
        else:
            self.sell(size=size)
            self.tf_initial_stop_loss = p + risk_ps
            self.lowest_low_in_trade = self.data.Low[-1]
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )

    def manage_trend_following_exit(self, p):
        atr = self.tf_atr[-1]
        if self.position.is_long:
            if p < self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade - atr * self.tf_chandelier_atr_multiplier
            )
            if p < self.chandelier_exit_level:
                self.close_position("TF_Chandelier")
        elif self.position.is_short:
            if p > self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade + atr * self.tf_chandelier_atr_multiplier
            )
            if p > self.chandelier_exit_level:
                self.close_position("TF_Chandelier")

    def open_mr_position(self, p, is_long):
        risk_ps = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * self.mr_risk_multiplier
        )
        if not 0 < size < 0.98:
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
            self.close_position("MR")

    def close_position(self, reason: str):
        eq_before = self.equity
        self.position.close()
        self.recent_trade_returns.append(self.equity / eq_before - 1)
        self.reset_trade_state()

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0:
            return 0
        return (risk_pct * self.equity) / (rps / p) / self.equity

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins, losses = [r for r in self.recent_trade_returns if r > 0], [
            r for r in self.recent_trade_returns if r < 0
        ]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        reward_ratio = (sum(wins) / len(wins)) / (abs(sum(losses) / len(losses)))
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


# --- 主程序入口 ---
if __name__ == "__main__":
    logger.info(f"🚀 (V41.05-MR-Enhanced) 开始运行 (使用预训练模型，无本地训练)...")

    warmup_days = 5
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_fetch_start_dt = backtest_start_dt - pd.Timedelta(days=warmup_days)

    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(
        f"为满足预热需求，数据获取将从 {data_fetch_start_dt.strftime('%Y-%m-%d')} 开始。"
    )

    raw_data = {
        s: fetch_binance_klines(
            s, CONFIG["interval"], data_fetch_start_dt, CONFIG["backtest_end_date"]
        )
        for s in CONFIG["symbols_to_test"]
    }
    raw_data = {s: d for s, d in raw_data.items() if not d.empty}
    if not raw_data:
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    logger.info("### 模式: 已禁用本地训练 (使用 models/ 目录下的预训练模型) ###")
    logger.info(f"### 准备回测数据 (包含预热期计算) ###")
    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        logger.info(f"为 {symbol} 预处理完整数据（包括预热期）...")
        processed_full_data = preprocess_data_for_strategy(data, symbol)
        backtest_period_slice = processed_full_data.loc[
            CONFIG["backtest_start_date"] :
        ].copy()
        if not backtest_period_slice.empty:
            processed_backtest_data[symbol] = backtest_period_slice
            logger.info(f"[{symbol}] 预热期计算完成，回测数据准备就绪。")
        else:
            logger.warning(f"[{symbol}] 在截取回测时间段后数据为空。")

    if not processed_backtest_data:
        logger.error("无最终回测数据，程序终止。")
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
        override_keys = [
            "strategy_class",
            "score_entry_threshold",
            "score_weights_tf",
            "ml_weights",
            "ml_weighted_threshold",
        ]
        for key in override_keys:
            if key in asset_overrides:
                bt_params[f"{key}_override"] = asset_overrides[key]
        bt = Backtest(
            data,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            finalize_trades=True,
        )
        stats = bt.run(**bt_params)
        all_stats[symbol], total_equity = (
            stats,
            total_equity + stats["Equity Final [$]"],
        )
        print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
        print(stats)  # 直接打印完整的 stats 对象

        # <<< 已移除 >>> bt.plot() 的调用

    if all_stats:
        initial_total = CONFIG["initial_cash"] * len(all_stats)
        ret = ((total_equity - initial_total) / initial_total) * 100
        print(f"\n{'#'*80}\n                 组合策略表现总览\n{'#'*80}")
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n"
                f"    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n"
                f"    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A')}"
            )
        print(
            f"\n--- 投资组合整体表现 ---\n"
            f"总初始资金: ${initial_total:,.2f}\n"
            f"总最终权益: ${total_equity:,.2f}\n"
            f"组合总回报率: {ret:.2f}%"
        )
