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
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    import tensorflow as tf

    ADVANCED_ML_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_ML_LIBS_INSTALLED = False

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 日志与字体设置 ---
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
    "backtest_end_date": "2025-11-05",
    "initial_cash": 500_000,
    "commission": 0.00075,
    "spread": 0.0002,
    "show_plots": False,
    "data_lookback_days": 90,
    "enable_ml_component": True,
}
# --- Keras模型文件路径配置 ---
KERAS_MODEL_PATH = "models/eth_trend_model_v1_15m.keras"
SCALER_PATH = "models/eth_trend_scaler_v1_15m.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns_15m.joblib"
KERAS_SEQUENCE_LENGTH = 60

# --- 策略参数 (采纳您参考代码中的优化版本) ---
STRATEGY_PARAMS = {
    "tsl_enabled": True,
    "tsl_activation_atr_mult": 1.8,
    "tsl_trailing_atr_mult": 2.2,
    "kelly_trade_history": 25,
    "default_risk_pct": 0.012,
    "max_risk_pct": 0.035,
    "regime_adx_period": 14,
    "regime_atr_period": 14,
    "regime_atr_slope_period": 6,
    "regime_rsi_period": 14,
    "regime_rsi_vol_period": 14,
    "regime_norm_period": 252,
    "regime_hurst_period": 80,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.6,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 40,
    "score_entry_threshold": 0.5,
    "score_weights_tf": {
        "breakout": 0.22,
        "momentum": 0.18,
        "mtf": 0.12,
        "ml": 0.23,
        "advanced_ml": 0.25,
    },
    "tf_donchian_period": 24,
    "tf_ema_fast_period": 21,
    "tf_ema_slow_period": 60,
    "tf_adx_confirm_threshold": 20,
    "tf_chandelier_atr_multiplier": 3.0,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_stop_loss_atr_multiplier": 1.5,
}

# --- 分状态参数优化配置 ---
# 这里的参数将覆盖上面的默认值，以实现更精细的控制
REGIME_SPECIFIC_PARAMS = {
    # 状态0: 震荡市
    0: {
        "score_entry_threshold": 999,  # 禁用趋势交易
        "mr_bb_period": 20,
        "mr_bb_std": 1.8,
        "mr_stop_loss_atr_multiplier": 1.2,
        "mr_risk_multiplier": 0.7,
    },
    # 状态1: 趋势市
    1: {
        "score_entry_threshold": 0.40,  # 启用趋势交易
        "tf_donchian_period": 25,
        "tf_ema_fast_period": 20,
        "tf_ema_slow_period": 60,
        "tf_stop_loss_atr_multiplier": 2.8,  # 在趋势市中给予更宽的初始止损
        "tf_chandelier_atr_multiplier": 2.5,
        "tsl_trailing_atr_mult": 1.8,  # 更积极地追踪利润
        "mr_risk_multiplier": 0.0,  # 禁用均值回归
    },
}


# --- 数据处理函数 (无变化) ---
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
    df["feature_bb_width_norm"] = norm(
        (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    )
    df["feature_vol_pct_change_norm"] = norm(df.Volume.pct_change(periods=1).abs())
    return df


def find_optimal_clusters(data: np.ndarray, max_clusters=8):
    logger.info(f"正在使用轮廓系数寻找最佳市场状态数量 (2 to {max_clusters})...")
    scores, best_score, best_n = {}, -1, 2
    sample_size = min(len(data), 5000)
    sample_indices = np.random.choice(len(data), sample_size, replace=False)
    for n_clusters in range(2, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data[sample_indices], labels[sample_indices])
            scores[n_clusters] = score
            if score > best_score:
                best_score, best_n = score, n_clusters
        except Exception:
            continue
    logger.info(f"轮廓系数得分: { {k: round(v, 4) for k, v in scores.items()} }")
    logger.info(f"✅ 找到最佳市场状态数量: {best_n} (得分: {best_score:.4f})")
    return best_n


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [
        "feature_adx_norm",
        "feature_atr_slope_norm",
        "feature_rsi_vol_norm",
        "feature_hurst",
        "feature_bb_width_norm",
        "feature_vol_pct_change_norm",
    ]
    if any(col not in df.columns for col in feature_columns):
        logger.error("聚类特征缺失，跳过。")
        df["market_regime"] = 1
        return df
    features_df = df[feature_columns].copy()
    features_df.ffill(inplace=True)
    features_df.dropna(inplace=True)
    if len(features_df) < 50:
        logger.warning("特征数据过少，跳过聚类。")
        df["market_regime"] = 1
        return df
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    optimal_n = find_optimal_clusters(scaled_features)
    logger.info(f"正在使用 K-Means (n_clusters={optimal_n}) 定义市场状态...")
    kmeans = KMeans(n_clusters=optimal_n, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(scaled_features)
    df["market_regime_cluster"] = np.nan
    df.loc[features_df.index, "market_regime_cluster"] = labels
    df["market_regime_cluster"] = df["market_regime_cluster"].ffill()
    centers_df = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_columns
    )
    centers_df["regime_label"] = centers_df.index
    logger.info("K-Means 聚类中心 (市场状态解读):")
    centers_df["trend_score"] = (
        centers_df["feature_adx_norm"] + (centers_df["feature_hurst"] - 0.5) * 2
    )
    centers_df = centers_df.sort_values(by="trend_score", ascending=False).reset_index(
        drop=True
    )
    num_trending_regimes = int(np.ceil(optimal_n / 2))
    trending_regimes = centers_df.head(num_trending_regimes)["regime_label"].tolist()
    logger.info(f"以下状态被识别为 '趋势市': {trending_regimes}")
    print(centers_df[["regime_label", "trend_score"] + feature_columns].round(4))
    df["market_regime"] = np.where(
        df["market_regime_cluster"].isin(trending_regimes), 1, -1
    )
    logger.info("✅ K-Means 市场状态定义完成。")
    return df


def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("正在为 Keras 模型生成特定特征...")
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
    logger.info("✅ Keras 特征生成完成。")
    return df


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(
        f"[{symbol}] 开始数据预处理 (范围: {df.index.min()} to {df.index.max()})..."
    )
    rsi_filter = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["ai_filter_signal"] = (
        (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
        .clip(-1, 1)
        .fillna(0)
    )

    df["volatility"] = df["Close"].pct_change().rolling(24 * 7).std() * np.sqrt(
        24 * 365
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
    logger.info(f"[{symbol}] 数据预处理完成。行数: {len(df)}")
    return df


def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    sequences = []
    data_len = len(data)
    for i in range(data_len - sequence_length + 1):
        sequences.append(data[i : i + sequence_length])
    return np.array(sequences)


# --- 策略类定义 ---
class UltimateStrategy(Strategy):
    symbol: str = "ETHUSDT"

    def init(self):
        self.params = STRATEGY_PARAMS.copy()
        self.regime_params = {
            idx: {**self.params, **overrides}
            for idx, overrides in REGIME_SPECIFIC_PARAMS.items()
        }

        close, high, low = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )
        self.recent_trade_returns = deque(maxlen=self.params["kelly_trade_history"])
        self.reset_trade_state()

        self.market_regime = self.I(lambda: self.data.market_regime)
        self.market_regime_cluster = self.I(lambda: self.data.market_regime_cluster)
        self.mtf_signal = self.I(lambda: self.data.mtf_signal)
        self.advanced_ml_signal = self.I(lambda: self.data.advanced_ml_signal)
        self.macro_trend = self.I(lambda: self.data.macro_trend_filter)
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.params["tf_atr_period"]
            ).average_true_range()
        )

        self.indicators = {}
        all_params_to_calc = [self.params] + list(self.regime_params.values())
        unique_periods = {
            "tf_donchian": set(p["tf_donchian_period"] for p in all_params_to_calc),
            "tf_ema_fast": set(p["tf_ema_fast_period"] for p in all_params_to_calc),
            "tf_ema_slow": set(p["tf_ema_slow_period"] for p in all_params_to_calc),
            "mr_bb": set(
                (p["mr_bb_period"], p["mr_bb_std"]) for p in all_params_to_calc
            ),
        }

        for period in unique_periods["tf_donchian"]:
            self.indicators[f"donchian_h_{period}"] = self.I(
                lambda period=period: high.rolling(period).max().shift(1)
            )
            self.indicators[f"donchian_l_{period}"] = self.I(
                lambda period=period: low.rolling(period).min().shift(1)
            )

        for period in unique_periods["tf_ema_fast"]:
            self.indicators[f"ema_fast_{period}"] = self.I(
                lambda period=period: ta.trend.EMAIndicator(
                    close, period
                ).ema_indicator()
            )

        for period in unique_periods["tf_ema_slow"]:
            self.indicators[f"ema_slow_{period}"] = self.I(
                lambda period=period: ta.trend.EMAIndicator(
                    close, period
                ).ema_indicator()
            )

        for period, std in unique_periods["mr_bb"]:
            bb = ta.volatility.BollingerBands(close, window=period, window_dev=std)
            self.indicators[f"bb_upper_{period}_{std:.1f}"] = self.I(bb.bollinger_hband)
            self.indicators[f"bb_lower_{period}_{std:.1f}"] = self.I(bb.bollinger_lband)
            self.indicators[f"bb_mid_{period}_{std:.1f}"] = self.I(bb.bollinger_mavg)

        stoch_rsi = ta.momentum.StochRSIIndicator(
            close, window=14, smooth1=3, smooth2=3
        )
        self.stoch_rsi_k = self.I(stoch_rsi.stochrsi_k)
        self.stoch_rsi_d = self.I(stoch_rsi.stochrsi_d)

        self.keras_model, self.scaler, self.feature_columns = (
            self._load_keras_model_and_dependencies()
        )
        self.keras_signal = self.I(self._calculate_keras_predictions)

    def next(self):
        current_regime_idx = int(self.market_regime_cluster[-1])
        p = self.regime_params.get(current_regime_idx, self.params)

        if self.position:
            self.manage_open_position(self.data.Close[-1], p)
        else:
            is_trending_regime = (
                self.market_regime[-1] == 1 and p["score_entry_threshold"] < 999
            )
            is_mr_allowed = p["mr_risk_multiplier"] > 0

            if self.macro_trend[-1] == 1:
                if is_trending_regime:
                    score = self._calculate_entry_score(p)
                    if score > p["score_entry_threshold"]:
                        self.open_tf_position(self.data.Close[-1], True, score, p)
                elif is_mr_allowed and self._define_mr_entry_signal(p) == 1:
                    self.open_mr_position(self.data.Close[-1], True, p)
            elif self.macro_trend[-1] == -1:
                if is_trending_regime:
                    score = self._calculate_entry_score(p)
                    if score < -p["score_entry_threshold"]:
                        self.open_tf_position(self.data.Close[-1], False, abs(score), p)
                elif is_mr_allowed and self._define_mr_entry_signal(p) == -1:
                    self.open_mr_position(self.data.Close[-1], False, p)

    def _calculate_entry_score(self, p) -> float:
        w = self.params["score_weights_tf"]
        donchian_h = self.indicators[f"donchian_h_{p['tf_donchian_period']}"][-1]
        donchian_l = self.indicators[f"donchian_l_{p['tf_donchian_period']}"][-1]
        ema_fast = self.indicators[f"ema_fast_{p['tf_ema_fast_period']}"][-1]
        ema_slow = self.indicators[f"ema_slow_{p['tf_ema_slow_period']}"][-1]

        b_s = (
            1
            if self.data.High[-1] > donchian_h
            else -1 if self.data.Low[-1] < donchian_l else 0
        )
        mo_s = 1 if ema_fast > ema_slow else -1
        return (
            b_s * w["breakout"]
            + mo_s * w["momentum"]
            + self.mtf_signal[-1] * w["mtf"]
            + self.keras_signal[-1] * w["ml"]
            + self.advanced_ml_signal[-1] * w["advanced_ml"]
        )

    def _define_mr_entry_signal(self, p) -> int:
        bb_lower = self.indicators[f"bb_lower_{p['mr_bb_period']}_{p['mr_bb_std']:.1f}"]
        bb_upper = self.indicators[f"bb_upper_{p['mr_bb_period']}_{p['mr_bb_std']:.1f}"]

        if (
            crossover(self.stoch_rsi_k, self.stoch_rsi_d)
            and self.stoch_rsi_k[-1] < 40
            and self.data.Close[-1] > bb_lower[-1]
        ):
            return 1
        if (
            crossover(self.stoch_rsi_d, self.stoch_rsi_k)
            and self.stoch_rsi_k[-1] > 60
            and self.data.Close[-1] < bb_upper[-1]
        ):
            return -1
        return 0

    def manage_open_position(self, price, p):
        # <<< 核心修复：严格隔离平仓逻辑 >>>
        if self.active_sub_strategy == "TF":
            self.manage_tf_exit(price, p)
        elif self.active_sub_strategy == "MR":
            self.manage_mr_exit(price, p)

    def manage_tf_exit(self, price, p):
        # 阶段一：检查是否触及初始硬止损
        if (self.position.is_long and price < self.stop_loss_price) or (
            self.position.is_short and price > self.stop_loss_price
        ):
            self.position.close()
            return

        # 阶段二：检查是否激活利润保护机制
        self._activate_trailing_stop(p)

        # 阶段三：如果已激活，则计算并更新动态止损
        if self.trailing_stop_active:
            tsl_price = self._get_trailing_stop_price(p)
            chandelier_price = self._get_chandelier_price(p)

            if self.position.is_long:
                new_sl = max(tsl_price, chandelier_price)
                # 保证止损只进不退
                if new_sl > self.stop_loss_price:
                    self.stop_loss_price = new_sl
            else:  # Short
                new_sl = min(tsl_price, chandelier_price)
                # 保证止损只进不退
                if new_sl < self.stop_loss_price:
                    self.stop_loss_price = new_sl

    def manage_mr_exit(self, price, p):
        # MR策略只有两种退出方式：达到中轨止盈，或触及初始止损
        bb_mid = self.indicators[f"bb_mid_{p['mr_bb_period']}_{p['mr_bb_std']:.1f}"][-1]

        # 检查止盈
        if (self.position.is_long and price >= bb_mid) or (
            self.position.is_short and price <= bb_mid
        ):
            self.position.close()
            return

        # 检查止损
        if (self.position.is_long and price < self.stop_loss_price) or (
            self.position.is_short and price > self.stop_loss_price
        ):
            self.position.close()

    def _activate_trailing_stop(self, p):
        if self.trailing_stop_active or not self.params["tsl_enabled"]:
            return

        atr = self.tf_atr[-1]
        activation_dist = atr * p["tsl_activation_atr_mult"]
        entry_price = self.trades[-1].entry_price

        if (
            self.position.is_long
            and self.data.Close[-1] >= entry_price + activation_dist
        ) or (
            not self.position.is_long
            and self.data.Close[-1] <= entry_price - activation_dist
        ):
            self.trailing_stop_active = True
            logger.info(f"利润保护机制在价格 {self.data.Close[-1]} 被激活")

    def _get_trailing_stop_price(self, p):
        trailing_dist = self.tf_atr[-1] * p["tsl_trailing_atr_mult"]
        return (
            self.data.Close[-1] - trailing_dist
            if self.position.is_long
            else self.data.Close[-1] + trailing_dist
        )

    def _get_chandelier_price(self, p):
        if self.position.is_long:
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            return (
                self.highest_high_in_trade
                - self.tf_atr[-1] * p["tf_chandelier_atr_multiplier"]
            )
        else:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            return (
                self.lowest_low_in_trade
                + self.tf_atr[-1] * p["tf_chandelier_atr_multiplier"]
            )

    def open_tf_position(self, price, is_long, confidence_factor, params):
        risk_ps = self.tf_atr[-1] * params["tf_stop_loss_atr_multiplier"]
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            price, risk_ps, self._calculate_dynamic_risk() * confidence_factor
        )
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        self.stop_loss_price = price - risk_ps if is_long else price + risk_ps
        # 我们不再向框架传递sl，由我们自己的逻辑全权管理
        if is_long:
            self.buy(size=size)
        else:
            self.sell(size=size)

    def open_mr_position(self, price, is_long, params):
        risk_ps = self.tf_atr[-1] * params["mr_stop_loss_atr_multiplier"]
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            price,
            risk_ps,
            self._calculate_dynamic_risk() * params["mr_risk_multiplier"],
        )
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        self.stop_loss_price = price - risk_ps if is_long else price + risk_ps
        # 我们不再向框架传递sl，由我们自己的逻辑全权管理
        if is_long:
            self.buy(size=size)
        else:
            self.sell(size=size)

    def _load_keras_model_and_dependencies(self):
        if not CONFIG["enable_ml_component"] or not ADVANCED_ML_LIBS_INSTALLED:
            return None, None, None
        try:
            model, scaler, feature_columns = (
                tf.keras.models.load_model(KERAS_MODEL_PATH),
                joblib.load(SCALER_PATH),
                joblib.load(FEATURE_COLUMNS_PATH),
            )
            logger.info(f"✅ [{self.symbol}] 成功加载Keras模型。")
            return model, scaler, feature_columns
        except Exception as e:
            logger.error(f"[{self.symbol}] 加载Keras模型失败: {e}")
            return None, None, None

    def _calculate_keras_predictions(self):
        if self.keras_model is None:
            return np.zeros(len(self.data.Close))
        try:
            features_df = self.data.df[self.feature_columns].fillna(0)
        except KeyError as e:
            logger.error(
                f"Keras预测所需特征缺失: {e}。请检查模型与数据处理流程是否一致。"
            )
            return np.zeros(len(self.data.Close))
        scaled_features_2d = self.scaler.transform(features_df)
        scaled_features_3d = create_sequences(scaled_features_2d, KERAS_SEQUENCE_LENGTH)
        predictions_proba = self.keras_model.predict(
            scaled_features_3d, verbose=0
        ).flatten()
        final_signals = np.zeros(len(self.data.Close))
        final_signals[len(self.data.Close) - len(predictions_proba) :] = (
            predictions_proba - 0.5
        ) * 2
        return final_signals

    def reset_trade_state(self):
        self.active_sub_strategy = None
        self.trailing_stop_active = False
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")
        self.stop_loss_price = 0

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0:
            return 0
        units = (self.equity * risk_pct) / rps
        if units * p > self.equity:
            units = (self.equity * 0.95) / p
        return max(0, int(units))

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.params["kelly_trade_history"]:
            return self.params["default_risk_pct"]
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return self.params["default_risk_pct"]
        win_rate, avg_win, avg_loss = (
            len(wins) / len(self.recent_trade_returns),
            sum(wins) / len(wins),
            abs(sum(losses) / len(losses)),
        )
        if avg_loss == 0:
            return self.params["default_risk_pct"]
        reward_ratio = avg_win / avg_loss
        if reward_ratio == 0:
            return self.params["default_risk_pct"]
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5), self.params["max_risk_pct"])


if __name__ == "__main__":
    logger.info(f"🚀 (V46.0-Let-Profits-Run) 开始运行...")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_lookback = timedelta(days=CONFIG["data_lookback_days"])
    data_fetch_start_date = (backtest_start_dt - data_lookback).strftime("%Y-%m-%d")
    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期: {data_fetch_start_date}")

    symbol = CONFIG["symbols_to_test"][0]
    raw_data = fetch_binance_klines(
        symbol, CONFIG["interval"], data_fetch_start_date, CONFIG["backtest_end_date"]
    )

    if raw_data.empty:
        logger.error("数据获取失败，终止。")
        exit()

    logger.info(f"### 准备回测数据 (开始日期: {CONFIG['backtest_start_date']}) ###")
    full_processed_data = preprocess_data_for_strategy(raw_data, symbol)
    final_data = full_processed_data.loc[CONFIG["backtest_start_date"] :].copy()

    if final_data.empty:
        logger.error("无回测数据，终止。")
        exit()

    logger.info(f"### 进入回测模式 ###")

    bt = Backtest(
        final_data,
        UltimateStrategy,
        cash=CONFIG["initial_cash"],
        commission=CONFIG["commission"],
    )
    stats = bt.run(symbol=symbol)

    print(f"\n{'#'*80}\n                 最终策略表现总览\n{'#'*80}")
    print(stats)
    if CONFIG["show_plots"]:
        bt.plot()
