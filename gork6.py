# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V41.01-No-News)

版本更新：
- (V41.01-No-News) 根据用户要求，完全移除了新闻情绪获取和分析功能，以简化流程并消除API依赖。
- (V41.00-Walk-Forward) 实施了前向展开(Walk-Forward)训练和验证框架以解决时间序列过拟合问题。
    - 废弃了 `train_test_split`，采用严格按时间顺序的训练和验证方式。
    - 模型现在会在回测期间，基于一个滚动的历史数据窗口被定期重新训练。
    - 策略实现了动态模型加载机制，以在回测中自动使用最新的、合适的模型。
    - 新增了用于控制前向展开训练流程的配置选项。
"""

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

# 检查机器学习库是否安装
try:
    import lightgbm as lgb
    from sklearn.metrics import classification_report

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False

# 检查高级机器学习库是否安装
try:
    import tensorflow as tf

    ADVANCED_ML_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_ML_LIBS_INSTALLED = False

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 日志和字体配置 ---
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
    "symbols_to_test": ["BTCUSDT", "ETHUSDT"],
    "interval": "1h",
    "backtest_start_date": "2025-01-01",  # 回测开始日期
    "backtest_end_date": "2025-10-13",  # 回测结束日期
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,
    "run_monte_carlo": False,
    "show_plots": False,
    # --- 前向展开训练 (Walk-Forward Training) 配置 ---
    "enable_walk_forward_training": True,  # 设为 True 以启用周期性模型重训练
    "training_window_days": 365 * 2,  # 用于训练每个模型的历史数据天数 (例如 2 年)
    "retrain_every_days": 90,  # 每隔多少天重训练一次模型 (例如每季度)
}
ML_HORIZONS = [4, 8, 12]

# --- 参数与类定义 (无变动) ---
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
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 30,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "score_entry_threshold": 0.4,
    "score_weights_tf": {
        "breakout": 0.20,
        "momentum": 0.15,
        "mtf": 0.10,
        "ml": 0.30,
        "advanced_ml": 0.25,
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "BTCUSDT": {
        "strategy_class": "BTCStrategy",
        "ml_weights": {"4h": 0.2, "8h": 0.3, "12h": 0.5},
        "ml_weighted_threshold": 0.45,
        "score_entry_threshold": 0.5,
        "score_weights_tf": {
            "breakout": 0.15,
            "momentum": 0.20,
            "mtf": 0.15,
            "ml": 0.25,
            "advanced_ml": 0.25,
        },
    },
    "ETHUSDT": {
        "strategy_class": "ETHStrategy",
        "ml_weights": {"4h": 0.25, "8h": 0.35, "12h": 0.4},
        "ml_weighted_threshold": 0.2,
        "score_entry_threshold": 0.35,
    },
    "SOLUSDT": {
        "strategy_class": "SOLStrategy",
        "ml_weights": {"4h": 0.3, "8h": 0.4, "12h": 0.3},
        "ml_weighted_threshold": 0.25,
        "score_entry_threshold": 0.4,
        "score_weights_tf": {
            "breakout": 0.3,
            "momentum": 0.1,
            "mtf": 0.1,
            "ml": 0.25,
            "advanced_ml": 0.25,
        },
    },
}


class StrategyMemory:  # 无变动
    def __init__(self, filepath="strategy_memory.csv"):
        self.filepath = filepath
        self.columns = [
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

    def record_optimization(self, timestamp, symbol, regime, best_params, performance):
        new_records = [
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "regime": regime,
                "param_key": key,
                "param_value": value,
                "performance": performance,
            }
            for key, value in best_params.items()
        ]
        new_df = pd.DataFrame(new_records)
        self.memory_df = pd.concat([self.memory_df, new_df], ignore_index=True)
        self.memory_df.sort_values(by="timestamp", inplace=True)
        self.memory_df.drop_duplicates(
            subset=["timestamp", "symbol", "regime", "param_key"],
            keep="last",
            inplace=True,
        )
        self.memory_df.to_csv(self.filepath, index=False)
        logger.info(f"🧠 记忆库已更新: {symbol} 在 {regime} 状态下的最优参数已记录。")

    def get_best_params(self, timestamp, symbol, regime):
        relevant_memory = self.memory_df[
            (self.memory_df["symbol"] == symbol)
            & (self.memory_df["regime"] == regime)
            & (self.memory_df["timestamp"] <= timestamp)
        ]
        if relevant_memory.empty:
            return None
        latest_timestamp = relevant_memory["timestamp"].max()
        latest_params_df = relevant_memory[
            relevant_memory["timestamp"] == latest_timestamp
        ]
        return pd.Series(
            latest_params_df.param_value.values, index=latest_params_df.param_key
        ).to_dict()


def fetch_binance_klines(  # 无变动
    symbol: str, interval: str, start_str: str, end_str: str = None, limit: int = 1000
) -> pd.DataFrame:
    url, columns = "https://api.binance.com/api/v3/klines", [
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
    start_ts, end_ts = int(pd.to_datetime(start_str).timestamp() * 1000), (
        int(pd.to_datetime(end_str).timestamp() * 1000)
        if end_str
        else int(time.time() * 1000)
    )
    all_data, retries, last_exception = [], 5, None
    while start_ts < end_ts:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit,
        }
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                if not data:
                    start_ts = end_ts
                    break
                all_data.extend(data)
                start_ts = data[-1][0] + 1
                break
            except requests.exceptions.RequestException as e:
                last_exception = e
                time.sleep(2**attempt)
        else:
            logger.error(
                f"获取 {symbol} 数据在 {retries} 次尝试后彻底失败. 最后错误: {last_exception}"
            )
            return pd.DataFrame()
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=columns)[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    logger.info(f"✅ 获取 {symbol} 数据成功：{len(df)} 条")
    return df


def compute_hurst(ts, max_lag=100):  # 无变动
    if len(ts) < 10:
        return 0.5
    lags, tau, valid_lags = range(2, min(max_lag, len(ts) // 2 + 1)), [], []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)
        if std_diff > 0:
            tau.append(std_diff)
            valid_lags.append(lag)
    if len(tau) < 2:
        return 0.5
    try:
        return max(0.0, min(1.0, np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]))
    except:
        return 0.5


def run_advanced_model_inference(df: pd.DataFrame) -> pd.DataFrame:  # 无变动
    logger.info("正在运行高级模型推理 (模拟)...")
    if not ADVANCED_ML_LIBS_INSTALLED:
        logger.warning("TensorFlow/PyTorch 未安装。高级模型信号将为中性(0)。")
        df["advanced_ml_signal"] = 0.0
        return df
    if "ai_filter_signal" in df.columns:
        df["advanced_ml_signal"] = (
            df["ai_filter_signal"].rolling(window=24, min_periods=12).mean().fillna(0)
        )
    else:
        df["advanced_ml_signal"] = 0.0
    logger.info("高级模型推理 (模拟) 完成。")
    return df


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    p = STRATEGY_PARAMS
    adx = ta.trend.ADXIndicator(df.High, df.Low, df.Close, p["regime_adx_period"]).adx()
    atr = ta.volatility.AverageTrueRange(
        df.High, df.Low, df.Close, p["regime_atr_period"]
    ).average_true_range()
    rsi = ta.momentum.RSIIndicator(df.Close, p["regime_rsi_period"]).rsi()
    norm = lambda s: (
        (s - s.rolling(p["regime_norm_period"]).min())
        / (
            s.rolling(p["regime_norm_period"]).max()
            - s.rolling(p["regime_norm_period"]).min()
        )
    ).fillna(0.5)
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
    # **注意**: 此处已移除 feature_news_sentiment 的创建
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["feature_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:  # 无变动
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
    df["market_regime_final"] = (
        df["trend_regime"].astype(str) + "_" + df["volatility_regime"].astype(str)
    )
    df["market_regime"] = np.where(df["trend_regime"] == "Trending", 1, -1)
    return df


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(
        f"[{symbol}] 开始数据预处理 (数据范围: {df.index.min()} to {df.index.max()})..."
    )

    # **注意**: 此处已移除对 get_news_sentiment 的调用

    rsi_filter = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["ai_filter_signal"] = (
        (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
        .clip(-1, 1)
        .fillna(0)
    )
    df = run_advanced_model_inference(df)
    df = add_ml_features(df)
    df = add_market_regime_features(df)

    # 获取与当前小时数据匹配的日线数据
    daily_start = df.index.min().normalize() - pd.Timedelta(
        days=STRATEGY_PARAMS["mtf_period"] + 1
    )
    daily_end = df.index.max().normalize()
    data_1d = fetch_binance_klines(
        symbol,
        "1d",
        daily_start.strftime("%Y-%m-%d"),
        daily_end.strftime("%Y-%m-%d"),
    )
    if data_1d is not None and not data_1d.empty:
        sma = ta.trend.SMAIndicator(
            data_1d["Close"], window=STRATEGY_PARAMS["mtf_period"]
        ).sma_indicator()
        mtf_signal_1d = pd.Series(
            np.where(data_1d["Close"] > sma, 1, -1), index=data_1d.index
        )
        df["mtf_signal"] = mtf_signal_1d.reindex(df.index, method="ffill").fillna(0)
    else:
        logger.warning(f"[{symbol}] 未能获取日线数据用于MTF信号计算。")
        df["mtf_signal"] = 0

    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


# --- 机器学习模型训练 (Walk-Forward 版) ---
def train_and_save_model(
    training_data: pd.DataFrame, symbol: str, training_end_date: pd.Timestamp
):
    """
    使用严格按时间顺序的数据进行模型训练和保存，以防止过拟合。
    """
    if not ML_LIBS_INSTALLED:
        logger.warning("缺少 LightGBM 或 scikit-learn 库，跳过模型训练。")
        return
    logger.info(
        f"--- 🤖 [Walk-Forward] 开始为 {symbol} 训练模型 (数据截止于: {training_end_date.date()}) ---"
    )

    features = [col for col in training_data.columns if "feature_" in col]
    if not features:
        logger.error(f"[{symbol}] 找不到任何特征列 (feature_) 用于训练。")
        return
    logger.info(f"[{symbol}] 使用以下特征进行训练: {features}")

    for h in ML_HORIZONS:
        logger.info(f"正在为 {h}h 预测窗口准备数据...")
        data = training_data.copy()
        data[f"target_{h}h"] = (data["Close"].shift(-h) > data["Close"]).astype(int)
        df_train = data.dropna(subset=[f"target_{h}h"] + features)

        X = df_train[features]
        y = df_train[f"target_{h}h"]

        if len(X) < 200 or len(y.unique()) < 2:
            logger.warning(
                f"[{symbol}-{h}h] 数据不足或目标类别单一，跳过此预测窗口的训练。"
            )
            continue

        # --- 关键改动：不再使用 train_test_split ---
        # 使用训练数据的最后10%作为验证集，以监控过拟合，同时保持时间顺序
        eval_size = int(len(X) * 0.1)
        X_train, X_eval = X[:-eval_size], X[-eval_size:]
        y_train, y_eval = y[:-eval_size], y[-eval_size:]

        logger.info(f"开始训练 {symbol} 的 {h}h 模型...")
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=100, random_state=42
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )

        y_pred = model.predict(X_eval)
        logger.info(
            f"[{symbol}-{h}h] 模型在时间顺序验证集上的评估报告:\n{classification_report(y_eval, y_pred)}"
        )

        # 使用训练结束日期来命名模型文件
        date_str = training_end_date.strftime("%Y%m%d")
        model_filename = f"directional_model_{symbol}_{h}h_{date_str}.joblib"
        joblib.dump(model, model_filename)
        logger.info(f"✅ [{symbol}-{h}h] 模型训练完成并已保存至: {model_filename}")


class BaseAssetStrategy:  # 无变动
    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy

    def _calculate_entry_score(self) -> float:
        main = self.main
        weights = main.score_weights_tf
        breakout_score = (
            1
            if main.data.High[-1] > main.tf_donchian_h[-1]
            else -1 if main.data.Low[-1] < main.tf_donchian_l[-1] else 0
        )
        momentum_score = 1 if main.tf_ema_fast[-1] > main.tf_ema_slow[-1] else -1
        mtf_score = main.mtf_signal[-1]
        ml_score = main.get_ml_confidence_score()
        advanced_ml_score = main.advanced_ml_signal[-1]
        return (
            breakout_score * weights.get("breakout", 0)
            + momentum_score * weights.get("momentum", 0)
            + mtf_score * weights.get("mtf", 0)
            + ml_score * weights.get("ml", 0)
            + advanced_ml_score * weights.get("advanced_ml", 0)
        )

    def _define_mr_entry_signal(self) -> int:
        main = self.main
        return (
            1
            if crossover(main.data.Close, main.mr_bb_lower)
            and main.mr_rsi[-1] < main.mr_rsi_oversold
            else (
                -1
                if crossover(main.mr_bb_upper, main.data.Close)
                and main.mr_rsi[-1] > main.mr_rsi_overbought
                else 0
            )
        )


class BTCStrategy(BaseAssetStrategy):  # 无变动
    def _calculate_entry_score(self) -> float:
        if self.main.tf_adx[-1] > 20:
            return super()._calculate_entry_score()
        return 0


class ETHStrategy(BaseAssetStrategy):  # 无变动
    pass


class SOLStrategy(BaseAssetStrategy):  # 无变动
    def _calculate_entry_score(self) -> float:
        base_score = super()._calculate_entry_score()
        try:
            volume_series = pd.Series(self.main.data.Volume)
            if len(volume_series) < 20:
                return base_score
            current_volume = volume_series.iloc[-1]
            mean_volume = volume_series.rolling(20).mean().iloc[-2]
            volume_spike = current_volume > mean_volume * 1.5
            if volume_spike and abs(base_score) > 0:
                return base_score * 1.1
        except Exception:
            return base_score
        return base_score


STRATEGY_MAPPING = {
    "BaseAssetStrategy": BaseAssetStrategy,
    "BTCStrategy": BTCStrategy,
    "ETHStrategy": ETHStrategy,
    "SOLStrategy": SOLStrategy,
}


class UltimateStrategy(Strategy):
    strategy_class_override, memory_instance = None, None
    score_entry_threshold_override, score_weights_tf_override = None, None
    ml_weights_override, ml_weighted_threshold_override = None, None
    for key, value in STRATEGY_PARAMS.items():
        exec(f"{key} = {value}")
    vol_weight, symbol = 1.0, None

    def init(self):
        # --- 参数和资产特定策略的初始化 (基本无变动) ---
        self.ml_weighted_threshold = getattr(
            self,
            "ml_weighted_threshold_override",
            ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get(
                "ml_weighted_threshold", 0.3
            ),
        )
        self.regime_score_threshold = getattr(
            self,
            "regime_score_threshold_override",
            STRATEGY_PARAMS["regime_score_threshold"],
        )
        self.score_entry_threshold = getattr(
            self,
            "score_entry_threshold_override",
            ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get(
                "score_entry_threshold", 0.4
            ),
        )
        score_weights_override = getattr(self, "score_weights_tf_override", None)
        if score_weights_override is not None:
            self.score_weights_tf = score_weights_override
        else:
            self.score_weights_tf = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get(
                "score_weights_tf", STRATEGY_PARAMS["score_weights_tf"]
            )
        self.ml_weights_dict = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get(
            "ml_weights"
        )
        strategy_class_name = getattr(
            self, "strategy_class_override", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(
            strategy_class_name, BaseAssetStrategy
        )(self)

        close, high, low = (
            pd.Series(self.data.Close, index=self.data.index),
            pd.Series(self.data.High, index=self.data.index),
            pd.Series(self.data.Low, index=self.data.index),
        )
        (
            self.recent_trade_returns,
            self.equity_peak,
            self.global_stop_triggered,
        ) = (
            deque(maxlen=self.kelly_trade_history),
            self.equity,
            False,
        )
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
            self.I(bb.bollinger_hband),
            self.I(bb.bollinger_lband),
            self.I(bb.bollinger_mavg),
        )
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(close, self.mr_rsi_period).rsi()
        )

        # --- 新增：动态模型加载机制 ---
        self.ml_models = {}
        self.model_files_map = []
        self.next_model_load_idx = 0
        self._discover_and_prepare_models()
        self._check_and_load_model()  # 初始加载

        self.last_checked_day = -1
        self.current_params = {"score_entry_threshold": self.score_entry_threshold}

    def _discover_and_prepare_models(self):
        """在回测开始时，发现所有可用的、已训练好的模型文件，并按日期排序。"""
        if self.symbol and ML_LIBS_INSTALLED:
            model_pattern = f"directional_model_{self.symbol}_*_*.joblib"
            model_files = glob.glob(model_pattern)

            parsed_files = []
            for f in model_files:
                try:
                    # 从文件名解析日期，例如 '..._BTCUSDT_4h_20240101.joblib'
                    date_str = f.split("_")[-1].split(".")[0]
                    model_date = pd.to_datetime(date_str, format="%Y%m%d")
                    horizon_str = f.split("_")[-2]  # '4h', '8h', '12h'
                    parsed_files.append(
                        {"date": model_date, "horizon": horizon_str, "path": f}
                    )
                except (IndexError, ValueError):
                    logger.warning(f"无法解析模型文件名格式: {f}")

            if parsed_files:
                # 按日期分组，然后按日期排序
                df = pd.DataFrame(parsed_files)
                grouped = df.groupby("date")
                for date, group in sorted(grouped, key=lambda x: x[0]):
                    self.model_files_map.append(
                        {"date": date, "models": group.to_dict("records")}
                    )

            if self.model_files_map:
                logger.info(
                    f"[{self.symbol}] 发现了 {len(self.model_files_map)} 个训练周期的模型。"
                )

    def _check_and_load_model(self):
        """检查当前时间是否需要加载新的模型。"""
        current_timestamp = self.data.index[-1]

        # 如果还有待加载的模型，并且当前时间已经超过了下一个模型的生效日期
        if (
            self.next_model_load_idx < len(self.model_files_map)
            and current_timestamp
            >= self.model_files_map[self.next_model_load_idx]["date"]
        ):

            model_set = self.model_files_map[self.next_model_load_idx]
            effective_date = model_set["date"].date()
            logger.info(f"[{self.symbol}] 动态加载于 {effective_date} 训练的新模型...")

            loaded_count = 0
            for model_info in model_set["models"]:
                try:
                    h = int(model_info["horizon"][:-1])  # '4h' -> 4
                    self.ml_models[h] = joblib.load(model_info["path"])
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"加载模型文件 {model_info['path']} 失败: {e}")

            if loaded_count > 0:
                logger.info(f"✅ 成功加载 {loaded_count} 个新模型。")
            self.next_model_load_idx += 1

    def next(self):
        # 在每个bar，检查是否需要更新模型
        self._check_and_load_model()

        # --- 核心交易逻辑 (无变动) ---
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            if self.data.market_regime[-1] == 1:
                self.run_scoring_system_entry(self.data.Close[-1])
            else:
                self.run_mean_reversion_entry(self.data.Close[-1])

    # --- 其余所有策略方法 (`run_scoring_system_entry`, `get_ml_confidence_score` 等) 均无变动 ---

    def run_scoring_system_entry(self, price):
        final_score = self.asset_strategy._calculate_entry_score()
        is_long = final_score > self.score_entry_threshold
        is_short = final_score < -self.score_entry_threshold
        if not (is_long or is_short):
            return
        self.open_tf_position(
            price, is_long=is_long, score=1.0, confidence_factor=abs(final_score)
        )

    def run_mean_reversion_entry(self, price):
        base_signal = self.asset_strategy._define_mr_entry_signal()
        if base_signal != 0:
            self.open_mr_position(price, is_long=(base_signal == 1))

    def get_ml_confidence_score(self) -> float:
        if not self.ml_weights_dict or not self.ml_models:
            return 0.0
        features = [col for col in self.data.df.columns if "feature_" in col]
        if self.data.df[features].iloc[-1].isnull().any():
            return 0.0
        current_features = self.data.df[features].iloc[-1:]
        confidence_score = 0.0
        for h, model in self.ml_models.items():
            try:
                pred = 1 if model.predict(current_features)[0] == 1 else -1
                confidence_score += pred * self.ml_weights_dict.get(f"{h}h", 0)
            except Exception:  # 如果模型未加载，则跳过
                pass
        return confidence_score

    def reset_trade_state(self):
        self.active_sub_strategy = None
        self.chandelier_exit_level = 0.0
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")
        self.mr_stop_loss = 0.0
        self.tf_initial_stop_loss = 0.0

    def manage_open_position(self, price):
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(price)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(price)

    def open_tf_position(self, price, is_long, score, confidence_factor):
        risk_per_share = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_per_share <= 0:
            return
        final_risk = self._calculate_dynamic_risk() * score * confidence_factor
        size = self._calculate_position_size(price, risk_per_share, final_risk)
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.tf_initial_stop_loss = price - risk_per_share
            self.highest_high_in_trade = self.data.High[-1]
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
        else:
            self.sell(size=size)
            self.tf_initial_stop_loss = price + risk_per_share
            self.lowest_low_in_trade = self.data.Low[-1]
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )

    def manage_trend_following_exit(self, price):
        atr = self.tf_atr[-1]
        if self.position.is_long:
            if price < self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade - atr * self.tf_chandelier_atr_multiplier
            )
            if price < self.chandelier_exit_level:
                self.close_position("TF_Chandelier")
        elif self.position.is_short:
            if price > self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade + atr * self.tf_chandelier_atr_multiplier
            )
            if price > self.chandelier_exit_level:
                self.close_position("TF_Chandelier")

    def open_mr_position(self, price, is_long):
        risk_per_share = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
        if risk_per_share <= 0:
            return
        size = self._calculate_position_size(
            price,
            risk_per_share,
            self._calculate_dynamic_risk() * self.mr_risk_multiplier,
        )
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        if is_long:
            self.buy(size=size)
            self.mr_stop_loss = price - risk_per_share
        else:
            self.sell(size=size)
            self.mr_stop_loss = price + risk_per_share

    def manage_mean_reversion_exit(self, price):
        if (
            self.position.is_long
            and (price >= self.mr_bb_mid[-1] or price <= self.mr_stop_loss)
        ) or (
            self.position.is_short
            and (price <= self.mr_bb_mid[-1] or price >= self.mr_stop_loss)
        ):
            self.close_position("MR")

    def close_position(self, reason: str):
        equity_before_close = self.equity
        self.position.close()
        self.recent_trade_returns.append((self.equity / equity_before_close) - 1)
        self.reset_trade_state()

    def _calculate_position_size(self, price, risk_per_share, target_risk_pct):
        if risk_per_share <= 0 or price <= 0:
            return 0
        return (target_risk_pct * self.equity) / (risk_per_share / price) / self.equity

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


if __name__ == "__main__":
    logger.info(f"🚀 (V41.01-No-News) 开始运行...")

    # --- 1. 计算所需的最早数据日期 ---
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    training_window = timedelta(days=CONFIG["training_window_days"])
    data_fetch_start_date = (backtest_start_dt - training_window).strftime("%Y-%m-%d")

    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期 (包含训练窗口): {data_fetch_start_date}")

    # --- 2. 一次性获取所有需要的历史数据 ---
    raw_data = {
        symbol: fetch_binance_klines(
            symbol,
            CONFIG["interval"],
            data_fetch_start_date,
            CONFIG["backtest_end_date"],
        )
        for symbol in CONFIG["symbols_to_test"]
    }
    raw_data = {s: d for s, d in raw_data.items() if not d.empty}
    if not raw_data:
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    # --- 3. 执行前向展开训练 (如果启用) ---
    if CONFIG["enable_walk_forward_training"]:
        logger.info("### 模式: 前向展开(Walk-Forward)模型训练 ###")

        retrain_interval = timedelta(days=CONFIG["retrain_every_days"])
        current_training_date = backtest_start_dt

        while current_training_date <= pd.to_datetime(CONFIG["backtest_end_date"]):
            training_data_start = current_training_date - training_window

            logger.info("=" * 50)
            logger.info(f"准备训练周期，训练数据截止于: {current_training_date.date()}")
            logger.info(
                f"训练数据窗口: {training_data_start.date()} -> {current_training_date.date()}"
            )
            logger.info("=" * 50)

            for symbol, data in raw_data.items():
                training_slice = data.loc[
                    training_data_start:current_training_date
                ].copy()
                if training_slice.empty:
                    logger.warning(f"[{symbol}] 在此训练周期内无数据，跳过。")
                    continue

                processed_training_data = preprocess_data_for_strategy(
                    training_slice, symbol
                )
                if not processed_training_data.empty:
                    train_and_save_model(
                        processed_training_data, symbol, current_training_date
                    )

            current_training_date += retrain_interval
    else:
        logger.info("### 模式: 跳过模型训练 ###")

    # --- 4. 准备完整的回测数据 ---
    logger.info(f"### 准备完整回测数据 (开始日期: {CONFIG['backtest_start_date']}) ###")
    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        backtest_period_slice = data.loc[CONFIG["backtest_start_date"] :].copy()
        if not backtest_period_slice.empty:
            logger.info(f"为 {symbol} 预处理回测数据...")
            processed_backtest_data[symbol] = preprocess_data_for_strategy(
                backtest_period_slice, symbol
            )

    processed_backtest_data = {
        s: d for s, d in processed_backtest_data.items() if not d.empty
    }
    if not processed_backtest_data:
        logger.error("回测时间段内没有可用的预处理数据，程序终止。")
        exit()

    # --- 5. 执行回测 ---
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
        print("\n" + "=" * 80 + f"\n正在回测品种: {symbol}\n" + "=" * 80)
        final_params = {
            f"{k}_override": v
            for k, v in ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).items()
        }

        bt = Backtest(
            data,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            finalize_trades=True,
        )
        stats = bt.run(
            symbol=symbol, vol_weight=vol_weights.get(symbol, 1.0), **final_params
        )
        all_stats[symbol], total_equity = (
            stats,
            total_equity + stats["Equity Final [$]"],
        )
        print(
            "\n" + "-" * 40 + f"\n          {symbol} 回测结果摘要\n" + "-" * 40,
            stats,
        )
        if CONFIG["show_plots"]:
            bt.plot()

    if all_stats:
        initial_total = CONFIG["initial_cash"] * len(all_stats)
        ret = ((total_equity - initial_total) / initial_total) * 100
        print("\n" + "#" * 80 + "\n                 组合策略表现总览\n" + "#" * 80)
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A')}"
            )
        print(
            f"\n--- 投资组合整体表现 ---\n总初始资金: ${initial_total:,.2f}\n总最终权益: ${total_equity:,.2f}\n组合总回报率: {ret:.2f}%"
        )
