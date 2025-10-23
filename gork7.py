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

try:
    import lightgbm as lgb
    from sklearn.metrics import classification_report

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

# --- 核心配置 --- "BTCUSDT"
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],
    "interval": "15m",
    "backtest_start_date": "2025-10-01",
    "backtest_end_date": "2025-10-19",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,
    "show_plots": False,
    "training_window_days": 365 * 1.5,
    "enable_ml_component": True,
}
ML_HORIZONS = [4, 8, 12]  # <-- 修复此行

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
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 25,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "score_entry_threshold": 0.4,
    # MODIFIED: Rebalanced weights. ML total is now 40%, non-ML is 60%.
    "score_weights_tf": {
        "breakout": 0.2667,  # Was 0.20
        "momentum": 0.20,  # Was 0.15
        "mtf": 0.1333,  # Was 0.10
        "ml": 0.2182,  # Was 0.30
        "advanced_ml": 0.1818,  # Was 0.25
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "BTCUSDT": {
        "strategy_class": "BTCStrategy",
        "ml_weights": {"4h": 0.25, "8h": 0.35, "12h": 0.4},
        "ml_weighted_threshold": 0.2,
        "score_entry_threshold": 0.35,
    },
    "ETHUSDT": {
        "strategy_class": "ETHStrategy",
        "ml_weights": {"4h": 0.15, "8h": 0.3, "12h": 0.55},
        "ml_weighted_threshold": 0.2,
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
    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


def train_and_save_model(
    training_data: pd.DataFrame, symbol: str, training_end_date: pd.Timestamp
):
    if not ML_LIBS_INSTALLED:
        logger.warning("缺少 ML 库，跳过训练。")
        return
    logger.info(
        f"--- 🤖 [单次训练] 开始为 {symbol} 训练模型 (数据截止于: {training_end_date.date()}) ---"
    )
    features = [col for col in training_data.columns if "feature_" in col]
    if not features:
        logger.error(f"[{symbol}] 找不到特征列。")
        return
    logger.info(f"[{symbol}] 使用以下特征进行训练: {features}")
    for h in ML_HORIZONS:
        logger.info(f"正在为 {h}h 预测窗口准备数据...")
        data = training_data.copy()
        data[f"target_{h}h"] = (data["Close"].shift(-h) > data["Close"]).astype(int)
        df_train = data.dropna(subset=[f"target_{h}h"] + features)
        X, y = df_train[features], df_train[f"target_{h}h"]
        if len(X) < 200 or y.nunique() < 2:
            logger.warning(f"[{symbol}-{h}h] 数据不足，跳过训练。")
            continue
        eval_size = int(len(X) * 0.1)
        X_train, X_eval, y_train, y_eval = (
            X[:-eval_size],
            X[-eval_size:],
            y[:-eval_size],
            y[-eval_size:],
        )
        logger.info(f"开始训练 {symbol} 的 {h}h 模型...")
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        y_pred = model.predict(X_eval)
        logger.info(
            f"[{symbol}-{h}h] 模型评估报告:\n{classification_report(y_eval, y_pred, zero_division=0)}"
        )
        model_filename = f"directional_model_{symbol}_{h}h.joblib"
        joblib.dump(model, model_filename)
        logger.info(f"✅ [{symbol}-{h}h] 模型训练完成并已保存至: {model_filename}")


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
        return (
            1
            if crossover(m.data.Close, m.mr_bb_lower)
            and m.mr_rsi[-1] < m.mr_rsi_oversold
            else (
                -1
                if crossover(m.mr_bb_upper, m.data.Close)
                and m.mr_rsi[-1] > m.mr_rsi_overbought
                else 0
            )
        )


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
        # 1. 首先加载全局默认参数
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)

        # 2. 获取该品种的特定配置
        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {})

        # 3. 应用从 bt.run() 传递过来的覆盖参数
        if self.score_entry_threshold_override is not None:
            self.score_entry_threshold = self.score_entry_threshold_override

        if self.score_weights_tf_override is not None:
            self.score_weights_tf = self.score_weights_tf_override

        if self.ml_weights_override is not None:
            self.ml_weights_dict = self.ml_weights_override
        else:
            self.ml_weights_dict = asset_overrides.get("ml_weights")

        if self.ml_weighted_threshold_override is not None:
            self.ml_weighted_threshold = self.ml_weighted_threshold_override
        else:
            self.ml_weighted_threshold = asset_overrides.get("ml_weighted_threshold")

        strategy_class_name = self.strategy_class_override or asset_overrides.get(
            "strategy_class", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(
            strategy_class_name, BaseAssetStrategy
        )(self)

        # 准备数据序列和指标
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
            lambda: ta.momentum.RSIIndicator(close, self.mr_rsi_period).rsi()
        )

        # 模型加载
        self.ml_models = {}
        self._load_models()

    def _load_models(self):
        # MODIFIED: Removed check for 'enable_ml_component'.
        # The strategy will now always attempt to load models if they exist.
        if not self.symbol or not ML_LIBS_INSTALLED:
            logger.warning(
                f"[{self.symbol}] 缺少 ML 库 (scikit-learn, lightgbm)，跳过模型加载。"
            )
            return

        backtest_start_str = pd.to_datetime(CONFIG["backtest_start_date"]).strftime(
            "%Y%m%d"
        )
        loaded_count = 0
        for h in ML_HORIZONS:
            model_files = glob.glob(f"directional_model_{self.symbol}_{h}h.joblib")
            if model_files:
                try:
                    self.ml_models[h] = joblib.load(model_files[0])
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"加载模型 {model_files[0]} 失败: {e}")
        if loaded_count > 0:
            logger.info(f"✅ [{self.symbol}] 成功加载 {loaded_count} 个模型。")
        else:
            logger.warning(
                f"[{self.symbol}] 未找到与日期 {backtest_start_str} 匹配的模型。"
            )

    def next(self):
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            if self.data.market_regime[-1] == 1:
                self.run_scoring_system_entry(self.data.Close[-1])
            else:
                self.run_mean_reversion_entry(self.data.Close[-1])

    def get_ml_confidence_score(self) -> float:
        # MODIFIED: Removed check for 'enable_ml_component'.
        # The score is now calculated if models were successfully loaded, regardless of the config setting.
        if not self.ml_models or not self.ml_weights_dict:
            return 0.0

        features = [col for col in self.data.df.columns if "feature_" in col]
        if self.data.df[features].iloc[-1].isnull().any():
            return 0.0
        current_features = self.data.df[features].iloc[-1:]
        score = 0.0
        for h, model in self.ml_models.items():
            try:
                pred = 1 if model.predict(current_features)[0] == 1 else -1
                score += pred * self.ml_weights_dict.get(f"{h}h", 0)
            except Exception:
                pass
        return score

    # (其他策略方法无变动)
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
        self.active_sub_strategy = None
        self.chandelier_exit_level = 0.0
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")
        self.mr_stop_loss = 0.0
        self.tf_initial_stop_loss = 0.0

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


if __name__ == "__main__":
    logger.info(f"🚀 (V41.04c-FinalFix) 开始运行...")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_fetch_start_date = data_fetch_start_date = CONFIG["backtest_start_date"]
    if CONFIG["enable_ml_component"]:
        training_window = timedelta(days=CONFIG["training_window_days"])
        data_fetch_start_date = (backtest_start_dt - training_window).strftime(
            "%Y-%m-%d"
        )
    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期 (包含训练窗口): {data_fetch_start_date}")
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

    if CONFIG.get("enable_ml_component", False):
        logger.info("### 模式: 执行模型训练 (已启用) ###")
        training_data_end_dt = backtest_start_dt - pd.Timedelta(seconds=1)
        logger.info(
            "=" * 50
            + f"\n准备训练周期，训练数据截止于: {training_data_end_dt.date()}\n"
            + f"训练数据窗口: {data_fetch_start_date} -> {training_data_end_dt.date()}\n"
            + "=" * 50
        )
        for symbol, data in raw_data.items():
            training_slice = data.loc[data_fetch_start_date:training_data_end_dt].copy()
            if training_slice.empty:
                logger.warning(f"[{symbol}] 无训练数据，跳过。")
                continue
            processed_training_data = preprocess_data_for_strategy(
                training_slice, symbol
            )
            if not processed_training_data.empty:
                train_and_save_model(processed_training_data, symbol, backtest_start_dt)
    else:
        logger.info("### 模式: 跳过模型训练 (已禁用) ###")

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
        print("\n" + "=" * 80 + f"\n正在回测品种: {symbol}\n" + "=" * 80)

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
        print(
            "\n" + "-" * 40 + f"\n          {symbol} 回测结果摘要\n" + "-" * 40, stats
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
