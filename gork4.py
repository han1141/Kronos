# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V38.1-Hurst-Fix)

版本更新：
- 修复了 `compute_hurst` 函数中因错误的元组赋值导致的 `UnboundLocalError`。
- (V38.0 的所有架构和功能保持不变)
"""

# --- 1. 导入库与配置 (保持不变) ---
import pandas as pd
import requests
import time
from datetime import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.font_manager
import joblib

try:
    from gnews import GNews
    from textblob import TextBlob

    NEWS_LIBS_INSTALLED = True
except ImportError:
    NEWS_LIBS_INSTALLED = False
try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 日志和字体配置 (保持不变) ---
# ... (代码省略)
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


# --- 核心配置 (保持不变) ---
CONFIG = {
    "symbols_to_test": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "interval": "1h",
    "start_date": "2023-01-01",
    "end_date": "2025-10-08",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,
    "run_monte_carlo": True,
    "show_plots": False,
    "train_new_model": False,
    "ml_training_start_date": "2023-01-01",
    "run_optimization": False,
}
NEWS_CONFIG = {
    "gnews_api_key": "439183c4b004dd34c1f940f0dabb44f8",
    "search_keywords": {
        "BTCUSDT": "Bitcoin OR BTC crypto",
        "ETHUSDT": "Ethereum OR ETH crypto",
        "SOLUSDT": "Solana OR SOL crypto",
    },
    "sentiment_rolling_period": 24,
}
ML_HORIZONS = [4, 8, 12]
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
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 30,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "score_mtf_bonus": 0.5,
    "ai_filter_rsi_period": 14,
    "ai_filter_fast_ma": 3,
    "ai_filter_slow_ma": 10,
    "ai_filter_confidence_threshold": 0.2,
    "score_ai_bonus": 0.5,
    "score_nonlinear_factor": 2.0,
    "volatility_norm_period": 100,
}
ASSET_SPECIFIC_OVERRIDES = {
    "BTCUSDT": {
        "strategy_class": "BTCStrategy",
        "ml_weights": {"4h": 0.2, "8h": 0.3, "12h": 0.5},
        "ml_weighted_threshold": 0.45,
    },
    "ETHUSDT": {
        "strategy_class": "ETHStrategy",
        "ml_weights": {"4h": 0.25, "8h": 0.35, "12h": 0.4},
        "ml_weighted_threshold": 0.2,
    },
    "SOLUSDT": {
        "strategy_class": "SOLStrategy",
        "ml_weights": {"4h": 0.3, "8h": 0.4, "12h": 0.3},
        "ml_weighted_threshold": 0.25,
    },
}


# --- ✅✅✅ 修复点: 修改 compute_hurst 函数 ✅✅✅ ---
def compute_hurst(ts, max_lag=100):
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau, valid_lags = [], []
    for lag in lags:
        # --- 原错误行 ---
        # diff, std_diff = ts[lag:] - ts[:-lag], np.std(diff)

        # --- 修正后代码 ---
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)

        if std_diff > 0:
            tau.append(std_diff)
            valid_lags.append(lag)
    if len(tau) < 2:
        return 0.5
    try:
        return max(0.0, min(1.0, np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]))
    except Exception:
        return 0.5


# --- 其他数据与指标函数 (保持不变) ---
def fetch_binance_klines(
    symbol: str, interval: str, start_str: str, end_str: str = None, limit: int = 1000
) -> pd.DataFrame:  # ...
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


def fetch_gnews_data(
    keywords: str, start_date_str: str, end_date_str: str
) -> pd.DataFrame:  # ...
    if (
        not NEWS_LIBS_INSTALLED
        or not NEWS_CONFIG["gnews_api_key"]
        or NEWS_CONFIG["gnews_api_key"] == "在此处粘贴您的GNews_API密钥"
    ):
        return pd.DataFrame()
    try:
        start_date_dt, end_date_dt = datetime.strptime(
            start_date_str, "%Y-%m-%d"
        ), datetime.strptime(end_date_str, "%Y-%m-%d")
        google_news = GNews(
            start_date=(start_date_dt.year, start_date_dt.month, start_date_dt.day),
            end_date=(end_date_dt.year, end_date_dt.month, end_date_dt.day),
        )
        google_news.api_key = NEWS_CONFIG["gnews_api_key"]
        news_list = google_news.get_news(keywords)
        if not news_list:
            return pd.DataFrame()
        df = pd.DataFrame(news_list)
        df["published date"] = pd.to_datetime(
            df["published date"], format="mixed", utc=True
        )
        df = df[["published date", "title"]].rename(
            columns={"published date": "timestamp"}
        )
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.error(f"获取GNews数据时出错: {e}")
        return pd.DataFrame()


def analyze_sentiment_and_merge(
    main_df: pd.DataFrame, news_df: pd.DataFrame
) -> pd.DataFrame:  # ...
    if news_df.empty:
        main_df["sentiment_score"] = 0.0
        return main_df
    news_df["sentiment_score"] = news_df["title"].apply(
        lambda title: TextBlob(title).sentiment.polarity
    )
    sentiment_series = news_df["sentiment_score"].resample("1h").mean().ffill()
    main_df = main_df.tz_localize("UTC") if main_df.index.tz is None else main_df
    main_df = main_df.join(sentiment_series)
    main_df["sentiment_score"] = main_df["sentiment_score"].fillna(0.0)
    main_df = main_df.tz_convert(None)
    return main_df


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:  # ...
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
    df["feature_adx_norm"], df["feature_atr_slope_norm"], df["feature_rsi_vol_norm"] = (
        norm(adx),
        norm(
            (atr - atr.shift(p["regime_atr_slope_period"]))
            / atr.shift(p["regime_atr_slope_period"])
        ),
        1 - norm(rsi.rolling(p["regime_rsi_vol_period"]).std()),
    )
    df["feature_hurst"] = (
        df.Close.rolling(p["regime_hurst_period"])
        .apply(lambda x: compute_hurst(np.log(x + 1e-9)), raw=False)
        .fillna(0.5)
    )
    df["feature_momentum_5"] = df.Close.pct_change(5).shift(1)
    if "sentiment_score" in df.columns:
        df["feature_sentiment_score"] = (df["sentiment_score"] + 1) / 2
        df["feature_sentiment_rolling_avg"] = (
            df["feature_sentiment_score"]
            .rolling(window=NEWS_CONFIG["sentiment_rolling_period"])
            .mean()
        )
    else:
        df["feature_sentiment_score"], df["feature_sentiment_rolling_avg"] = 0.5, 0.5
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["feature_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    return df


def train_and_save_model(symbol: str, horizon: int):  # ...
    if not ML_LIBS_INSTALLED:
        return
    end_date_str = (
        datetime.strptime(CONFIG["start_date"], "%Y-%m-%d") - pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    data = fetch_binance_klines(
        symbol, "1h", CONFIG["ml_training_start_date"], end_date_str
    )
    if data.empty or len(data) < 500:
        return
    news_data = fetch_gnews_data(
        NEWS_CONFIG["search_keywords"].get(symbol, symbol),
        CONFIG["ml_training_start_date"],
        end_date_str,
    )
    ml_data = prepare_ml_data(
        analyze_sentiment_and_merge(data, news_data), look_forward_period=horizon
    )
    if ml_data.empty:
        return
    features = [col for col in ml_data.columns if col.startswith("feature_")]
    X, y = ml_data[features], ml_data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = lgb.LGBMClassifier(
        objective="binary", random_state=42, n_estimators=100, learning_rate=0.05
    )
    model.fit(X_train, y_train, feature_name=features)
    joblib.dump(model, f"directional_model_{symbol}_{horizon}h.joblib")


# --- 策略架构 (保持不变) ---
class BaseAssetStrategy:  # ...
    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy

    def _define_tf_entry_signal(self) -> int:
        main = self.main
        is_breakout_up, is_breakout_down = (
            main.data.High[-1] > main.tf_donchian_h[-1],
            main.data.Low[-1] < main.tf_donchian_l[-1],
        )
        is_momentum_long, is_momentum_short = (
            main.tf_ema_fast[-1] > main.tf_ema_slow[-1],
            main.tf_ema_fast[-1] < main.tf_ema_slow[-1],
        )
        is_trend_strong = main.tf_adx[-1] > main.tf_adx_confirm_threshold
        if is_trend_strong and is_breakout_up and is_momentum_long:
            return 1
        elif is_trend_strong and is_breakout_down and is_momentum_short:
            return -1
        return 0

    def _define_mr_entry_signal(self) -> int:
        main = self.main
        is_oversold = (
            crossover(main.data.Close, main.mr_bb_lower)
            and main.mr_rsi[-1] < main.mr_rsi_oversold
        )
        is_overbought = (
            crossover(main.mr_bb_upper, main.data.Close)
            and main.mr_rsi[-1] > main.mr_rsi_overbought
        )
        if is_oversold:
            return 1
        elif is_overbought:
            return -1
        return 0


class BTCStrategy(BaseAssetStrategy):  # ...
    def _define_tf_entry_signal(self) -> int:
        main, btc_adx_threshold = self.main, 20
        is_breakout_up, is_breakout_down = (
            main.data.High[-1] > main.tf_donchian_h[-1],
            main.data.Low[-1] < main.tf_donchian_l[-1],
        )
        is_momentum_long, is_momentum_short = (
            main.tf_ema_fast[-1] > main.tf_ema_slow[-1],
            main.tf_ema_fast[-1] < main.tf_ema_slow[-1],
        )
        is_trend_strong = main.tf_adx[-1] > btc_adx_threshold
        if is_trend_strong and is_breakout_up and is_momentum_long:
            return 1
        elif is_trend_strong and is_breakout_down and is_momentum_short:
            return -1
        return 0


class ETHStrategy(BaseAssetStrategy):
    pass


class SOLStrategy(BaseAssetStrategy):  # ...
    def _define_tf_entry_signal(self) -> int:
        main, sol_adx_threshold = self.main, 17
        is_breakout_up, is_breakout_down = (
            main.data.High[-1] > main.tf_donchian_h[-1],
            main.data.Low[-1] < main.tf_donchian_l[-1],
        )
        is_momentum_long, is_momentum_short = (
            main.tf_ema_fast[-1] > main.tf_ema_slow[-1],
            main.tf_ema_fast[-1] < main.tf_ema_slow[-1],
        )
        is_trend_strong = main.tf_adx[-1] > sol_adx_threshold
        volume_spike = (
            main.data.Volume[-1] > main.data.Volume.rolling(20).mean()[-2] * 1.5
        )
        if is_trend_strong and is_breakout_up and is_momentum_long and volume_spike:
            return 1
        elif (
            is_trend_strong and is_breakout_down and is_momentum_short and volume_spike
        ):
            return -1
        return 0


STRATEGY_MAPPING = {
    "BaseAssetStrategy": BaseAssetStrategy,
    "BTCStrategy": BTCStrategy,
    "ETHStrategy": ETHStrategy,
    "SOLStrategy": SOLStrategy,
}


class UltimateStrategy(Strategy):  # ...
    strategy_class_override = None
    (
        ml_use_weighted_logic_override,
        ml_weights_override,
        ml_weighted_threshold_override,
    ) = (None, None, None)
    ml_use_weighted_logic, ml_weights, ml_weighted_threshold = False, None, 0.5
    (
        tf_donchian_period_dynamic,
        tf_chandelier_atr_multiplier_dynamic,
        mr_bb_std_dynamic,
    ) = (None, None, None)
    (
        max_risk_pct_override,
        regime_score_threshold_override,
        ml_consensus_level_override,
    ) = (None, None, None)
    for key, value in STRATEGY_PARAMS.items():
        exec(f"{key} = {value}")
    vol_weight, symbol, ml_consensus_level = 1.0, None, 3

    def init(self):
        self.tf_donchian_period, self.tf_chandelier_atr_multiplier, self.mr_bb_std = (
            int(getattr(self, "tf_donchian_period_dynamic", self.tf_donchian_period)),
            getattr(
                self,
                "tf_chandelier_atr_multiplier_dynamic",
                self.tf_chandelier_atr_multiplier,
            ),
            getattr(self, "mr_bb_std_dynamic", self.mr_bb_std),
        )
        self.regime_score_threshold, self.max_risk_pct = getattr(
            self, "regime_score_threshold_override", self.regime_score_threshold
        ), getattr(self, "max_risk_pct_override", self.max_risk_pct)
        self.ml_use_weighted_logic, self.ml_weights_dict, self.ml_weighted_threshold = (
            getattr(self, "ml_use_weighted_logic_override", self.ml_use_weighted_logic),
            getattr(self, "ml_weights_override", self.ml_weights),
            getattr(self, "ml_weighted_threshold_override", self.ml_weighted_threshold),
        )
        strategy_class_name = getattr(
            self, "strategy_class_override", "BaseAssetStrategy"
        )
        StrategyClass = STRATEGY_MAPPING.get(strategy_class_name)
        if not StrategyClass:
            raise ValueError(f"未找到名为 '{strategy_class_name}' 的策略类")
        self.asset_strategy = StrategyClass(self)
        close, high, low = (
            pd.Series(self.data.Close, index=self.data.index),
            pd.Series(self.data.High, index=self.data.index),
            pd.Series(self.data.Low, index=self.data.index),
        )
        (
            self.recent_trade_returns,
            self.equity_peak,
            self.global_stop_triggered,
            self.ml_models,
        ) = (deque(maxlen=self.kelly_trade_history), self.equity, False, {})
        self.reset_trade_state()
        self.market_regime, self.mtf_signal, self.ai_filter_signal = (
            self.I(lambda: self.data.market_regime),
            self.I(lambda: self.data.mtf_signal),
            self.I(lambda: self.data.ai_filter_signal),
        )
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.tf_atr_period
            ).average_true_range()
        )
        self.tf_donchian_h, self.tf_donchian_l = self.I(
            lambda: high.rolling(self.tf_donchian_period).max().shift(1)
        ), self.I(lambda: low.rolling(self.tf_donchian_period).min().shift(1))
        self.tf_ema_fast, self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(
                close, self.tf_ema_fast_period
            ).ema_indicator()
        ), self.I(
            lambda: ta.trend.EMAIndicator(
                close, self.tf_ema_slow_period
            ).ema_indicator()
        )
        self.tf_adx = self.I(
            lambda: ta.trend.ADXIndicator(
                high, low, close, self.tf_adx_confirm_period
            ).adx()
        )
        bb_indicator = ta.volatility.BollingerBands(
            close, self.mr_bb_period, self.mr_bb_std
        )
        self.mr_bb_upper, self.mr_bb_lower, self.mr_bb_mid = (
            self.I(lambda: bb_indicator.bollinger_hband()),
            self.I(lambda: bb_indicator.bollinger_lband()),
            self.I(lambda: bb_indicator.bollinger_mavg()),
        )
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(close, self.mr_rsi_period).rsi()
        )
        if self.symbol and ML_LIBS_INSTALLED:
            for h in ML_HORIZONS:
                try:
                    self.ml_models[h] = joblib.load(
                        f"directional_model_{self.symbol}_{h}h.joblib"
                    )
                except FileNotFoundError:
                    logger.warning(f"未找到 {self.symbol} 的 {h}h ML模型")

    def reset_trade_state(self):
        (
            self.active_sub_strategy,
            self.chandelier_exit_level,
            self.highest_high_in_trade,
            self.lowest_low_in_trade,
            self.mr_stop_loss,
        ) = (None, 0.0, 0, float("inf"), 0.0)

    def next(self):
        current_bar = len(self.data.Close) - 1
        if current_bar > self.dd_grace_period_bars:
            decay_progress = min(
                (current_bar - self.dd_grace_period_bars) / self.dd_decay_bars, 1.0
            )
            current_dd_pct = self.dd_initial_pct - (
                decay_progress * (self.dd_initial_pct - self.dd_final_pct)
            )
            self.equity_peak = max(self.equity_peak, self.equity)
            if self.equity < self.equity_peak * (1 - current_dd_pct):
                if not self.global_stop_triggered and self.position:
                    self.position.close()
                self.global_stop_triggered = True
                return
        if self.global_stop_triggered:
            return
        price = self.data.Close[-1]
        if self.position:
            self.manage_open_position(price)
        else:
            if self.data.market_regime[-1] == 1:
                self.run_trend_following_entry(price)
            else:
                self.run_mean_reversion_entry(price)

    def get_ml_confidence_score(self) -> float:
        if (
            not self.ml_use_weighted_logic
            or not self.ml_weights_dict
            or len(self.ml_models) != len(self.ml_weights_dict)
        ):
            return 0.0
        features = [col for col in self.data.df.columns if col.startswith("feature_")]
        if any(pd.isna(self.data.df[features].iloc[-1])):
            return 0.0
        current_features_df = self.data.df[features].iloc[-1:]
        confidence_score = 0.0
        for h, model in self.ml_models.items():
            prediction = 1 if model.predict(current_features_df)[0] == 1 else -1
            confidence_score += prediction * self.ml_weights_dict.get(f"{h}h", 0)
        return confidence_score

    def run_trend_following_entry(self, price):
        base_signal, ml_score = (
            self.asset_strategy._define_tf_entry_signal(),
            self.get_ml_confidence_score(),
        )
        if base_signal == 0 or not (
            (base_signal == 1 and ml_score > self.ml_weighted_threshold)
            or (base_signal == -1 and ml_score < -self.ml_weighted_threshold)
        ):
            return
        score = 1.0
        if (base_signal == 1 and self.mtf_signal[-1] == 1) or (
            base_signal == -1 and self.mtf_signal[-1] == -1
        ):
            score += self.score_mtf_bonus
        if (
            base_signal == 1
            and self.ai_filter_signal[-1] > self.ai_filter_confidence_threshold
        ) or (
            base_signal == -1
            and self.ai_filter_signal[-1] < -self.ai_filter_confidence_threshold
        ):
            score += self.score_ai_bonus
        self.open_tf_position(
            price,
            is_long=(base_signal == 1),
            score=score**self.score_nonlinear_factor,
            ml_confidence_factor=abs(ml_score),
        )

    def run_mean_reversion_entry(self, price):
        base_signal, ml_score = (
            self.asset_strategy._define_mr_entry_signal(),
            self.get_ml_confidence_score(),
        )
        if (
            base_signal == 0
            or (base_signal == 1 and ml_score < -self.ml_weighted_threshold)
            or (base_signal == -1 and ml_score > self.ml_weighted_threshold)
        ):
            return
        self.open_mr_position(price, is_long=(base_signal == 1))

    def manage_open_position(self, price):
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(price)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(price)

    def open_tf_position(self, price, is_long, score, ml_confidence_factor):
        risk_per_share = self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
        if risk_per_share <= 0:
            return
        final_risk_pct = self._calculate_dynamic_risk() * score * ml_confidence_factor
        size = self._calculate_position_size(price, risk_per_share, final_risk_pct)
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.highest_high_in_trade, self.chandelier_exit_level = (
                self.data.High[-1],
                self.data.High[-1] - risk_per_share,
            )
        else:
            self.sell(size=size)
            self.lowest_low_in_trade, self.chandelier_exit_level = (
                self.data.Low[-1],
                self.data.Low[-1] + risk_per_share,
            )

    def manage_trend_following_exit(self, price):
        current_atr = self.tf_atr[-1]
        if self.position.is_long:
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - current_atr * self.tf_chandelier_atr_multiplier
            )
            if price < self.chandelier_exit_level:
                self.close_position("TF")
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + current_atr * self.tf_chandelier_atr_multiplier
            )
            if price > self.chandelier_exit_level:
                self.close_position("TF")

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
        win_rate, reward_ratio = len(wins) / len(self.recent_trade_returns), (
            sum(wins) / len(wins)
        ) / (abs(sum(losses) / len(losses)))
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight
        kelly_fraction = win_rate - (1 - win_rate) / reward_ratio
        return min(
            max(0.005, kelly_fraction * 0.5) * self.vol_weight, self.max_risk_pct
        )


# --- 辅助与主程序逻辑 (保持不变) ---
def run_monte_carlo(trades_df, initial_cash, symbol: str, n_simulations=1000):  # ...
    if trades_df.empty:
        return
    returns = trades_df["ReturnPct"]
    final_equities = [
        initial_cash
        * (1 + np.random.choice(returns, size=len(returns), replace=True)).prod()
        for _ in range(n_simulations)
    ]
    mean_equity, median_equity, var_5_pct, best_95_pct = (
        np.mean(final_equities),
        np.median(final_equities),
        np.percentile(final_equities, 5),
        np.percentile(final_equities, 95),
    )
    logger.info(
        f"--- MC模拟结果 ---\n次数: {n_simulations}\n均值: ${mean_equity:,.2f}\n中位数: ${median_equity:,.2f}\n5% VaR: ${var_5_pct:,.2f}\n95% Best: ${best_95_pct:,.2f}"
    )


def generate_dynamic_params(volatility: float, baseline_vol: float) -> dict:  # ...
    vol_factor = np.clip(volatility / baseline_vol, 0.8, 1.2)
    p = STRATEGY_PARAMS
    return {
        "tf_chandelier_atr_multiplier_dynamic": np.round(
            p["tf_chandelier_atr_multiplier"] * vol_factor, 2
        ),
        "tf_donchian_period_dynamic": int(
            np.round(p["tf_donchian_period"] * vol_factor)
        ),
        "mr_bb_std_dynamic": np.round(p["mr_bb_std"] * vol_factor),
        "max_risk_pct_override": np.round(
            np.clip(p["max_risk_pct"] / vol_factor, 0.02, 0.05), 4
        ),
    }


if __name__ == "__main__":  # ...
    logger.info(f"🚀 (V38.1-Hurst-Fix) 开始运行...")
    if CONFIG["train_new_model"]:
        for symbol in CONFIG["symbols_to_test"]:
            for horizon in ML_HORIZONS:
                train_and_save_model(symbol, horizon)
    all_data, volatilities = {}, {}
    for symbol in CONFIG["symbols_to_test"]:
        data = fetch_binance_klines(
            symbol, CONFIG["interval"], CONFIG["start_date"], CONFIG["end_date"]
        )
        if not data.empty:
            all_data[symbol], volatilities[symbol] = data, data.Close.resample(
                "D"
            ).last().pct_change().std() * np.sqrt(365)
    if not volatilities:
        exit()
    if CONFIG["run_optimization"]:
        symbol_to_optimize = "ETHUSDT"
        data_processed = add_ml_features(all_data.get(symbol_to_optimize).copy())
        data_processed.dropna(inplace=True)
        (
            data_processed["market_regime"],
            data_processed["mtf_signal"],
            data_processed["ai_filter_signal"],
        ) = (0, 0, 0)
        bt_opt = Backtest(
            data_processed,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
        )
        optimization_params = {
            "ml_weighted_threshold_override": np.arange(0.1, 0.4, 0.05)
        }
        base_params = ASSET_SPECIFIC_OVERRIDES.get(symbol_to_optimize, {})
        stats, heatmap = bt_opt.optimize(
            **optimization_params,
            **{f"{k}_override": v for k, v in base_params.items()},
            symbol=symbol_to_optimize,
            maximize="Equity Final [$]",
            return_heatmap=True,
        )
        print("--- 优化结果 ---\n", stats, "\n--- 最优参数组合 ---\n", stats._strategy)
    else:
        all_stats, total_final_equity = {}, 0
        inverse_vol = {s: 1 / v for s, v in volatilities.items() if v > 0}
        vol_weights = {
            s: (iv / sum(inverse_vol.values())) * len(volatilities)
            for s, iv in inverse_vol.items()
        }
        for symbol in CONFIG["symbols_to_test"]:
            print("\n" + "=" * 80 + f"\n正在回测品种: {symbol}\n" + "=" * 80)
            data_4h = all_data.get(symbol).copy()
            if data_4h is None:
                continue
            news_data = fetch_gnews_data(
                NEWS_CONFIG["search_keywords"].get(symbol, symbol),
                CONFIG["start_date"],
                CONFIG["end_date"],
            )
            data_4h = analyze_sentiment_and_merge(data_4h, news_data)
            symbol_volatility = volatilities.get(symbol, 0.7)
            dynamic_baseline_vol = data_4h.Close.resample(
                "D"
            ).last().pct_change().rolling(252).std().mean() * np.sqrt(365)
            if pd.isna(dynamic_baseline_vol) or dynamic_baseline_vol <= 0:
                dynamic_baseline_vol = 0.7
            final_params = generate_dynamic_params(
                symbol_volatility, dynamic_baseline_vol
            )
            if asset_overrides := ASSET_SPECIFIC_OVERRIDES.get(symbol, {}):
                for key, value in asset_overrides.items():
                    final_params[f"{key}_override"] = value
            data_4h = add_ml_features(data_4h)
            data_1d = fetch_binance_klines(
                symbol, "1d", CONFIG["start_date"], CONFIG["end_date"]
            )
            if data_1d is not None and not data_1d.empty:
                sma_indicator = ta.trend.SMAIndicator(
                    data_1d["Close"], window=STRATEGY_PARAMS["mtf_period"]
                ).sma_indicator()
                mtf_signal_1d = pd.Series(
                    np.where(data_1d["Close"] > sma_indicator, 1, -1),
                    index=data_1d.index,
                )
                data_4h["mtf_signal"] = mtf_signal_1d.reindex(
                    data_4h.index, method="ffill"
                ).fillna(0)
            else:
                data_4h["mtf_signal"] = 0
            final_regime_threshold = final_params.get(
                "regime_score_threshold_override",
                STRATEGY_PARAMS["regime_score_threshold"],
            )
            data_4h["market_regime"] = np.where(
                data_4h["feature_regime_score"] > final_regime_threshold, 1, -1
            )
            rsi_filter = ta.momentum.RSIIndicator(data_4h.Close, 14).rsi()
            data_4h["ai_filter_signal"] = (
                (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
                .clip(-1, 1)
                .fillna(0)
            )
            data_4h.dropna(inplace=True)
            bt = Backtest(
                data_4h,
                UltimateStrategy,
                cash=CONFIG["initial_cash"],
                commission=CONFIG["commission"],
                spread=CONFIG["spread"],
                finalize_trades=True,
            )
            stats = bt.run(
                vol_weight=vol_weights.get(symbol, 1.0), symbol=symbol, **final_params
            )
            all_stats[symbol], total_final_equity = (
                stats,
                total_final_equity + stats["Equity Final [$]"],
            )
            print(
                "\n" + "-" * 40 + f"\n          {symbol} 回测结果摘要\n" + "-" * 40,
                stats,
            )
            print("\n--- 🔍 策略健康度监控 ---")
            if not np.isnan(sqn := stats.get("SQN")):
                print(f"SQN (系统质量数): {sqn:.2f}")
            if not np.isnan(kelly := stats.get("Kelly Criterion")):
                print(f"凯利准则: {kelly:.4f}")
            if not np.isnan(calmar := stats.get("Calmar Ratio")):
                print(f"卡玛比率 (Cal-Ratio): {calmar:.3f}")
            if CONFIG["run_monte_carlo"] and not stats["_trades"].empty:
                run_monte_carlo(stats["_trades"], CONFIG["initial_cash"], symbol)
        if (num_assets := len(all_stats)) > 0:
            total_initial_cash = CONFIG["initial_cash"] * num_assets
            portfolio_return = (
                (total_final_equity - total_initial_cash) / total_initial_cash
            ) * 100
            print("\n" + "#" * 80 + "\n                 组合策略表现总览\n" + "#" * 80)
            for symbol, stats in all_stats.items():
                print(
                    f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A'):.3f}"
                )
            print(
                f"\n--- 投资组合整体表现 ---\n总初始资金: ${total_initial_cash:,.2f}\n总最终权益: ${total_final_equity:,.2f}\n组合总回报率: {portfolio_return:.2f}%"
            )
