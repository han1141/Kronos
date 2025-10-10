# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V40.12-Cache-Fix)

版本更新：
- (V40.12-Cache-Fix) 优化了新闻缓存逻辑：当检测到缓存文件为空（包含0条新闻）时，系统将自动尝试
                       重新从API获取，实现了对过去API错误的自我修复，确保数据质量。
- (V40.11-Style-Fix) 修复了 Pylance linter 因单行多语句导致的报错，提升了代码可读性。
- (V40.10-Risk-Stab-Fix) 增强新闻获取稳定性并为趋势跟踪策略加入了硬性初始止损。
- (V40.9-Flow-Optimize) 优化了主程序逻辑，提高了仅回测模式下的启动速度。
- (V40.8-News-Cache) 实现了新闻缓存功能。
"""

# --- 1. 导入库与配置 ---
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
import os
import json

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
    "symbols_to_test": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "interval": "1h",
    "training_start_date": "2021-01-01",
    "start_date": "2023-01-01",
    "end_date": "2025-10-08",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,
    "run_monte_carlo": True,
    "show_plots": False,
    "train_new_model": False,
    "run_learning_phase": False,
    "run_adaptive_backtest": True,
}
NEWS_CONFIG = {
    "gnews_api_key": "YOUR_GNEWS_API_KEY",
    "search_keywords": {
        "BTCUSDT": "Bitcoin OR BTC crypto",
        "ETHUSDT": "Ethereum OR ETH crypto",
        "SOLUSDT": "Solana OR SOL crypto",
    },
    "sentiment_rolling_period": 72,
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


class StrategyMemory:
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


def fetch_binance_klines(
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
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = (
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


def compute_hurst(ts, max_lag=100):
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


# ✅✅✅ (V40.12) 更新: 优化空缓存处理逻辑 ✅✅✅
def get_news_sentiment(symbol: str, data_index: pd.DatetimeIndex) -> pd.Series:
    logger.info(f"[{symbol}] 正在获取新闻情绪...")
    cache_dir = "news_cache"
    os.makedirs(cache_dir, exist_ok=True)
    start_date_str = data_index.min().strftime("%Y%m%d")
    end_date_str = data_index.max().strftime("%Y%m%d")
    cache_file = os.path.join(
        cache_dir, f"{symbol}_{start_date_str}_{end_date_str}.json"
    )

    news_items = None

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # 检查缓存数据是否为非空列表
            if isinstance(cached_data, list) and cached_data:
                news_items = cached_data
                logger.info(
                    f"✅ [{symbol}] 从缓存文件 {cache_file} 成功加载了 {len(news_items)} 条新闻。"
                )
            else:
                logger.warning(
                    f"缓存文件 {cache_file} 为空或格式不正确。将尝试重新从API获取。"
                )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"读取缓存文件 {cache_file} 失败: {e}。将重新从API获取。")

    if news_items is None:
        api_key = NEWS_CONFIG.get("gnews_api_key")
        if not api_key or api_key == "YOUR_GNEWS_API_KEY":
            logger.warning("GNews API 密钥无效或未设置。")
        elif not NEWS_LIBS_INSTALLED:
            logger.warning("GNews 或 TextBlob 未安装，无法获取新闻情绪。")
        else:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"正在通过 GNews API 获取 '{symbol}' 的新闻 (尝试 {attempt + 1}/{max_retries})..."
                    )
                    gnews = GNews()
                    gnews.api_key = api_key
                    keyword = NEWS_CONFIG["search_keywords"].get(symbol, symbol)
                    start_date_news = data_index.min().to_pydatetime()
                    end_date_news = data_index.max().to_pydatetime()
                    gnews.start_date = (
                        start_date_news.year,
                        start_date_news.month,
                        start_date_news.day,
                    )
                    gnews.end_date = (
                        end_date_news.year,
                        end_date_news.month,
                        end_date_news.day,
                    )
                    news_items_fetched = gnews.get_news(f'"{keyword}"')

                    if not news_items_fetched:
                        logger.warning(f"[{symbol}] API 未返回任何新闻条目。")
                        news_items = []
                    else:
                        news_items = news_items_fetched

                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(news_items, f, ensure_ascii=False, indent=4)
                    logger.info(
                        f"✅ [{symbol}] 从 API 获取了 {len(news_items)} 条新闻并已缓存至 {cache_file}。"
                    )
                    break
                except Exception as e:
                    logger.error(f"[{symbol}] API 请求失败 (尝试 {attempt + 1}): {e}。")
                    if attempt < max_retries - 1:
                        sleep_time = (attempt + 1) * 5
                        logger.info(f"将在 {sleep_time} 秒后重试...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"[{symbol}] 达到最大重试次数，获取新闻失败。")
                        news_items = None

    if news_items is not None:
        if not news_items:
            logger.warning(f"[{symbol}] 无可用新闻数据。返回中性情绪。")
            return pd.Series(0, index=data_index, name="news_sentiment")

        sentiments = []
        for item in news_items:
            title = item.get("title", "") or ""
            description = item.get("description", "") or ""
            text_to_analyze = title + " " + description
            if text_to_analyze.strip():
                sentiment = TextBlob(text_to_analyze).sentiment.polarity
                pub_date = item.get("published date")
                if pub_date:
                    sentiments.append(
                        {
                            "date": pd.to_datetime(pub_date)
                            .tz_localize(None)
                            .floor("D"),
                            "sentiment": sentiment,
                        }
                    )

        if not sentiments:
            logger.warning(f"[{symbol}] 新闻内容为空或无法进行情绪分析。返回中性情绪。")
            return pd.Series(0, index=data_index, name="news_sentiment")

        sentiment_df = pd.DataFrame(sentiments).groupby("date")["sentiment"].mean()
        daily_index_range = pd.date_range(
            start=data_index.min().floor("D"), end=data_index.max().floor("D")
        )
        daily_sentiment = sentiment_df.reindex(daily_index_range).fillna(0)
        hourly_sentiment = daily_sentiment.reindex(data_index, method="ffill").fillna(0)
        return (
            hourly_sentiment.rolling(
                window=NEWS_CONFIG["sentiment_rolling_period"], min_periods=1
            )
            .mean()
            .fillna(0)
        )

    logger.info(f"[{symbol}] 正在生成模拟新闻情绪作为最终备用方案...")
    daily_index = pd.date_range(
        start=data_index.min().floor("D"), end=data_index.max().floor("D"), freq="D"
    )
    daily_sentiment_scores = np.random.uniform(-0.5, 0.5, size=len(daily_index))
    daily_sentiment_scores = np.cumsum(daily_sentiment_scores) / len(daily_index)
    daily_sentiment_series = pd.Series(daily_sentiment_scores, index=daily_index)
    hourly_sentiment = daily_sentiment_series.reindex(
        data_index, method="ffill"
    ).fillna(0)
    return (
        hourly_sentiment.rolling(window=NEWS_CONFIG["sentiment_rolling_period"])
        .mean()
        .fillna(0)
    )


def run_advanced_model_inference(df: pd.DataFrame) -> pd.DataFrame:
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
    if "news_sentiment" in df.columns:
        df["feature_news_sentiment"] = df["news_sentiment"].clip(-1, 1)
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
    df["market_regime_final"] = (
        df["trend_regime"].astype(str) + "_" + df["volatility_regime"].astype(str)
    )
    df["market_regime"] = np.where(df["trend_regime"] == "Trending", 1, -1)
    return df


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(f"[{symbol}] 开始数据预处理...")
    df["news_sentiment"] = get_news_sentiment(symbol, df.index)
    rsi_filter = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["ai_filter_signal"] = (
        (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
        .clip(-1, 1)
        .fillna(0)
    )
    df = run_advanced_model_inference(df)
    df = add_ml_features(df)
    df = add_market_regime_features(df)
    data_1d = fetch_binance_klines(
        symbol,
        "1d",
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
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
        df["mtf_signal"] = 0
    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


def train_and_save_model(data: pd.DataFrame, symbol: str):
    if not ML_LIBS_INSTALLED:
        logger.warning("缺少 LightGBM 或 scikit-learn 库，跳过模型训练。")
        return
    logger.info(f"--- 🤖 开始为 {symbol} 训练新的机器学习模型 ---")
    features = [col for col in data.columns if "feature_" in col]
    if not features:
        logger.error(f"[{symbol}] 找不到任何特征列 (feature_) 用于训练。")
        return
    logger.info(f"[{symbol}] 使用以下特征进行训练: {features}")
    for h in ML_HORIZONS:
        logger.info(f"正在为 {h}h 预测窗口准备数据...")
        data[f"target_{h}h"] = (data["Close"].shift(-h) > data["Close"]).astype(int)
        df_train = data.dropna(subset=[f"target_{h}h"] + features)
        X = df_train[features]
        y = df_train[f"target_{h}h"]
        if len(X) < 200 or len(y.unique()) < 2:
            logger.warning(
                f"[{symbol}-{h}h] 数据不足或目标类别单一，跳过此预测窗口的训练。"
            )
            continue
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"开始训练 {symbol} 的 {h}h 模型...")
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=100, random_state=42
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        y_pred = model.predict(X_test)
        logger.info(
            f"[{symbol}-{h}h] 模型评估报告:\n{classification_report(y_test, y_pred)}"
        )
        model_filename = f"directional_model_{symbol}_{h}h.joblib"
        joblib.dump(model, model_filename)
        logger.info(f"✅ [{symbol}-{h}h] 模型训练完成并已保存至: {model_filename}")


class BaseAssetStrategy:
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


class BTCStrategy(BaseAssetStrategy):
    def _calculate_entry_score(self) -> float:
        if self.main.tf_adx[-1] > 20:
            return super()._calculate_entry_score()
        return 0


class ETHStrategy(BaseAssetStrategy):
    pass


class SOLStrategy(BaseAssetStrategy):
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
            self.ml_models,
        ) = (deque(maxlen=self.kelly_trade_history), self.equity, False, {})
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
        if self.symbol and ML_LIBS_INSTALLED:
            for h in ML_HORIZONS:
                try:
                    self.ml_models[h] = joblib.load(
                        f"directional_model_{self.symbol}_{h}h.joblib"
                    )
                except FileNotFoundError:
                    logger.warning(f"未找到 {self.symbol} 的 {h}h ML模型")
        self.last_checked_day = -1
        self.current_params = {"score_entry_threshold": self.score_entry_threshold}

    def next(self):
        current_day = self.data.index[-1].dayofyear
        if self.last_checked_day != current_day:
            self.last_checked_day = current_day
            if self.memory_instance and CONFIG["run_adaptive_backtest"]:
                current_regime = self.data.market_regime_final[-1]
                current_timestamp = self.data.index[-1]
                best_params = self.memory_instance.get_best_params(
                    current_timestamp, self.symbol, current_regime
                )
                if best_params:
                    typed_params = {k: float(v) for k, v in best_params.items()}
                    if typed_params != self.current_params:
                        self.current_params = typed_params
                        self.score_entry_threshold = self.current_params.get(
                            "score_entry_threshold_override", self.score_entry_threshold
                        )
                        logger.debug(
                            f"🧠 {current_timestamp}: 策略自适应更新 for {self.symbol} in {current_regime}. 新参数: {self.current_params}"
                        )
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            if self.data.market_regime[-1] == 1:
                self.run_scoring_system_entry(self.data.Close[-1])
            else:
                self.run_mean_reversion_entry(self.data.Close[-1])

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
            pred = 1 if model.predict(current_features)[0] == 1 else -1
            confidence_score += pred * self.ml_weights_dict.get(f"{h}h", 0)
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
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        reward_ratio = (sum(wins) / len(wins)) / (abs(sum(losses) / len(losses)))
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


if __name__ == "__main__":
    logger.info(f"🚀 (V40.12-Cache-Fix) 开始运行...")
    strategy_memory = StrategyMemory()

    load_start_date = (
        CONFIG["training_start_date"]
        if CONFIG["train_new_model"]
        else CONFIG["start_date"]
    )
    logger.info(f"数据加载期间: {load_start_date} to {CONFIG['end_date']}")

    raw_data = {
        symbol: fetch_binance_klines(
            symbol, CONFIG["interval"], load_start_date, CONFIG["end_date"]
        )
        for symbol in CONFIG["symbols_to_test"]
    }
    raw_data = {s: d for s, d in raw_data.items() if not d.empty}
    if not raw_data:
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    if CONFIG["train_new_model"]:
        logger.info("### 进入模型训练模式 ###")
        for symbol, data in raw_data.items():
            logger.info(f"为 {symbol} 预处理训练数据...")
            processed_training_data = preprocess_data_for_strategy(data.copy(), symbol)
            if not processed_training_data.empty:
                train_and_save_model(processed_training_data, symbol)
    else:
        logger.info("跳过模型训练环节 (train_new_model=False)。")

    logger.info(f"### 准备回测数据 (开始日期: {CONFIG['start_date']}) ###")
    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        backtest_period_slice = data.loc[CONFIG["start_date"] :].copy()
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

    if CONFIG["run_learning_phase"]:
        logger.info("### 进入学习与记忆模式 ###")
    else:
        mode = "自适应" if CONFIG["run_adaptive_backtest"] else "标准"
        logger.info(f"### 进入{mode}回测模式 ###")
        all_stats, total_equity = {}, 0
        vols = {
            s: d.Close.resample("D").last().pct_change().std() * np.sqrt(365)
            for s, d in processed_backtest_data.items()
        }
        inv_vols = {s: 1 / v for s, v in vols.items() if v > 0}
        vol_weights = {
            s: (iv / sum(inv_vols.values())) * len(inv_vols)
            for s, iv in inv_vols.items()
        }

        for symbol, data in processed_backtest_data.items():
            print("\n" + "=" * 80 + f"\n正在回测品种: {symbol}\n" + "=" * 80)
            final_params = {
                f"{k}_override": v
                for k, v in ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).items()
            }
            final_params["memory_instance"] = strategy_memory
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
            if CONFIG["run_monte_carlo"] and not stats["_trades"].empty:
                pass

        if all_stats:
            initial_total = CONFIG["initial_cash"] * len(all_stats)
            ret = ((total_equity - initial_total) / initial_total) * 100
            print("\n" + "#" * 80 + "\n                 组合策略表现总览\n" + "#" * 80)
            for symbol, stats in all_stats.items():
                print(
                    f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A'):.3f}"
                )
            print(
                f"\n--- 投资组合整体表现 ---\n总初始资金: ${initial_total:,.2f}\n总最终权益: ${total_equity:,.2f}\n组合总回报率: {ret:.2f}%"
            )
