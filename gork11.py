# -*- coding: utf-8 -*-
# V57.4-Final-Numba-Fix

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
from scipy.stats import linregress

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
except ImportError:
    lgb = None

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


from backtesting import Backtest, Strategy
import ta

# --- 日志与字体配置 (无变化) ---
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


# --- 核心配置 (无变化) ---
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],
    "interval": "15m",
    "backtest_start_date": "2023-01-01",
    "backtest_end_date": "2023-12-31",
    "initial_cash": 500_000,
    "commission": 0.00075,
    "spread": 0.0002,
    "show_plots": False,
    "data_lookback_days": 350,
}

# --- 模型文件路径配置 (无变化) ---
ML_MODEL_PATH = "models/eth_trend_model_lgb_4h.joblib"
# ... (其他路径不变)

# --- 策略参数 (无变化) ---
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
    "score_entry_threshold": 0.4,
}

# --- 辅助变量 (无变化) ---
ASSET_SPECIFIC_OVERRIDES = {}
ML_LIBS_INSTALLED = lgb is not None


# --- 函数定义 ---
def fetch_binance_klines(s, i, st, en=None, l=1000):  # ... (无变化)
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


### <<< 核心修正: 手动实现线性回归，移除对 lstsq 的依赖 >>> ###
@jit
def get_hurst_exponent_numba(ts, max_lag=100):
    lags = np.arange(2, max_lag)
    tau = np.empty(len(lags))
    for i, lag in enumerate(lags):
        diff = ts[lag:] - ts[:-lag]
        tau[i] = np.sqrt(np.nanstd(diff))

    valid_indices = np.where(tau > 0)[0]
    if len(valid_indices) < 2:
        return 0.5

    x = np.log(lags[valid_indices])
    y = np.log(tau[valid_indices])

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0.5  # 避免除以0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope * 2.0


@jit
def rolling_hurst_numba(ts_array, window):
    n = len(ts_array)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_slice = ts_array[i - window + 1 : i + 1]
        if not np.any(np.isnan(window_slice)):
            result[i] = get_hurst_exponent_numba(window_slice, max_lag=window)
    return result


def get_hurst_exponent(ts, max_lag=100):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(f"[{symbol}] 开始市场状态识别与特征工程 (高性能版)...")
    p = STRATEGY_PARAMS

    df["regime_adx"] = ta.trend.ADXIndicator(
        df["High"], df["Low"], df["Close"], window=p["regime_adx_period"]
    ).adx()
    atr = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=p["regime_atr_period"]
    ).average_true_range()
    df["regime_atr_norm"] = (atr / df["Close"]) * 100
    rsi = ta.momentum.RSIIndicator(df["Close"], window=p["regime_rsi_period"]).rsi()
    df["regime_rsi_vol"] = rsi.rolling(p["regime_rsi_vol_period"]).std()
    df["regime_atr_slope"] = atr.rolling(p["regime_atr_slope_period"]).apply(
        lambda x: linregress(np.arange(len(x)), x).slope, raw=False
    )

    if NUMBA_INSTALLED:
        logger.info("Numba已安装，使用JIT加速Hurst指数计算...")
        df["regime_hurst"] = rolling_hurst_numba(
            df["Close"].to_numpy(), p["regime_hurst_period"]
        )
    else:
        logger.warning("Numba未安装，Hurst指数计算会非常慢...")
        df["regime_hurst"] = (
            df["Close"]
            .rolling(p["regime_hurst_period"])
            .apply(get_hurst_exponent, raw=False)
        )

    norm_period = p["regime_norm_period"]
    for col in ["regime_adx", "regime_atr_slope", "regime_rsi_vol"]:
        rolling_min = df[col].rolling(norm_period).min()
        rolling_max = df[col].rolling(norm_period).max()
        denominator = rolling_max - rolling_min
        norm_col_name = col.replace("regime_", "") + "_norm"
        df[norm_col_name] = (df[col] - rolling_min) / denominator.replace(0, np.nan)
        if col == "regime_rsi_vol":
            df[norm_col_name] = 1 - df[norm_col_name]

    df["hurst_norm"] = (df["regime_hurst"] - 0.5).clip(0)

    df["regime_score"] = (
        df["adx_norm"].fillna(0) * p["regime_score_weight_adx"]
        + df["atr_slope_norm"].fillna(0) * p["regime_score_weight_atr"]
        + df["rsi_vol_norm"].fillna(0) * p["regime_score_weight_rsi"]
        + df["hurst_norm"].fillna(0) * p["regime_score_weight_hurst"]
    )

    df["market_regime"] = np.where(
        df["regime_score"] > p["regime_score_threshold"], 1, 0
    )

    df_4h = df.resample("4H").agg({"Close": "last"}).dropna()
    ema_4h = ta.trend.EMAIndicator(
        df_4h["Close"], window=p["mtf_period"]
    ).ema_indicator()
    df_4h["mtf_signal"] = np.where(df_4h["Close"] > ema_4h, 1, -1).shift(1)
    df["mtf_signal"] = df_4h["mtf_signal"].reindex(df.index, method="ffill").fillna(0)

    df["advanced_ml_signal"] = np.random.choice(
        [1, -1, 0], size=len(df), p=[0.1, 0.1, 0.8]
    )

    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。")
    return df


# --- 策略类与主函数 (完全无变化) ---
class BaseAssetStrategy:
    def __init__(self, strategy):
        self.strategy = strategy
        self.data = strategy.data

    def _calculate_entry_score(self):
        s = self.strategy
        score = 0
        if self.data.Close[-1] > s.tf_donchian_h[-1]:
            score += 0.3
        if self.data.Close[-1] < s.tf_donchian_l[-1]:
            score -= 0.3
        if s.tf_ema_fast[-1] > s.tf_ema_slow[-1]:
            score += 0.2
        else:
            score -= 0.2
        if s.tf_adx[-1] > s.tf_adx_confirm_threshold:
            score += 0.2 * np.sign(s.tf_ema_fast[-1] - s.tf_ema_slow[-1])
        score += 0.3 * s.mtf_signal[-1]
        score += 0.5 * s.advanced_ml_signal[-1]
        return score

    def _define_mr_entry_signal(self):
        s = self.strategy
        if self.data.Close[-1] < s.mr_bb_lower[-1] and s.mr_rsi[-1] < s.mr_rsi_oversold:
            return 1
        if (
            self.data.Close[-1] > s.mr_bb_upper[-1]
            and s.mr_rsi[-1] > s.mr_rsi_overbought
        ):
            return -1
        return 0


STRATEGY_MAPPING = {"BaseAssetStrategy": BaseAssetStrategy}


class UltimateStrategy(Strategy):
    symbol = None
    vol_weight = 1.0

    def init(self):
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)
        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {})
        strategy_class_name = asset_overrides.get("strategy_class", "BaseAssetStrategy")
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
            lambda: ta.momentum.RSIIndicator(close, self.mr_rsi_period).rsi()
        )

    def next(self):
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            if self.market_regime[-1] == 1:
                self.run_scoring_system_entry(self.data.Close[-1])
            else:
                self.run_mean_reversion_entry(self.data.Close[-1])

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
            self.tf_initial_stop_loss = p - risk_ps
            self.highest_high_in_trade = self.data.High[-1]
        else:
            self.sell(size=size)
            self.tf_initial_stop_loss = p + risk_ps
            self.lowest_low_in_trade = self.data.Low[-1]

    def manage_trend_following_exit(self, p):
        atr = self.tf_atr[-1]
        if self.position.is_long:
            if p < self.tf_initial_stop_loss:
                self.position.close()
                return
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade - atr * self.tf_chandelier_atr_multiplier
            )
            if p < self.chandelier_exit_level:
                self.position.close()
        elif self.position.is_short:
            if p > self.tf_initial_stop_loss:
                self.position.close()
                return
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade + atr * self.tf_chandelier_atr_multiplier
            )
            if p > self.chandelier_exit_level:
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
        return (self.equity * risk_pct) / rps

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        returns = np.array(list(self.recent_trade_returns))
        if np.all(returns >= 0) or np.all(returns <= 0):
            return self.default_risk_pct * self.vol_weight
        win_rate = np.mean(returns > 0)
        if win_rate == 0 or win_rate == 1:
            return self.default_risk_pct
        avg_win = np.mean(returns[returns > 0])
        avg_loss = np.abs(np.mean(returns[returns < 0]))
        if avg_loss == 0:
            return self.default_risk_pct
        reward_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


if __name__ == "__main__":
    logger.info(f"🚀 (V57.4-Final-Numba-Fix) 开始运行...")

    import sys

    if len(sys.argv) == 3:
        CONFIG["backtest_start_date"] = sys.argv[1]
        CONFIG["backtest_end_date"] = sys.argv[2]

    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_fetch_start_date = (
        backtest_start_dt - timedelta(days=CONFIG["data_lookback_days"])
    ).strftime("%Y-%m-%d")

    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期: {data_fetch_start_date}")

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
            CONFIG["backtest_start_date"] : CONFIG["backtest_end_date"]
        ].copy()
        if not backtest_period_slice.empty:
            processed_backtest_data[symbol] = backtest_period_slice

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
