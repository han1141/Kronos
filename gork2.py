# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V38.0 - “宏观牛熊引擎”)

终极架构升级：引入宏观牛熊状态，动态调整策略的攻击性。
1.  引入“宏观牛熊状态”(`macro_regime`): 基于200日均线判断市场宏观趋势，
    为每个交易对标记出牛市(1)或熊市(-1)的大背景。
2.  策略逻辑动态化：
    - `FilteredTF_MR`策略在牛市中只做多，在熊市中只做空，旨在顺应大势，
      解决V37过于保守、错失牛市行情的核心痛点。
    - `VolatilityBreakout`策略在牛市中只捕捉向上突破，在熊市中只捕捉
      向下突破，提高信号的有效性。
3.  实现“看大势、做小势”：系统进化为多策略、多周期、多维度的元框架，
    能够在熊市中自动切换至防御模式，在牛市中大胆进攻。
"""

# --- 1. 导入库与配置 ---
import pandas as pd
import requests
import time
from datetime import datetime
import logging
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.font_manager
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 日志系统配置 ---
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
        logger.warning("未找到指定的中文字体，绘图可能出现乱码。")
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")


set_chinese_font()


# --- 全局配置 ---
CONFIG = {
    "symbols_to_test": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "interval": "4h",
    "start_date": "2020-01-01",  # 恢复长周期回测以验证新框架的有效性
    "end_date": "2025-10-08",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,
    "run_monte_carlo": True,
    "show_plots": False,
}

# --- FilteredTF_MR 策略的全局默认参数 ---
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
    "regime_hurst_period": 100,
    "regime_norm_period": 252,
    "regime_vov_period": 20,
    "regime_score_weight_adx": 0.50,
    "regime_score_weight_atr": 0.25,
    "regime_score_weight_rsi": 0.05,
    "regime_score_weight_hurst": 0.05,
    "regime_score_weight_vov": 0.15,
    "regime_score_upper_threshold": 0.60,
    "regime_score_lower_threshold": 0.40,
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_adx_confirm_period": 14,
    "tf_adx_confirm_threshold": 18,
    "tf_chandelier_period": 22,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
    "tf_volume_period": 20,
    "tf_volume_multiplier": 1.5,
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
    "macro_ma_period": 200,  # 新增：宏观牛熊线周期
}


# --- 辅助函数 ---
def fetch_binance_klines(
    symbol: str, interval: str, start_str: str, end_str: str = None, limit: int = 1000
) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    columns = [
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
    all_data, retries = [], 5
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
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not data:
                    break
                all_data.extend(data)
                start_ts = data[-1][0] + 1
                break
            except requests.exceptions.RequestException as e:
                wait = 2**attempt
                logger.warning(f"请求失败: {e}，{wait}s后重试...")
                time.sleep(wait)
        else:
            logger.error(f"多次重试后仍无法获取数据，终止。")
            break
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=columns)
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    logger.info(
        f"✅ 获取 {symbol} 数据成功：{len(df)} 条，从 {df.index[0]} 到 {df.index[-1]}"
    )
    return df


def compute_hurst(ts, max_lag=100):
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau = [
        np.std(ts[lag:] - ts[:-lag]) for lag in lags if np.std(ts[lag:] - ts[:-lag]) > 0
    ]
    if len(tau) < 2:
        return 0.5
    valid_lags = [lag for lag in lags if np.std(ts[lag:] - ts[:-lag]) > 0]
    try:
        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return max(0.0, min(1.0, poly[0]))
    except:
        return 0.5


def run_monte_carlo(trades_df, initial_cash, symbol: str, n_simulations=1000):
    if trades_df.empty:
        logger.warning("没有交易数据，无法进行蒙特卡洛模拟。")
        return
    returns = trades_df["ReturnPct"]
    final_equities = [
        initial_cash
        * (1 + np.random.choice(returns, size=len(returns), replace=True)).prod()
        for _ in range(n_simulations)
    ]
    mean_equity, median_equity = np.mean(final_equities), np.median(final_equities)
    var_5_pct, best_95_pct = np.percentile(final_equities, 5), np.percentile(
        final_equities, 95
    )
    logger.info(
        f"--- 蒙特卡洛模拟结果 ---\n模拟次数: {n_simulations}, 平均最终权益: ${mean_equity:,.2f}, 中位数: ${median_equity:,.2f}\n5% VaR: ${var_5_pct:,.2f}, 95% 最好情况: ${best_95_pct:,.2f}"
    )


# ==============================================================================
# --- 策略库 (STRATEGY HUB) ---
# ==============================================================================


# --- 策略A: 稳健型全天候策略 ---
class FilteredTF_MR(Strategy):
    for key, value in STRATEGY_PARAMS.items():
        exec(f"{key} = {value}")
    tf_donchian_period_dynamic = None
    tf_chandelier_atr_multiplier_dynamic = None
    mr_bb_std_dynamic = None
    max_risk_pct_override = None
    regime_score_upper_threshold_override = None
    regime_score_lower_threshold_override = None
    use_volume_filter_override = None
    use_engulfing_filter_override = None
    vol_weight = 1.0

    def is_bullish_engulfing(self):
        if len(self.data.Close) < 2:
            return False
        prev_open, prev_close, curr_open, curr_close = (
            self.data.Open[-2],
            self.data.Close[-2],
            self.data.Open[-1],
            self.data.Close[-1],
        )
        return (
            prev_open > prev_close
            and curr_close > curr_open
            and curr_close > prev_open
            and curr_open < prev_close
        )

    def is_bearish_engulfing(self):
        if len(self.data.Close) < 2:
            return False
        prev_open, prev_close, curr_open, curr_close = (
            self.data.Open[-2],
            self.data.Close[-2],
            self.data.Open[-1],
            self.data.Close[-1],
        )
        return (
            prev_close > prev_open
            and curr_open > curr_close
            and curr_open > prev_close
            and curr_close < prev_open
        )

    def init(self):
        self.tf_donchian_period = int(
            getattr(self, "tf_donchian_period_dynamic", self.tf_donchian_period)
        )
        self.tf_chandelier_atr_multiplier = getattr(
            self,
            "tf_chandelier_atr_multiplier_dynamic",
            self.tf_chandelier_atr_multiplier,
        )
        self.mr_bb_std = getattr(self, "mr_bb_std_dynamic", self.mr_bb_std)
        self.max_risk_pct = getattr(self, "max_risk_pct_override", self.max_risk_pct)
        self.regime_score_upper_threshold = getattr(
            self,
            "regime_score_upper_threshold_override",
            self.regime_score_upper_threshold,
        )
        self.regime_score_lower_threshold = getattr(
            self,
            "regime_score_lower_threshold_override",
            self.regime_score_lower_threshold,
        )
        self.use_volume_filter = getattr(self, "use_volume_filter_override", True)
        self.use_engulfing_filter = getattr(self, "use_engulfing_filter_override", True)
        close, high, low, volume = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
            pd.Series(self.data.Volume),
        )
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()
        self.market_regime, self.mtf_signal, self.ai_filter_signal = (
            self.I(lambda: self.data.market_regime),
            self.I(lambda: self.data.mtf_signal),
            self.I(lambda: self.data.ai_filter_signal),
        )
        self.macro_regime = self.I(lambda: self.data.macro_regime)  # 新增：加载宏观状态
        self.tf_atr = self.I(
            ta.volatility.average_true_range, high, low, close, self.tf_atr_period
        )
        self.tf_donchian_h = self.I(
            lambda: high.rolling(self.tf_donchian_period).max().shift(1)
        )
        self.tf_donchian_l = self.I(
            lambda: low.rolling(self.tf_donchian_period).min().shift(1)
        )
        self.tf_ema_fast = self.I(
            ta.trend.ema_indicator, close, self.tf_ema_fast_period
        )
        self.tf_ema_slow = self.I(
            ta.trend.ema_indicator, close, self.tf_ema_slow_period
        )
        self.tf_adx = self.I(ta.trend.adx, high, low, close, self.tf_adx_confirm_period)
        self.tf_avg_volume = self.I(
            lambda: volume.rolling(self.tf_volume_period).mean()
        )
        bb_indicator = ta.volatility.BollingerBands(
            close, self.mr_bb_period, self.mr_bb_std
        )
        self.mr_bb_upper, self.mr_bb_lower, self.mr_bb_mid = (
            self.I(bb_indicator.bollinger_hband),
            self.I(bb_indicator.bollinger_lband),
            self.I(bb_indicator.bollinger_mavg),
        )
        self.mr_rsi = self.I(ta.momentum.rsi, close, self.mr_rsi_period)
        self.long_term_atr = self.I(
            ta.volatility.average_true_range,
            high,
            low,
            close,
            self.volatility_norm_period,
        )
        self.equity_peak, self.global_stop_triggered = self.equity, False

    def reset_trade_state(self):
        (
            self.active_sub_strategy,
            self.chandelier_exit_level,
            self.highest_high_in_trade,
            self.lowest_low_in_trade,
            self.mr_stop_loss,
        ) = (None, 0.0, 0, float("inf"), 0.0)

    def next(self):
        if len(self.data.Close) - 1 > self.dd_grace_period_bars:
            decay = min(
                (len(self.data.Close) - 1 - self.dd_grace_period_bars)
                / self.dd_decay_bars,
                1.0,
            )
            dd_pct = self.dd_initial_pct - (
                decay * (self.dd_initial_pct - self.dd_final_pct)
            )
            self.equity_peak = max(self.equity_peak, self.equity)
            if self.equity < self.equity_peak * (1 - dd_pct):
                if not self.global_stop_triggered and self.position:
                    self.position.close()
                self.global_stop_triggered = True
        if self.global_stop_triggered:
            return
        if not self.position:
            regime = self.market_regime[-1]
            if regime == 1:
                self.run_trend_following_entry(self.data.Close[-1])
            elif regime == -1:
                self.run_mean_reversion_entry(self.data.Close[-1])
        else:
            self.manage_open_position(self.data.Close[-1])

    def manage_open_position(self, price):
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(price)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(price)

    def run_trend_following_entry(self, price):
        if self.use_volume_filter and not (
            self.data.Volume[-1] > self.tf_avg_volume[-1] * self.tf_volume_multiplier
        ):
            return
        sig = 0
        if (
            self.tf_adx[-1] > self.tf_adx_confirm_threshold
            and self.data.High[-1] > self.tf_donchian_h[-1]
            and self.tf_ema_fast[-1] > self.tf_ema_slow[-1]
        ):
            sig = 1
        elif (
            self.tf_adx[-1] > self.tf_adx_confirm_threshold
            and self.data.Low[-1] < self.tf_donchian_l[-1]
            and self.tf_ema_fast[-1] < self.tf_ema_slow[-1]
        ):
            sig = -1
        if sig == 0:
            return
        # 新增：宏观牛熊过滤
        if (self.macro_regime[-1] == 1 and sig == -1) or (
            self.macro_regime[-1] == -1 and sig == 1
        ):
            return
        score = 1.0 + self.score_mtf_bonus * (1 if sig == self.mtf_signal[-1] else 0)
        score += self.score_ai_bonus * (
            1
            if (
                sig == 1
                and self.ai_filter_signal[-1] > self.ai_filter_confidence_threshold
            )
            or (
                sig == -1
                and self.ai_filter_signal[-1] < -self.ai_filter_confidence_threshold
            )
            else 0
        )
        self.open_tf_position(
            price, is_long=(sig == 1), score=score**self.score_nonlinear_factor
        )

    def run_mean_reversion_entry(self, price):
        is_oversold = (
            crossover(self.data.Close, self.mr_bb_lower)
            and self.mr_rsi[-1] < self.mr_rsi_oversold
        )
        is_overbought = (
            crossover(self.mr_bb_upper, self.data.Close)
            and self.mr_rsi[-1] > self.mr_rsi_overbought
        )
        # 新增：宏观牛熊过滤
        if (self.macro_regime[-1] == 1 and is_overbought) or (
            self.macro_regime[-1] == -1 and is_oversold
        ):
            return
        if self.use_engulfing_filter:
            if is_oversold and self.is_bullish_engulfing():
                self.open_mr_position(price, True)
            elif is_overbought and self.is_bearish_engulfing():
                self.open_mr_position(price, False)
        else:
            if is_oversold:
                self.open_mr_position(price, True)
            elif is_overbought:
                self.open_mr_position(price, False)

    def open_tf_position(self, price, is_long, score):
        risk_per_share = self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
        if risk_per_share <= 0:
            return
        risk_pct = self._calculate_dynamic_risk()
        if len(self.long_term_atr) > 1 and self.long_term_atr[-1] > 0:
            risk_pct *= 1 / max(self.tf_atr[-1] / self.long_term_atr[-1], 0.5)
        size = self._calculate_position_size(price, risk_per_share, risk_pct * score)
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
        risk_per_share = self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
        if self.position.is_long:
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            if price < self.highest_high_in_trade - risk_per_share:
                self.close_position("TF(钱德勒)")
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            if price > self.lowest_low_in_trade + risk_per_share:
                self.close_position("TF(钱德勒)")

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
        if (self.position.is_long and price >= self.mr_bb_mid[-1]) or (
            self.position.is_short and price <= self.mr_bb_mid[-1]
        ):
            self.close_position("MR(回归中轨)")
        elif (self.position.is_long and price <= self.mr_stop_loss) or (
            self.position.is_short and price >= self.mr_stop_loss
        ):
            self.close_position("MR(ATR止损)")

    def close_position(self, reason: str):
        if self.position.pl != 0:
            self.recent_trade_returns.append(
                self.position.pl / (self.equity - self.position.pl)
            )
        self.position.close()
        self.reset_trade_state()

    def _calculate_position_size(self, price, risk_per_share, target_risk_pct):
        return (target_risk_pct * self.equity) / (risk_per_share / price * self.equity)

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins, losses = [r for r in self.recent_trade_returns if r > 0], [
            r for r in self.recent_trade_returns if r < 0
        ]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win_loss_ratio = (sum(wins) / len(wins)) / (abs(sum(losses) / len(losses)))
        kelly = (
            win_rate - (1 - win_rate) / avg_win_loss_ratio
            if avg_win_loss_ratio > 0
            else 0
        )
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


# --- 策略C: 波动率收缩突破策略 ---
class VolatilityBreakout(Strategy):
    squeeze_period = 100
    breakout_period = 20
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    sl_atr_multiplier = 2.0
    tp_atr_multiplier = 4.0
    max_risk_pct = 0.025
    vol_weight = 1.0

    squeeze_period_override = None
    breakout_period_override = None
    sl_atr_multiplier_override = None
    tp_atr_multiplier_override = None
    max_risk_pct_override = None

    def init(self):
        self.squeeze_period = int(
            getattr(self, "squeeze_period_override", self.squeeze_period)
        )
        self.breakout_period = int(
            getattr(self, "breakout_period_override", self.breakout_period)
        )
        self.sl_atr_multiplier = getattr(
            self, "sl_atr_multiplier_override", self.sl_atr_multiplier
        )
        self.tp_atr_multiplier = getattr(
            self, "tp_atr_multiplier_override", self.tp_atr_multiplier
        )
        self.max_risk_pct = getattr(self, "max_risk_pct_override", self.max_risk_pct)

        close, high, low = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )
        self.macro_regime = self.I(lambda: self.data.macro_regime)  # 新增：加载宏观状态

        bb_indicator = ta.volatility.BollingerBands(close, self.bb_period, self.bb_std)
        bbw_series = (
            bb_indicator.bollinger_hband() - bb_indicator.bollinger_lband()
        ) / bb_indicator.bollinger_mavg()
        rolling_min_bbw_series = bbw_series.rolling(self.squeeze_period).min()

        self.bbw, self.rolling_min_bbw = self.I(lambda: bbw_series), self.I(
            lambda: rolling_min_bbw_series
        )
        self.breakout_high = self.I(
            lambda: high.rolling(self.breakout_period).max().shift(1)
        )
        self.breakout_low = self.I(
            lambda: low.rolling(self.breakout_period).min().shift(1)
        )
        self.atr = self.I(
            ta.volatility.average_true_range, high, low, close, self.atr_period
        )
        self.sl_level, self.tp_level = 0.0, 0.0

    def next(self):
        price = self.data.Close[-1]
        if self.position:
            if (
                self.position.is_long
                and (price <= self.sl_level or price >= self.tp_level)
            ) or (
                self.position.is_short
                and (price >= self.sl_level or price <= self.tp_level)
            ):
                self.position.close()
            return

        is_in_squeeze = (
            self.bbw[-1] <= self.rolling_min_bbw[-1] * 1.05
            if not np.isnan(self.rolling_min_bbw[-1])
            else False
        )
        if not is_in_squeeze:
            return

        risk_per_share = self.atr[-1] * self.sl_atr_multiplier
        if risk_per_share <= 0:
            return
        size = (self.max_risk_pct * self.equity * self.vol_weight) / (
            risk_per_share / price * self.equity
        )
        if not (0 < size < 0.98):
            return

        # 新增：宏观牛熊过滤
        if self.macro_regime[-1] == 1 and crossover(
            self.data.Close, self.breakout_high[-1]
        ):
            self.buy(size=size)
            self.sl_level, self.tp_level = (
                price - risk_per_share,
                price + self.atr[-1] * self.tp_atr_multiplier,
            )
        elif self.macro_regime[-1] == -1 and crossover(
            self.breakout_low[-1], self.data.Close
        ):
            self.sell(size=size)
            self.sl_level, self.tp_level = (
                price + risk_per_share,
                price - self.atr[-1] * self.tp_atr_multiplier,
            )


# --- 策略注册中心 ---
STRATEGY_HUB = {
    "FilteredTF_MR": FilteredTF_MR,
    "VolatilityBreakout": VolatilityBreakout,
}

# --- 资产个性化配置中心 ---
ASSET_SPECIFIC_OVERRIDES = {
    "BTCUSDT": {"strategy_name": "FilteredTF_MR"},
    "ETHUSDT": {"strategy_name": "FilteredTF_MR"},
    "SOLUSDT": {
        "strategy_name": "VolatilityBreakout",
        "squeeze_period_override": 120,
        "breakout_period_override": 24,
        "sl_atr_multiplier_override": 1.8,
        "tp_atr_multiplier_override": 3.5,
        "max_risk_pct_override": 0.03,
    },
}


def generate_dynamic_params(volatility: float, baseline_vol: float) -> dict:
    volatility_factor = np.clip(volatility / baseline_vol, 0.7, 1.5)
    p = STRATEGY_PARAMS
    return {
        "tf_chandelier_atr_multiplier_dynamic": np.round(
            p["tf_chandelier_atr_multiplier"] * volatility_factor, 2
        ),
        "tf_donchian_period_dynamic": int(
            np.round(p["tf_donchian_period"] * volatility_factor)
        ),
        "mr_bb_std_dynamic": np.round(p["mr_bb_std"] * volatility_factor, 2),
        "max_risk_pct_override": np.round(
            np.clip(p["max_risk_pct"] / volatility_factor, 0.015, 0.055), 4
        ),
    }


if __name__ == "__main__":
    all_stats, total_final_equity = {}, 0
    logger.info(f'🚀 (V38.0: "宏观牛熊引擎") 开始运行...')
    logger.info(f"详细交易日志将保存在文件: {log_filename}")

    all_data, volatilities = {}, {}
    for symbol in CONFIG["symbols_to_test"]:
        data = fetch_binance_klines(
            symbol, CONFIG["interval"], CONFIG["start_date"], CONFIG["end_date"]
        )
        if not data.empty:
            all_data[symbol] = data
            volatilities[symbol] = data["Close"].resample(
                "D"
            ).last().pct_change().std() * np.sqrt(365)
    if not volatilities:
        exit("所有品种数据获取失败。")
    inverse_vol = {s: 1 / v for s, v in volatilities.items() if v > 0}
    total_inverse_vol = sum(inverse_vol.values())
    vol_weights = {
        s: (iv / total_inverse_vol) * len(volatilities) for s, iv in inverse_vol.items()
    }

    for symbol in CONFIG["symbols_to_test"]:
        print("\n" + "=" * 80 + f"\n正在回测品种: {symbol}\n" + "=" * 80)
        data_4h = all_data.get(symbol)
        if data_4h is None:
            continue

        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(symbol, {})
        strategy_name_to_use = asset_overrides.get("strategy_name", "FilteredTF_MR")
        StrategyClass = STRATEGY_HUB.get(strategy_name_to_use)

        if not StrategyClass:
            logger.error(
                f"错误：在策略库中未找到名为 '{strategy_name_to_use}' 的策略。跳过 {symbol}。"
            )
            continue
        logger.info(f"✅ 为 {symbol} 加载策略: {strategy_name_to_use}")

        final_params = {}
        p = STRATEGY_PARAMS

        # --- 全局数据预处理：宏观牛熊状态 ---
        logger.info("开始计算宏观牛熊状态...")
        data_1d = fetch_binance_klines(
            symbol, "1d", CONFIG["start_date"], CONFIG["end_date"]
        )
        if not data_1d.empty and len(data_1d) > p["macro_ma_period"]:
            macro_ma = ta.trend.sma_indicator(
                data_1d["Close"], window=p["macro_ma_period"]
            )
            data_1d["macro_regime"] = np.where(data_1d["Close"] > macro_ma, 1, -1)
            data_4h["macro_regime"] = (
                data_1d["macro_regime"].reindex(data_4h.index, method="ffill").fillna(0)
            )
        else:
            logger.warning(f"日线数据不足或获取失败，宏观过滤器将停用。")
            data_4h["macro_regime"] = 0

        if StrategyClass == FilteredTF_MR:
            symbol_volatility = volatilities.get(symbol, 0.7)
            daily_vol_series = data_4h["Close"].resample(
                "D"
            ).last().pct_change().rolling(252).std() * np.sqrt(365)
            dynamic_baseline_vol = (
                daily_vol_series.mean()
                if pd.notna(daily_vol_series.mean()) and daily_vol_series.mean() > 0
                else 0.7
            )
            dynamic_params_for_symbol = generate_dynamic_params(
                symbol_volatility, dynamic_baseline_vol
            )
            final_params.update(dynamic_params_for_symbol)

            logger.info("开始为 FilteredTF_MR 进行微观数据预处理...")
            if not data_1d.empty:
                sma_1d = ta.trend.SMAIndicator(
                    data_1d["Close"], window=p["mtf_period"]
                ).sma_indicator()
                data_1d["mtf_signal"] = np.where(data_1d["Close"] > sma_1d, 1, -1)
                data_4h["mtf_signal"] = (
                    data_1d["mtf_signal"]
                    .reindex(data_4h.index, method="ffill")
                    .fillna(0)
                )
            else:
                data_4h["mtf_signal"] = 0

            adx, atr, rsi = (
                ta.trend.adx(
                    data_4h.High, data_4h.Low, data_4h.Close, p["regime_adx_period"]
                ),
                ta.volatility.average_true_range(
                    data_4h.High, data_4h.Low, data_4h.Close, p["regime_atr_period"]
                ),
                ta.momentum.rsi(data_4h.Close, p["regime_rsi_period"]),
            )
            adx_norm = (adx - adx.rolling(p["regime_norm_period"]).min()) / (
                adx.rolling(p["regime_norm_period"]).max()
                - adx.rolling(p["regime_norm_period"]).min()
            )
            atr_slope_norm = (
                atr.pct_change(p["regime_atr_slope_period"])
                - atr.pct_change(p["regime_atr_slope_period"])
                .rolling(p["regime_norm_period"])
                .min()
            ) / (
                atr.pct_change(p["regime_atr_slope_period"])
                .rolling(p["regime_norm_period"])
                .max()
                - atr.pct_change(p["regime_atr_slope_period"])
                .rolling(p["regime_norm_period"])
                .min()
            )
            rsi_vol = rsi.rolling(p["regime_rsi_vol_period"]).std()
            rsi_vol_norm = 1 - (
                rsi_vol - rsi_vol.rolling(p["regime_norm_period"]).min()
            ) / (
                rsi_vol.rolling(p["regime_norm_period"]).max()
                - rsi_vol.rolling(p["regime_norm_period"]).min()
            )
            hurst_series = (
                data_4h["Close"]
                .rolling(p["regime_hurst_period"])
                .apply(lambda x: compute_hurst(np.log(x.replace(0, 1e-9))), raw=False)
            )
            hurst_norm = np.clip((hurst_series - 0.3) / 0.7, 0, 1)
            vov = atr.rolling(p["regime_vov_period"]).std()
            vov_norm = 1 - (vov - vov.rolling(p["regime_norm_period"]).min()) / (
                vov.rolling(p["regime_norm_period"]).max()
                - vov.rolling(p["regime_norm_period"]).min()
            )

            regime_score = (
                adx_norm.fillna(0.5) * p["regime_score_weight_adx"]
                + atr_slope_norm.fillna(0.5) * p["regime_score_weight_atr"]
                + rsi_vol_norm.fillna(0.5) * p["regime_score_weight_rsi"]
                + hurst_norm.fillna(0.5) * p["regime_score_weight_hurst"]
                + vov_norm.fillna(0.5) * p["regime_score_weight_vov"]
            )
            upper_thresh = asset_overrides.get(
                "regime_score_upper_threshold_override",
                p["regime_score_upper_threshold"],
            )
            lower_thresh = asset_overrides.get(
                "regime_score_lower_threshold_override",
                p["regime_score_lower_threshold"],
            )
            data_4h["market_regime"] = np.select(
                [regime_score > upper_thresh, regime_score < lower_thresh],
                [1, -1],
                default=0,
            )

            rsi_filter = ta.momentum.RSIIndicator(
                data_4h.Close, p["ai_filter_rsi_period"]
            ).rsi()
            data_4h["ai_filter_signal"] = (
                (
                    (
                        rsi_filter.rolling(p["ai_filter_fast_ma"]).mean()
                        - rsi_filter.rolling(p["ai_filter_slow_ma"]).mean()
                    )
                    / 50
                )
                .clip(-1, 1)
                .fillna(0)
            )
            logger.info("数据预处理完成。")

        elif StrategyClass == VolatilityBreakout:
            logger.info("VolatilityBreakout 策略仅需宏观状态，无需额外微观预处理。")
            pass

        if asset_overrides:
            logger.info(f"--- 为 {symbol} 应用个性化参数 ---")
            for key, value in asset_overrides.items():
                if key != "strategy_name":
                    final_params[key] = value
                    logger.info(f"  - {key}: {value}")

        bt = Backtest(
            data_4h,
            StrategyClass,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            spread=CONFIG["spread"],
            finalize_trades=True,
        )
        stats = bt.run(vol_weight=vol_weights.get(symbol, 1.0), **final_params)
        all_stats[symbol] = stats
        total_final_equity += stats["Equity Final [$]"]

        print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}\n{stats}")
        sqn, kelly, calmar = (
            stats.get("SQN"),
            stats.get("Kelly Criterion"),
            stats.get("Calmar Ratio"),
        )
        if sqn is not None and not np.isnan(sqn):
            print(
                f"\n--- 🔍 策略健康度监控 ---\nSQN: {sqn:.2f}\n凯利准则: {kelly:.4f}\n卡玛比率: {calmar:.3f}"
            )
        if CONFIG["run_monte_carlo"] and not stats["_trades"].empty:
            run_monte_carlo(stats["_trades"], CONFIG["initial_cash"], symbol)

    if all_stats:
        total_initial_cash = CONFIG["initial_cash"] * len(all_stats)
        portfolio_return = (
            (total_final_equity - total_initial_cash) / total_initial_cash * 100
        )
        print("\n" + "#" * 80 + "\n                 组合策略表现总览\n" + "#" * 80)
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A'):.3f}"
            )
        print(
            f"\n--- 投资组合整体表现 ---\n总初始资金: ${total_initial_cash:,.2f}\n总最终权益: ${total_final_equity:,.2f}\n组合总回报率: {portfolio_return:.2f}%"
        )
