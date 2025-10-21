# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V34.2.1 - Hotfix版)

修正V34.2版本中因代码重构导致的`KeyError: 'mtf_signal'`错误。
- 恢复了在`data_1d`数据帧中创建`mtf_signal`列的逻辑，确保多时间周期信号能够被正确计算和应用。
- 其他所有在V34.2中引入的“Patience and Stability”优化保持不变。
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

# import optuna

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

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 全局配置 ---
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],
    "interval": "5m",
    "start_date": "2025-08-01",
    "end_date": "2025-10-19",
    "initial_cash": 500_000,
    "commission": 0.001,
    "spread": 0.0005,
    "run_monte_carlo": True,
    "show_plots": False,
}

# --- 策略核心参数 (V34.2 结构化) ---
STRATEGY_PARAMS = {
    "risk_management": {
        "kelly_trade_history": 20,
        "default_risk_pct": 0.015,
        "max_risk_pct": 0.04,
        "dd_grace_period_bars": 240,
        "dd_initial_pct": 0.35,
        "dd_final_pct": 0.25,
        "dd_decay_bars": 4320,
    },
    "regime_filter": {
        "adx_period": 14,
        "atr_period": 14,
        "atr_slope_period": 5,
        "rsi_period": 14,
        "rsi_vol_period": 14,
        "norm_period": 252,
        "hurst_period": 100,
        "score_smoothing_period": 5,
        "weight_adx": 0.6,
        "weight_atr": 0.3,
        "weight_rsi": 0.05,
        "weight_hurst": 0.05,
        "score_threshold": 0.45,
    },
    "tf_params": {
        "donchian_period": 40,
        "ema_fast_period": 20,
        "ema_slow_period": 75,
        "adx_confirm_period": 14,
        "adx_confirm_threshold": 22,
        "chandelier_period": 22,
        "chandelier_atr_multiplier": 3.0,
        "atr_period": 14,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
    },
    "mr_params": {
        "bb_period": 25,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "stop_loss_atr_multiplier": 1.5,
        "risk_multiplier": 0.5,
        "volume_ma_period": 20,
        "tp_bb_width_factor": 0.8,
    },
    "signal_enhancement": {
        "mtf_period": 50,
        "score_mtf_bonus": 0.5,
        "ai_filter_rsi_period": 14,
        "ai_filter_fast_ma": 3,
        "ai_filter_slow_ma": 10,
        "ai_filter_confidence_threshold": 0.2,
        "score_ai_bonus": 0.5,
        "score_nonlinear_factor": 2.0,
    },
    "trade_management": {
        "volatility_norm_period": 100,
        "max_holding_period": 2880,
    },
}
ASSET_SPECIFIC_OVERRIDES = {"BTCUSDT": {}, "SOLUSDT": {}, "ETHUSDT": {}}


# ... (辅助函数保持不变)
def get_interval_timedelta(interval_str: str) -> timedelta:
    unit = interval_str[-1]
    value = int(interval_str[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    raise ValueError(f"不支持的时间间隔: {interval_str}")


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
                response = requests.get(url, params=params, timeout=20)
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
            logger.error("多次重试后仍无法获取数据，终止。")
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
    if len(df) > 1:
        expected_interval = get_interval_timedelta(interval)
        time_diffs = df.index.to_series().diff().iloc[1:]
        gaps = time_diffs[time_diffs > expected_interval * 1.1]
        if not gaps.empty:
            logger.warning(
                f"在 {symbol} 数据中检测到 {len(gaps)} 处数据缺失。最长缺失时段: {gaps.max()}"
            )
    logger.info(
        f"✅ 获取 {symbol} 数据成功：{len(df)} 条，从 {df.index[0]} 到 {df.index[-1]}"
    )
    return df


def compute_hurst(ts, max_lag=100):
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau, valid_lags = [], []
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


class UltimateStrategy(Strategy):
    # (策略类的实现与V34.2完全相同，无需修改)
    tf_donchian_period_dynamic = None
    tf_chandelier_atr_multiplier_dynamic = None
    mr_bb_std_dynamic = None
    max_risk_pct_override = None
    regime_score_threshold_override = None

    p_risk = STRATEGY_PARAMS["risk_management"]
    p_regime = STRATEGY_PARAMS["regime_filter"]
    p_tf = STRATEGY_PARAMS["tf_params"]
    p_mr = STRATEGY_PARAMS["mr_params"]
    p_signal = STRATEGY_PARAMS["signal_enhancement"]
    p_trade = STRATEGY_PARAMS["trade_management"]
    vol_weight = 1.0

    def init(self):
        self.tf_donchian_period = int(
            getattr(self, "tf_donchian_period_dynamic", self.p_tf["donchian_period"])
        )
        self.tf_chandelier_atr_multiplier = getattr(
            self,
            "tf_chandelier_atr_multiplier_dynamic",
            self.p_tf["chandelier_atr_multiplier"],
        )
        self.mr_bb_std = getattr(self, "mr_bb_std_dynamic", self.p_mr["bb_std"])
        self.regime_score_threshold = getattr(
            self, "regime_score_threshold_override", self.p_regime["score_threshold"]
        )
        self.max_risk_pct = getattr(
            self, "max_risk_pct_override", self.p_risk["max_risk_pct"]
        )

        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        volume = pd.Series(self.data.Volume, index=self.data.index)

        self.recent_trade_returns = deque(maxlen=self.p_risk["kelly_trade_history"])
        self.reset_trade_state()

        self.market_regime = self.I(lambda: self.data.market_regime)
        self.mtf_signal = self.I(lambda: self.data.mtf_signal)
        self.ai_filter_signal = self.I(lambda: self.data.ai_filter_signal)

        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.p_tf["atr_period"]
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
                close, self.p_tf["ema_fast_period"]
            ).ema_indicator()
        )
        self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(
                close, self.p_tf["ema_slow_period"]
            ).ema_indicator()
        )
        self.tf_adx = self.I(
            lambda: ta.trend.ADXIndicator(
                high, low, close, self.p_tf["adx_confirm_period"]
            ).adx()
        )
        macd = ta.trend.MACD(
            close,
            self.p_tf["macd_slow"],
            self.p_tf["macd_fast"],
            self.p_tf["macd_signal"],
        )
        self.tf_macd = self.I(lambda: macd.macd())
        self.tf_macd_signal = self.I(lambda: macd.macd_signal())

        bb = ta.volatility.BollingerBands(close, self.p_mr["bb_period"], self.mr_bb_std)
        self.mr_bb_upper = self.I(lambda: bb.bollinger_hband())
        self.mr_bb_lower = self.I(lambda: bb.bollinger_lband())
        self.mr_bb_mid = self.I(lambda: bb.bollinger_mavg())
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(close, self.p_mr["rsi_period"]).rsi()
        )
        self.mr_volume_ma = self.I(
            lambda: volume.rolling(self.p_mr["volume_ma_period"]).mean()
        )

        self.long_term_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.p_trade["volatility_norm_period"]
            ).average_true_range()
        )
        self.equity_peak = self.equity
        self.global_stop_triggered = False

    def reset_trade_state(self):
        self.active_sub_strategy, self.chandelier_exit_level, self.mr_stop_loss = (
            None,
            0.0,
            0.0,
        )
        self.highest_high_in_trade, self.lowest_low_in_trade, self.trade_entry_bar = (
            0,
            float("inf"),
            0,
        )

    def next(self):
        current_bar = len(self.data.Close) - 1
        p = self.p_risk
        if current_bar > p["dd_grace_period_bars"]:
            decay = min(
                (current_bar - p["dd_grace_period_bars"]) / p["dd_decay_bars"], 1.0
            )
            dd_pct = p["dd_initial_pct"] - (
                decay * (p["dd_initial_pct"] - p["dd_final_pct"])
            )
            self.equity_peak = max(self.equity_peak, self.equity)
            if self.equity < self.equity_peak * (1 - dd_pct):
                if not self.global_stop_triggered:
                    logger.warning(f"🚨 智能熔断触发！")
                if self.position:
                    self.position.close()
                self.global_stop_triggered = True
                return
        if self.global_stop_triggered:
            return
        if self.position:
            self.manage_open_position()
        else:
            if len(self.market_regime) > 0 and self.market_regime[-1] == 1:
                self.run_trend_following_entry()
            else:
                self.run_mean_reversion_entry()

    def manage_open_position(self):
        if (
            len(self.data) - 1 - self.trade_entry_bar
            > self.p_trade["max_holding_period"]
        ):
            self.close_position("时间退出")
            return
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit()
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit()

    def run_trend_following_entry(self):
        p = self.p_tf
        is_trend = self.tf_adx[-1] > p["adx_confirm_threshold"]
        is_break_up = self.data.High[-1] > self.tf_donchian_h[-1]
        is_break_down = self.data.Low[-1] < self.tf_donchian_l[-1]
        is_mom_long = self.tf_ema_fast[-1] > self.tf_ema_slow[-1]
        is_mom_short = self.tf_ema_fast[-1] < self.tf_ema_slow[-1]
        is_macd_long = self.tf_macd[-1] > self.tf_macd_signal[-1]
        is_macd_short = self.tf_macd[-1] < self.tf_macd_signal[-1]
        base_signal = 0
        if is_trend and is_break_up and is_mom_long and is_macd_long:
            base_signal = 1
        elif is_trend and is_break_down and is_mom_short and is_macd_short:
            base_signal = -1
        if base_signal == 0:
            return
        p_sig = self.p_signal
        score = 1.0
        if (base_signal == 1 and self.mtf_signal[-1] == 1) or (
            base_signal == -1 and self.mtf_signal[-1] == -1
        ):
            score += p_sig["score_mtf_bonus"]
        if (
            base_signal == 1
            and self.ai_filter_signal[-1] > p_sig["ai_filter_confidence_threshold"]
        ) or (
            base_signal == -1
            and self.ai_filter_signal[-1] < -p_sig["ai_filter_confidence_threshold"]
        ):
            score += p_sig["score_ai_bonus"]
        self.open_tf_position(
            is_long=(base_signal == 1), score=score ** p_sig["score_nonlinear_factor"]
        )

    def open_tf_position(self, is_long, score):
        price = self.data.Close[-1]
        initial_atr = self.tf_atr[-1]
        risk_per_share = initial_atr * self.tf_chandelier_atr_multiplier
        if risk_per_share <= 0:
            return
        target_risk = self._calculate_dynamic_risk()
        if len(self.long_term_atr) > 1 and self.long_term_atr[-1] > 0:
            target_risk *= 1 / max(initial_atr / self.long_term_atr[-1], 0.5)
        final_risk = target_risk * score
        size = self._calculate_position_size(price, risk_per_share, final_risk)
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        self.trade_entry_bar = len(self.data) - 1
        if is_long:
            self.buy(size=size)
            self.highest_high_in_trade = self.data.High[-1]
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - initial_atr * self.tf_chandelier_atr_multiplier
            )
        else:
            self.sell(size=size)
            self.lowest_low_in_trade = self.data.Low[-1]
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + initial_atr * self.tf_chandelier_atr_multiplier
            )

    def manage_trend_following_exit(self):
        price = self.data.Close[-1]
        current_atr = self.tf_atr[-1]
        should_close, reason = False, ""
        if self.position.is_long:
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - current_atr * self.tf_chandelier_atr_multiplier
            )
            if price < self.chandelier_exit_level:
                should_close, reason = True, "钱德勒止损"
            elif crossover(self.tf_ema_slow, self.tf_ema_fast):
                should_close, reason = True, "均线死叉"
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + current_atr * self.tf_chandelier_atr_multiplier
            )
            if price > self.chandelier_exit_level:
                should_close, reason = True, "钱德勒止损"
            elif crossover(self.tf_ema_fast, self.tf_ema_slow):
                should_close, reason = True, "均线金叉"
        if should_close:
            self.close_position(f"TF({reason})")

    def run_mean_reversion_entry(self):
        p = self.p_mr
        is_vol = self.data.Volume[-1] > self.mr_volume_ma[-1]
        is_os = (
            crossover(self.data.Close, self.mr_bb_lower)
            and self.mr_rsi[-1] < p["rsi_oversold"]
            and is_vol
        )
        is_ob = (
            crossover(self.mr_bb_upper, self.data.Close)
            and self.mr_rsi[-1] > p["rsi_overbought"]
            and is_vol
        )
        if is_os:
            self.open_mr_position(is_long=True)
        elif is_ob:
            self.open_mr_position(is_long=False)

    def open_mr_position(self, is_long):
        price = self.data.Close[-1]
        initial_atr = self.tf_atr[-1]
        risk_per_share = initial_atr * self.p_mr["stop_loss_atr_multiplier"]
        if risk_per_share <= 0:
            return
        target_risk = self._calculate_dynamic_risk() * self.p_mr["risk_multiplier"]
        size = self._calculate_position_size(price, risk_per_share, target_risk)
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        self.trade_entry_bar = len(self.data) - 1
        if is_long:
            self.buy(size=size)
            self.mr_stop_loss = price - risk_per_share
        else:
            self.sell(size=size)
            self.mr_stop_loss = price + risk_per_share

    def manage_mean_reversion_exit(self):
        price = self.data.Close[-1]
        p = self.p_mr
        should_close, reason = False, ""
        bb_width = self.mr_bb_upper[-1] - self.mr_bb_lower[-1]
        long_tp = self.mr_bb_mid[-1] + bb_width * p["tp_bb_width_factor"]
        short_tp = self.mr_bb_mid[-1] - bb_width * p["tp_bb_width_factor"]
        if self.position.is_long:
            if price >= long_tp:
                should_close, reason = True, "动态止盈"
            elif price < self.mr_bb_mid[-1]:
                should_close, reason = True, "回归失败"
            elif price <= self.mr_stop_loss:
                should_close, reason = True, "ATR硬止损"
        elif self.position.is_short:
            if price <= short_tp:
                should_close, reason = True, "动态止盈"
            elif price > self.mr_bb_mid[-1]:
                should_close, reason = True, "回归失败"
            elif price >= self.mr_stop_loss:
                should_close, reason = True, "ATR硬止损"
        if should_close:
            self.close_position(f"MR({reason})")

    def close_position(self, reason: str):
        equity_before = self.equity
        self.position.close()
        self.recent_trade_returns.append((self.equity / equity_before) - 1)
        self.reset_trade_state()

    def _calculate_position_size(self, price, risk_per_share, target_risk_pct):
        risk_capital = target_risk_pct * self.equity
        pos_value = risk_capital / (risk_per_share / price)
        return pos_value / self.equity

    def _calculate_dynamic_risk(self):
        p = self.p_risk
        if len(self.recent_trade_returns) < p["kelly_trade_history"]:
            return p["default_risk_pct"] * self.vol_weight
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return p["default_risk_pct"] * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        if reward_ratio == 0:
            return p["default_risk_pct"] * self.vol_weight
        kelly = win_rate - (1 - win_rate) / reward_ratio
        if kelly <= 0:
            return p["default_risk_pct"] * 0.1 * self.vol_weight
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


# ... (辅助函数保持不变)
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
    logger.info("--- 蒙特卡洛模拟结果 ---")
    logger.info(f"模拟次数: {n_simulations}")
    logger.info(f"平均最终权益: ${np.mean(final_equities):,.2f}")
    logger.info(f"最终权益中位数: ${np.median(final_equities):,.2f}")
    logger.info(f"5% 最差情况权益 (VaR): ${np.percentile(final_equities, 5):,.2f}")
    logger.info(f"95% 最好情况权益: ${np.percentile(final_equities, 95):,.2f}")


def generate_dynamic_params(volatility: float, baseline_vol: float) -> dict:
    vol_factor = np.clip(volatility / baseline_vol, 0.5, 1.5)
    p_tf = STRATEGY_PARAMS["tf_params"]
    p_mr = STRATEGY_PARAMS["mr_params"]
    p_risk = STRATEGY_PARAMS["risk_management"]
    return {
        "tf_chandelier_atr_multiplier_dynamic": np.round(
            p_tf["chandelier_atr_multiplier"] * vol_factor, 2
        ),
        "tf_donchian_period_dynamic": int(
            np.round(p_tf["donchian_period"] * vol_factor)
        ),
        "mr_bb_std_dynamic": np.round(p_mr["bb_std"] * vol_factor, 2),
        "max_risk_pct_override": np.round(
            np.clip(p_risk["max_risk_pct"] / vol_factor, 0.02, 0.06), 4
        ),
    }


# --- 主执行逻辑 ---
if __name__ == "__main__":
    logger.info(f"🚀 (V34.2.1: Hotfix版) 开始运行...")
    logger.info(f"详细交易日志将保存在文件: {log_filename}")

    all_data = {
        sym: fetch_binance_klines(
            sym, CONFIG["interval"], CONFIG["start_date"], CONFIG["end_date"]
        )
        for sym in CONFIG["symbols_to_test"]
    }
    all_data = {sym: df for sym, df in all_data.items() if not df.empty}
    if not all_data:
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    volatilities = {
        sym: max(
            np.log(df["Close"].resample("D").last().pct_change() + 1).std()
            * np.sqrt(365),
            0.01,
        )
        for sym, df in all_data.items()
    }
    inverse_vol = {sym: 1 / vol for sym, vol in volatilities.items() if vol > 0}
    vol_weights = {
        sym: (inv_vol / sum(inverse_vol.values())) * len(volatilities)
        for sym, inv_vol in inverse_vol.items()
    }
    logger.info("--- 波动率风险平价权重 (基于对数日收益率) ---")
    for symbol, weight in vol_weights.items():
        logger.info(
            f"  - {symbol}: {weight:.2f} (年化波动率: {volatilities.get(symbol, 0):.2%})"
        )

    all_stats = {}
    for symbol in CONFIG["symbols_to_test"]:
        print("\n" + "=" * 80 + f"\n正在回测品种: {symbol}\n" + "=" * 80)
        data = all_data.get(symbol)
        if data is None:
            continue

        symbol_volatility = volatilities.get(symbol, 0.7)
        daily_vol_series = (
            np.log(data["Close"].resample("D").last().pct_change() + 1)
        ).rolling(252).std() * np.sqrt(365)
        dynamic_baseline_vol = daily_vol_series.mean()
        if pd.isna(dynamic_baseline_vol) or dynamic_baseline_vol <= 0:
            dynamic_baseline_vol = 0.7
        logger.info(f"使用动态波动率基线: {dynamic_baseline_vol:.2%}")

        final_params = generate_dynamic_params(symbol_volatility, dynamic_baseline_vol)
        logger.info(f"--- 为 {symbol} 应用动态参数 ---")
        for key, value in final_params.items():
            logger.info(
                f"  - {key.replace('_dynamic', '').replace('_override', '')}: {value}"
            )

        logger.info("开始进行数据预处理 (多因子信号)...")
        p_sig, p_regime = (
            STRATEGY_PARAMS["signal_enhancement"],
            STRATEGY_PARAMS["regime_filter"],
        )

        data_1d = fetch_binance_klines(
            symbol, "1d", CONFIG["start_date"], CONFIG["end_date"]
        )
        if not data_1d.empty:
            sma_1d = ta.trend.SMAIndicator(
                data_1d["Close"], window=p_sig["mtf_period"]
            ).sma_indicator()
            # 核心修正: 重新加入这一行来创建mtf_signal列
            data_1d["mtf_signal"] = np.where(data_1d["Close"] > sma_1d, 1, -1)
            data["mtf_signal"] = (
                data_1d["mtf_signal"].reindex(data.index, method="ffill").fillna(0)
            )
        else:
            data["mtf_signal"] = 0

        adx = ta.trend.ADXIndicator(
            data["High"], data["Low"], data["Close"], p_regime["adx_period"]
        ).adx()
        adx_norm = (adx - adx.rolling(p_regime["norm_period"]).min()) / (
            adx.rolling(p_regime["norm_period"]).max()
            - adx.rolling(p_regime["norm_period"]).min()
        )
        atr = ta.volatility.AverageTrueRange(
            data["High"], data["Low"], data["Close"], p_regime["atr_period"]
        ).average_true_range()
        atr_slope = (atr - atr.shift(p_regime["atr_slope_period"])) / atr.shift(
            p_regime["atr_slope_period"]
        )
        atr_slope_norm = (
            atr_slope - atr_slope.rolling(p_regime["norm_period"]).min()
        ) / (
            atr_slope.rolling(p_regime["norm_period"]).max()
            - atr_slope.rolling(p_regime["norm_period"]).min()
        )
        rsi = ta.momentum.RSIIndicator(
            data["Close"], window=p_regime["rsi_period"]
        ).rsi()
        rsi_vol = rsi.rolling(p_regime["rsi_vol_period"]).std()
        rsi_vol_norm = (rsi_vol - rsi_vol.rolling(p_regime["norm_period"]).min()) / (
            rsi_vol.rolling(p_regime["norm_period"]).max()
            - rsi_vol.rolling(p_regime["norm_period"]).min()
        )

        dynamic_hurst_period = int(
            np.round(
                p_regime["hurst_period"]
                / np.clip(symbol_volatility / dynamic_baseline_vol, 0.5, 1.5)
            )
        )
        hurst = pd.Series(
            [
                compute_hurst(
                    np.log(
                        data["Close"]
                        .iloc[i - dynamic_hurst_period : i]
                        .replace(0, 1e-9)
                    ).values
                )
                for i in range(dynamic_hurst_period, len(data))
            ],
            index=data.index[dynamic_hurst_period:],
        ).fillna(0.5)
        hurst_norm = np.clip((hurst - 0.3) / 0.7, 0, 1)

        regime_score = (
            adx_norm.fillna(0.5) * p_regime["weight_adx"]
            + atr_slope_norm.fillna(0.5) * p_regime["weight_atr"]
            + (1 - rsi_vol_norm.fillna(0.5)) * p_regime["weight_rsi"]
            + hurst_norm * p_regime["weight_hurst"]
        )
        regime_score_smoothed = regime_score.ewm(
            span=p_regime["score_smoothing_period"], adjust=False
        ).mean()
        data["market_regime"] = np.where(
            regime_score_smoothed > p_regime["score_threshold"], 1, -1
        )

        rsi_f = ta.momentum.RSIIndicator(
            data["Close"], window=p_sig["ai_filter_rsi_period"]
        ).rsi()
        data["ai_filter_signal"] = (
            (
                (
                    rsi_f.rolling(p_sig["ai_filter_fast_ma"]).mean()
                    - rsi_f.rolling(p_sig["ai_filter_slow_ma"]).mean()
                )
                / 50
            )
            .clip(-1, 1)
            .fillna(0)
        )
        logger.info("数据预处理完成。")

        bt = Backtest(
            data,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            spread=CONFIG["spread"],
            finalize_trades=True,
        )
        stats = bt.run(vol_weight=vol_weights.get(symbol, 1.0), **final_params)
        all_stats[symbol] = stats

        print("\n" + "-" * 40 + f"\n          {symbol} 回测结果摘要\n" + "-" * 40)
        print(stats)
        print("\n--- 🔍 策略健康度监控 ---")
        sqn = stats.get("SQN")
        kelly = stats.get("Kelly Criterion")
        calmar = stats.get("Calmar Ratio")
        if sqn is not None and not np.isnan(sqn):
            print(f"SQN (系统质量数): {sqn:.2f}")
        if kelly is not None and not np.isnan(kelly):
            print(f"凯利准则: {kelly:.4f}")
        if calmar is not None and not np.isnan(calmar):
            print(f"卡玛比率: {calmar:.3f}")

        if CONFIG["show_plots"]:
            bt.plot()
        if CONFIG["run_monte_carlo"] and not stats["_trades"].empty:
            run_monte_carlo(stats["_trades"], CONFIG["initial_cash"], symbol)

    if len(all_stats) > 0:
        total_final_equity = sum(s["Equity Final [$]"] for s in all_stats.values())
        total_initial_cash = CONFIG["initial_cash"] * len(all_stats)
        portfolio_return = (
            (total_final_equity - total_initial_cash) / total_initial_cash * 100
        )
        print("\n" + "#" * 80 + "\n                 组合策略表现总览\n" + "#" * 80)
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)"
                f"\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A'):.3f}"
            )
        print("\n--- 投资组合整体表现 ---")
        print(f"总初始资金: ${total_initial_cash:,.2f}")
        print(f"总最终权益: ${total_final_equity:,.2f}")
        print(f"组合总回报率: {portfolio_return:.2f}%")
