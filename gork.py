# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V33.1 - “系统稳定器增强”版，修复版)

基于V33.1，修复KeyError和RuntimeWarning：
- 包装Hurst日志访问以防数组键错误。
- 优化compute_hurst：使用std_diff（无sqrt），过滤std>0，避免log(0)警告；返回poly[0]作为H（标准方法）。
- 调整Hurst标准化：clip((hurst-0.3)/0.7, 0,1)以覆盖0-1范围，>0.5偏趋势。
- 确保无HTML输出（show_plots=False）。
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
    "symbols_to_test": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "interval": "4h",
    "start_date": "2024-01-01",
    "end_date": "2025-10-08",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,  # 滑点模拟 (0.05%)
    "run_monte_carlo": True,
    "show_plots": False,  # 确保不打开HTML界面
}

# --- 策略核心参数 (全局默认值) ---
STRATEGY_PARAMS = {
    # 资金管理与风控
    "kelly_trade_history": 20,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "dd_grace_period_bars": 240,
    "dd_initial_pct": 0.35,
    "dd_final_pct": 0.25,
    "dd_decay_bars": 4320,
    # 市场状态合成评分
    "regime_adx_period": 14,
    "regime_atr_period": 14,
    "regime_atr_slope_period": 5,
    "regime_rsi_period": 14,
    "regime_rsi_vol_period": 14,
    "regime_norm_period": 252,
    "regime_hurst_period": 100,  # Hurst指数滚动窗口
    "regime_score_weight_adx": 0.6,
    "regime_score_weight_atr": 0.3,
    "regime_score_weight_rsi": 0.05,  # 降低以容纳Hurst
    "regime_score_weight_hurst": 0.05,  # Hurst权重
    "regime_score_threshold": 0.45,
    # 子策略1: 趋势跟随 (TF)
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_adx_confirm_period": 14,
    "tf_adx_confirm_threshold": 18,
    "tf_chandelier_period": 22,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
    # 子策略2: 均值回归 (MR)
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 30,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    # 信号评分
    "mtf_period": 50,
    "score_mtf_bonus": 0.5,
    "ai_filter_rsi_period": 14,
    "ai_filter_fast_ma": 3,
    "ai_filter_slow_ma": 10,
    "ai_filter_confidence_threshold": 0.2,
    "score_ai_bonus": 0.5,
    "score_nonlinear_factor": 2.0,
    # 波动率调整
    "volatility_norm_period": 100,
}

# --- 资产个性化配置中心 ---
ASSET_SPECIFIC_OVERRIDES = {
    "BTCUSDT": {"regime_score_threshold": 0.45},
    "SOLUSDT": {"regime_score_threshold": 0.45},
    "ETHUSDT": {},
}


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
    logger.info(
        f"✅ 获取 {symbol} 数据成功：{len(df)} 条，从 {df.index[0]} 到 {df.index[-1]}"
    )
    return df


# 修复: Hurst指数计算函数（避免log(0)，使用std_diff作为tau，poly[0]=H）
def compute_hurst(ts, max_lag=100):
    """
    计算Hurst指数，用于检测均值回归（H<0.5）或趋势（H>0.5）。
    ts: log价格序列 (np.array)
    """
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau = []
    valid_lags = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)
        if std_diff > 0:  # 过滤零std，避免log(0)
            tau.append(std_diff)
            valid_lags.append(lag)
    if len(tau) < 2:
        return 0.5  # 默认随机游走
    try:
        poly = np.polyfit(np.log(valid_lags), np.log(tau), 1)
        return max(0.0, min(1.0, poly[0]))  # 夹紧0-1
    except:
        return 0.5


class UltimateStrategy(Strategy):
    # 为动态参数声明占位符
    tf_donchian_period_dynamic = None
    tf_chandelier_atr_multiplier_dynamic = None
    mr_bb_std_dynamic = None
    max_risk_pct_override = None

    # 为个性化参数声明占位符
    regime_score_threshold_override = None

    for key, value in STRATEGY_PARAMS.items():
        exec(f"{key} = {value}")
    vol_weight = 1.0

    def init(self):
        # 参数适配 (动态参数优先，其次是个性化参数，最后是全局默认)
        self.tf_donchian_period = int(
            getattr(self, "tf_donchian_period_dynamic", self.tf_donchian_period)
        )
        self.tf_chandelier_atr_multiplier = getattr(
            self,
            "tf_chandelier_atr_multiplier_dynamic",
            self.tf_chandelier_atr_multiplier,
        )
        self.mr_bb_std = getattr(self, "mr_bb_std_dynamic", self.mr_bb_std)
        self.regime_score_threshold = getattr(
            self, "regime_score_threshold_override", self.regime_score_threshold
        )
        self.max_risk_pct = getattr(self, "max_risk_pct_override", self.max_risk_pct)

        # --- 现有初始化代码 ---
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()
        self.market_regime = self.I(lambda: self.data.market_regime)
        self.mtf_signal = self.I(lambda: self.data.mtf_signal)
        self.ai_filter_signal = self.I(lambda: self.data.ai_filter_signal)
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
        bb_indicator = ta.volatility.BollingerBands(
            close, self.mr_bb_period, self.mr_bb_std
        )
        self.mr_bb_upper = self.I(lambda: bb_indicator.bollinger_hband())
        self.mr_bb_lower = self.I(lambda: bb_indicator.bollinger_lband())
        self.mr_bb_mid = self.I(lambda: bb_indicator.bollinger_mavg())
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(close, self.mr_rsi_period).rsi()
        )
        self.long_term_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.volatility_norm_period
            ).average_true_range()
        )
        self.equity_peak = self.equity
        self.global_stop_triggered = False

        # 修复: 包装Hurst日志访问，防止KeyError
        try:
            if len(self.data.hurst) > 0:
                logger.info(f"当前Hurst指数: {self.data.hurst[-1]:.2f}")
            else:
                logger.info("Hurst未计算（序列为空）")
        except Exception as e:
            logger.info(f"Hurst未计算: {e}")

    def reset_trade_state(self):
        self.active_sub_strategy = None
        self.chandelier_exit_level = 0.0
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")
        self.mr_stop_loss = 0.0

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
            stop_loss_level = self.equity_peak * (1 - current_dd_pct)
            if self.equity < stop_loss_level:
                if not self.global_stop_triggered:
                    logger.warning(
                        f"🚨 智能熔断触发！(当前阈值: {current_dd_pct:.2%}, 衰减进度: {decay_progress:.1%})"
                    )
                if self.position:
                    self.position.close()
                self.global_stop_triggered = True
                return
        if self.global_stop_triggered:
            return
        price = self.data.Close[-1]
        if not self.position:
            if len(self.market_regime) > 0 and self.market_regime[-1] == 1:
                self.run_trend_following_entry(price)
            else:
                self.run_mean_reversion_entry(price)
        else:
            self.manage_open_position(price)

    def manage_open_position(self, price):
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(price)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(price)

    def run_trend_following_entry(self, price):
        is_trend_strong = self.tf_adx[-1] > self.tf_adx_confirm_threshold
        is_breakout_up = self.data.High[-1] > self.tf_donchian_h[-1]
        is_breakout_down = self.data.Low[-1] < self.tf_donchian_l[-1]
        is_momentum_long = self.tf_ema_fast[-1] > self.tf_ema_slow[-1]
        is_momentum_short = self.tf_ema_fast[-1] < self.tf_ema_slow[-1]
        base_signal = 0
        if is_trend_strong and is_breakout_up and is_momentum_long:
            base_signal = 1
        elif is_trend_strong and is_breakout_down and is_momentum_short:
            base_signal = -1
        if base_signal == 0:
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
        nonlinear_score = score**self.score_nonlinear_factor
        self.open_tf_position(price, is_long=(base_signal == 1), score=nonlinear_score)

    def open_tf_position(self, price, is_long, score):
        initial_atr = self.tf_atr[-1]
        risk_per_share = initial_atr * self.tf_chandelier_atr_multiplier
        if risk_per_share <= 0:
            return
        target_risk_pct = self._calculate_dynamic_risk()
        if len(self.long_term_atr) > 1 and self.long_term_atr[-1] > 0:
            vol_ratio = initial_atr / self.long_term_atr[-1]
            volatility_dampener = 1 / max(vol_ratio, 0.5)
            target_risk_pct *= volatility_dampener
        final_risk_pct = target_risk_pct * score
        size = self._calculate_position_size(price, risk_per_share, final_risk_pct)
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.highest_high_in_trade = self.data.High[-1]
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - initial_atr * self.tf_chandelier_atr_multiplier
            )
            logger.debug(
                f"[TF] 📈 开多仓 (评分: {score:.2f}): {size:.4f} @ {price:.2f}, SL: {self.chandelier_exit_level:.2f}"
            )
        else:
            self.sell(size=size)
            self.lowest_low_in_trade = self.data.Low[-1]
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + initial_atr * self.tf_chandelier_atr_multiplier
            )
            logger.debug(
                f"[TF] 📉 开空仓 (评分: {score:.2f}): {size:.4f} @ {price:.2f}, SL: {self.chandelier_exit_level:.2f}"
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
                self.close_position("TF(钱德勒)")
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + current_atr * self.tf_chandelier_atr_multiplier
            )
            if price > self.chandelier_exit_level:
                self.close_position("TF(钱德勒)")

    def run_mean_reversion_entry(self, price):
        is_oversold = (
            crossover(self.data.Close, self.mr_bb_lower)
            and self.mr_rsi[-1] < self.mr_rsi_oversold
        )
        is_overbought = (
            crossover(self.mr_bb_upper, self.data.Close)
            and self.mr_rsi[-1] > self.mr_rsi_overbought
        )
        if is_oversold:
            self.open_mr_position(price, is_long=True)
        elif is_overbought:
            self.open_mr_position(price, is_long=False)

    def open_mr_position(self, price, is_long):
        initial_atr = self.tf_atr[-1]
        risk_per_share = initial_atr * self.mr_stop_loss_atr_multiplier
        if risk_per_share <= 0:
            return
        target_risk_pct = self._calculate_dynamic_risk() * self.mr_risk_multiplier
        size = self._calculate_position_size(price, risk_per_share, target_risk_pct)
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        if is_long:
            self.buy(size=size)
            self.mr_stop_loss = price - risk_per_share
            logger.debug(
                f"[MR] 📈 开多仓: {size:.4f} @ {price:.2f}, SL: {self.mr_stop_loss:.2f}"
            )
        else:
            self.sell(size=size)
            self.mr_stop_loss = price + risk_per_share
            logger.debug(
                f"[MR] 📉 开空仓: {size:.4f} @ {price:.2f}, SL: {self.mr_stop_loss:.2f}"
            )

    def manage_mean_reversion_exit(self, price):
        should_close = False
        reason = ""
        if self.position.is_long and price >= self.mr_bb_mid[-1]:
            should_close = True
            reason = "回归中轨"
        elif self.position.is_short and price <= self.mr_bb_mid[-1]:
            should_close = True
            reason = "回归中轨"
        if self.position.is_long and price <= self.mr_stop_loss:
            should_close = True
            reason = "ATR止损"
        elif self.position.is_short and price >= self.mr_stop_loss:
            should_close = True
            reason = "ATR止损"
        if should_close:
            self.close_position(f"MR({reason})")

    def close_position(self, reason: str):
        price = self.data.Close[-1]
        direction_str = "多头" if self.position.is_long else "空头"
        size_before_close = self.position.size
        equity_before_close = self.equity
        self.position.close()
        pnl_pct = (self.equity / equity_before_close) - 1
        self.recent_trade_returns.append(pnl_pct)
        self.reset_trade_state()
        logger.debug(
            f"✅ 仓位平仓({reason}): {direction_str} {abs(size_before_close):.4f} @ {price:.2f}"
        )

    def _calculate_position_size(self, price, risk_per_share, target_risk_pct):
        risk_capital = target_risk_pct * self.equity
        position_value = risk_capital / (risk_per_share / price)
        return position_value / self.equity

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight
        kelly_fraction = win_rate - (1 - win_rate) / reward_ratio
        dynamic_risk = max(0.005, kelly_fraction * 0.5)
        return min(dynamic_risk * self.vol_weight, self.max_risk_pct)


def run_monte_carlo(trades_df, initial_cash, symbol: str, n_simulations=1000):
    if trades_df.empty:
        logger.warning("没有交易数据，无法进行蒙特卡洛模拟。")
        return
    returns = trades_df["ReturnPct"]
    final_equities = []
    for _ in range(n_simulations):
        sim_returns = np.random.choice(returns, size=len(returns), replace=True)
        final_equities.append(initial_cash * (1 + sim_returns).prod())
    mean_equity = np.mean(final_equities)
    median_equity = np.median(final_equities)
    var_5_pct = np.percentile(final_equities, 5)
    best_95_pct = np.percentile(final_equities, 95)
    logger.info("--- 蒙特卡洛模拟结果 ---")
    logger.info(f"模拟次数: {n_simulations}")
    logger.info(f"平均最终权益: ${mean_equity:,.2f}")
    logger.info(f"最终权益中位数: ${median_equity:,.2f}")
    logger.info(f"5% 最差情况权益 (VaR): ${var_5_pct:,.2f}")
    logger.info(f"95% 最好情况权益: ${best_95_pct:,.2f}")
    results_data = {
        "Symbol": [symbol],
        "Simulations": [n_simulations],
        "Initial Cash": [f"${initial_cash:,.2f}"],
        "Mean Final Equity": [f"${mean_equity:,.2f}"],
        "Median Final Equity": [f"${median_equity:,.2f}"],
        "5% VaR Equity": [f"${var_5_pct:,.2f}"],
        "95% Best Case Equity": [f"${best_95_pct:,.2f}"],
    }
    results_df = pd.DataFrame(results_data)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monte_carlo_results_{symbol}_{timestamp_str}.csv"
    try:
        results_df.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.info(f"💾 蒙特卡洛模拟结果已保存至: {filename}")
    except Exception as e:
        logger.error(f"保存蒙特卡洛CSV文件时出错: {e}")


def generate_dynamic_params(volatility: float, baseline_vol: float) -> dict:
    """根据年化波动率和动态基线，为单个品种生成一套动态参数。"""
    volatility_factor = volatility / baseline_vol
    volatility_factor = np.clip(volatility_factor, 0.8, 1.2)  # 收紧范围，防止异变
    p = STRATEGY_PARAMS

    dynamic_chandelier = p["tf_chandelier_atr_multiplier"] * volatility_factor
    dynamic_donchian = p["tf_donchian_period"] * volatility_factor
    dynamic_bb_std = p["mr_bb_std"] * volatility_factor
    dynamic_max_risk_pct = p["max_risk_pct"] / volatility_factor
    dynamic_max_risk_pct = np.clip(dynamic_max_risk_pct, 0.02, 0.05)

    params = {
        "tf_chandelier_atr_multiplier_dynamic": np.round(dynamic_chandelier, 2),
        "tf_donchian_period_dynamic": int(np.round(dynamic_donchian)),
        "mr_bb_std_dynamic": np.round(dynamic_bb_std, 2),
        "max_risk_pct_override": np.round(dynamic_max_risk_pct, 4),
    }
    return params


if __name__ == "__main__":
    all_stats = {}
    total_final_equity = 0
    logger.info(f"🚀 (V33.1: “系统稳定器增强”版，修复) 开始运行...")
    logger.info(f"详细交易日志将保存在文件: {log_filename}")

    all_data = {}
    volatilities = {}
    for symbol in CONFIG["symbols_to_test"]:
        data = fetch_binance_klines(
            symbol, CONFIG["interval"], CONFIG["start_date"], CONFIG["end_date"]
        )
        if not data.empty:
            all_data[symbol] = data
            daily_returns = data["Close"].resample("D").last().pct_change()
            volatilities[symbol] = daily_returns.std() * np.sqrt(365)
    if not volatilities:
        logger.error("所有品种数据获取失败，无法计算波动率权重。")
        exit()

    inverse_vol = {sym: 1 / vol for sym, vol in volatilities.items() if vol > 0}
    total_inverse_vol = sum(inverse_vol.values())
    vol_weights = {
        sym: (inv_vol / total_inverse_vol) * len(volatilities)
        for sym, inv_vol in inverse_vol.items()
    }
    logger.info("--- 波动率风险平价权重 ---")
    for symbol, weight in vol_weights.items():
        logger.info(
            f"  - {symbol}: {weight:.2f} (年化波动率: {volatilities.get(symbol, 0):.2%})"
        )

    for symbol in CONFIG["symbols_to_test"]:
        print("\n" + "=" * 80)
        logger.info(f"正在回测品种: {symbol}")
        print("=" * 80)

        data_4h = all_data.get(symbol)
        if data_4h is None:
            logger.error(f"❌ 未找到 {symbol} 的数据，跳过。")
            continue

        symbol_volatility = volatilities.get(symbol, 0.7)

        # 计算稳健的动态基线
        daily_vol_series = all_data[symbol]["Close"].resample(
            "D"
        ).last().pct_change().rolling(252).std() * np.sqrt(365)
        dynamic_baseline_vol = daily_vol_series.mean()
        if pd.isna(dynamic_baseline_vol) or dynamic_baseline_vol <= 0:
            dynamic_baseline_vol = 0.7
        logger.info(f"使用动态波动率基线: {dynamic_baseline_vol:.2%}")

        dynamic_params_for_symbol = generate_dynamic_params(
            symbol_volatility, dynamic_baseline_vol
        )
        final_params = dynamic_params_for_symbol.copy()

        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(symbol, {})
        if asset_overrides:
            logger.info(f"--- 为 {symbol} 应用个性化参数 ---")
            for key, value in asset_overrides.items():
                final_params[f"{key}_override"] = value
                logger.info(f"  - {key}: {value} (覆盖默认)")

        logger.info(f"--- 为 {symbol} 应用动态参数 ---")
        for key, value in dynamic_params_for_symbol.items():
            logger.info(
                f"  - {key.replace('_dynamic', '').replace('_override', '')}: {value}"
            )

        final_regime_threshold = final_params.get(
            "regime_score_threshold_override", STRATEGY_PARAMS["regime_score_threshold"]
        )

        logger.info("开始进行数据预处理 (多因子信号)...")
        data_1d = fetch_binance_klines(
            symbol, "1d", CONFIG["start_date"], CONFIG["end_date"]
        )
        if not data_1d.empty:
            sma_1d = ta.trend.SMAIndicator(
                data_1d["Close"], window=STRATEGY_PARAMS["mtf_period"]
            ).sma_indicator()
            data_1d["mtf_signal"] = np.where(data_1d["Close"] > sma_1d, 1, -1)
            data_4h["mtf_signal"] = (
                data_1d["mtf_signal"].reindex(data_4h.index, method="ffill").fillna(0)
            )
        else:
            logger.warning("未能获取日线数据，MTF过滤器将停用。")
            data_4h["mtf_signal"] = 0

        logger.info("  - 计算市场状态合成评分...")
        adx = ta.trend.ADXIndicator(
            data_4h["High"],
            data_4h["Low"],
            data_4h["Close"],
            STRATEGY_PARAMS["regime_adx_period"],
        ).adx()
        adx_norm = (adx - adx.rolling(STRATEGY_PARAMS["regime_norm_period"]).min()) / (
            adx.rolling(STRATEGY_PARAMS["regime_norm_period"]).max()
            - adx.rolling(STRATEGY_PARAMS["regime_norm_period"]).min()
        )
        adx_norm = adx_norm.fillna(0.5)
        atr = ta.volatility.AverageTrueRange(
            data_4h["High"],
            data_4h["Low"],
            data_4h["Close"],
            STRATEGY_PARAMS["regime_atr_period"],
        ).average_true_range()
        atr_slope = (
            atr - atr.shift(STRATEGY_PARAMS["regime_atr_slope_period"])
        ) / atr.shift(STRATEGY_PARAMS["regime_atr_slope_period"])
        atr_slope_norm = (
            atr_slope - atr_slope.rolling(STRATEGY_PARAMS["regime_norm_period"]).min()
        ) / (
            atr_slope.rolling(STRATEGY_PARAMS["regime_norm_period"]).max()
            - atr_slope.rolling(STRATEGY_PARAMS["regime_norm_period"]).min()
        )
        atr_slope_norm = atr_slope_norm.fillna(0.5)
        rsi = ta.momentum.RSIIndicator(
            data_4h["Close"], window=STRATEGY_PARAMS["regime_rsi_period"]
        ).rsi()
        rsi_vol = rsi.rolling(STRATEGY_PARAMS["regime_rsi_vol_period"]).std()
        rsi_vol_norm = (
            rsi_vol - rsi_vol.rolling(STRATEGY_PARAMS["regime_norm_period"]).min()
        ) / (
            rsi_vol.rolling(STRATEGY_PARAMS["regime_norm_period"]).max()
            - rsi_vol.rolling(STRATEGY_PARAMS["regime_norm_period"]).min()
        )
        rsi_vol_norm = 1 - rsi_vol_norm.fillna(0.5)

        # 修复: 计算滚动Hurst指数（使用log价格，避免警告）
        hurst_series = pd.Series(index=data_4h.index, dtype=float)
        period = STRATEGY_PARAMS["regime_hurst_period"]
        for i in range(period, len(data_4h)):
            price_slice = data_4h["Close"].iloc[i - period : i]
            if (price_slice <= 0).any():
                ts = np.log(price_slice + 1e-8)  # 避免log(0)
            else:
                ts = np.log(price_slice)
            hurst_series.iloc[i] = compute_hurst(ts.values)
        hurst_series = hurst_series.fillna(0.5)
        data_4h["hurst"] = hurst_series
        # 调整标准化：H>0.5 favoring趋势 (TF=1)，范围0-1
        hurst_norm = np.clip((hurst_series - 0.3) / 0.7, 0, 1)

        # 更新regime_score: 添加Hurst项
        regime_score = (
            adx_norm * STRATEGY_PARAMS["regime_score_weight_adx"]
            + atr_slope_norm * STRATEGY_PARAMS["regime_score_weight_atr"]
            + rsi_vol_norm * STRATEGY_PARAMS["regime_score_weight_rsi"]
            + hurst_norm * STRATEGY_PARAMS["regime_score_weight_hurst"]
        )
        data_4h["market_regime"] = np.where(
            regime_score > final_regime_threshold, 1, -1
        )

        logger.info("  - 模拟AI信号过滤器...")
        rsi_filter = ta.momentum.RSIIndicator(
            data_4h["Close"], window=STRATEGY_PARAMS["ai_filter_rsi_period"]
        ).rsi()
        rsi_fast = rsi_filter.rolling(
            window=STRATEGY_PARAMS["ai_filter_fast_ma"]
        ).mean()
        rsi_slow = rsi_filter.rolling(
            window=STRATEGY_PARAMS["ai_filter_slow_ma"]
        ).mean()
        data_4h["ai_filter_signal"] = (rsi_fast - rsi_slow) / 50
        data_4h["ai_filter_signal"] = data_4h["ai_filter_signal"].clip(-1, 1).fillna(0)
        logger.info("数据预处理完成。")

        bt = Backtest(
            data_4h,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            spread=CONFIG["spread"],  # 滑点
            finalize_trades=True,
        )
        stats = bt.run(vol_weight=vol_weights.get(symbol, 1.0), **final_params)

        all_stats[symbol] = stats
        total_final_equity += stats["Equity Final [$]"]
        print("\n" + "-" * 40)
        print(f"          {symbol} 回测结果摘要")
        print("-" * 40)
        print(stats)
        sqn = stats.get("SQN")
        kelly = stats.get("Kelly Criterion")
        calmar = stats.get("Calmar Ratio")
        print("\n--- 🔍 策略健康度监控 ---")
        if sqn is not None and not np.isnan(sqn):
            print(f"SQN (系统质量数): {sqn:.2f}")
        if kelly is not None and not np.isnan(kelly):
            print(f"凯利准则: {kelly:.4f}")
        if calmar is not None and not np.isnan(calmar):
            print(f"卡玛比率 (Cal-Ratio): {calmar:.3f}")
        if CONFIG["show_plots"]:
            try:
                bt.plot(
                    resample="D",
                    supertitle=f"{symbol} Equity Curve",
                    open_browser=False,
                )
            except TypeError:
                logger.warning(
                    "您的 backtesting.py 库版本较旧，不支持 supertitle/open_browser 参数。"
                )
                bt.plot()
        if CONFIG["run_monte_carlo"] and not stats["_trades"].empty:
            run_monte_carlo(stats["_trades"], CONFIG["initial_cash"], symbol)

    num_assets = len(all_stats)
    if num_assets > 0:
        total_initial_cash = CONFIG["initial_cash"] * num_assets
        portfolio_return = (
            (total_final_equity - total_initial_cash) / total_initial_cash * 100
        )
        print("\n" + "#" * 80)
        print("                 组合策略表现总览")
        print("#" * 80)
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A'):.3f}"
            )
        print("\n--- 投资组合整体表现 ---")
        print(f"总初始资金: ${total_initial_cash:,.2f}")
        print(f"总最终权益: ${total_final_equity:,.2f}")
        print(f"组合总回报率: {portfolio_return:.2f}%")
