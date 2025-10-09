# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V19.2 - 清爽终端版)

本次升级:
- 【核心】实现日志分级，将所有逐笔交易详情 (开仓, 平仓, 止损等) 降级为 DEBUG 级别。
- 【优化】调整日志系统配置，使控制台只显示 INFO 级别以上的关键摘要信息，
          而日志文件则完整记录所有 DEBUG 级别的详细交易过程。
- 【效果】运行回测时终端输出将非常干净，所有交易细节均保存在日志文件中，便于复盘。
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

# --- V19.2 核心修改: 分级日志系统 ---
# 1. 创建 logger，并设置最低级别为 DEBUG 以捕获所有信息
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 2. 创建文件处理器 (FileHandler)，记录 DEBUG 及以上级别
log_filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)  # 文件记录所有细节

# 3. 创建控制台处理器 (StreamHandler)，只显示 INFO 及以上级别
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # 控制台只显示关键信息

# 4. 定义日志格式
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# 5. 将处理器添加到 logger (防止重复添加)
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
    "start_date": "2020-01-01",
    "end_date": "2025-10-08",
    "initial_cash": 500_000,
    "commission": 0.0005,
    "run_monte_carlo": True,
}

# --- 策略核心参数 ---
STRATEGY_PARAMS = {
    "kelly_trade_history": 20,
    "default_risk_pct": 0.02,
    "max_risk_pct": 0.04,
    "max_equity_drawdown_pct": 0.20,
    "donchian_period": 20,
    "ema_fast_period": 20,
    "ema_slow_period": 50,
    "adx_confirm_period": 14,
    "adx_confirm_threshold": 18,
    "chandelier_period": 22,
    "chandelier_atr_multiplier": 3.0,
    "atr_period": 14,
    "sma_ultra_long_period": 400,
    "vol_regime_period_long": 100,
    "vol_high_threshold": 1.5,
    "vol_low_threshold": 0.7,
    "atr_multiplier_high_vol": 3.5,
    "atr_multiplier_low_vol": 1.8,
    "max_pyramid_count": 3,
    "pyramid_atr_distance": 1.5,
}


def fetch_binance_klines(
    symbol: str, interval: str, start_str: str, end_str: str = None, limit: int = 1000
) -> pd.DataFrame:
    # ... 此函数与之前版本完全相同 ...
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


class UltimateTrendStrategy(Strategy):
    for key, value in STRATEGY_PARAMS.items():
        exec(f"{key} = {value}")

    def init(self):
        # ... 此函数与V19.0版本完全相同 ...
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()
        self.atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.atr_period
            ).average_true_range()
        )
        self.donchian_h = self.I(
            lambda: high.rolling(self.donchian_period).max().shift(1)
        )
        self.donchian_l = self.I(
            lambda: low.rolling(self.donchian_period).min().shift(1)
        )
        self.ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(close, self.ema_fast_period).ema_indicator()
        )
        self.ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(close, self.ema_slow_period).ema_indicator()
        )
        self.adx = self.I(
            lambda: ta.trend.ADXIndicator(
                high, low, close, self.adx_confirm_period
            ).adx()
        )
        self.sma_ultra_long = self.I(
            lambda: ta.trend.SMAIndicator(
                close, self.sma_ultra_long_period
            ).sma_indicator()
        )
        self.equity_peak = self.equity
        self.global_stop_triggered = False

    def reset_trade_state(self):
        # ... 此函数与V19.0版本完全相同 ...
        self.chandelier_exit_level = 0.0
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")

    def next(self):
        # ... 此函数与V19.0版本完全相同 ...
        price = self.data.Close[-1]
        self.equity_peak = max(self.equity_peak, self.equity)
        stop_loss_level = self.equity_peak * (1 - self.max_equity_drawdown_pct)
        if self.equity < stop_loss_level:
            if not self.global_stop_triggered:
                logger.warning(
                    f"🚨 全局最大回撤止损触发！(阈值: {self.max_equity_drawdown_pct:.2%})"
                )
                if self.position:
                    self.position.close()
                self.global_stop_triggered = True
            return
        if self.global_stop_triggered:
            return
        if not self.position:
            self.run_confirmed_breakout_logic(price)
        else:
            self.manage_open_position(price)

    def manage_open_position(self, price):
        # ... 此函数与V19.0版本完全相同 ...
        current_atr = self.atr[-1]
        if self.position.is_long:
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - current_atr * self.chandelier_atr_multiplier
            )
            if price < self.chandelier_exit_level:
                self.close_position("钱德勒止损")
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade + current_atr * self.chandelier_atr_multiplier
            )
            if price > self.chandelier_exit_level:
                self.close_position("钱德勒止损")

    def run_confirmed_breakout_logic(self, price):
        # ... 此函数与V19.0版本完全相同 ...
        is_trend_strong = self.adx[-1] > self.adx_confirm_threshold
        is_breakout_up = self.data.High[-1] > self.donchian_h[-1]
        is_breakout_down = self.data.Low[-1] < self.donchian_l[-1]
        is_momentum_long = self.ema_fast[-1] > self.ema_slow[-1]
        is_momentum_short = self.ema_fast[-1] < self.ema_slow[-1]
        is_macro_bull = (
            price > self.sma_ultra_long[-1] if len(self.sma_ultra_long) > 0 else True
        )
        if is_trend_strong and is_breakout_up and is_momentum_long and is_macro_bull:
            self.open_position(price, is_long=True)
        elif (
            is_trend_strong
            and is_breakout_down
            and is_momentum_short
            and not is_macro_bull
        ):
            self.open_position(price, is_long=False)

    def open_position(self, price, is_long):
        # --- V19.2 修改: 将日志级别改为 DEBUG ---
        initial_atr = self.atr[-1]
        risk_per_share = initial_atr * self.chandelier_atr_multiplier
        if risk_per_share <= 0:
            return
        target_risk_pct = self._calculate_dynamic_risk()
        size = self._calculate_position_size(price, risk_per_share, target_risk_pct)
        if not (0 < size < 0.98):
            return
        self.reset_trade_state()
        if is_long:
            self.buy(size=size)
            self.highest_high_in_trade = self.data.High[-1]
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - initial_atr * self.chandelier_atr_multiplier
            )
            logger.debug(
                f"📈 开多仓: {size:.4f} @ {price:.2f}, 初始钱德勒止损: {self.chandelier_exit_level:.2f}"
            )
        else:
            self.sell(size=size)
            self.lowest_low_in_trade = self.data.Low[-1]
            self.chandelier_exit_level = (
                self.lowest_low_in_trade + initial_atr * self.chandelier_atr_multiplier
            )
            logger.debug(
                f"📉 开空仓: {size:.4f} @ {price:.2f}, 初始钱德勒止损: {self.chandelier_exit_level:.2f}"
            )

    def close_position(self, reason: str):
        # --- V19.2 修改: 将日志级别改为 DEBUG ---
        price = self.data.Close[-1]
        direction_str = "多头" if self.position.is_long else "空头"
        size_before_close = self.position.size
        equity_before_close = self.equity
        self.position.close()
        pnl_pct = (self.equity / equity_before_close) - 1
        self.recent_trade_returns.append(pnl_pct)
        self.reset_trade_state()
        logger.debug(
            f"✅ 仓位全部平仓({reason}): {direction_str} {abs(size_before_close):.4f} @ {price:.2f}"
        )

    def _calculate_position_size(self, price, risk_per_share, target_risk_pct):
        # ... 此函数与V19.0版本完全相同 ...
        risk_capital = target_risk_pct * self.equity
        position_value = risk_capital / (risk_per_share / price)
        return position_value / self.equity

    def _calculate_dynamic_risk(self):
        # ... 此函数与V19.0版本完全相同 ...
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return self.default_risk_pct
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        if reward_ratio == 0:
            return self.default_risk_pct
        kelly_fraction = win_rate - (1 - win_rate) / reward_ratio
        dynamic_risk = max(0.005, kelly_fraction * 0.5)
        return min(dynamic_risk, self.max_risk_pct)


def run_monte_carlo(trades_df, initial_cash, symbol: str, n_simulations=1000):
    # ... 此函数与V17.4版本完全相同 ...
    if trades_df.empty:
        logger.warning("没有交易数据，无法进行蒙特卡洛模拟。")
        return
    returns = trades_df["ReturnPct"]
    n_trades, final_equities = len(returns), []
    for _ in range(n_simulations):
        sim_returns = np.random.choice(returns, size=n_trades, replace=True)
        current_equity = initial_cash * (1 + sim_returns).prod()
        final_equities.append(current_equity)
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


if __name__ == "__main__":
    all_stats = {}
    total_final_equity = 0
    logger.info(f"🚀 (V19.1: 专业日志版) 开始运行...")
    logger.info(f"详细交易日志将保存在文件: {log_filename}")

    for symbol in CONFIG["symbols_to_test"]:
        print("\n" + "=" * 80)
        logger.info(f"正在回测品种: {symbol}")
        print("=" * 80)
        data = fetch_binance_klines(
            symbol, CONFIG["interval"], CONFIG["start_date"], CONFIG["end_date"]
        )
        if data.empty:
            logger.error(f"❌ 无法获取 {symbol} 数据，跳过该品种。")
            continue
        bt = Backtest(
            data,
            UltimateTrendStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
        )
        stats = bt.run()
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
        logger.info(f"绘图功能已禁用。回测HTML文件已在当前目录生成。")
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
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats['Sharpe Ratio']:.3f}"
            )
        print("\n--- 投资组合整体表现 ---")
        print(f"总初始资金: ${total_initial_cash:,.2f}")
        print(f"总最终权益: ${total_final_equity:,.2f}")
        print(f"组合总回报率: {portfolio_return:.2f}%")
