# -*- coding: utf-8 -*-
"""
🚀 终极优化版加密货币趋势交易系统 (V40.1 - “最终修正”版)

这是我们策略开发迭代之旅的毕业作品。它象征着一次“螺旋式上升”的顿悟：我们保留了
后期版本中所有被验证为成功的专业模块，但将它们嫁接到一个更纯粹、更强大、更符合
市场本质的“纯趋势跟踪”内核之上。

 V40.1 更新日志:
- 【修正】: 修复了因重构代码时遗留的“幽灵”变量而导致的TypeError。彻底移除了与
            动态ADX阈值相关的占位符和适配逻辑，确保策略内核的纯粹性和稳定性。

 V40.0 最终版特性:
- 【心脏】: 回归纯粹的趋势跟踪内核 (V19.1)
- 【利刃】: 动态参数引擎 (V33.0)
- 【护盾】: 动态风险平价 (V33.0)
- 【堡垒】: 多层动态风控 (V31.0)
- 【智慧】: 资产个性化配置 & 组合管理 (V34.0)
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
    "symbols_to_test": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],  # 最终决策: 专注于高期望资产
    "interval": "1h",
    "start_date": "2020-01-01",
    "end_date": "2025-10-08",
    "initial_cash": 500_000,
    "commission": 0.0005,
    "run_monte_carlo": True,
    "show_plots": False,
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
    # 纯趋势跟随策略 (TF)
    "tf_donchian_period": 20,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 50,
    "tf_adx_confirm_period": 14,
    "tf_adx_confirm_threshold": 25,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
}

# --- 资产个性化配置中心 ---
ASSET_SPECIFIC_OVERRIDES = {
    # "BTCUSDT": { "tf_adx_confirm_threshold": 22 },
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


class UltimateStrategy(Strategy):
    # 为动态和个性化参数声明占位符
    tf_donchian_period_dynamic = None
    tf_chandelier_atr_multiplier_dynamic = None
    max_risk_pct_override = None
    # <-- 修正: 彻底移除 tf_adx_confirm_threshold_override 占位符

    for key, value in STRATEGY_PARAMS.items():
        exec(f"{key} = {value}")
    vol_weight = 1.0

    def init(self):
        # 参数适配
        self.tf_donchian_period = int(
            getattr(self, "tf_donchian_period_dynamic", self.tf_donchian_period)
        )
        self.tf_chandelier_atr_multiplier = getattr(
            self,
            "tf_chandelier_atr_multiplier_dynamic",
            self.tf_chandelier_atr_multiplier,
        )
        self.max_risk_pct = getattr(self, "max_risk_pct_override", self.max_risk_pct)
        # <-- 修正: 彻底移除对 tf_adx_confirm_threshold_override 的适配

        # --- 初始化指标 ---
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()

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

        self.equity_peak = self.equity
        self.global_stop_triggered = False

    def reset_trade_state(self):
        self.chandelier_exit_level = 0.0
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")

    def next(self):
        # 智能熔断
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

        # 核心交易逻辑
        if not self.position:
            self.run_trend_entry(price)
        else:
            self.manage_trend_exit(price)

    def run_trend_entry(self, price):
        is_trend_strong = self.tf_adx[-1] > self.tf_adx_confirm_threshold
        is_breakout_up = self.data.High[-1] > self.tf_donchian_h[-1]
        is_breakout_down = self.data.Low[-1] < self.tf_donchian_l[-1]
        is_momentum_long = self.tf_ema_fast[-1] > self.tf_ema_slow[-1]
        is_momentum_short = self.tf_ema_fast[-1] < self.tf_ema_slow[-1]

        if is_trend_strong and is_breakout_up and is_momentum_long:
            self.open_position(price, is_long=True)
        elif is_trend_strong and is_breakout_down and is_momentum_short:
            self.open_position(price, is_long=False)

    def open_position(self, price, is_long):
        initial_atr = self.tf_atr[-1]
        risk_per_share = initial_atr * self.tf_chandelier_atr_multiplier
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
                - initial_atr * self.tf_chandelier_atr_multiplier
            )
            logger.debug(
                f"📈 开多仓: {size:.4f} @ {price:.2f}, SL: {self.chandelier_exit_level:.2f}"
            )
        else:
            self.sell(size=size)
            self.lowest_low_in_trade = self.data.Low[-1]
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + initial_atr * self.tf_chandelier_atr_multiplier
            )
            logger.debug(
                f"📉 开空仓: {size:.4f} @ {price:.2f}, SL: {self.chandelier_exit_level:.2f}"
            )

    def manage_trend_exit(self, price):
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
                self.close_position("钱德勒止盈")
        elif self.position.is_short:
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + current_atr * self.tf_chandelier_atr_multiplier
            )
            if price > self.chandelier_exit_level:
                self.close_position("钱德勒止盈")

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
    # ... 此函数与之前版本完全相同 ...
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
    # ... 此函数与V33.0版本完全相同 ...
    volatility_factor = volatility / baseline_vol
    volatility_factor = np.clip(volatility_factor, 0.7, 1.5)
    p = STRATEGY_PARAMS
    dynamic_chandelier = p["tf_chandelier_atr_multiplier"] * volatility_factor
    dynamic_donchian = p["tf_donchian_period"] * volatility_factor
    dynamic_max_risk_pct = p["max_risk_pct"] / volatility_factor
    dynamic_max_risk_pct = np.clip(dynamic_max_risk_pct, 0.02, 0.05)
    params = {
        "tf_chandelier_atr_multiplier_dynamic": np.round(dynamic_chandelier, 2),
        "tf_donchian_period_dynamic": int(np.round(dynamic_donchian)),
        "max_risk_pct_override": np.round(dynamic_max_risk_pct, 4),
    }
    return params


if __name__ == "__main__":
    all_stats = {}
    total_final_equity = 0
    logger.info(f"🚀 (V40.1: “最终修正”版) 开始运行...")
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

        bt = Backtest(
            data_4h,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
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
