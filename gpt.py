#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版：EMA + RSI 双核共振 + 趋势过滤（适用于加密货币1分钟数据）
- [修复] 入场条件过于严苛导致信号稀少
- [新增] ADX 趋势强度过滤，避免震荡市频繁交易
- [新增] 动态 RSI 阈值（基于ATR波动率调整）
- [增强] 更合理的止损/止盈逻辑（基于ATR）
- [调试] 添加交易日志输出
- [优化] 使用 pd.Series 直接计算，提升性能和可读性
"""
import os
import time
import logging
import requests
import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy
from backtesting.lib import crossover, plot_heatmaps
from backtesting.lib import resample_apply

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# fetch_binance_klines（稳健版）
# ---------------------------
def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: str = None,
    limit: int = 1000,
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
    all_data = []
    retries = 5

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
                    start_ts = end_ts
                    break
                all_data.extend(data)
                start_ts = data[-1][0] + 1
                break
            except Exception as e:
                wait = 2**attempt
                logger.warning(f"请求失败: {e}，{wait}s后重试...")
                time.sleep(wait)
        else:
            logger.error("多次重试后仍无法获取数据，终止。")
            break

    if not all_data:
        logger.error("未获取到任何数据。")
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


# ---------------------------
# 技术指标（直接返回 pd.Series，更高效）
# ---------------------------
def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int = 14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    tr0 = high - low
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = pd.concat(
        [high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1
    ).max(axis=1)

    atr_val = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / atr_val
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / atr_val
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
    return dx.rolling(period).mean()


# ---------------------------
# 改进版 EMA + RSI 策略（V2）
# ---------------------------
class EMA_RSI_Strategy_V2(Strategy):
    # --- 参数配置 ---
    ema_fast = 10
    ema_slow = 25
    rsi_period = 14
    rsi_oversold_base = 35
    rsi_overbought_base = 65
    atr_period = 14
    adx_period = 14
    adx_threshold = 20  # 只有ADX > 20才认为有趋势

    size_fraction = 0.95
    stop_loss_atr_multiplier = 1.5
    take_profit_atr_multiplier = 2.0

    min_bars_between_entries = 5

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.ema_fast_v = self.I(ema, close, self.ema_fast)
        self.ema_slow_v = self.I(ema, close, self.ema_slow)
        self.rsi_v = self.I(rsi, close, self.rsi_period)
        self.atr_v = self.I(atr, high, low, close, self.atr_period)
        self.adx_v = self.I(adx, high, low, close, self.adx_period)

        self._last_entry_bar = -9999
        self._log_trades = True  # 开启交易日志

    def _bars_since_last_entry(self):
        return len(self.data) - 1 - self._last_entry_bar

    def next(self):
        price = self.data.Close[-1]
        ef = self.ema_fast_v[-1]
        es = self.ema_slow_v[-1]
        r = self.rsi_v[-1]
        atr = self.atr_v[-1]
        adx = self.adx_v[-1]

        # --- 动态RSI阈值（波动大时放宽）---
        rsi_oversold = self.rsi_oversold_base - (atr / price * 100)
        rsi_overbought = self.rsi_overbought_base + (atr / price * 100)

        # --- 趋势判断 ---
        uptrend = ef > es
        downtrend = ef < es

        # --- 仅在趋势明确时交易 ---
        strong_trend = adx > self.adx_threshold

        enter_long = (
            crossover(self.ema_fast_v, self.ema_slow_v)
            and r <= rsi_oversold
            and uptrend
            and strong_trend
        )
        exit_long = (not uptrend) or (r >= self.rsi_overbought_base)

        enter_short = (
            crossover(self.ema_slow_v, self.ema_fast_v)
            and r >= rsi_overbought
            and downtrend
            and strong_trend
        )
        exit_short = (not downtrend) or (r <= self.rsi_oversold_base)

        # --- 交易间隔控制 ---
        if self._bars_since_last_entry() < self.min_bars_between_entries:
            enter_long = False
            enter_short = False

        # --- 平仓逻辑 ---
        if self.position:
            if self.position.is_long and exit_long:
                self.position.close()
            elif self.position.is_short and exit_short:
                self.position.close()

        # --- 开仓逻辑 ---
        if not self.position:
            if enter_long:
                sl = price - self.stop_loss_atr_multiplier * atr
                tp = price + self.take_profit_atr_multiplier * atr
                self.buy(size=self.size_fraction, sl=sl, tp=tp)
                self._last_entry_bar = len(self.data) - 1
                if self._log_trades:
                    logger.info(
                        f"{self.data.index[-1]} | LONG  @ {price:.2f} | SL={sl:.2f} | TP={tp:.2f}"
                    )

            elif enter_short:
                sl = price + self.stop_loss_atr_multiplier * atr
                tp = price - self.take_profit_atr_multiplier * atr
                self.sell(size=self.size_fraction, sl=sl, tp=tp)
                self._last_entry_bar = len(self.data) - 1
                if self._log_trades:
                    logger.info(
                        f"{self.data.index[-1]} | SHORT @ {price:.2f} | SL={sl:.2f} | TP={tp:.2f}"
                    )


# ---------------------------
# 主函数示例
# ---------------------------
if __name__ == "__main__":
    SYMBOL = "ETHUSDT"
    INTERVAL = "1m"
    START = "2024-01-01"
    END = "2024-02-28"

    # 参数配置
    strat_params = {
        "ema_fast": 10,
        "ema_slow": 25,
        "rsi_period": 14,
        "rsi_oversold_base": 40,
        "rsi_overbought_base": 60,
        "adx_threshold": 20,
        "stop_loss_atr_multiplier": 1.5,
        "take_profit_atr_multiplier": 2.0,
        "min_bars_between_entries": 5,
        "size_fraction": 0.95,
    }

    # 获取数据
    df = fetch_binance_klines(SYMBOL, INTERVAL, START, END)
    if df.empty:
        raise RuntimeError("无法获取数据")

    # 运行回测
    bt = Backtest(
        df,
        EMA_RSI_Strategy_V2,
        cash=10_000,
        commission=0.0004,
        margin=0.05,  # 20x leverage
        trade_on_close=True,
        exclusive_orders=True,
    )

    # 执行回测
    stats = bt.run(**strat_params)
    print(stats)

    # 绘图
    bt.plot(filename="EMA_RSI_Strategy_V2.html", open_browser=False)

    # 可选：参数热力图优化
    # params_grid = {
    #     'ema_fast': range(8, 16, 2),
    #     'ema_slow': range(20, 31, 5),
    #     'rsi_period': [10, 14],
    #     'adx_threshold': [15, 20, 25]
    # }
    # heatmap_stats, _ = bt.optimize(**params_grid, maximize='Equity Final [$]', return_heatmap=True)
    # plot_heatmaps(heatmap_stats, filename="heatmap.html")
