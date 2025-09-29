import pandas as pd
import requests
import time
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta
import numpy as np


# ========== 数据获取 (代码不变) ==========
def get_full_historical_data(
    symbol: str, interval: str, start_str: str, end_str: str = None
) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    start_time = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_time = (
        int(pd.to_datetime(end_str).timestamp() * 1000)
        if end_str
        else int(time.time() * 1000)
    )
    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            start_time = data[-1][0] + 1
        except requests.exceptions.RequestException as e:
            print(f"网络错误: {e}, 5秒后重试...")
            time.sleep(5)

    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(
        f"✅ 成功获取 {symbol} 从 {df.index[0]} 到 {df.index[-1]} 的 {len(df)} 条数据。"
    )
    return df


class BullBearRangeStrategy(Strategy):
    # <<< 调整1：大幅降低牛市仓位 >>>
    bull_position_size = 0.5  # 从0.95降低到0.5

    # --- 其他参数保持不变 ---
    ema_fast_period = 12
    ema_slow_period = 26
    ema_long_period = 50
    ema_market_period = 200
    rsi_period = 14
    rsi_bull_lower = 25
    rsi_bull_upper = 80
    atr_period = 14
    atr_mult_bull = 2.0
    adx_period = 14
    adx_threshold = 20
    ema_diff_threshold = 0.003

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(close, self.ema_fast_period).ema_indicator()
        )
        self.ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(close, self.ema_slow_period).ema_indicator()
        )
        self.ema_long = self.I(
            lambda: ta.trend.EMAIndicator(close, self.ema_long_period).ema_indicator()
        )
        self.ema_market = self.I(
            lambda: ta.trend.EMAIndicator(close, self.ema_market_period).ema_indicator()
        )
        self.rsi = self.I(
            lambda: ta.momentum.RSIIndicator(close, self.rsi_period).rsi()
        )
        self.atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                high, low, close, self.atr_period
            ).average_true_range()
        )
        self.adx = self.I(
            lambda: ta.trend.ADXIndicator(high, low, close, self.adx_period).adx()
        )
        self.trailing_sl = 0.0

    def is_range_market(self) -> bool:
        adx_val = self.adx[-1]
        ema_fast_val = self.ema_fast[-1]
        ema_slow_val = self.ema_slow[-1]
        ema_diff = abs(ema_fast_val - ema_slow_val) / ema_slow_val
        return adx_val < self.adx_threshold and ema_diff < self.ema_diff_threshold

    def next(self):
        price = self.data.Close[-1]
        current_rsi = self.rsi[-1]

        # <<< 调整3：增加震荡市平仓逻辑 >>>
        if self.is_range_market():
            if self.position:
                self.position.close()  # 如果进入震荡，平掉现有仓位
            return  # 震荡期不进行任何操作

        # 定义市场状态
        is_bull_market = price > self.ema_market[-1]

        # 移动止损
        if self.position:
            if self.position.is_long:
                if price <= self.trailing_sl:
                    self.position.close()
                    return
                new_sl = price - self.atr[-1] * self.atr_mult_bull
                self.trailing_sl = max(self.trailing_sl, new_sl)

        # <<< 调整2：只在牛市中交易 >>>
        if not is_bull_market:
            # 如果是熊市，且有持仓，则平仓（确保不会持有仓位进入熊市）
            if self.position:
                self.position.close()
            return  # 熊市不做任何开仓操作

        # 入场逻辑 (只保留牛市部分)
        if not self.position:
            is_uptrend = (
                price > self.ema_long[-1] and self.ema_fast[-1] > self.ema_slow[-1]
            )

            if is_uptrend and self.rsi_bull_lower < current_rsi < self.rsi_bull_upper:
                initial_sl = price - self.atr[-1] * self.atr_mult_bull
                self.buy(size=self.bull_position_size, sl=initial_sl)
                self.trailing_sl = initial_sl


if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    TIME_INTERVAL = "4h"
    START_DATE = "2022-01-01"
    END_DATE = "2025-09-27"
    INITIAL_CASH = 500_000
    COMMISSION_FEE = 0.0005

    print("📡 获取历史数据...")
    data = get_full_historical_data(
        symbol=SYMBOL, interval=TIME_INTERVAL, start_str=START_DATE, end_str=END_DATE
    )

    if not data.empty:
        bt = Backtest(
            data,
            BullBearRangeStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION_FEE,
            exclusive_orders=True,
            trade_on_close=True,
            finalize_trades=True,
        )

        stats = bt.run()
        print("✅ 回测完成！\n")
        print(stats)

        print("\n📊 生成回测图表...")
        bt.plot(filename="bull_bear_range_strategy_v2.html", open_browser=False)
        print("✅ 图表已保存: bull_bear_range_strategy_v2.html")
