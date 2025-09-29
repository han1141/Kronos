# backtest_strategy.py - 全新重建的趋势跟踪策略 (最终修正版)
import pandas as pd
import requests
import time
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 1. 数据获取函数 (保持不变) ---
def get_full_historical_data(symbol: str, interval: str, start_str: str, end_str: str = None) -> pd.DataFrame:
    # ... (此处代码与之前版本完全相同，为简洁省略)
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    start_time = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_time = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)
    all_data = []
    while start_time < end_time:
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "endTime": end_time, "limit": limit}
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            if not data: break
            all_data.extend(data)
            start_time = data[-1][0] + 1
        except requests.exceptions.RequestException as e:
            print(f"网络错误: {e}, 5秒后重试...")
            time.sleep(5)
    df = pd.DataFrame(all_data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(f"成功获取 {symbol} 从 {df.index[0]} 到 {df.index[-1]} 的 {len(df)} 条数据。")
    return df

# --- 2. 全新的、逻辑一致的趋势跟踪策略 (已修正) ---
class EMATrendStrategy(Strategy):
    """
    EMA 黄金交叉趋势跟踪策略 (带ATR移动止损)
    """
    ema_fast_period = 50
    ema_slow_period = 100
    ema_long_period = 200
    atr_period = 14
    atr_sl_multiplier = 2.5

    def init(self):
        """预计算所有需要的指标，并初始化移动止损变量"""
        # 计算指标
        self.ema_fast = self.I(lambda: ta.trend.EMAIndicator(close=pd.Series(self.data.Close), window=self.ema_fast_period).ema_indicator())
        self.ema_slow = self.I(lambda: ta.trend.EMAIndicator(close=pd.Series(self.data.Close), window=self.ema_slow_period).ema_indicator())
        self.ema_long = self.I(lambda: ta.trend.EMAIndicator(close=pd.Series(self.data.Close), window=self.ema_long_period).ema_indicator())
        self.atr = self.I(lambda: ta.volatility.AverageTrueRange(high=pd.Series(self.data.High), low=pd.Series(self.data.Low), close=pd.Series(self.data.Close), window=self.atr_period).average_true_range())
        
        # === 关键修正 1：初始化一个变量来存储移动止损价格 ===
        self.trailing_sl = 0.0

    def next(self):
        """在每个时间点执行交易决策"""
        price = self.data.Close[-1]

        # === 关键修正 2：手动实现移动止损的更新和检查逻辑 ===
        # 如果当前有持仓，则更新并检查移动止损
        if self.position:
            if self.position.is_long:
                # 检查是否触及移动止损
                if price <= self.trailing_sl:
                    self.position.close()
                    return # 触及后平仓并结束当前K线的操作
                # 如果没触及，则更新移动止损位（只上移，不下移）
                new_sl = price - self.atr[-1] * self.atr_sl_multiplier
                self.trailing_sl = max(self.trailing_sl, new_sl)

            else: # is_short
                # 检查是否触及移动止损
                if price >= self.trailing_sl:
                    self.position.close()
                    return # 触及后平仓并结束当前K线的操作
                # 如果没触及，则更新移动止损位（只下移，不上移）
                new_sl = price + self.atr[-1] * self.atr_sl_multiplier
                self.trailing_sl = min(self.trailing_sl, new_sl)

        # 如果当前没有持仓，则检查入场信号
        else:
            is_bull_market = price > self.ema_long[-1]
            is_bear_market = price < self.ema_long[-1]
            fast_cross_above_slow = crossover(self.ema_fast, self.ema_slow)
            fast_cross_below_slow = crossover(self.ema_slow, self.ema_fast)

            # 买入条件：牛市 + 黄金交叉
            if is_bull_market and fast_cross_above_slow:
                initial_sl = price - self.atr[-1] * self.atr_sl_multiplier
                self.buy(sl=initial_sl)
                # === 关键修正 3：开仓后，初始化移动止损价格 ===
                self.trailing_sl = initial_sl

            # 卖出条件：熊市 + 死亡交叉
            elif is_bear_market and fast_cross_below_slow:
                initial_sl = price + self.atr[-1] * self.atr_sl_multiplier
                self.sell(sl=initial_sl)
                # === 关键修正 3：开仓后，初始化移动止损价格 ===
                self.trailing_sl = initial_sl

# --- 3. 主执行模块 (保持不变) ---
if __name__ == '__main__':
    SYMBOL = "BTCUSDT"
    TIME_INTERVAL = "4h"
    START_DATE = "2023-01-01"
    END_DATE = "2025-09-27"
    INITIAL_CASH = 100_000
    COMMISSION_FEE = 0.001

    data = get_full_historical_data(symbol=SYMBOL, interval=TIME_INTERVAL, start_str=START_DATE, end_str=END_DATE)

    if not data.empty:
        bt = Backtest(data, EMATrendStrategy, cash=INITIAL_CASH, commission=COMMISSION_FEE, exclusive_orders=True, trade_on_close=True)
        print("\n开始执行【全新EMA趋势策略】回测...")
        stats = bt.run()
        print("回测完成！")
        print("\n" + "="*80)
        print(" " * 28 + "全新策略 - 回测性能报告")
        print("="*80)
        print(stats)
        print("\n正在生成回测图表HTML文件...")
        bt.plot(open_browser=False)
        print("图表已保存至当前目录，请手动打开查看。")
    else:
        print("未能获取到数据，回测无法进行。")