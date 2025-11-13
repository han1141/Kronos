# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import requests
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# 数据获取函数
def fetch_binance_klines(symbol, interval, start_date, end_date=None):
    url = "https://api.binance.com/api/v3/klines"
    cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]

    start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_date).timestamp() * 1000) if end_date else None

    all_data = []
    current_ts = start_ts

    while current_ts < end_ts if end_ts else True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_ts,
            "limit": 1000,
        }
        if end_ts:
            params["endTime"] = end_ts

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            current_ts = data[-1][0] + 1
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[*cols, "c1", "c2", "c3", "c4", "c5", "c6"])[
        cols
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.set_index("timestamp").sort_index()


# 自定义技术指标计算
def calculate_bbands(prices, period=25, std_mult=4):
    """计算布林带指标"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper_band = sma + (std * std_mult)
    lower_band = sma - (std * std_mult)

    return upper_band.values, sma.values, lower_band.values


def calculate_atr(high, low, close, period=14):
    """计算ATR指标"""
    df = pd.DataFrame({"high": high, "low": low, "close": close})

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift(1))
    tr3 = abs(df["low"] - df["close"].shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.values


def calculate_adx(high, low, close, period=14):
    """计算ADX指标"""
    df = pd.DataFrame({"high": high, "low": low, "close": close})

    # 计算+DI和-DI
    high_low = df["high"] - df["low"]
    high_close_prev = abs(df["high"] - df["close"].shift(1))
    low_close_prev = abs(df["low"] - df["close"].shift(1))

    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff().abs()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    plus_di = 100 * (
        plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean()
    )
    minus_di = 100 * (
        minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean()
    )

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return adx.values


class BB4xADXStrategy(Strategy):
    # 策略参数
    bb_period = 25
    bb_std_mult = 4
    adx_period = 14
    atr_period = 14
    risk_per_trade = 0.02
    adx_trend_threshold = 20  # ADX趋势阈值

    def init(self):
        close_prices = pd.Series(self.data.Close)
        high_prices = pd.Series(self.data.High)
        low_prices = pd.Series(self.data.Low)

        # 计算技术指标
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            calculate_bbands, close_prices, self.bb_period, self.bb_std_mult
        )
        self.adx_values = self.I(
            calculate_adx, high_prices, low_prices, close_prices, self.adx_period
        )
        self.atr_values = self.I(
            calculate_atr, high_prices, low_prices, close_prices, self.atr_period
        )

        # 交易统计
        self.total_trades = 0
        self.entry_price = None
        self.current_stop = None

    def calculate_position_size(self, price, stop_loss_price):
        """基于风险的仓位计算"""
        if price <= stop_loss_price:
            return 0

        risk_amount = self.equity * self.risk_per_trade
        risk_per_unit = abs(price - stop_loss_price)
        units = risk_amount / risk_per_unit

        max_units = (self.equity * 0.95) / price
        units = min(units, max_units)
        units = max(1, int(units))

        return units

    def is_trending_market(self):
        """判断市场是否处于趋势中"""
        if len(self.adx_values) == 0:
            return False
        return self.adx_values[-1] > self.adx_trend_threshold

    def next(self):
        if len(self.data.Close) < self.bb_period + 5:
            return

        price = self.data.Close[-1]
        atr = self.atr_values[-1] if len(self.atr_values) > 0 else price * 0.01

        # 持仓管理
        if self.position:
            if self.entry_price is None:
                return

            if self.position.is_long:
                profit_pct = (price - self.entry_price) / self.entry_price

                # 移动止损 - 基于布林带中轨
                if len(self.bb_middle) > 0:
                    bb_middle = self.bb_middle[-1]
                    # 使用布林带中轨作为移动止损
                    new_stop = bb_middle
                    self.current_stop = max(getattr(self, "current_stop", 0), new_stop)

                # 止盈止损
                if hasattr(self, "current_stop") and price <= self.current_stop:
                    self.position.close()
                    self.entry_price = None
                    logger.info(
                        f"多头止损: 价格{price:.2f}触及移动止损{self.current_stop:.2f}"
                    )
                elif profit_pct > 0.08:  # 盈利8%止盈
                    self.position.close()
                    self.entry_price = None
                    logger.info(f"多头止盈: 盈利{profit_pct*100:.2f}%")
                elif profit_pct < -0.04:  # 亏损4%硬止损
                    self.position.close()
                    self.entry_price = None
                    logger.info(f"多头硬止损: 亏损{profit_pct*100:.2f}%")

            else:  # short
                profit_pct = (self.entry_price - price) / self.entry_price

                # 移动止损 - 基于布林带中轨
                if len(self.bb_middle) > 0:
                    bb_middle = self.bb_middle[-1]
                    # 使用布林带中轨作为移动止损
                    new_stop = bb_middle
                    self.current_stop = min(
                        getattr(self, "current_stop", float("inf")), new_stop
                    )

                # 止盈止损
                if hasattr(self, "current_stop") and price >= self.current_stop:
                    self.position.close()
                    self.entry_price = None
                    logger.info(
                        f"空头止损: 价格{price:.2f}触及移动止损{self.current_stop:.2f}"
                    )
                elif profit_pct > 0.08:  # 盈利8%止盈
                    self.position.close()
                    self.entry_price = None
                    logger.info(f"空头止盈: 盈利{profit_pct*100:.2f}%")
                elif profit_pct < -0.04:  # 亏损4%硬止损
                    self.position.close()
                    self.entry_price = None
                    logger.info(f"空头硬止损: 亏损{profit_pct*100:.2f}%")

            return

        # 检查指标是否存在
        if (
            len(self.bb_upper) == 0
            or len(self.bb_lower) == 0
            or len(self.bb_middle) == 0
            or len(self.adx_values) == 0
        ):
            return

        bb_upper = self.bb_upper[-1]
        bb_middle = self.bb_middle[-1]
        bb_lower = self.bb_lower[-1]

        # 检查是否在趋势市场中
        if not self.is_trending_market():
            return  # 不在趋势市场，不交易

        # 使用布林带中线作为趋势开关
        # 价格在中线上方 - 只考虑多头
        # 价格在下方下方 - 只考虑空头

        # 多头入场条件
        if (
            price > bb_middle  # 价格在中线上方
            and price > bb_upper  # 价格突破上轨
            and self.data.Volume[-1] > np.mean(self.data.Volume[-10:])  # 成交量放大
            and price > self.data.Close[-2]
        ):  # 确认突破

            # 多头信号 - 突破4倍布林带上轨且在中线上方
            stop_loss = bb_middle  # 以中线作为止损
            size = self.calculate_position_size(price, stop_loss)

            if size > 0:
                self.buy(size=size)
                self.current_stop = stop_loss
                self.entry_price = price
                self.total_trades += 1
                logger.info(
                    f"多头入场: 价格{price:.2f}突破布林带上轨{bb_upper:.2f}, 中线{bb_middle:.2f}"
                )

        # 空头入场条件
        elif (
            price < bb_middle  # 价格在中线下方
            and price < bb_lower  # 价格跌破下轨
            and self.data.Volume[-1] > np.mean(self.data.Volume[-10:])  # 成交量放大
            and price < self.data.Close[-2]
        ):  # 确认跌破

            # 空头信号 - 跌破4倍布林带下轨且在中线下方
            stop_loss = bb_middle  # 以中线作为止损
            size = self.calculate_position_size(price, stop_loss)

            if size > 0:
                self.sell(size=size)
                self.current_stop = stop_loss
                self.entry_price = price
                self.total_trades += 1
                logger.info(
                    f"空头入场: 价格{price:.2f}跌破布林带下轨{bb_lower:.2f}, 中线{bb_middle:.2f}"
                )


def run_bb4x_adx_backtest():
    """运行4倍布林带+ADX策略回测"""
    logger.info("====== 运行4倍布林带+ADX策略回测 ======")

    symbol = "ETHUSDT"
    interval = "15m"
    start_date = "2025-01-01"
    end_date = "2025-12-31"

    logger.info(f"获取 {symbol} 数据...")
    data = fetch_binance_klines(symbol, interval, start_date, end_date)

    if data.empty:
        logger.error("数据获取失败")
        return

    logger.info(f"数据获取成功: {len(data)} 条记录")

    bt = Backtest(
        data, BB4xADXStrategy, cash=100000, commission=0.00075, exclusive_orders=True
    )

    stats = bt.run()

    print(f"\n{'='*50}")
    print(f"4倍布林带+ADX策略回测结果 - {symbol} {interval}")
    print(f"测试期间: {start_date} 至 {end_date}")
    print(f"{'='*50}")

    key_metrics = {
        "最终权益": f"${stats['Equity Final [$]']:,.2f}",
        "总收益率": f"{stats['Return [%]']:.2f}%",
        "年化收益率": f"{stats.get('Return (Ann.) [%]', 0):.2f}%",
        "夏普比率": f"{stats.get('Sharpe Ratio', 0):.2f}",
        "最大回撤": f"{stats.get('Max. Drawdown [%]', 0):.2f}%",
        "交易次数": f"{stats.get('# Trades', 0)}",
        "胜率": f"{stats.get('Win Rate [%]', 0):.2f}%",
        "盈亏比": f"{stats.get('Profit Factor', 0):.2f}",
    }

    for metric, value in key_metrics.items():
        print(f"{metric}: {value}")

    # 交易分析
    try:
        trades = stats["_trades"]
        if not trades.empty:
            print(f"\n交易分析:")
            print(f"平均持仓时间: {trades['Duration'].mean()}")
            print(f"最大盈利交易: {trades['PnL'].max():.2f}")
            print(f"最大亏损交易: {trades['PnL'].min():.2f}")

            # 计算平均盈亏比
            winning_trades = trades[trades["PnL"] > 0]
            losing_trades = trades[trades["PnL"] < 0]
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                avg_win = winning_trades["PnL"].mean()
                avg_loss = abs(losing_trades["PnL"].mean())
                print(f"平均盈亏比: {avg_win/avg_loss:.2f}")
    except:
        pass


if __name__ == "__main__":
    run_bb4x_adx_backtest()
