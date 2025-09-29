import pandas as pd
import requests
import time
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta


# --- Data retrieval function remains unchanged ---
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
            print(f"Network error: {e}, retrying in 5 seconds...")
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
    if df.empty:
        print(f"❌ No data found for {symbol} in the specified date range.")
        return df
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    print(
        f"✅ Successfully retrieved {len(df)} data points for {symbol} from {df.index[0]} to {df.index[-1]}."
    )
    return df


class FinalProfitableStrategy(Strategy):
    """
    The final, synthesized strategy combining the best of all previous versions:
    1. Trend Filter: Only trade when price > SMA200.
    2. High-Probability Entry: Buy the Dip (cross above Middle Band).
    3. Clear Take-Profit: Sell at the Upper Band.
    4. Dynamic Stop-Loss: Place stop-loss at the Lower Band.
    """

    bb_period = 20
    bb_std = 2.0
    sma_long_period = 200
    position_size_pct = 0.9

    def init(self):
        close = pd.Series(self.data.Close)
        self.bb_indicator = ta.volatility.BollingerBands(
            close=close, window=self.bb_period, window_dev=self.bb_std
        )
        self.bb_upper = self.I(self.bb_indicator.bollinger_hband)
        self.bb_middle = self.I(self.bb_indicator.bollinger_mavg)
        self.bb_lower = self.I(self.bb_indicator.bollinger_lband)
        self.sma_long = self.I(
            lambda: ta.trend.SMAIndicator(close, self.sma_long_period).sma_indicator()
        )

    def next(self):
        if len(self.data.Close) < self.sma_long_period:
            return

        price = self.data.Close[-1]
        is_uptrend = price > self.sma_long[-1]

        # --- EXIT LOGIC (TAKE PROFIT) ---
        # If we have a long position, check to see if it has hit the upper band.
        if self.position.is_long:
            if price >= self.bb_upper[-1]:
                self.position.close()

        # --- ENTRY LOGIC ("BUY THE DIP") ---
        # Only look for entries if we are in an uptrend and have no open position.
        elif is_uptrend:
            # High-probability entry: Buy when the price crosses *above* the middle band.
            if crossover(self.data.Close, self.bb_middle):
                # Dynamic stop-loss: Place the SL at the current lower band.
                sl_price = self.bb_lower[-1]
                self.buy(size=self.position_size_pct, sl=sl_price)


if __name__ == "__main__":
    SYMBOL = "BTCUSDT"
    TIME_INTERVAL = "1h"

    # Backtest on the current year (2025) data
    START_DATE = "2025-01-01"
    END_DATE = "2025-09-29"

    INITIAL_CASH = 500_000
    COMMISSION_FEE = 0.0005

    print("📡 Retrieving historical data for the FINAL PROFITABLE Strategy...")
    data = get_full_historical_data(
        symbol=SYMBOL, interval=TIME_INTERVAL, start_str=START_DATE, end_str=END_DATE
    )

    if not data.empty:
        print("\n🚀 Starting backtest...")
        bt = Backtest(
            data,
            FinalProfitableStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION_FEE,
            exclusive_orders=True,
            trade_on_close=True,
            finalize_trades=True,
        )

        stats = bt.run()
        print("✅ Backtest complete!\n")
        print(stats)
        print("\n📊 Generating plot...")
        bt.plot()
    else:
        print(
            "\nBacktest skipped. Please check your date range and internet connection."
        )
