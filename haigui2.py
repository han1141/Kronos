import pandas as pd
import numpy as np
import requests
import os
import time
import logging
from datetime import timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================= 🔧 参数修改区 =================
SYMBOL = "ETHUSDT"
INTERVAL = "5m"
START_DATE = "2024-01-01"
END_DATE = "2025-11-25"

# 费率
FEE_RATE = 0.00035  # 0.035%
SLIPPAGE_PCT = 0.0001  # 0.01%

# 策略参数
DOJI_RATIO = 0.15  # 十字星定义
MIN_DOJI_RANGE = 0.001  # 最小波动
RISK_REWARD_RATIO = 1.5  # 盈亏比 1.5
MA_PERIOD = 20  # 15m MA5
EMA_TREND_PERIOD = 200  # 🔥 新增：200周期趋势线

# CHOP 参数
CHOP_PERIOD = 14
CHOP_THRESHOLD = 61.8

# ADX 参数 (仅用于高位止盈，不再用于低位止损)
ADX_PERIOD = 14


# ================= 1. 数据获取 =================
def _download_binance_klines(symbol, interval, start_dt, end_dt):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_ts = int(pd.to_datetime(start_dt).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_dt).timestamp() * 1000)
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": 1000,
        }
        try:
            r = requests.get(url, params=params, timeout=5).json()
            if not r or not isinstance(r, list):
                break
            all_data.extend(r)
            start_ts = r[-1][0] + 1
            time.sleep(0.1)
        except:
            break
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "x",
            "x",
            "x",
            "x",
            "x",
            "x",
        ],
    ).iloc[:, :6]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c])
    return df.set_index("timestamp").sort_index()


def fetch_binance_klines(symbol, interval, start_str, end_str):
    if not os.path.exists("data"):
        os.makedirs("data")
    full_cache = f"data/{symbol}_{interval}_full.csv"
    if os.path.exists(full_cache):
        return (
            pd.read_csv(full_cache, index_col="timestamp", parse_dates=True)
            .sort_index()
            .drop_duplicates()
            .loc[start_str:end_str]
        )
    else:
        df = _download_binance_klines(symbol, interval, start_str, end_str)
        if not df.empty:
            df.to_csv(full_cache)
        return df


# ================= 2. 指标计算 =================
def calculate_indicators(df):
    df = df.copy()
    # ADX
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["+DM"] = np.where(
        (df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]),
        df["High"] - df["High"].shift(1),
        0,
    )
    df["+DM"] = np.where(df["+DM"] < 0, 0, df["+DM"])
    df["-DM"] = np.where(
        (df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)),
        df["Low"].shift(1) - df["Low"],
        0,
    )
    df["-DM"] = np.where(df["-DM"] < 0, 0, df["-DM"])
    df["TR_smooth"] = df["TR"].ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    df["+DM_smooth"] = df["+DM"].ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    df["-DM_smooth"] = df["-DM"].ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    df["+DI"] = 100 * (df["+DM_smooth"] / df["TR_smooth"])
    df["-DI"] = 100 * (df["-DM_smooth"] / df["TR_smooth"])
    df["DX"] = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
    df["ADX"] = df["DX"].ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()

    # CHOP
    df["Sum_TR"] = df["TR"].rolling(window=CHOP_PERIOD).sum()
    df["Max_High"] = df["High"].rolling(window=CHOP_PERIOD).max()
    df["Min_Low"] = df["Low"].rolling(window=CHOP_PERIOD).min()
    df["Range_n"] = df["Max_High"] - df["Min_Low"]
    df["CHOP"] = 100 * np.log10(df["Sum_TR"] / df["Range_n"]) / np.log10(CHOP_PERIOD)

    return df


# ================= 3. 策略引擎 =================
def run_strategy(df_5m, df_15m, symbol, fee_rate=FEE_RATE, slippage_pct=SLIPPAGE_PCT):
    logger.info(
        f"🚀 开始回测 (15m十字星+MA5 | 顺EMA{EMA_TREND_PERIOD}趋势 | 无ADX低位止损)..."
    )

    # 1. 15m 数据处理
    df_15m = calculate_indicators(df_15m)
    df_15m["MA"] = df_15m["Close"].rolling(window=MA_PERIOD).mean()

    # 🔥 新增：EMA 200 趋势线
    df_15m["EMA_Trend"] = (
        df_15m["Close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    )

    # 十字星定义
    df_15m["Body"] = abs(df_15m["Close"] - df_15m["Open"])
    df_15m["Range"] = df_15m["High"] - df_15m["Low"]
    df_15m["Ratio"] = df_15m["Body"] / df_15m["Range"]
    df_15m["Is_Doji"] = (df_15m["Ratio"] <= DOJI_RATIO) & (
        df_15m["Range"] / df_15m["Open"] >= MIN_DOJI_RANGE
    )

    # 映射 15m 指标到 5m
    df_5m = df_5m.copy()
    df_5m["ADX_15m_Ref"] = df_15m["ADX"].reindex(df_5m.index, method="ffill")
    df_5m["CHOP_15m_Ref"] = df_15m["CHOP"].reindex(df_5m.index, method="ffill")

    trades = []
    last_exit_time = None

    for ts, candle in df_5m.iterrows():
        if last_exit_time is not None and ts <= last_exit_time:
            continue

        # K1 (15m)
        k1_time = ts - timedelta(minutes=15)
        if k1_time not in df_15m.index:
            continue
        k1_candle = df_15m.loc[k1_time]

        # 1. 震荡过滤
        current_chop = df_5m.loc[ts, "CHOP_15m_Ref"]
        if not pd.isna(current_chop) and current_chop > CHOP_THRESHOLD:
            continue

        # 2. 十字星过滤
        if not k1_candle["Is_Doji"]:
            continue

        # 3. MA5 搭载过滤
        k1_ma = k1_candle["MA"]
        if pd.isna(k1_ma):
            continue
        k1_low = k1_candle["Low"]
        k1_high = k1_candle["High"]
        k1_close = k1_candle["Close"]
        k1_trend = k1_candle["EMA_Trend"]  # 趋势线

        if pd.isna(k1_trend):
            continue

        is_supported = (k1_low <= k1_ma * 1.0005) and (k1_close >= k1_ma * 0.9995)
        is_resisted = (k1_high >= k1_ma * 0.9995) and (k1_close <= k1_ma * 1.0005)

        if not is_supported and not is_resisted:
            continue

        # 4. K2 确认 + 趋势过滤
        k2_close = candle["Close"]
        entry_signal = None

        # 🔥 顺势做多：价格 > EMA200 且 十字星踩MA5
        if is_supported and k2_close > k1_high:
            if k1_close > k1_trend:  # 趋势过滤
                entry_signal = "long"
                stop_loss = k1_low

        # 🔥 顺势做空：价格 < EMA200 且 十字星压MA5
        elif is_resisted and k2_close < k1_low:
            if k1_close < k1_trend:  # 趋势过滤
                entry_signal = "short"
                stop_loss = k1_high

        if entry_signal:
            entry_price = k2_close
            risk = abs(entry_price - stop_loss)
            if risk == 0:
                continue

            if entry_signal == "long":
                target_tp = entry_price + (risk * RISK_REWARD_RATIO)
            else:
                target_tp = entry_price - (risk * RISK_REWARD_RATIO)

            trade_result = simulate_trade(
                df_5m, ts, entry_price, stop_loss, target_tp, entry_signal
            )

            exit_price = trade_result["exit_price"]
            last_exit_time = trade_result["exit_time"]

            if entry_signal == "long":
                gross_pct = (exit_price - entry_price) / entry_price
            else:
                gross_pct = (entry_price - exit_price) / entry_price

            net_pct = gross_pct - 2 * (fee_rate + slippage_pct)
            pnl_value = entry_price * net_pct

            trades.append(
                {
                    "date": ts.date(),
                    "side": entry_signal,
                    "entry_time": ts + timedelta(minutes=5),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl_value,
                    "pnl_pct": net_pct * 100,
                    "reason": trade_result["reason"],
                }
            )

    return pd.DataFrame(trades)


def simulate_trade(day_data, start_time, entry_price, hard_stop, target_tp, side):
    future_data = day_data[day_data.index > start_time]
    prev_adx = day_data.loc[start_time, "ADX_15m_Ref"]
    if pd.isna(prev_adx):
        prev_adx = 0

    for ts, row in future_data.iterrows():
        current_adx = row["ADX_15m_Ref"]

        if side == "long":
            # 硬止损 (十字星另一头)
            if row["Low"] <= hard_stop:
                return {"exit_time": ts, "exit_price": hard_stop, "reason": "Hard SL"}

            # 🔥 删除 ADX < 20 止损逻辑，给行情空间 🔥

            # ADX 止盈 (高位力竭)
            if not pd.isna(current_adx) and prev_adx > 45 and current_adx < prev_adx:
                return {
                    "exit_time": ts,
                    "exit_price": row["Close"],
                    "reason": "TP (ADX Peak)",
                }

            # 固定止盈 (1.5R)
            if row["High"] >= target_tp:
                return {"exit_time": ts, "exit_price": target_tp, "reason": "TP (1.5R)"}

        else:  # short
            if row["High"] >= hard_stop:
                return {"exit_time": ts, "exit_price": hard_stop, "reason": "Hard SL"}

            if not pd.isna(current_adx) and prev_adx > 45 and current_adx < prev_adx:
                return {
                    "exit_time": ts,
                    "exit_price": row["Close"],
                    "reason": "TP (ADX Peak)",
                }

            if row["Low"] <= target_tp:
                return {"exit_time": ts, "exit_price": target_tp, "reason": "TP (1.5R)"}

        if not pd.isna(current_adx):
            prev_adx = current_adx

    last_candle = day_data.iloc[-1] if not day_data.empty else None
    if last_candle is not None:
        return {
            "exit_time": last_candle.name,
            "exit_price": last_candle["Close"],
            "reason": "End of Data",
        }
    return {"exit_time": start_time, "exit_price": entry_price, "reason": "Error"}


# ================= 4. 分析 =================
def analyze_results(trades_df, symbol):
    if trades_df.empty:
        logger.warning("⚠️ 没有产生交易。")
        return

    trades_df["cum_pnl_pct"] = trades_df["pnl_pct"].cumsum()
    total_trades = len(trades_df)
    win_rate = (trades_df["pnl"] > 0).mean() * 100
    total_return = trades_df["cum_pnl_pct"].iloc[-1]

    print("=" * 40)
    print(f"📊 策略回测结果: {symbol}")
    print(f"修正: 顺EMA{EMA_TREND_PERIOD} | 移除ADX止损 | CHOP过滤")
    print("=" * 40)
    print(f"交易总数: {total_trades}")
    print(f"🔥 胜率: {win_rate:.2f}%")
    print(f"累计收益率: {total_return:.2f}%")

    print("-" * 40)
    print("离场原因统计:")
    print(trades_df["reason"].value_counts())
    print("=" * 40)


if __name__ == "__main__":
    df_5m = fetch_binance_klines(SYMBOL, "5m", START_DATE, END_DATE)
    df_15m = fetch_binance_klines(SYMBOL, "15m", START_DATE, END_DATE)
    if not df_5m.empty and not df_15m.empty:
        results = run_strategy(df_5m, df_15m, SYMBOL)
        analyze_results(results, SYMBOL)
