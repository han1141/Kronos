import pandas as pd
import numpy as np
import os
import requests
import time
import logging
from datetime import datetime
import pandas_ta as ta

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ================= 1. 数据获取 =================
def fetch_binance_klines(symbol, interval, start_str, end_str):
    filename = f"{symbol}_{interval}_{start_str.replace('-', '')}_{end_str.replace('-', '')}.csv"
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col="timestamp", parse_dates=True)
            logger.info(f"📂 加载缓存: {filename}")
            return df.sort_index()
        except Exception as e:
            logger.warning(f"缓存加载失败: {e}")

    logger.info(f"🌐 下载数据: {symbol} ({start_str} ~ {end_str})")
    url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    all_data = []
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not data or not isinstance(data, list):
                break
            all_data.extend(data)
            start_ts = data[-1][0] + 1
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"下载错误: {e}")
            time.sleep(2)

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
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col])
    return df.set_index("timestamp").sort_index().drop_duplicates()


# ================= 2. 指标计算 =================
def calculate_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(window=period).mean()


def calculate_adx(df, period=14):
    df = df.copy()
    df["up_move"] = df["High"] - df["High"].shift(1)
    df["down_move"] = df["Low"].shift(1) - df["Low"]
    df["plus_dm"] = np.where(
        (df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0
    )
    df["minus_dm"] = np.where(
        (df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0
    )

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift(1))
    low_close = np.abs(df["Low"] - df["Close"].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = tr.rolling(window=period).mean()
    atr = atr.replace(0, np.nan)

    df["plus_di"] = 100 * (df["plus_dm"].rolling(window=period).mean() / atr)
    df["minus_di"] = 100 * (df["minus_dm"].rolling(window=period).mean() / atr)
    sum_di = df["plus_di"] + df["minus_di"]
    dx = 100 * np.abs(df["plus_di"] - df["minus_di"]) / sum_di
    adx = dx.rolling(window=period).mean()
    return adx


def add_multi_tf_rsi(df, rsi_period: int = 14):
    """
    为原始 K 线加入日线 / 周线 RSI (以收盘价 resample 后计算，再前向填充回当前周期)。
    """
    df = df.copy()
    close = df["Close"]

    daily_close = close.resample("1D").last()
    weekly_close = close.resample("1W").last()

    rsi_daily = ta.rsi(daily_close, length=rsi_period)
    rsi_weekly = ta.rsi(weekly_close, length=rsi_period)

    df["rsi_daily"] = rsi_daily.reindex(df.index, method="ffill")
    df["rsi_weekly"] = rsi_weekly.reindex(df.index, method="ffill")

    return df


def compute_ultimate_stats(
    df,
    lookahead: int = 1,
    weekly_rsi_high: float = 68.0,  # 周线超买阈值（做空）
    weekly_rsi_low: float = 32.0,  # 周线超卖阈值（做多）
):
    """
    终极信号统计（十字星在 7MA 附近 + 周线 RSI 过滤）:

    - 做空:
        上一根为 inside bar + 十字星，且收盘在 7MA 上方
        当前 K 线收盘跌破 7MA
        周线 RSI > weekly_rsi_high

    - 做多:
        上一根为 inside bar + 十字星，且收盘在 7MA 下方
        当前 K 线收盘上穿 7MA
        周线 RSI < weekly_rsi_low
    """
    df = df.copy()

    # 确保有 7MA
    if "ma_7" not in df.columns:
        df["ma_7"] = df["Close"].rolling(window=7).mean()

    # 确保有周线 RSI
    if "rsi_weekly" not in df.columns:
        df = add_multi_tf_rsi(df)

    close = df["Close"]
    ma7 = df["ma_7"]

    # inside bar & 十字星（在每一根 K 上）
    is_inside = (df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))
    body = (df["Close"] - df["Open"]).abs()
    rng = df["High"] - df["Low"]
    is_doji = (rng > 0) & (body / rng <= 0.2)

    # 以上一根 K 线作为十字 inside，当前 K 线作为入场
    is_inside_prev = is_inside.shift(1)
    is_doji_prev = is_doji.shift(1)
    ib_close_prev = close.shift(1)
    ma7_prev = ma7.shift(1)

    current_close = close
    current_ma7 = ma7
    weekly_rsi = df["rsi_weekly"]

    # 做空：上方十字 → 当根下破 7MA 且周线超买
    mask_short = (
        is_inside_prev
        & is_doji_prev
        & (ib_close_prev > ma7_prev)
        & (current_close < current_ma7)
        & (weekly_rsi > weekly_rsi_high)
    ).fillna(False)

    # 做多：下方十字 → 当根上破 7MA 且周线超卖
    mask_long = (
        is_inside_prev
        & is_doji_prev
        & (ib_close_prev < ma7_prev)
        & (current_close > current_ma7)
        & (weekly_rsi < weekly_rsi_low)
    ).fillna(False)

    def _side_stats(mask, long_side: bool):
        if mask.sum() == 0:
            return 0, 0.0, 0.0, 0.0

        entry = current_close[mask]
        future = close.shift(-lookahead)[mask]

        valid = future.notna()
        entry = entry[valid]
        future = future[valid]
        if len(entry) == 0:
            return 0, 0.0, 0.0, 0.0

        ret = (future - entry) / entry
        if long_side:
            win = (future > entry).mean()
            big_meat = (ret > 0.01).mean()
        else:
            win = (future < entry).mean()
            big_meat = (ret < -0.01).mean()
        avg_ret = ret.mean()
        return int(len(entry)), win, big_meat, avg_ret

    long_n, long_win, long_big, long_avg = _side_stats(mask_long, long_side=True)

    stats = {
        # 只保留终极做多相关统计（按你的要求去掉终极做空）
        "终极做多信号数": long_n,
        "终极做多胜率": f"{long_win:.2%}",
        "终极做多平均收益": f"{long_avg:.3%}",
        "终极做多>1% 超级大肉概率": f"{long_big:.2%}",
    }
    return stats


# ================= 3. 纯 Inside Bar 方向预测测试 =================
def inside_bar_direction_test(df, lookahead=3, require_ma7_cross=False):
    """
    纯 Inside Bar，测试方向预测准确性：
    - 信号定义：上一根 K 线为 Mother Bar，当前 K 为 Inside Bar
    - 方向定义：Inside Bar 之后第一根 K (i+1) 向上/向下突破 Mother High/Low
    - 正确与否：在 lookahead 根 K 之后的收盘价，相对于突破价方向是否正确
    """
    df = df.copy()
    # 7 周期简单均线
    df["ma_7"] = df["Close"].rolling(window=7).mean()
    df["is_inside"] = (df["High"] < df["High"].shift(1)) & (
        df["Low"] > df["Low"].shift(1)
    )

    signals = []
    n = len(df)

    # 从第 2 根开始，到倒数 lookahead+1 根结束
    for i in range(2, n - lookahead):
        if not df["is_inside"].iloc[i]:
            continue

        # 与 7MA 相关的特征
        ma7_val = df["ma_7"].iloc[i]
        ma7_prev = df["ma_7"].iloc[i - 1]
        ma7_cross = (
            (not np.isnan(ma7_val))
            and (df["Low"].iloc[i] <= ma7_val <= df["High"].iloc[i])
        )
        above_ma7_before = (
            (not np.isnan(ma7_prev)) and (df["Close"].iloc[i - 1] > ma7_prev)
        )

        # 可选过滤：仅保留“inside bar 高低区间包含 7MA”的情况
        if require_ma7_cross and not ma7_cross:
            continue

        mother_high = df["High"].iloc[i - 1]
        mother_low = df["Low"].iloc[i - 1]

        # 突破发生在 Inside 之后第一根 K
        bh = df["High"].iloc[i + 1]
        bl = df["Low"].iloc[i + 1]

        up_break = bh > mother_high
        down_break = bl < mother_low

        # 同时上下突破或都没突破，视为无效信号
        if up_break == down_break:
            continue

        direction = 1 if up_break else -1
        entry_price = mother_high if direction == 1 else mother_low

        future_close = df["Close"].iloc[i + lookahead]
        ret = (future_close - entry_price) / entry_price

        if direction == 1:
            correct = future_close > entry_price
        else:
            correct = future_close < entry_price

        signals.append(
            {
                "time": df.index[i],
                "direction": direction,
                "entry": entry_price,
                "future_close": future_close,
                "ret": ret,
                "correct": correct,
                "ma_7": ma7_val,
                "ma7_cross": ma7_cross,
                "above_ma7_before": above_ma7_before,
            }
        )

    if not signals:
        results = {
            "总信号数": 0,
            "看多信号数": 0,
            "看空信号数": 0,
            "总体准确率": "0.00%",
            "看多准确率": "0.00%",
            "看空准确率": "0.00%",
            "平均未来收益": "0.00%",
        }
        return df, results, []

    sig_df = pd.DataFrame(signals)
    total = len(sig_df)
    long_df = sig_df[sig_df["direction"] == 1]
    short_df = sig_df[sig_df["direction"] == -1]

    acc = sig_df["correct"].mean()
    long_acc = long_df["correct"].mean() if len(long_df) > 0 else 0.0
    short_acc = short_df["correct"].mean() if len(short_df) > 0 else 0.0
    # 标的本身的平均变动（不考虑方向）
    avg_ret = sig_df["ret"].mean()
    # 按预测方向持仓的平均收益（方向 * 变动）
    sig_df["ret_dir"] = sig_df["direction"] * sig_df["ret"]
    avg_ret_dir = sig_df["ret_dir"].mean()

    # 与 7MA 的关系（按入场价相对 7MA 分组）
    sig_df["above_ma7"] = sig_df["entry"] > sig_df["ma_7"]
    long_above = sig_df[(sig_df["direction"] == 1) & (sig_df["above_ma7"])]
    long_below = sig_df[(sig_df["direction"] == 1) & (~sig_df["above_ma7"])]
    short_below = sig_df[(sig_df["direction"] == -1) & (~sig_df["above_ma7"])]
    short_above = sig_df[(sig_df["direction"] == -1) & (sig_df["above_ma7"])]

    def _acc(df_subset):
        return df_subset["correct"].mean() if len(df_subset) > 0 else 0.0

    long_above_acc = _acc(long_above)
    long_below_acc = _acc(long_below)
    short_below_acc = _acc(short_below)
    short_above_acc = _acc(short_above)

    # 不同收益阈值的统计（按方向收益 ret_dir）
    thresholds = [0.001, 0.002, 0.003, 0.005, 0.01]  # 0.1%, 0.2%, 0.3%, 0.5%, 1%
    long_dir_ret = sig_df[sig_df["direction"] == 1]["ret_dir"]
    short_dir_ret = sig_df[sig_df["direction"] == -1]["ret_dir"]

    results = {
        "总信号数": total,
        "看多信号数": len(long_df),
        "看空信号数": len(short_df),
        "总体准确率": f"{acc:.2%}",
        "看多准确率": f"{long_acc:.2%}",
        "看空准确率": f"{short_acc:.2%}",
        "平均未来收益(不看方向)": f"{avg_ret:.2%}",
        "按预测方向的平均收益": f"{avg_ret_dir:.2%}",
        "多单在MA7上方准确率": f"{long_above_acc:.2%}",
        "多单在MA7下方准确率": f"{long_below_acc:.2%}",
        "空单在MA7下方准确率": f"{short_below_acc:.2%}",
        "空单在MA7上方准确率": f"{short_above_acc:.2%}",
        "预测窗口": f"{lookahead} 根K线后收盘",
    }

    # 为不同收益阈值添加统计结果
    for th in thresholds:
        label = f"{th*100:.1f}%"
        if len(long_dir_ret) > 0:
            cnt_l = (long_dir_ret > th).sum()
            ratio_l = cnt_l / len(long_dir_ret)
            results[f"多单收益>{label} 次数"] = cnt_l
            results[f"多单收益>{label} 占比"] = f"{ratio_l:.2%}"
        if len(short_dir_ret) > 0:
            cnt_s = (short_dir_ret > th).sum()
            ratio_s = cnt_s / len(short_dir_ret)
            results[f"空单收益>{label} 次数"] = cnt_s
            results[f"空单收益>{label} 占比"] = f"{ratio_s:.2%}"

    # 返回前 10 条样本，方便人工查看
    samples = sig_df.head(10).to_dict(orient="records")
    return df, results, samples


# ================= 4. 固定止盈 0.3% 策略（保留，但主程序不再使用） =================
def strategy_inside_bar_scalping(
    df,
    initial_capital=10000,
    risk_per_trade=0.02,
    trend_ema_period=144,  # 使用较稳的均线
    adx_threshold=25,  # 趋势强度过滤
    target_pct=0.003,  # 单侧风险约 0.3%（以价格为基准）
    reward_r_multiple=1.0,  # 默认盈亏比 1:1
):
    df = df.copy()

    # 指标计算
    df["ema_trend"] = df["Close"].ewm(span=trend_ema_period, adjust=False).mean()
    df["ATR"] = calculate_atr(df)
    df["ADX"] = calculate_adx(df)
    df["is_inside"] = (df["High"] < df["High"].shift(1)) & (
        df["Low"] > df["Low"].shift(1)
    )

    capital = initial_capital
    position = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    unit_size = 0.0

    trades = []
    equity_curve = [initial_capital] * len(df)
    start_idx = max(trend_ema_period, 50) + 1

    for i in range(start_idx, len(df)):
        curr_idx = df.index[i]
        curr_open = df["Open"].iloc[i]
        curr_high = df["High"].iloc[i]
        curr_low = df["Low"].iloc[i]

        # 历史数据
        prev_is_inside = df["is_inside"].iloc[i - 1]
        mother_high = df["High"].iloc[i - 2]
        mother_low = df["Low"].iloc[i - 2]
        mother_close = df["Close"].iloc[i - 2]

        trend_val = df["ema_trend"].iloc[i - 1]
        adx_val = df["ADX"].iloc[i - 1]
        atr_val = df["ATR"].iloc[i - 1]

        # === 持仓管理 (固定百分比止损 + R:R 止盈) ===
        if position != 0:
            pnl = 0
            exit_type = ""

            if position == 1:  # 多单
                if curr_low <= stop_loss:
                    exit_price = min(curr_open, stop_loss)
                    pnl = (exit_price - entry_price) * unit_size
                    exit_type = "SL"
                    position = 0
                elif curr_high >= take_profit:
                    # 注意：如果开盘价直接跳空超过止盈位，按开盘价止盈（赚更多）
                    # 否则按设定止盈位离场
                    exit_price = max(curr_open, take_profit)
                    pnl = (exit_price - entry_price) * unit_size
                    exit_type = "TP(Scalp)"
                    position = 0

            elif position == -1:  # 空单
                if curr_high >= stop_loss:
                    exit_price = max(curr_open, stop_loss)
                    pnl = (entry_price - exit_price) * unit_size
                    exit_type = "SL"
                    position = 0
                elif curr_low <= take_profit:
                    exit_price = min(curr_open, take_profit)
                    pnl = (entry_price - exit_price) * unit_size
                    exit_type = "TP(Scalp)"
                    position = 0

            if position == 0:
                capital += pnl
                trades.append(
                    {
                        "time": curr_idx,
                        "type": exit_type,
                        "pnl": pnl,
                        "balance": capital,
                    }
                )

        # === 开仓信号 ===
        if position == 0 and prev_is_inside and (adx_val > adx_threshold):
            risk_dist = mother_high - mother_low

            # ATR 过滤 (太小的不要)
            if risk_dist > 0.2 * atr_val:

                # --- 做多 ---
                if (mother_close > trend_val) and (curr_high > mother_high):
                    entry_price = max(curr_open, mother_high)
                    # 固定百分比止损（约 0.3%）
                    stop_loss = entry_price * (1 - target_pct)
                    real_risk = entry_price - stop_loss  # ≈ entry_price * target_pct

                    if real_risk > 0:
                        # 1.5R 止盈
                        profit_dist = real_risk * reward_r_multiple
                        take_profit = entry_price + profit_dist

                        risk_amt = capital * risk_per_trade
                        unit_size = risk_amt / real_risk
                        position = 1
                        trades.append(
                            {
                                "time": curr_idx,
                                "type": "BUY_SCALP",
                                "price": entry_price,
                                "sl": stop_loss,
                                "tp": take_profit,
                            }
                        )

                # --- 做空 ---
                elif (mother_close < trend_val) and (curr_low < mother_low):
                    entry_price = min(curr_open, mother_low)
                    # 固定百分比止损（约 0.3%）
                    stop_loss = entry_price * (1 + target_pct)
                    real_risk = stop_loss - entry_price  # ≈ entry_price * target_pct

                    if real_risk > 0:
                        # 1.5R 止盈
                        profit_dist = real_risk * reward_r_multiple
                        take_profit = entry_price - profit_dist

                        risk_amt = capital * risk_per_trade
                        unit_size = risk_amt / real_risk
                        position = -1
                        trades.append(
                            {
                                "time": curr_idx,
                                "type": "SELL_SCALP",
                                "price": entry_price,
                                "sl": stop_loss,
                                "tp": take_profit,
                            }
                        )

        # 记录资金
        if position != 0:
            curr_price = df["Close"].iloc[i]
            unrealized = (
                (curr_price - entry_price) * unit_size
                if position == 1
                else (entry_price - curr_price) * unit_size
            )
            equity_curve[i] = capital + unrealized
        else:
            equity_curve[i] = capital

    df["equity"] = equity_curve

    total_return = (capital / initial_capital) - 1
    pnl_list = [t["pnl"] for t in trades if "pnl" in t]
    wins = [p for p in pnl_list if p > 0]
    win_rate = len(wins) / len(pnl_list) if len(pnl_list) > 0 else 0

    results = {
        "总回报": f"{total_return:.2%}",
        "最终资金": f"${capital:.2f}",
        "交易次数": len(pnl_list),
        "胜率": f"{win_rate:.2%}",
        "模式": f"R:R≈{reward_r_multiple}:1, 风险约 {target_pct*100:.2f}%",
    }

    return df, results, trades


# ================= 5. 主程序：纯 Inside Bar 方向预测 =================
if __name__ == "__main__":
    symbol = "ETHUSDT"
    interval = "15m"
    start_str = "2025-01-01"
    end_str = "2025-12-31"

    lookahead = 1  # 默认观察 Inside Bar 之后第 1 根 K 的方向

    print("🚀 启动：纯 Inside Bar 方向预测测试")

    df = fetch_binance_klines(symbol, interval, start_str, end_str)

    if not df.empty and len(df) > 100:
        # 只使用：inside bar 且蜡烛高低区间包含 7MA 的信号
        df_ib, results_ma7, samples_ma7 = inside_bar_direction_test(
            df, lookahead=lookahead, require_ma7_cross=True
        )

        print("\n" + "=" * 40)
        print("📊 Inside Bar 方向预测结果（仅十字星在7MA上）")
        print("=" * 40)
        for k, v in results_ma7.items():
            print(f"{k}: {v}")

        # 加入日线 / 周线 RSI，并统计十字星在 7MA 上下穿 + RSI 过滤的终极多空信号
        df_with_rsi = add_multi_tf_rsi(df_ib)
        ultimate_stats = compute_ultimate_stats(df_with_rsi, lookahead=lookahead)

        print("\n" + "=" * 40)
        print("📊 终极十字星信号（周线 RSI 过滤）")
        print("=" * 40)
        for k, v in ultimate_stats.items():
            print(f"{k}: {v}")

        print("\n📝 十字星在7MA上样本 (前 5 条):")
        for s in samples_ma7[:5]:
            dir_str = "UP" if s["direction"] == 1 else "DOWN"
            flag = "✅" if s["correct"] else "❌"
            print(
                f"[{s['time']}] 方向: {dir_str}, "
                f"入场: {s['entry']:.2f}, 未来收盘: {s['future_close']:.2f}, "
                f"收益: {s['ret']:.2%} {flag}"
            )
    else:
        print("❌ 数据不足")
