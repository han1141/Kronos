# -*- coding-utf-8 -*-
"""
【三K核心版 - ETHUSDT 15m】
只保留最核心的三K结构：
K1 触及中轨的小实体 + K2 越过中轨的有力实体，K3 右侧入场。
去掉 ADX / CHOP / ER / R² / 固定 TP/SL 等所有额外过滤和复杂回测逻辑。
"""
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ================= 配置参数（单一版本，去掉模式分支） =================
SYMBOL = "ETHUSDT"
INTERVAL = "15m"
START_DATE = "2023-01-01"
END_DATE = "2025-11-20"

# ================= 策略参数（三K核心结构 + R² + 1H 过滤） =================
# 布林带中轨作为成本线（15m）
BB_LENGTH = 15
BB_STD = 4.0  # 中轨作为成本线

# K1: 触及中轨的小实体
DOJI_RATIO = 0.10  # K1 小实体：Body <= DOJI_RATIO * Range

# K2: 收盘越过中轨即可（不过分强调实体长度，避免过度收缩样本）
K2_MIN_ATR = 0.0  # 设为 0 关闭实体长度过滤，只保留“越过中轨”的条件

# R²（决定系数）过滤震荡：只在价格对时间拟合度较高时使用三K结构
R2_LENGTH = 14
R2_THRESHOLD = 0.20

# 1 小时时间框架方向过滤：1H 收盘 vs 1H 均线
HTF_MA_LENGTH = 20  # 1H 均线长度

# 15m 小 R:R 固定止盈止损参数（只改出场，不改信号）
SCALP_TP_PCT = 0.0035  # +0.25% 止盈
SCALP_SL_PCT = 0.0060  # -0.15% 止损


# ================= 1. 数据读取 =================
def get_data_from_binance():
    filename = f"{SYMBOL}_{INTERVAL}_{START_DATE}_{END_DATE}_ma_pure.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col="timestamp", parse_dates=True)
    print("请先确保有数据文件！(运行 v29 或 v30 下载)")
    return pd.DataFrame()


# ================= 2. 信号计算 =================
def calc_simple_squeeze_signals(df):
    """
    三K核心结构 + R² + 1H 过滤:
    - K1: i-2, 影线触及中轨的小实体 K（多空均衡）
    - K2: i-1, 实体明显放大、且收盘越过中轨（宣判）
    - K3: i,   右侧入场 K（信号打在 K3 上）
    - R²: 使用 K2 时点的价格-时间线性拟合决定系数过滤震荡，仅在拟合度较高段落出手
    - 1H: 使用 K2 时点对应的 1 小时收盘相对于 1H 均线的方向过滤大级别趋势
    """
    # 1. 基础指标：布林带中轨 + ATR
    bb = ta.bbands(df["C"], length=BB_LENGTH, std=BB_STD)
    df["BB_Mid"] = bb[f"BBM_{BB_LENGTH}_{BB_STD}"]
    df["ATR14"] = ta.atr(df["H"], df["L"], df["C"], length=14)

    # R²：基于时间序列的线性回归判定系数（价格对时间的滚动相关系数平方）
    t_index = pd.Series(np.arange(len(df)), index=df.index)
    corr = df["C"].rolling(R2_LENGTH).corr(t_index)
    r2 = (corr**2).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["R2"] = r2

    # 1 小时方向过滤：基于 1H 收盘与其均线的关系
    df_1h = df["C"].resample("1H").last().to_frame("C_1H")
    df_1h["C_1H_MA"] = df_1h["C_1H"].rolling(HTF_MA_LENGTH).mean()
    # 映射回 15m：对每个 15m K，取最近一根已完成的 1H 收盘及其均线
    df["C_1H"] = df_1h["C_1H"].reindex(df.index, method="ffill")
    df["C_1H_MA"] = df_1h["C_1H_MA"].reindex(df.index, method="ffill")

    # K 线形态
    df["Body"] = (df["C"] - df["O"]).abs()
    df["Range"] = df["H"] - df["L"]

    O = df["O"]
    H = df["H"]
    L = df["L"]
    C = df["C"]
    mid = df["BB_Mid"]
    body = df["Body"]
    rng = df["Range"]
    atr = df["ATR14"]

    # --- K1: 触及中轨的小实体（i-2）---
    O1 = O.shift(2)
    C1 = C.shift(2)
    H1 = H.shift(2)
    L1 = L.shift(2)
    mid1 = mid.shift(2)
    body1 = body.shift(2)
    range1 = rng.shift(2)

    k1_touch_mid = (H1 >= mid1) & (L1 <= mid1)
    k1_small_body = body1 <= (range1 * DOJI_RATIO)
    k1_valid = k1_touch_mid & k1_small_body

    # --- K2: 收盘越过中轨的有力实体（i-1）---
    O2 = O.shift(1)
    C2 = C.shift(1)
    body2 = body.shift(1)
    mid2 = mid.shift(1)
    atr2 = atr.shift(1)

    # 多头: 阳线，收盘在中轨之上（不再要求实体 ≥ 某个 ATR 倍数）
    k2_bull = (C2 > O2) & (C2 > mid2)
    # 空头: 阴线，收盘在中轨之下
    k2_bear = (C2 < O2) & (C2 < mid2)

    # R² 过滤：使用 K2 时点的 R²（再向后一个 K 才入场，避免前瞻）
    r2_2 = df["R2"].shift(1)
    r2_trend = r2_2 >= R2_THRESHOLD

    # 1H 方向过滤：使用 K2 时点对应的 1H 收盘相对 1H 均线的方向
    c1h_2 = df["C_1H"].shift(1)
    ma1h_2 = df["C_1H_MA"].shift(1)
    htf_long = c1h_2 > ma1h_2
    htf_short = c1h_2 < ma1h_2

    pattern_long = k1_valid & k2_bull & r2_trend & htf_long
    pattern_short = k1_valid & k2_bear & r2_trend & htf_short

    # --- K3: 右侧入场 K（当前K，信号打在 K3 上）---
    df["Signal"] = 0
    df.loc[pattern_long, "Signal"] = 1
    df.loc[pattern_short, "Signal"] = -1

    df.dropna(inplace=True)
    return df


# ================= 3. 验证 =================
def verify_strategy(df):
    signals = df[df["Signal"] != 0].copy()

    print("\n" + "=" * 60)
    print(f"【三K核心版】 {SYMBOL} {INTERVAL}")
    print(
        f"结构: K1 触及布林中轨的小实体 (Body ≤ {DOJI_RATIO:.2f} * Range) + "
        f"K2 收盘越过中轨，K3 为右侧入场 K；"
        f"K2 时点叠加 R² 过滤震荡 (len={R2_LENGTH}, R² ≥ {R2_THRESHOLD:.2f}) + "
        f"1H 均线方向过滤 (1H 收盘 vs MA{HTF_MA_LENGTH})。"
    )
    print(f"样本数: {len(signals)} 次")
    print("=" * 60)

    if len(signals) == 0:
        print("无信号。")
        return

    # 为避免 Look-ahead Bias，所有交易在信号 K 收盘后，
    # 以「下一根 K 的开盘价」作为入场价。
    # 这里只统计 15 分钟（1 根 K）方向准确率。
    h = 1  # 1 根 = 15m

    # entry: 下一根开盘
    entry_price = df["O"].shift(-1).loc[signals.index]
    # exit: 从入场 K 开始，往后 h 根的收盘价
    exit_price = df["C"].shift(-(1 + h)).loc[signals.index]

    mask = (~entry_price.isna()) & (~exit_price.isna())
    if mask.sum() == 0:
        print("15 分钟方向判断: 无有效样本。")
        return

    ep = entry_price[mask]
    ex = exit_price[mask]
    sig = signals["Signal"][mask]

    pnl = (ex - ep) * sig
    win_rate = (pnl > 0).mean()

    status = "🔥 强" if win_rate > 0.55 else ("✅ 稳" if win_rate > 0.50 else "❌ 弱")
    print(f"15 mins      | 方向正确占比: {win_rate:.2%} | {status}")

    # ========== 15 分钟固定 TP/SL 微调回测（只改出场，不改信号） ==========
    print("-" * 60)
    scalp_setups = [
        ("紧凑_1", 0.0018, 0.0008),  # TP=0.18%, SL=0.08%
        ("紧凑_2", 0.0020, 0.0010),  # TP=0.20%, SL=0.10%
        ("原版",  SCALP_TP_PCT, SCALP_SL_PCT),  # 当前 0.25% / 0.15% 组合
    ]

    for name, tp_pct, sl_pct in scalp_setups:
        print(
            f"[15m 固定 TP/SL 回测 - {name}] TP=+{tp_pct*100:.2f}%, "
            f"SL=-{sl_pct*100:.2f}% (入场=下一根开盘)"
        )

        tp_count = sl_count = timeout_count = 0
        rets = []

        for ts, row in signals.iterrows():
            side_val = int(row["Signal"])

            try:
                idx = df.index.get_loc(ts)
            except KeyError:
                continue

            entry_idx = idx + 1
            if entry_idx >= len(df):
                continue  # 最后一根无法入场

            entry_ts = df.index[entry_idx]
            entry_price = df.at[entry_ts, "O"]

            # 只看入场这一根 15m K 内是否触及 TP/SL，否则按该根收盘价平仓
            high = df.at[entry_ts, "H"]
            low = df.at[entry_ts, "L"]
            close_price = df.at[entry_ts, "C"]

            if side_val == 1:
                tp_level = entry_price * (1 + tp_pct)
                sl_level = entry_price * (1 - sl_pct)
                hit_tp = high >= tp_level
                hit_sl = low <= sl_level
            else:
                tp_level = entry_price * (1 - tp_pct)
                sl_level = entry_price * (1 + sl_pct)
                hit_tp = low <= tp_level
                hit_sl = high >= sl_level

            if hit_tp and hit_sl:
                # 保守处理：同一根内 TP/SL 都触及，按止损计
                exit_price = sl_level
                sl_count += 1
            elif hit_tp:
                exit_price = tp_level
                tp_count += 1
            elif hit_sl:
                exit_price = sl_level
                sl_count += 1
            else:
                exit_price = close_price
                timeout_count += 1

            ret = (exit_price - entry_price) * side_val / entry_price
            rets.append(ret)

        n_trades = len(rets)
        if n_trades == 0:
            print("  固定 TP/SL 模式下无有效样本。")
        else:
            rets = np.array(rets)
            win_rate_s = (rets > 0).mean()
            avg_ret = rets.mean()
            std_ret = rets.std(ddof=1) if n_trades > 1 else 0.0

            print(
                f"  总样本: {n_trades} | TP 命中: {tp_count} | SL 命中: {sl_count} | "
                f"未触及 TP/SL: {timeout_count}"
            )
            print(
                f"  TP/SL 胜率: {win_rate_s:.2%} | "
                f"单笔平均收益: {avg_ret*100:.3f}% | "
                f"单笔收益标准差: {std_ret*100:.3f}%"
            )

            qs = np.percentile(rets, [0, 25, 50, 75, 100])
            print(
                "  收益分布 (单位: %): "
                f"min={qs[0]*100:.3f}, "
                f"25%={qs[1]*100:.3f}, "
                f"50%={qs[2]*100:.3f}, "
                f"75%={qs[3]*100:.3f}, "
                f"max={qs[4]*100:.3f}"
            )
        print("-" * 40)


if __name__ == "__main__":
    try:
        df = pd.read_csv(
            f"{SYMBOL}_{INTERVAL}_{START_DATE}_{END_DATE}_ma_pure.csv",
            index_col="timestamp",
            parse_dates=True,
        )
        df_sig = calc_simple_squeeze_signals(df)
        verify_strategy(df_sig)
    except Exception as e:
        print(f"错误: {e}")
