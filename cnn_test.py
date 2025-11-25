# -*- coding-utf-8 -*-
"""
【v34.0 (ADX+CHOP+三K结构版)】
核心思路：
1. 用 4 倍布林带中轨作为成本线。
2. K1 为“触及中轨的小实体K”，代表多空均衡。
3. K2 为“收盘越过中轨、不过长”的宣判K（价格突破成本线）。
4. ADX 过滤弱趋势，CHOP + 布林带宽度过滤极端震荡环境。
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

# 核心参数
BB_LENGTH = 14
BB_STD = 4.0  # 4倍布林带，中轨作为“成本线”
DOJI_RATIO = 0.2  # K1 小实体：Body <= DOJI_RATIO * Range（实体不超过全长 20%）
ADX_THRESHOLD = 22  # ADX 阈值：只保留较强趋势段
CHOP_LENGTH = 12  # CHOP 周期：衡量趋势 vs 震荡
CHOP_TREND_MAX = 58.0  # 越低越趋势化，这里允许到 62

# 三K结构额外参数
K2_EXHAUST_ATR = 0.8  # 不过耗：K2 收盘离中轨不超过 0.8 ATR
MID_FLAT_ATR = 0.8  # 中轨走平：中轨斜率相对 ATR 要较小
K2_MIN_ATR = 0.5  # K2 实体至少 0.5 ATR，过滤噪音K


# ================= 1. 数据读取 =================
def get_data_from_binance():
    filename = f"{SYMBOL}_{INTERVAL}_{START_DATE}_{END_DATE}_ma_pure.csv"
    if os.path.exists(filename):
        return pd.read_csv(filename, index_col="timestamp", parse_dates=True)
    print("请先确保有数据文件！(运行 v29 或 v30 下载)")
    return pd.DataFrame()


# ================= 2. 信号计算 =================
def calc_adx_signals(df):
    # 1. 基础指标：布林带（中轨+宽度）
    bb = ta.bbands(df["C"], length=BB_LENGTH, std=BB_STD)
    df["BB_Mid"] = bb[f"BBM_{BB_LENGTH}_{BB_STD}"]
    # 布林带宽度 & 成交量均线（用于过滤震荡与无量假信号）
    df["bb_width"] = bb[f"BBB_{BB_LENGTH}_{BB_STD}"]

    # 计算 ATR
    df["ATR14"] = ta.atr(df["H"], df["L"], df["C"], length=14)

    # 计算 ADX（趋势力度）
    adx = ta.adx(df["H"], df["L"], df["C"], length=14)
    df["ADX"] = adx["ADX_14"]

    # 计算 Choppiness Index (CHOP) —— 趋势 vs 震荡
    # TR = max(High-Low, |High-prevClose|, |Low-prevClose|)
    prev_close = df["C"].shift(1)
    tr1 = df["H"] - df["L"]
    tr2 = (df["H"] - prev_close).abs()
    tr3 = (df["L"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    tr_sum = tr.rolling(CHOP_LENGTH).sum()
    high_max = df["H"].rolling(CHOP_LENGTH).max()
    low_min = df["L"].rolling(CHOP_LENGTH).min()
    chop = 100 * np.log10(tr_sum / (high_max - low_min + 1e-9)) / np.log10(CHOP_LENGTH)
    df["CHOP"] = chop

    # 形态
    df["Body"] = (df["C"] - df["O"]).abs()
    df["Range"] = df["H"] - df["L"]

    # ========== 严格三K结构 ==========
    # 约定:
    # K1: i-2, 骑在中轨的小实体
    # K2: i-1, 实体明显放大、突破 K1 高/低
    # K3: i,   右侧入场K（Signal 打在 K3 上）

    O = df["O"]
    H = df["H"]
    L = df["L"]
    C = df["C"]
    mid = df["BB_Mid"]
    body = df["Body"]
    rng = df["Range"]
    atr = df["ATR14"]
    bb_width = df["bb_width"]

    # --- K1: 战场 ---
    O1 = O.shift(2)
    C1 = C.shift(2)
    H1 = H.shift(2)
    L1 = L.shift(2)
    mid1 = mid.shift(2)
    body1 = body.shift(2)
    range1 = rng.shift(2)

    # 影线触及中轨：只要高低价区间与中轨相交即可
    k1_touch_mid = (H1 >= mid1) & (L1 <= mid1)
    # 小实体
    k1_small_body = body1 <= (range1 * DOJI_RATIO)

    k1_valid = k1_touch_mid & k1_small_body

    # --- K2: 宣判 ---
    O2 = O.shift(1)
    C2 = C.shift(1)
    H2 = H.shift(1)
    L2 = L.shift(1)
    body2 = body.shift(1)
    mid2 = mid.shift(1)
    atr2 = atr.shift(1)

    # 不过耗: K2 收盘不能离中轨太远（避免已经拉太满的耗尽段）
    k2_not_exhausted = (atr2 > 0) & (np.abs(C2 - mid2) <= K2_EXHAUST_ATR * atr2)

    # 中轨走平: 中轨斜率相对 ATR 很小
    mid1 = mid.shift(2)
    mid_slope = (mid2 - mid1).abs()
    k2_mid_flat = (atr2 > 0) & (mid_slope <= MID_FLAT_ATR * atr2)

    # 布林带宽度过滤：仅剔除最极端收缩（震荡）环境（更宽松）
    bb_not_squeeze = bb_width > bb_width.rolling(100).quantile(0.15)

    adx2 = df["ADX"].shift(1)
    chop2 = df["CHOP"].shift(1)
    # 趋势环境过滤：ADX 衡量力度，CHOP 衡量趋势 vs 震荡，再叠加布林宽度
    trend_ok = (adx2 >= ADX_THRESHOLD) & (chop2 < CHOP_TREND_MAX) & bb_not_squeeze

    # K2 实体不能太小（严格 15m 模式下启用），避免小噪音 K
    if K2_MIN_ATR > 0:
        k2_body_not_tiny = (atr2 > 0) & (body2 >= K2_MIN_ATR * atr2)
    else:
        k2_body_not_tiny = np.ones_like(body2, dtype=bool)

    # 多头: K1 多空均衡之后，K2 收盘在中轨之上，且为阳线、不过长、成本线走平
    k2_bull = (
        (C2 > O2) & (C2 > mid2) & k2_not_exhausted & k2_mid_flat & k2_body_not_tiny
    )
    # 空头: K1 多空均衡之后，K2 收盘在中轨之下，且为阴线、不过长、成本线走平
    k2_bear = (
        (C2 < O2) & (C2 < mid2) & k2_not_exhausted & k2_mid_flat & k2_body_not_tiny
    )

    pattern_long = k1_valid & k2_bull & trend_ok
    pattern_short = k1_valid & k2_bear & trend_ok

    # --- K3: 右侧入场 ---
    # Signal 打在 K3 (当前 K) 上，后续用 K3 开盘价 / 回踩中轨作为入场价
    df["Signal"] = 0
    df.loc[pattern_long, "Signal"] = 1
    df.loc[pattern_short, "Signal"] = -1

    df.dropna(inplace=True)
    return df


# ================= 3. 验证 =================
def verify_strategy(df):
    signals = df[df["Signal"] != 0].copy()

    print("\n" + "=" * 60)
    print(f"【v34.0 ADX+CHOP+三K结构版】 ETHUSDT 15m")
    print(
        f"过滤: (ADX ≥ {ADX_THRESHOLD} & CHOP < {CHOP_TREND_MAX}) + "
        f"(K1触及中轨小实体 + K2 收盘越过中轨不过长 + 中轨走平)"
    )
    print(f"样本数: {len(signals)} 次")
    print("=" * 60)

    if len(signals) == 0:
        print("无信号。")
        return

    # 预测窗口: 1 根K (15min) 和 2 根K (30min)
    horizons = [1, 2]

    print(f"{'预测窗口':<12} | {'胜率 (Win Rate)':<20} | {'评价'}")
    print("-" * 60)

    for h in horizons:
        # 入场价: K3 开盘价；若当根有回踩中轨，则按中轨成交
        o_k3 = signals["O"]
        mid_k3 = signals["BB_Mid"]
        h_k3 = signals["H"]
        l_k3 = signals["L"]

        touch_mid = (h_k3 >= mid_k3) & (l_k3 <= mid_k3)

        # 多头: 取更低的价格（开盘或中轨）
        long_entry = np.where(
            (signals["Signal"] == 1) & touch_mid,
            mid_k3,
            o_k3,
        )
        # 空头: 取更高的价格（开盘或中轨）
        short_entry = np.where(
            (signals["Signal"] == -1) & touch_mid,
            mid_k3,
            o_k3,
        )

        entry_price = pd.Series(long_entry, index=signals.index)
        entry_price[signals["Signal"] == -1] = short_entry[signals["Signal"] == -1]

        future_close = df["C"].shift(-h)
        exit_price = future_close.loc[signals.index]

        pnl = (exit_price - entry_price) * signals["Signal"]
        win_rate = (pnl > 0).mean()

        status = (
            "🔥 强" if win_rate > 0.55 else ("✅ 稳" if win_rate > 0.50 else "❌ 弱")
        )
        time_str = f"{h*15} mins"

        print(f"{time_str:<12} | {win_rate:.2%}             | {status}")

    print("-" * 60)

    # 拆解
    l_wins = signals[signals["Signal"] == 1]
    s_wins = signals[signals["Signal"] == -1]

    h_target = 2
    print(f"[30分钟拆解]")
    if len(l_wins) > 0:
        l_acc = ((df["C"].shift(-h_target).loc[l_wins.index] - l_wins["C"]) > 0).mean()
        print(f"多头 (Trend+CHOP): {l_acc:.2%} (Count: {len(l_wins)})")
    if len(s_wins) > 0:
        s_acc = ((s_wins["C"] - df["C"].shift(-h_target).loc[s_wins.index]) > 0).mean()
        print(f"空头 (Trend+CHOP): {s_acc:.2%} (Count: {len(s_wins)})")


if __name__ == "__main__":
    try:
        df = pd.read_csv(
            f"{SYMBOL}_{INTERVAL}_{START_DATE}_{END_DATE}_ma_pure.csv",
            index_col="timestamp",
            parse_dates=True,
        )
        df_sig = calc_adx_signals(df)
        verify_strategy(df_sig)
    except Exception as e:
        print(f"错误: {e}")
