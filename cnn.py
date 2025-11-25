# -*- coding-utf-8 -*-
"""
【v19.3 (Cost-Cover) - 成本覆盖版】
核心改动：
1. 目标变更：预测交易是否能覆盖手续费和滑点 (R > 0.15)。
2. 模型升级：弃用 LSTM，改用 XGBoost (更适合表格型小样本二分类)。
3. 特征增强：加入 RSI 和 ATR 波动率特征，展平时间窗口。
"""
import os
import random
import time
import logging
import warnings
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# ================= 锁死随机性 =================
SEED_VALUE = 42
os.environ["PYTHONHASHSEED"] = str(SEED_VALUE)
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

warnings.filterwarnings("ignore")

# ================= 配置参数 =================
SYMBOL = "ETHUSDT"
INTERVAL = "15m"

# 为了避免只在单一年份上过拟合，这里改成多年份数据，
# 后面按“历史年份训练、目标年份测试”的方式滚动评估。
START_DATE = "2024-01-01"
END_DATE = "2025-11-20"

# 趋势系统
MA_MICRO_PERIOD = 200
EMA_MACRO_PERIOD = 960
BB_STD = 4.0
BB_PERIOD = 20

# --- 【智能分流参数】 ---
LARGE_CANDLE_THRESHOLD = 1.5
RETRACEMENT_LEVEL = 0.5

# 追踪止损
INITIAL_SL_ATR = 1.2
TRAILING_ACTIVATION = 1.0
TRAILING_CALLBACK = 1.0

# 训练参数
LOOK_FORWARD_TRAIN = 48
# XGBoost 不需要太长的时间序列，我们取最近 3 根 K 线的特征展开
WINDOW_SIZE = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ================= 1. 数据获取 =================
def fetch_binance_klines(symbol, interval, start_str, end_str):
    filename = f"{symbol}_{interval}_{start_str}_{end_str}.csv"
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename, index_col="timestamp", parse_dates=True)
            return df.sort_index()
        except:
            pass

    logger.info(f"🌐 下载数据: {symbol}...")
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
            r = requests.get(url, params=params, timeout=5)
            data = r.json()
            if not data or not isinstance(data, list):
                break
            all_data.extend(data)
            start_ts = data[-1][0] + 1
            time.sleep(0.05)
        except Exception as e:
            logger.error(f"下载错误: {e}")
            time.sleep(1)

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
    )
    df = df.iloc[:, :6]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c])
    df = df.set_index("timestamp").sort_index()
    df.to_csv(filename)
    return df


# ================= 2. 特征工程 (增强版) =================
def process_data(df):
    logger.info("计算特征 (XGBoost 增强版)...")

    # 基础趋势
    df["MA_Micro"] = df["Close"].rolling(MA_MICRO_PERIOD).mean()
    df["EMA_Macro"] = ta.ema(df["Close"], length=EMA_MACRO_PERIOD)

    # 布林带
    bb = ta.bbands(df["Close"], length=BB_PERIOD, std=BB_STD)
    df["BB_Mid"] = bb[f"BBM_{BB_PERIOD}_{BB_STD}"]
    df["bb_width"] = bb[f"BBB_{BB_PERIOD}_{BB_STD}"]

    # 核心特征
    df["ADX"] = ta.adx(df["High"], df["Low"], df["Close"], length=14)["ADX_14"]
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # --- 新增特征 ---
    # 1. RSI (动量)
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # 2. 价格相对于均线的位置 (归一化)
    df["Price_vs_Micro"] = (df["Close"] - df["MA_Micro"]) / (df["MA_Micro"] + 1e-9)

    # 3. 相对成交量
    df["relative_volume"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

    # 4. K线实体大小
    df["body_size_norm"] = (df["Close"] - df["Open"]).abs() / (df["ATR"] + 1e-9)

    # 5. 波动率趋势 (ATR 短期/长期)
    df["ATR_Trend"] = df["ATR"] / (df["ATR"].rolling(50).mean() + 1e-9)

    df.dropna(inplace=True)

    # 信号逻辑 (保持不变)
    prev_close = df["Close"].shift(1)
    prev_mid = df["BB_Mid"].shift(1)
    curr_close = df["Close"]
    curr_mid = df["BB_Mid"]

    trend_bullish = (curr_close > df["MA_Micro"]) & (curr_close > df["EMA_Macro"])
    trend_bearish = (curr_close < df["MA_Micro"]) & (curr_close < df["EMA_Macro"])

    df["Rule_Direction"] = 0
    df.loc[
        (prev_close < prev_mid) & (curr_close > curr_mid) & trend_bullish,
        "Rule_Direction",
    ] = 1
    df.loc[
        (prev_close > prev_mid) & (curr_close < curr_mid) & trend_bearish,
        "Rule_Direction",
    ] = -1

    return df


# ================= 3. 智能交易模拟器 =================
def simulate_smart_trade(direction, open_p, close_p, atr_val, body_norm, highs, lows):
    is_climax = body_norm >= LARGE_CANDLE_THRESHOLD
    mode = "LIMIT" if is_climax else "MARKET"

    entry_price = close_p
    start_idx = 0

    if is_climax:
        target_price = (open_p + close_p) / 2
        filled = False
        check_range = min(3, len(highs))
        for i in range(check_range):
            if direction == 1:
                if lows[i] <= target_price:
                    entry_price = target_price
                    filled = True
                    start_idx = i
                    break
            else:
                if highs[i] >= target_price:
                    entry_price = target_price
                    filled = True
                    start_idx = i
                    break
        if not filled:
            return 0.0, mode

    # 止损设置
    if direction == 1:
        current_sl = entry_price - (atr_val * INITIAL_SL_ATR)
        highest_price = entry_price
    else:
        current_sl = entry_price + (atr_val * INITIAL_SL_ATR)
        lowest_price = entry_price

    for i in range(start_idx, len(highs)):
        h = highs[i]
        l = lows[i]

        if direction == 1:
            if l <= current_sl:
                return (current_sl - entry_price) / atr_val, mode
            if h > highest_price:
                highest_price = h
                if (highest_price - entry_price) / atr_val > TRAILING_ACTIVATION:
                    new_sl = highest_price - (atr_val * TRAILING_CALLBACK)
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if h >= current_sl:
                return (entry_price - current_sl) / atr_val, mode
            if l < lowest_price:
                lowest_price = l
                if (entry_price - lowest_price) / atr_val > TRAILING_ACTIVATION:
                    new_sl = lowest_price + (atr_val * TRAILING_CALLBACK)
                    if new_sl < current_sl:
                        current_sl = new_sl

    exit_p = (highs[-1] + lows[-1]) / 2
    if direction == 1:
        r = (exit_p - entry_price) / atr_val
    else:
        r = (entry_price - exit_p) / atr_val

    return r, mode


# ================= 4. 构建 XGBoost 数据集 =================
def create_dataset_xgb(df):
    # 特征列表
    feature_cols = [
        "ADX",
        "bb_width",
        "body_size_norm",
        "Price_vs_Micro",
        "relative_volume",
        "RSI",
        "ATR_Trend",
    ]

    # 提取原始数据
    raw_data = df[feature_cols].values

    X, y = [], []
    indices, directions = [], []
    real_r_list, mode_list = [], []

    # 转换为 Numpy 以加速
    opens = df["Open"].values
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    atrs = df["ATR"].values
    body_norms = df["body_size_norm"].values
    rule_dirs = df["Rule_Direction"].values

    # 遍历
    for i in range(WINDOW_SIZE, len(df) - LOOK_FORWARD_TRAIN):
        d = rule_dirs[i]
        if d == 0:
            continue

        # 模拟交易
        w_highs = highs[i + 1 : i + 1 + LOOK_FORWARD_TRAIN]
        w_lows = lows[i + 1 : i + 1 + LOOK_FORWARD_TRAIN]

        r_result, mode = simulate_smart_trade(
            d, opens[i], closes[i], atrs[i], body_norms[i], w_highs, w_lows
        )

        # 【核心修改】 标签定义：是否覆盖手续费和滑点
        # 假设手续费+滑点约 0.15% 波动，对应约 0.15 R
        label = 1 if r_result > 0.15 else 0

        # 构建特征向量：展平 WINDOW_SIZE 窗口
        # 例如：[t-2特征, t-1特征, t特征] 拼接成一个长向量
        window_feat = raw_data[i - WINDOW_SIZE + 1 : i + 1].flatten()

        X.append(window_feat)
        y.append(label)
        indices.append(i)
        directions.append(d)
        real_r_list.append(r_result)
        mode_list.append(mode)

    return (
        np.array(X),
        np.array(y),
        np.array(indices),
        np.array(directions),
        np.array(real_r_list),
        np.array(mode_list),
    )


# ================= 5. 主程序 =================
def run():
    df_raw = fetch_binance_klines(SYMBOL, INTERVAL, START_DATE, END_DATE)
    if df_raw.empty:
        print("数据下载失败")
        return

    df = process_data(df_raw)

    # 1. 构建 XGBoost 专用数据集
    X, y, indices, _, real_r, modes = create_dataset_xgb(df)

    logger.info(f"总样本数: {len(X)}")
    logger.info(f"正样本比例 (覆盖成本): {np.mean(y):.2%}")

    if len(X) < 50:
        print("样本不足")
        return

    # 将样本索引映射回时间，用于按年份滚动评估
    years = df.index[indices].year

    # 统计战报辅助函数
    from collections import Counter

    def print_stats(r_vals, m_vals, title):
        if len(r_vals) == 0:
            print(f"\n>>>>>> {title} (无数据) <<<<<<")
            return

        r_arr = np.array(r_vals)
        win_rate = np.mean(r_arr > 0)  # 绝对盈利率
        cover_rate = np.mean(r_arr > 0.15)  # 覆盖成本比例 (R > 0.15)

        print(f"\n>>>>>> {title} <<<<<<")
        print(
            f"  交易数: {len(r_arr)} | 胜率(>0): {win_rate:.1%} | 成本覆盖率(>0.15): {cover_rate:.1%}"
        )
        print(f"  总收益: {np.sum(r_arr):.2f} R | 平均期望: {np.mean(r_arr):.3f} R")

        m_counts = Counter(m_vals)
        print(f"  模式分布: Market={m_counts['MARKET']}, Limit={m_counts['LIMIT']}")

    print("\n" + "=" * 60)
    print(f"【战报 V19.3】 XGBoost 成本覆盖版（多年份滚动评估）")
    print(f"目标: 预测 R > 0.15 (覆盖手续费+滑点)")
    print("=" * 60)

    threshold = 0.5
    all_ai_r = []
    all_ai_modes = []

    for year in sorted(set(years)):
        # 使用“历史年份 < year”作为训练，当年 == year 作为测试
        train_mask = years < year
        test_mask = years == year

        if np.sum(train_mask) < 50 or np.sum(test_mask) == 0:
            continue

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        r_test = real_r[test_mask]
        mode_test = modes[test_mask]

        # 训练 XGBoost（每个年份独立一套模型）
        print(f"\n训练年份 < {year} 的模型，用于评估 {year} 年...")
        ratio = (len(y_train) - np.sum(y_train)) / (np.sum(y_train) + 1e-9)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=SEED_VALUE,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        print(f"{year} 年测试集准确率: {acc:.2%}")

        # 当年基准表现（规则盘全量信号）
        print_stats(r_test, mode_test, f"{year} 年基准 (全量信号)")

        # AI 优选：Prob > 阈值
        ai_mask = y_prob > threshold
        ai_r = r_test[ai_mask]
        ai_modes = mode_test[ai_mask]

        print_stats(ai_r, ai_modes, f"{year} 年 AI 优选 (Prob > {threshold})")
        print("-" * 60)

        all_ai_r.extend(list(ai_r))
        all_ai_modes.extend(list(ai_modes))

    # 汇总所有年份的 AI 优选表现
    if all_ai_r:
        print("\n" + "=" * 60)
        print("【多年份汇总】AI 优选信号表现 (所有测试年份合并)")
        print("=" * 60)
        print_stats(np.array(all_ai_r), np.array(all_ai_modes), "AI 优选 (全部年份)")

    # 特征重要性：最后一次训练的模型（最新年份的模型）
    # 由于窗口展平，构造对应的特征名
    print("\n[特征重要性 Top 5] （以最后一次训练的模型为例）")
    cols = [
        "ADX",
        "bb_width",
        "body_size_norm",
        "Price_vs_Micro",
        "relative_volume",
        "RSI",
        "ATR_Trend",
    ]
    all_feat_names = []
    for w in range(WINDOW_SIZE):
        for c in cols:
            all_feat_names.append(f"{c}_t-{WINDOW_SIZE-1-w}")

    imps = model.feature_importances_
    sorted_idx = np.argsort(imps)[::-1]
    for i in sorted_idx[:5]:
        print(f"  {all_feat_names[i]}: {imps[i]:.4f}")


if __name__ == "__main__":
    run()
