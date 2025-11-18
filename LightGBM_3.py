import numpy as np
# 兼容性修复：部分 pandas_ta 版本从 numpy 导入 NaN
# 在新版本 numpy 中没有导出 NaN 符号，这里手动设置别名以避免 ImportError
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # noqa: N816 (保持与外部库兼容的大小写)
import pandas as pd
import requests
import time
import logging
import os
import joblib
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from scipy.signal import find_peaks

# <<< 新增导入 >>>
import numba

# --- 0. 设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 🚀 全局配置 ---
SYMBOL = "ETHUSDT"
INTERVAL = "30m"
DATA_START_DATE = "2017-01-01"
TRAIN_START = "2018-01-01"
VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = "2025-11-16"
LOOK_BACK = 60
# 使用步进抽样的方式降低展平维度，如步长为5则仅取每5根中的一根
LAG_STRIDE = 5  # 降维关键参数：5 -> 60窗口仅取12个滞后切片

# 可选：是否计算Hurst（默认关闭，避免高计算量与不稳定）
USE_HURST = False

# 可选：是否进行时间序列交叉验证（WFA/TS-CV），默认关闭以加快运行
ENABLE_TIME_SERIES_CV = False

# 可选：使用考虑交易成本与滑点的阈值选择（在验证集上最大化净收益）
ENABLE_COST_AWARE_THRESHOLD = True

# 可选：风控过滤器与持仓管理
ENABLE_RISK_FILTER = True
ADX_MIN = 15.0
ATR_NORM_MIN = 0.0012  # ATR_14 / Close
REQUIRE_TREND_CONFIRM = False  # 放宽：不强制MACD双周期共振
REQUIRE_PRICE_ABOVE_EMA_4H = False  # 放宽：允许回撤中的入场
ENFORCE_NO_OVERLAP = False  # 放宽：允许并行持仓
COOLDOWN_BARS = 2  # 轻度冷却，减少过密交易

# 固定止盈：涨到 target_return 直接止盈；止损按 max_drawdown_limit（默认做 1:1 或略微正向的盈亏比）
# 默认关闭 ATR 动态止盈/止损，先用简单、稳定的固定 TP/SL 结构
USE_ATR_BASED_EXITS = False
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.2

# 可选：概率期望为正才开仓（用近似期望：p*TP - (1-p)*SL - cost > 0）
REQUIRE_POSITIVE_EXPECTANCY = True

# 可选：阈值选择时要求验证集最少产生的交易数量，避免过拟合到极少数样本
MIN_VALIDATION_TRADES = 30

# 概率标定，提升p的可解释性（用于EV评估与阈值搜索/回测）
ENABLE_PROBA_CALIBRATION = True

# 每日Top-K筛选，限制每日交易为当日最高置信度的K笔
ENABLE_DAILY_TOP_K = True
DAILY_TOP_K = 2

# 每日最小成交数与兜底概率阈值（确保频率≈1–2 笔/天）
ENABLE_DAILY_MIN_TRADES = True
MIN_DAILY_TRADES = 1
MIN_DAILY_PROB_FLOOR = 0.35

# 破损保护：到达一定浮盈后将止损抬至保本，降低大亏比例
ENABLE_BREAK_EVEN = True
BE_TRIGGER_RET = 0.002   # 浮盈达到 +0.20% 时激活保本
BE_STOP_RET = 0.0        # 激活后止损抬到入场价（保本，未覆盖手续费/滑点）

# 回测相关：手续费与滑点设置（单边费率与单边滑点）
# 挂单 Maker 手续费（例如 0.02% -> 0.0002，可按实际费率调整）
FEE_RATE = 0.0002
SLIPPAGE_RATE = 0.0005  # 5 bps，若认为挂单几乎无滑点可进一步下调

# 账户层最大回撤监控阈值（例如 10%），仅用于回测期统计和告警，不强制停止交易
ACCOUNT_MAX_DRAWDOWN = 0.10
TREND_CONFIG = {
    # 目标：在一个相对合理的时间窗口内（约 6 小时）博取 0.75% 左右的收益，
    # 并允许更宽的回撤（约 1.2%），以便实际交易中能吃到更多“先跌后涨”的机会。
    "look_forward_steps": 12,     # 向前最多观察 12 根 30m K 线（约 6 小时）
    "target_return": 0.0075,      # 目标止盈 0.75%
    "max_drawdown_limit": 0.012,  # 最大容忍回撤 1.20%（TP:SL ≈ 1:1.6）
}
logger.info(
    f"训练目标盈利：{TREND_CONFIG['target_return']*100}%，最大回撤限制：{TREND_CONFIG['max_drawdown_limit']*100}%"
)

# --- 文件路径 ---
MODELS_DIR, DATA_DIR = "models_gbm2", "data"
MODEL_SAVE_PATH = os.path.join(
    MODELS_DIR, f"eth_model_high_precision_v4_{INTERVAL}.joblib"
)
SCALER_SAVE_PATH = os.path.join(
    MODELS_DIR, f"eth_scaler_high_precision_v4_{INTERVAL}.joblib"
)
FEATURE_COLUMNS_PATH = os.path.join(
    MODELS_DIR, f"feature_columns_high_precision_v4_{INTERVAL}.joblib"
)
FLATTENED_COLUMNS_PATH = os.path.join(
    MODELS_DIR, f"flattened_columns_high_precision_v4_{INTERVAL}.joblib"
)
CALIBRATOR_SAVE_PATH = os.path.join(
    MODELS_DIR, f"eth_calibrator_v4_{INTERVAL}.joblib"
)
DATA_CACHE_PATH = os.path.join(DATA_DIR, f"{SYMBOL.lower()}_{INTERVAL}_data.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# --- 数据获取与辅助函数 ---
def fetch_binance_klines(s, i, st, en=None, l=1000):
    # ... (此函数保持不变) ...
    url, cols = "https://api.binance.com/api/v3/klines", [
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
    sts, ets = int(pd.to_datetime(st).timestamp() * 1000), (
        int(pd.to_datetime(en).timestamp() * 1000) if en else int(time.time() * 1000)
    )
    all_d, retries, max_retries = [], 0, 5
    while sts < ets:
        try:
            r = requests.get(
                url,
                params={
                    "symbol": s.upper(),
                    "interval": i,
                    "startTime": sts,
                    "endTime": ets,
                    "limit": l,
                },
                timeout=15,
            )
            r.raise_for_status()
            d = r.json()
            if not d:
                break
            all_d.extend(d)
            sts = d[-1][0] + 1
            retries = 0
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries > max_retries:
                logger.error(f"获取数据失败超过最大重试次数: {e}")
                return pd.DataFrame()
            logger.warning(
                f"获取数据失败，正在重试 ({retries}/{max_retries})... Error: {e}"
            )
            time.sleep(retries * 2)
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d, columns=cols)[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


# <<< 修复版：Hurst函数（Numba加速，稳健取样，正确斜率缩放） >>>
@numba.njit(cache=True)
def compute_hurst_numba(ts):
    n = ts.shape[0]
    if n < 20:
        return 0.5

    # 稳健选择滞后：最多到 N/2，并限制采样点数量，避免大滞后样本过少导致不稳定
    max_lag = n // 2
    if max_lag < 3:
        return 0.5

    # 至多采样 ~25 个滞后点
    step = max(1, max_lag // 25)
    count = ((max_lag - 2) // step) + 1

    # 累积对数域的矩，用于线性回归（无需中间数组与过滤）
    valid = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0

    for i in range(count):
        lag = 2 + i * step
        if lag > max_lag:
            break
        # 差分
        m = 0.0
        ln = n - lag
        if ln <= 1:
            continue
        # 计算均值
        for k in range(ln):
            m += ts[lag + k] - ts[k]
        m /= ln
        # 计算方差（ddof=0）
        v = 0.0
        for k in range(ln):
            d = (ts[lag + k] - ts[k]) - m
            v += d * d
        v /= ln
        if v <= 0.0:
            continue
        tau = np.sqrt(v)
        x = np.log(lag)
        y = np.log(tau)
        valid += 1
        sum_x += x
        sum_y += y
        sum_x2 += x * x
        sum_xy += x * y

    if valid < 2:
        return 0.5

    mx = sum_x / valid
    my = sum_y / valid
    cov = (sum_xy / valid) - (mx * my)
    varx = (sum_x2 / valid) - (mx * mx)
    if varx <= 0.0:
        return 0.5
    slope = cov / varx  # Hurst 斜率（无需×2）
    # 夹紧范围，防溢出
    if slope < 0.0:
        slope = 0.0
    elif slope > 1.0:
        slope = 1.0
    return slope


def get_market_structure_features(df, order=5):
    # 删除未来信息泄露：find_peaks 需要左右两侧数据确认峰值
    # 这里使用“对称峰值确认后延迟order根”原则：
    # 先在全局上定位峰值，但将确认结果整体向后移动 order 根，
    # 保证在时刻t仅能看见 t-order 之前被确认的结构点。
    df = df.copy()
    high_peaks_idx, _ = find_peaks(
        df["High"].values, distance=order, prominence=max(df["High"].std() * 0.5, 1e-9)
    )
    low_peaks_idx, _ = find_peaks(
        (-df["Low"]).values, distance=order, prominence=max(df["Low"].std() * 0.5, 1e-9)
    )

    swing_high_raw = np.full(len(df), np.nan)
    swing_high_raw[high_peaks_idx] = df["High"].values[high_peaks_idx]
    swing_low_raw = np.full(len(df), np.nan)
    swing_low_raw[low_peaks_idx] = df["Low"].values[low_peaks_idx]

    # 将确认过的峰值整体后移 order 根，避免在t使用到t之后的数据
    df["swing_high_price"] = pd.Series(swing_high_raw, index=df.index).shift(order).ffill()
    df["swing_low_price"] = pd.Series(swing_low_raw, index=df.index).shift(order).ffill()

    df["is_uptrend"] = (
        (df["swing_high_price"] > df["swing_high_price"].shift(1))
        & (df["swing_low_price"] > df["swing_low_price"].shift(1))
    ).astype(int)
    df["is_downtrend"] = (
        (df["swing_high_price"] < df["swing_high_price"].shift(1))
        & (df["swing_low_price"] < df["swing_low_price"].shift(1))
    ).astype(int)
    df["market_structure"] = df["is_uptrend"] - df["is_downtrend"]
    return df[["market_structure"]]


def ema_series(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi_series(s: pd.Series, length: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # 边界处理
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi.rename(f"RSI_{length}")


def macd_df(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
            col_names: tuple | None = None) -> pd.DataFrame:
    ema_fast = ema_series(s, fast)
    ema_slow = ema_series(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    if col_names is None:
        cols = (f"MACD_{fast}_{slow}_{signal}", f"MACDh_{fast}_{slow}_{signal}", f"MACDs_{fast}_{slow}_{signal}")
    else:
        cols = col_names
    return pd.DataFrame({cols[0]: macd_line, cols[1]: hist, cols[2]: signal_line}, index=s.index)


def bbands_df(s: pd.Series, length: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    ma = s.rolling(window=length, min_periods=length).mean()
    sd = s.rolling(window=length, min_periods=length).std(ddof=0)
    lower = ma - std_mult * sd
    upper = ma + std_mult * sd
    return pd.DataFrame({
        f"BBL_{length}_{std_mult}": lower,
        f"BBM_{length}_{std_mult}": ma,
        f"BBU_{length}_{std_mult}": upper,
    }, index=s.index)


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return atr.rename(f"ATR_{length}")


def adx_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0.0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr)
    dx = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx.rename(f"ADX_{length}")


def obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = close.diff().fillna(0.0)
    sign = direction.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    obv = (sign * volume).cumsum()
    return obv.rename("OBV")


def feature_engineering(df):
    """整合所有特征计算 - V4.2 增加特征交叉"""
    df_copy = df.copy()
    logger.info("--- 开始计算特征 (V4.2 增强版 - 含特征交叉) ---")

    # 1. 基础指标（自实现）
    df_copy["RSI_14"] = rsi_series(df_copy["Close"], length=14)
    macd_cols = macd_df(df_copy["Close"], fast=12, slow=26, signal=9)
    df_copy = pd.concat([df_copy, macd_cols], axis=1)
    bb_cols = bbands_df(df_copy["Close"], length=20, std_mult=2.0)
    df_copy = pd.concat([df_copy, bb_cols], axis=1)
    df_copy["ADX_14"] = adx_series(df_copy["High"], df_copy["Low"], df_copy["Close"], length=14)
    df_copy["ATR_14"] = atr_series(df_copy["High"], df_copy["Low"], df_copy["Close"], length=14)
    df_copy["OBV"] = obv_series(df_copy["Close"], df_copy["Volume"]) 

    # 2. 市场结构与长周期趋势 (保持不变)
    market_structure_df = get_market_structure_features(df_copy)
    macd_long = macd_df(
        df_copy["Close"], fast=24, slow=52, signal=18,
        col_names=("MACD_long", "MACDh_long", "MACDs_long")
    )
    df_copy = pd.concat([df_copy, macd_long], axis=1)

    # 3. 市场状态与波动性（Hurst 默认关闭，避免不稳定与高计算量）
    if USE_HURST:
        logger.info("正在计算Hurst指数 (可能较慢)...")
        # 可选：在较低频率上计算更稳健的Hurst后再对齐
        # 这里仍提供原始实现的开关
        df_copy["hurst"] = (
            df_copy["Close"].rolling(window=100).apply(compute_hurst_numba, raw=True)
        )
        logger.info("Hurst指数计算完成。")
    else:
        df_copy["hurst"] = 0.5  # 关闭时使用0.5作为常量占位，避免噪声影响
    df_copy["volatility_log"] = (
        (np.log(df_copy["Close"] / df_copy["Close"].shift(1))).rolling(window=20).std()
    )

    # 4. "规则" 转化为 "特征" (保持不变)
    df_copy["macd_cross_signal"] = (
        df_copy["MACD_12_26_9"] > df_copy["MACDs_12_26_9"]
    ).astype(int)
    df_copy["macd_long_cross_signal"] = (
        df_copy["MACD_long"] > df_copy["MACDs_long"]
    ).astype(int)

    # 5. 多时间框架特征（修正4H泄露：仅使用已完成的4H周期）
    # 使用右闭合窗 + 右标签，确保时间戳代表“上一根已收盘的4H K线”
    close_4h = df_copy["Close"].resample("4h", label="right", closed="right").last()
    ema_4h = ema_series(close_4h, length=50)
    # 关键：整体后移一根4H，避免在15m的4H区间内看到当前未完成的4H数据
    ema_4h_shifted = ema_4h.shift(1)
    df_copy["ema_4h"] = ema_4h_shifted.reindex(df_copy.index, method="ffill")
    df_copy["price_above_ema_4h"] = (df_copy["Close"] > df_copy["ema_4h"]).astype(int)

    # 6. --- <<< 新增：特征交叉 (Feature Crossing) >>> ---
    logger.info("正在创建交叉特征...")
    # 示例1: 波动率与趋势强度的交互 (高ADX和高ATR可能意味着强力突破)
    df_copy["adx_x_atr_norm"] = (df_copy["ADX_14"] / 50) * (
        df_copy["ATR_14"] / df_copy["Close"]
    )
    # 示例2: RSI与市场状态的交互 (趋势市中的RSI vs 震荡市的RSI)
    df_copy["rsi_x_hurst"] = df_copy["RSI_14"] * df_copy["hurst"]
    # 示例3: 短期趋势与长周期趋势的确认 (两个MACD都看涨)
    df_copy["macd_cross_confirm"] = (
        df_copy["macd_cross_signal"] * df_copy["macd_long_cross_signal"]
    )
    logger.info("交叉特征创建完成。")

    # 7. 整合
    df_copy = pd.concat([df_copy, market_structure_df], axis=1)

    # 8. 选择特征列并进行清理
    feature_columns = [
        col
        for col in df_copy.columns
        if col
        not in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "swing_high_price",
            "swing_low_price",
            "ema_4h",
        ]
    ]

    # 附加低维、稳定的派生特征：收益率与多尺度动量
    df_copy["ret_1"] = df_copy["Close"].pct_change()
    df_copy["ret_4"] = df_copy["Close"].pct_change(4)
    df_copy["ret_16"] = df_copy["Close"].pct_change(16)
    df_copy["rsi_delta_1"] = df_copy["RSI_14"].diff(1)
    df_copy["macd_delta_1"] = df_copy["MACD_12_26_9"].diff(1)

    # 合并并清理
    all_features_df = df_copy[feature_columns + [
        "ret_1","ret_4","ret_16","rsi_delta_1","macd_delta_1"
    ]].replace([np.inf, -np.inf], np.nan).ffill()

    return all_features_df


# --- (其他函数如 create_trend_labels, train_and_validate 等都使用我们上一轮讨论的最新版本) ---
# ...
# The rest of the script (create_trend_labels, train_and_validate, run_backtest_and_evaluate, __main__)
# remains IDENTICAL to the one provided in the previous response ("给出修改后的完整版代码").
# You only need to replace the `compute_hurst` and `feature_engineering` functions
# and add `import numba` at the top.
#
# For completeness, I'll paste the rest of the script again.
#
def create_trend_labels(df, look_forward_steps, target_return, max_drawdown_limit):
    """
    标签 A 版本：只关注“未来是否有足够上行空间”，不在标签中强行约束回撤。

    定义：
      - label = 1 当且仅当：在未来 look_forward_steps 根 K 线内，
        最高价曾经触及或超过当前收盘价 * (1 + target_return)；
      - 否则 label = 0。

    说明：
      - max_drawdown_limit 参数在标签中不再使用，只作为策略/回测层 TP/SL 的风险控制；
      - 这样可以增加正样本数量，使模型专注于学习“后续有足够上行空间”的情形，
        回撤控制则在交易执行逻辑中通过止损/保本来完成。
    """
    df_copy = df.copy()
    df_copy["target_price"] = df_copy["Close"] * (1 + target_return)
    future_highs = (
        df_copy["High"]
        .rolling(window=look_forward_steps)
        .max()
        .shift(-look_forward_steps)
    )
    profit_reached = future_highs >= df_copy["target_price"]
    df_copy["label"] = profit_reached.astype(int)
    return df_copy


def create_flattened_sequences(data, labels, look_back=60, stride=1):
    """
    以步进方式展平序列，减少维度并降低冗余。
    data: np.ndarray [T, F]
    labels: np.ndarray [T]
    look_back: 历史窗口长度
    stride: 步长，>1 时仅采样稀疏历史切片，降维关键
    """
    X, y = [], []
    if stride < 1:
        stride = 1
    # 选择将 [i-look_back, i) 中按步长抽样
    for i in range(look_back, len(data)):
        window = data[i - look_back : i : stride, :]
        X.append(window.flatten())
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_and_validate(train_df, validation_df, look_back, trend_config):
    logger.info("--- 开始训练和验证流程 (V4) ---")
    X_train_full_df = feature_engineering(train_df).dropna()
    X_validation_full_df = feature_engineering(validation_df).dropna()
    train_labeled = create_trend_labels(train_df, **trend_config)
    validation_labeled = create_trend_labels(validation_df, **trend_config)
    y_train_full = train_labeled["label"].align(X_train_full_df, join="inner", axis=0)[
        0
    ]
    X_train_full_df = X_train_full_df.align(
        train_labeled["label"], join="inner", axis=0
    )[0]
    y_validation_full = validation_labeled["label"].align(
        X_validation_full_df, join="inner", axis=0
    )[0]
    X_validation_full_df = X_validation_full_df.align(
        validation_labeled["label"], join="inner", axis=0
    )[0]
    X_validation_full_df = X_validation_full_df[X_train_full_df.columns]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_full_df)
    X_validation_scaled = scaler.transform(X_validation_full_df)
    original_columns = X_train_full_df.columns
    # 基于步长的扁平化列名（例如 lag_59, lag_54, ..., lag_0）
    selected_lags = list(range(look_back - 1, -1, -LAG_STRIDE))
    flattened_columns = [
        f"{col}_lag_{lag}"
        for lag in selected_lags
        for col in original_columns
    ]
    joblib.dump(original_columns, FEATURE_COLUMNS_PATH)
    joblib.dump(flattened_columns, FLATTENED_COLUMNS_PATH)
    X_train_np, y_train = create_flattened_sequences(
        X_train_scaled, y_train_full.values, look_back, stride=LAG_STRIDE
    )
    X_validation_np, y_validation = create_flattened_sequences(
        X_validation_scaled, y_validation_full.values, look_back, stride=LAG_STRIDE
    )
    X_train_df = pd.DataFrame(X_train_np, columns=flattened_columns)
    X_validation_df = pd.DataFrame(X_validation_np, columns=flattened_columns)
    logger.info(f"训练样本: {len(X_train_df)}, 验证样本: {len(X_validation_df)}")
    train_label_counts = np.bincount(y_train)
    if train_label_counts.size < 2 or train_label_counts[1] == 0:
        logger.error("训练数据中没有正样本(label=1)，无法继续。")
        return None, None, None
    precision_focus_ratio = 0.3
    scale_pos_weight = (
        train_label_counts[0] / train_label_counts[1]
    ) * precision_focus_ratio
    logger.info(f"调整后的 scale_pos_weight (追求高胜率): {scale_pos_weight:.2f}")
    lgb_params = {
        "objective": "binary",
        "metric": "logloss",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 20,
        "max_depth": 5,
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1,
        "scale_pos_weight": scale_pos_weight,
        "colsample_bytree": 0.7,
        "subsample": 0.7,
        "reg_alpha": 0.1,
    }

    # 可选：时间序列交叉验证（WFA/TS-CV）以评估稳健性
    if ENABLE_TIME_SERIES_CV:
        logger.info("启动时间序列交叉验证 (TS-CV) 评估稳健性...")
        tscv = TimeSeriesSplit(n_splits=3)
        cv_precisions, cv_recalls = [], []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_df)):
            X_tr, y_tr = X_train_df.iloc[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train_df.iloc[va_idx], y_train[va_idx]

            cv_model = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": 500})
            cv_model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="logloss",
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            y_va_prob = cv_model.predict_proba(X_va)[:, 1]
            # 使用0.5阈值粗略评估
            y_va_pred = (y_va_prob > 0.5).astype(int)
            p = precision_score(y_va, y_va_pred, zero_division=0)
            r = recall_score(y_va, y_va_pred, zero_division=0)
            cv_precisions.append(p)
            cv_recalls.append(r)
            logger.info(f"TS-CV 折{fold+1}: Precision={p:.4f}, Recall={r:.4f}")
        logger.info(
            f"TS-CV 平均: Precision={np.mean(cv_precisions):.4f}, Recall={np.mean(cv_recalls):.4f}"
        )
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    logger.info("\n开始训练 LightGBM 模型 (V4)...")
    lgb_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_validation_df, y_validation)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    y_val_pred_probs = lgb_model.predict_proba(X_validation_df)[:, 1]
    if ENABLE_PROBA_CALIBRATION:
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(y_val_pred_probs, y_validation)
            y_val_pred_probs = calibrator.transform(y_val_pred_probs)
            joblib.dump(calibrator, CALIBRATOR_SAVE_PATH)
            logger.info("已完成概率标定(Isotonic)并保存校准器。")
        except Exception as e:
            logger.warning(f"概率标定失败，使用未标定概率。Error: {e}")

    # 统一的阈值变量与收益记录：
    # - 若成本感知搜索找到“验证集平均净收益 > 0”的阈值，则优先采用；
    # - 否则回退到基于精确率/F1 的阈值选择，避免强行锁定在亏损阈值上。
    best_threshold = 0.5
    best_avg_net = None

    if ENABLE_COST_AWARE_THRESHOLD:
        # 使用验证集基于净收益选择阈值
        logger.info("基于净收益选择最佳阈值（含手续费与滑点）...")
        # 构建与验证序列对齐的索引与价格序列
        val_feat_index = X_validation_full_df.index
        pred_index = val_feat_index[look_back : look_back + len(y_val_pred_probs)]
        v_close = validation_df["Close"].reindex(pred_index)
        v_high = validation_df["High"].reindex(pred_index)
        v_low = validation_df["Low"].reindex(pred_index)

        # 风控过滤器（仅在需要时启用）
        if ENABLE_RISK_FILTER:
            risk_mask = np.ones(len(pred_index), dtype=bool)
            try:
                adx_vals = X_validation_full_df["ADX_14"].reindex(pred_index).values
                atr_vals = X_validation_full_df["ATR_14"].reindex(pred_index).values
                macd_conf = X_validation_full_df["macd_cross_confirm"].reindex(pred_index).values.astype(bool)
                price_above = X_validation_full_df["price_above_ema_4h"].reindex(pred_index).values.astype(bool)
            except KeyError:
                # 兼容列缺失的情况
                adx_vals = np.full(len(pred_index), np.nan)
                atr_vals = np.full(len(pred_index), np.nan)
                macd_conf = np.zeros(len(pred_index), dtype=bool)
                price_above = np.zeros(len(pred_index), dtype=bool)

            if ADX_MIN is not None:
                risk_mask &= np.isfinite(adx_vals) & (adx_vals >= ADX_MIN)
            if ATR_NORM_MIN is not None:
                atr_norm = np.divide(atr_vals, v_close.values, out=np.zeros_like(atr_vals), where=np.isfinite(atr_vals) & np.isfinite(v_close.values) & (v_close.values > 0))
                risk_mask &= atr_norm >= ATR_NORM_MIN
            if REQUIRE_TREND_CONFIRM:
                risk_mask &= macd_conf
            if REQUIRE_PRICE_ABOVE_EMA_4H:
                risk_mask &= price_above
        else:
            risk_mask = np.ones(len(pred_index), dtype=bool)

        # 使用一组候选阈值（分位数）
        # 候选阈值更多集中在高分位，减少过多交易
        qs = np.concatenate(
            [
                np.linspace(0.35, 0.90, 24),
                np.linspace(0.91, 0.98, 8),
            ]
        )
        thresh_candidates = np.unique(np.quantile(y_val_pred_probs, qs))
        best_avg_net = -1e9
        look_forward = trend_config["look_forward_steps"]
        dd_limit = trend_config["max_drawdown_limit"]

        cost_per_trade = 2 * (FEE_RATE + SLIPPAGE_RATE)
        # 预计算每日索引
        day_index = pd.Index(pred_index).normalize()
        unique_days = np.unique(day_index.values)

        for th in thresh_candidates:
            sig_raw = (y_val_pred_probs > th)
            cand_mask = (sig_raw & risk_mask)

            # 每日Top-K + 每日最小成交数
            if ENABLE_DAILY_TOP_K or ENABLE_DAILY_MIN_TRADES:
                allow = np.zeros(len(pred_index), dtype=bool)
                for d in unique_days:
                    # 候选层级：cand(含EV)、prob+risk（不含EV）、prob-only、全量
                    day_mask_all = (day_index.values == d)
                    day_cand = np.where(day_mask_all & cand_mask)[0]
                    day_prob_risk = np.where(day_mask_all & sig_raw & risk_mask)[0]
                    day_prob_only = np.where(day_mask_all & sig_raw)[0]
                    day_all = np.where(day_mask_all)[0]

                    selected = []
                    # 先取 cand 中的Top-K
                    if day_cand.size > 0:
                        k = DAILY_TOP_K if ENABLE_DAILY_TOP_K else day_cand.size
                        k = min(k, day_cand.size)
                        topk = day_cand[np.argsort(y_val_pred_probs[day_cand])[-k:]]
                        selected.extend(topk.tolist())

                    # 若需要每日最小成交数，则按层级补足
                    need = 0
                    if ENABLE_DAILY_MIN_TRADES:
                        need = max(0, MIN_DAILY_TRADES - len(selected))
                    if need > 0 and day_prob_risk.size > 0:
                        # 去掉已选并补足
                        remain = np.setdiff1d(day_prob_risk, np.array(selected, dtype=int), assume_unique=False)
                        if remain.size > 0:
                            take = min(need, remain.size)
                            extra = remain[np.argsort(y_val_pred_probs[remain])[-take:]]
                            selected.extend(extra.tolist())
                            need = max(0, MIN_DAILY_TRADES - len(selected))
                    if need > 0 and day_prob_only.size > 0:
                        remain = np.setdiff1d(day_prob_only, np.array(selected, dtype=int), assume_unique=False)
                        if remain.size > 0:
                            # 仅在满足兜底概率阈值下补充
                            remain = remain[y_val_pred_probs[remain] >= MIN_DAILY_PROB_FLOOR]
                            if remain.size > 0:
                                take = min(need, remain.size)
                                extra = remain[np.argsort(y_val_pred_probs[remain])[-take:]]
                                selected.extend(extra.tolist())

                    # 落到allow
                    if len(selected) > 0:
                        allow[np.array(selected, dtype=int)] = True
                sig = allow.astype(int)
            else:
                sig = cand_mask.astype(int)
            pnl_list = []
            busy_until = -1
            for t, s in enumerate(sig):
                if s != 1:
                    continue
                if t <= busy_until:
                    continue
                if pd.isna(v_close.iloc[t]) or pd.isna(v_high.iloc[t]) or pd.isna(v_low.iloc[t]):
                    continue
                entry = v_close.iloc[t] * (1 + SLIPPAGE_RATE)
                # 动态止盈止损（受限于配置边界）
                if USE_ATR_BASED_EXITS and np.isfinite(atr_vals[t]) and v_close.iloc[t] > 0:
                    atr_n = float(atr_vals[t] / v_close.iloc[t])
                    tp_ret = max(trend_config["target_return"], TP_ATR_MULT * atr_n)
                    sl_ret = min(dd_limit, SL_ATR_MULT * atr_n)
                else:
                    tp_ret = trend_config["target_return"]
                    sl_ret = dd_limit

                # 期望为正过滤（可选）
                if REQUIRE_POSITIVE_EXPECTANCY:
                    p = float(y_val_pred_probs[t])
                    if (p * tp_ret - (1 - p) * sl_ret - cost_per_trade) <= 0:
                        continue

                target = entry * (1 + tp_ret)
                stop = entry * (1 - sl_ret)
                t_end = min(t + look_forward, len(pred_index) - 1)
                hit = False
                stopped = False
                be_active = False
                breakeven = False
                be_trigger = entry * (1 + BE_TRIGGER_RET) if ENABLE_BREAK_EVEN else 0.0
                be_stop = entry * (1 + BE_STOP_RET) if ENABLE_BREAK_EVEN else 0.0
                for j in range(t + 1, t_end + 1):
                    # 先检查是否触发止损，再检查是否命中目标
                    if v_low.iloc[j] <= stop:
                        exit_price = stop * (1 - SLIPPAGE_RATE)
                        stopped = True
                        break
                    if v_high.iloc[j] >= target:
                        exit_price = target * (1 - SLIPPAGE_RATE)
                        hit = True
                        break
                    if ENABLE_BREAK_EVEN and (not be_active) and (v_high.iloc[j] >= be_trigger):
                        be_active = True
                    if ENABLE_BREAK_EVEN and be_active and (v_low.iloc[j] <= be_stop):
                        exit_price = be_stop * (1 - SLIPPAGE_RATE)
                        breakeven = True
                        break
                if not hit and not stopped and not breakeven:
                    exit_price = v_close.iloc[t_end] * (1 - SLIPPAGE_RATE)
                    busy_until = t_end + COOLDOWN_BARS
                else:
                    busy_until = j + COOLDOWN_BARS
                gross = (exit_price - entry) / entry
                net = gross - 2 * FEE_RATE
                pnl_list.append(net)
            # 只有当交易数满足阈值时，才考虑该阈值，否则跳过以避免样本过少
            trade_cnt = len(pnl_list)
            if trade_cnt >= MIN_VALIDATION_TRADES and trade_cnt > 0:
                avg_net = float(np.mean(pnl_list))
                if avg_net > best_avg_net:
                    best_avg_net = avg_net
                    best_threshold = float(th)
        if best_avg_net == -1e9:
            # 如果所有候选阈值都未达到最小交易数要求，则放宽要求，选择平均净收益最高者
            logger.warning(
                f"未找到满足最小交易数({MIN_VALIDATION_TRADES})的阈值，放宽要求进行选择。"
            )
            best_avg_net = -1e9
            best_threshold = 0.5
            for th in thresh_candidates:
                sig_raw = y_val_pred_probs > th
                sig = (sig_raw & risk_mask).astype(int)
                pnl_list = []
                busy_until = -1
                for t, s in enumerate(sig):
                    if s != 1:
                        continue
                    if t <= busy_until:
                        continue
                    if pd.isna(v_close.iloc[t]) or pd.isna(v_high.iloc[t]) or pd.isna(
                        v_low.iloc[t]
                    ):
                        continue
                    entry = v_close.iloc[t] * (1 + SLIPPAGE_RATE)
                    if (
                        USE_ATR_BASED_EXITS
                        and np.isfinite(atr_vals[t])
                        and v_close.iloc[t] > 0
                    ):
                        atr_n = float(atr_vals[t] / v_close.iloc[t])
                        tp_ret = max(
                            trend_config["target_return"], TP_ATR_MULT * atr_n
                        )
                        sl_ret = min(dd_limit, SL_ATR_MULT * atr_n)
                    else:
                        tp_ret = trend_config["target_return"]
                        sl_ret = dd_limit
                    # 放宽阶段：不强制正期望过滤，避免无交易导致阈值退化
                    target = entry * (1 + tp_ret)
                    stop = entry * (1 - sl_ret)
                    t_end = min(t + look_forward, len(pred_index) - 1)
                    hit = False
                    stopped = False
                    for j in range(t + 1, t_end + 1):
                        if v_low.iloc[j] <= stop:
                            exit_price = stop * (1 - SLIPPAGE_RATE)
                            stopped = True
                            break
                        if v_high.iloc[j] >= target:
                            exit_price = target * (1 - SLIPPAGE_RATE)
                            hit = True
                            break
                    if not hit and not stopped:
                        exit_price = v_close.iloc[t_end] * (1 - SLIPPAGE_RATE)
                        busy_until = t_end + COOLDOWN_BARS
                    else:
                        busy_until = j + COOLDOWN_BARS
                    gross = (exit_price - entry) / entry
                    net = gross - 2 * FEE_RATE
                    pnl_list.append(net)
                if len(pnl_list) > 0:
                    avg_net = float(np.mean(pnl_list))
                    if avg_net > best_avg_net:
                        best_avg_net = avg_net
                        best_threshold = float(th)
        if best_avg_net == -1e9:
            logger.warning(
                "基于净收益选择未能找到有效阈值，将在回退路径中使用F1或默认阈值。"
            )
            best_avg_net = None
        elif best_avg_net <= 0:
            logger.warning(
                f"基于净收益搜索的最优阈值在验证集上的平均净收益仍为负: {best_avg_net:.5f}，将回退到基于精确率/F1 的阈值搜索。"
            )
        else:
            logger.info(
                f"基于净收益选择的最佳阈值: {best_threshold:.4f} (验证集平均净收益: {best_avg_net:.5f})"
            )

    # --- 统一的 F1/精确率 回退逻辑 ---
    # 触发条件：
    # 1) 没开启成本感知搜索；或
    # 2) 成本感知搜索找不到任何有效阈值；或
    # 3) 找到的最佳阈值在验证集上的平均净收益仍为负。
    if (not ENABLE_COST_AWARE_THRESHOLD) or (best_avg_net is None) or (
        best_avg_net is not None and best_avg_net <= 0
    ):
        MIN_PRECISION_TARGET = 0.55
        precisions, recalls, thresholds = precision_recall_curve(
            y_validation, y_val_pred_probs
        )
        valid_threshold_indices = np.where(precisions[:-1] >= MIN_PRECISION_TARGET)[0]
        if len(valid_threshold_indices) > 0:
            f1_scores = np.divide(
                2 * recalls * precisions,
                recalls + precisions,
                out=np.zeros_like(recalls),
                where=(recalls + precisions) != 0,
            )
            best_idx_within_valid = np.argmax(f1_scores[valid_threshold_indices])
            final_best_idx = valid_threshold_indices[best_idx_within_valid]
            best_threshold = thresholds[final_best_idx]
            logger.info(
                f"在满足胜率>{MIN_PRECISION_TARGET*100}%的条件下，找到最佳阈值: {best_threshold:.4f}"
            )
            logger.info(
                f"该阈值下的验证集表现: Precision={precisions[final_best_idx]:.4f}, Recall={recalls[final_best_idx]:.4f}"
            )
        else:
            logger.warning(
                f"未能找到任何阈值可以使验证集胜率达到 {MIN_PRECISION_TARGET*100}%。将使用最大化F1的阈值。"
            )
            f1_scores = np.divide(
                2 * recalls * precisions,
                recalls + precisions,
                out=np.zeros_like(recalls),
                where=(recalls + precisions) != 0,
            )
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = (
                thresholds[best_f1_idx] if len(thresholds) > best_f1_idx else 0.5
            )
            logger.info(f"在验证集上找到的最佳F1阈值: {best_threshold:.4f}")
    joblib.dump(lgb_model, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    logger.info(f"模型已保存到: {MODEL_SAVE_PATH}")
    return lgb_model, scaler, best_threshold


# --- 🚀 4. 升级版回测评估函数 (V4.3 - 增加回撤分布分析) ---
def run_backtest_and_evaluate(
    test_df, model, scaler, look_back, threshold, trend_config,
    fee_rate: float = FEE_RATE, slippage_rate: float = SLIPPAGE_RATE,
):
    logger.info(
        "\n" + "=" * 60 + "\n--- 开始在测试集上进行严格的回测评估 (V4) ---\n" + "=" * 60
    )
    original_columns = joblib.load(FEATURE_COLUMNS_PATH)
    flattened_columns = joblib.load(FLATTENED_COLUMNS_PATH)
    test_features_df = feature_engineering(test_df).dropna()
    test_features_aligned = test_features_df.reindex(
        columns=original_columns, fill_value=0
    )
    test_scaled = scaler.transform(test_features_aligned)
    logger.info("逐根K线遍历测试集进行预测...")

    # 预取用于风控过滤与收益计算的序列（与特征对齐）
    idx_full = test_features_df.index
    close_arr = test_df["Close"].reindex(idx_full).values
    high_arr = test_df["High"].reindex(idx_full).values
    low_arr = test_df["Low"].reindex(idx_full).values
    # 可能缺少的列使用安全默认值
    adx_arr = test_features_df.get("ADX_14", pd.Series(np.nan, index=idx_full)).reindex(idx_full).values
    atr_arr = test_features_df.get("ATR_14", pd.Series(np.nan, index=idx_full)).reindex(idx_full).values
    macd_conf_arr = test_features_df.get("macd_cross_confirm", pd.Series(0, index=idx_full)).reindex(idx_full).astype(bool).values
    price_above_arr = test_features_df.get("price_above_ema_4h", pd.Series(0, index=idx_full)).reindex(idx_full).astype(bool).values
    atr_norm = np.divide(atr_arr, close_arr, out=np.zeros_like(atr_arr), where=np.isfinite(atr_arr) & np.isfinite(close_arr) & (close_arr > 0))

    cost_per_trade = 2 * (fee_rate + slippage_rate)
    # 载入校准器（若存在）
    calibrator = None
    if ENABLE_PROBA_CALIBRATION and os.path.exists(CALIBRATOR_SAVE_PATH):
        try:
            calibrator = joblib.load(CALIBRATOR_SAVE_PATH)
            logger.info("已加载概率校准器用于回测预测。")
        except Exception as e:
            logger.warning(f"加载概率校准器失败，使用未标定概率。Error: {e}")

    # 一次性计算全部概率与动态TP/SL
    probs = np.zeros(len(test_scaled))
    for i in tqdm(range(look_back, len(test_scaled))):
        input_sequence = test_scaled[i - look_back : i : LAG_STRIDE, :]
        input_flattened_np = input_sequence.flatten().reshape(1, -1)
        input_df = pd.DataFrame(input_flattened_np, columns=flattened_columns)
        p = model.predict_proba(input_df)[0][1]
        probs[i] = float(calibrator.transform([p])[0]) if calibrator is not None else float(p)

    # 构造基础掩码（风险过滤 + 概率阈值 + 期望为正）
    n = len(test_scaled)
    base_mask = np.zeros(n, dtype=bool)
    tp_arr = np.full(n, TREND_CONFIG["target_return"], dtype=float)
    sl_arr = np.full(n, TREND_CONFIG["max_drawdown_limit"], dtype=float)
    if USE_ATR_BASED_EXITS:
        with np.errstate(divide="ignore", invalid="ignore"):
            atrn = np.divide(atr_arr, close_arr, out=np.zeros_like(atr_arr), where=np.isfinite(atr_arr) & (close_arr > 0))
        tp_arr = np.maximum(tp_arr, TP_ATR_MULT * atrn)
        sl_arr = np.minimum(sl_arr, SL_ATR_MULT * atrn)

    # 风险过滤
    risk_ok = np.ones(n, dtype=bool)
    if ENABLE_RISK_FILTER:
        if ADX_MIN is not None:
            risk_ok &= np.isfinite(adx_arr) & (adx_arr >= ADX_MIN)
        if ATR_NORM_MIN is not None:
            risk_ok &= np.isfinite(atr_norm) & (atr_norm >= ATR_NORM_MIN)
        if REQUIRE_TREND_CONFIRM:
            risk_ok &= macd_conf_arr
        if REQUIRE_PRICE_ABOVE_EMA_4H:
            risk_ok &= price_above_arr

    # 概率阈值
    prob_ok = probs > threshold
    # 期望为正
    if REQUIRE_POSITIVE_EXPECTANCY:
        ev_ok = (probs * tp_arr - (1 - probs) * sl_arr - cost_per_trade) > 0
    else:
        ev_ok = np.ones(n, dtype=bool)
    cand_mask = prob_ok & risk_ok & ev_ok

    # 应用每日Top-K与每日最小成交数（按原始索引的日期）
    selected_mask = np.zeros(n, dtype=bool)
    if ENABLE_DAILY_TOP_K or ENABLE_DAILY_MIN_TRADES:
        idx = np.arange(n)
        day_index_full = pd.Index(idx_full)
        for day, grp in pd.Series(idx, index=day_index_full).groupby(day_index_full.normalize()):
            idxs = grp.values
            idxs = idxs[(idxs >= look_back)]
            day_cand = idxs[cand_mask[idxs]]
            day_prob_risk = idxs[prob_ok[idxs] & risk_ok[idxs]]
            day_prob_only = idxs[prob_ok[idxs]]

            selected = []
            # 先取 cand 中的Top-K
            if day_cand.size > 0:
                k = DAILY_TOP_K if ENABLE_DAILY_TOP_K else day_cand.size
                k = min(k, day_cand.size)
                topk = day_cand[np.argsort(probs[day_cand])[-k:]]
                selected.extend(topk.tolist())

            # 按每日最小成交数兜底
            need = 0
            if ENABLE_DAILY_MIN_TRADES:
                need = max(0, MIN_DAILY_TRADES - len(selected))
            if need > 0 and day_prob_risk.size > 0:
                remain = np.setdiff1d(day_prob_risk, np.array(selected, dtype=int), assume_unique=False)
                if remain.size > 0:
                    take = min(need, remain.size)
                    extra = remain[np.argsort(probs[remain])[-take:]]
                    selected.extend(extra.tolist())
                    need = max(0, MIN_DAILY_TRADES - len(selected))
            if need > 0 and day_prob_only.size > 0:
                remain = np.setdiff1d(day_prob_only, np.array(selected, dtype=int), assume_unique=False)
                if remain.size > 0:
                    remain = remain[probs[remain] >= MIN_DAILY_PROB_FLOOR]
                    if remain.size > 0:
                        take = min(need, remain.size)
                        extra = remain[np.argsort(probs[remain])[-take:]]
                        selected.extend(extra.tolist())

            if len(selected) > 0:
                selected_mask[np.array(selected, dtype=int)] = True
    else:
        selected_mask = cand_mask

    # 若筛选后无任何信号，逐步放宽：先取消EV过滤，再取消风险过滤，最后仅保留阈值
    if not np.any(selected_mask[look_back:]):
        logger.warning("无交易信号（包含EV与风险过滤、Top-K）。放宽EV过滤后重试...")
        cand_no_ev = prob_ok & risk_ok  # 去掉EV过滤
        selected_mask = np.zeros(n, dtype=bool)
        if ENABLE_DAILY_TOP_K:
            idx = np.arange(n)
            day_index_full = pd.Index(idx_full)
            for day, grp in pd.Series(idx, index=day_index_full).groupby(day_index_full.normalize()):
                idxs = grp.values
                idxs = idxs[(idxs >= look_back)]
                day_candidates = idxs[cand_no_ev[idxs]]
                if day_candidates.size == 0:
                    continue
                k = min(DAILY_TOP_K, day_candidates.size)
                topk = day_candidates[np.argsort(probs[day_candidates])[-k:]]
                selected_mask[topk] = True
        else:
            selected_mask = cand_no_ev

    if not np.any(selected_mask[look_back:]):
        logger.warning("无交易信号（取消EV后仍为空）。放宽风险过滤后重试...")
        cand_no_risk = prob_ok  # 仅阈值
        selected_mask = np.zeros(n, dtype=bool)
        if ENABLE_DAILY_TOP_K:
            idx = np.arange(n)
            day_index_full = pd.Index(idx_full)
            for day, grp in pd.Series(idx, index=day_index_full).groupby(day_index_full.normalize()):
                idxs = grp.values
                idxs = idxs[(idxs >= look_back)]
                day_candidates = idxs[cand_no_risk[idxs]]
                if day_candidates.size == 0:
                    continue
                k = min(DAILY_TOP_K, day_candidates.size)
                topk = day_candidates[np.argsort(probs[day_candidates])[-k:]]
                selected_mask[topk] = True
        else:
            selected_mask = cand_no_risk

    if not np.any(selected_mask[look_back:]):
        logger.warning("无交易信号（取消风险后仍为空）。最终回退到阈值筛选且不使用Top-K。")
        selected_mask = prob_ok

    # 兜底保护：若在以上所有放宽后仍然在回测区间内没有任何信号，
    # 则退化为“仅使用风险过滤 + 每日 Top-K”的规则，完全移除概率阈值与期望过滤，
    # 以便在极端标签/阈值设置下仍能观察策略的大致行为。
    if not np.any(selected_mask[look_back:]):
        logger.warning(
            "回测最终仍无任何交易信号，将使用仅基于风险过滤和每日Top-K的兜底规则（不使用概率阈值与期望过滤）。"
        )
        selected_mask = np.zeros(n, dtype=bool)
        if ENABLE_DAILY_TOP_K:
            idx_all = np.arange(n)
            day_index_full = pd.Index(idx_full)
            for day, grp in pd.Series(idx_all, index=day_index_full).groupby(
                day_index_full.normalize()
            ):
                idxs = grp.values
                idxs = idxs[(idxs >= look_back)]
                day_candidates = idxs[risk_ok[idxs]]
                if day_candidates.size == 0:
                    continue
                k = min(DAILY_TOP_K, day_candidates.size)
                topk = day_candidates[np.argsort(probs[day_candidates])[-k:]]
                selected_mask[topk] = True
        else:
            selected_mask = risk_ok.copy()

    # 顺序执行交易并统计，同时输出 final_signals（仅记录多头信号，用于分类评估）
    final_signals = []
    pnl_list = []
    trade_count = 0
    busy_until = -1
    # 账户层最大回撤监控：以权益曲线为基准，仅用于统计和告警，不强制停止交易
    equity = 1.0
    equity_peak = 1.0
    max_dd_overall = 0.0

    for i in range(look_back, n):
        long_signal = bool(selected_mask[i])

        # 简单做空逻辑：模型未给多头信号 + ADX 足够强则考虑开空
        short_signal = False
        if (not long_signal) and np.isfinite(adx_arr[i]) and (adx_arr[i] >= ADX_MIN):
            short_signal = True

        if (ENFORCE_NO_OVERLAP or COOLDOWN_BARS > 0) and i <= busy_until:
            # 持有中或冷却期，不开新仓，多头信号标记为 0
            final_signals.append(0)
            continue

        if long_signal:
            # 多头交易
            final_signals.append(1)
            trade_count += 1
            entry_price = close_arr[i] * (1 + slippage_rate)
            target_price = entry_price * (1 + tp_arr[i])
            stop_price = entry_price * (1 - sl_arr[i])
            be_trigger = entry_price * (1 + BE_TRIGGER_RET) if ENABLE_BREAK_EVEN else 0.0
            be_stop = entry_price * (1 + BE_STOP_RET) if ENABLE_BREAK_EVEN else 0.0
            look_forward = trend_config["look_forward_steps"]
            i_end = min(i + look_forward, n - 1)
            hit = False
            stopped = False
            be_active = False
            breakeven = False
            exit_j = i_end
            for j in range(i + 1, i_end + 1):
                # 1) 单笔 TP/SL/保本逻辑
                if low_arr[j] <= stop_price:
                    exit_price = stop_price * (1 - slippage_rate)
                    stopped = True
                    exit_j = j
                    break
                if high_arr[j] >= target_price:
                    exit_price = target_price * (1 - slippage_rate)
                    hit = True
                    exit_j = j
                    break
                if ENABLE_BREAK_EVEN and (not be_active) and (high_arr[j] >= be_trigger):
                    be_active = True
                if ENABLE_BREAK_EVEN and be_active and (low_arr[j] <= be_stop):
                    exit_price = be_stop * (1 - slippage_rate)
                    breakeven = True
                    exit_j = j
                    break

                # 2) 账户层最大回撤：以当前收盘价估算权益，若回撤超过阈值则强制平仓
                mark_price = close_arr[j]
                if mark_price > 0:
                    open_gross_ret = (mark_price - entry_price) / entry_price
                    open_net_ret = open_gross_ret - 2 * fee_rate
                    temp_equity = equity * (1.0 + open_net_ret)
                    temp_peak = max(equity_peak, temp_equity)
                    if temp_peak > 0:
                        temp_dd = 1.0 - temp_equity / temp_peak
                        if temp_dd >= ACCOUNT_MAX_DRAWDOWN:
                            exit_price = mark_price * (1 - slippage_rate)
                            stopped = True
                            exit_j = j
                            logger.warning(
                                f"账户层回撤达到 {temp_dd*100:.2f}% (阈值 {ACCOUNT_MAX_DRAWDOWN*100:.2f}%)，在 {idx_full[j]} 强制平仓。"
                            )
                            break

            if not hit and not stopped and not breakeven:
                exit_price = close_arr[i_end] * (1 - slippage_rate)
                exit_j = i_end

            gross_ret = (exit_price - entry_price) / entry_price
            net_ret = gross_ret - 2 * fee_rate
            pnl_list.append(net_ret)

            # 更新账户权益与最大回撤监控
            equity *= (1.0 + net_ret)
            if equity > equity_peak:
                equity_peak = equity
            if equity_peak > 0:
                cur_dd = 1.0 - equity / equity_peak
                if cur_dd > max_dd_overall:
                    max_dd_overall = cur_dd

            if ENFORCE_NO_OVERLAP or COOLDOWN_BARS > 0:
                busy_until = exit_j + COOLDOWN_BARS

        elif short_signal:
            # 空头交易：仅计入收益，不影响多头分类评估（final_signals 记为 0）
            final_signals.append(0)
            trade_count += 1
            entry_price = close_arr[i] * (1 - slippage_rate)  # 做空按卖出价入场
            target_price = entry_price * (1 - tp_arr[i])      # 价格下跌获利
            stop_price = entry_price * (1 + sl_arr[i])        # 上涨触发止损
            be_trigger = entry_price * (1 - BE_TRIGGER_RET) if ENABLE_BREAK_EVEN else 0.0
            be_stop = entry_price * (1 - BE_STOP_RET) if ENABLE_BREAK_EVEN else 0.0
            look_forward = trend_config["look_forward_steps"]
            i_end = min(i + look_forward, n - 1)
            hit = False
            stopped = False
            be_active = False
            breakeven = False
            exit_j = i_end
            for j in range(i + 1, i_end + 1):
                # 1) 单笔 TP/SL/保本逻辑（空头方向）
                if high_arr[j] >= stop_price:
                    exit_price = stop_price * (1 + slippage_rate)
                    stopped = True
                    exit_j = j
                    break
                if low_arr[j] <= target_price:
                    exit_price = target_price * (1 + slippage_rate)
                    hit = True
                    exit_j = j
                    break
                if ENABLE_BREAK_EVEN and (not be_active) and (low_arr[j] <= be_trigger):
                    be_active = True
                if ENABLE_BREAK_EVEN and be_active and (high_arr[j] >= be_stop):
                    exit_price = be_stop * (1 + slippage_rate)
                    breakeven = True
                    exit_j = j
                    break

                # 2) 账户层最大回撤：以当前收盘价估算权益，若回撤超过阈值则强制平仓（空头方向）
                mark_price = close_arr[j]
                if mark_price > 0:
                    open_gross_ret = (entry_price - mark_price) / entry_price
                    open_net_ret = open_gross_ret - 2 * fee_rate
                    temp_equity = equity * (1.0 + open_net_ret)
                    temp_peak = max(equity_peak, temp_equity)
                    if temp_peak > 0:
                        temp_dd = 1.0 - temp_equity / temp_peak
                        if temp_dd >= ACCOUNT_MAX_DRAWDOWN:
                            exit_price = mark_price * (1 + slippage_rate)
                            stopped = True
                            exit_j = j
                            logger.warning(
                                f"账户层回撤达到 {temp_dd*100:.2f}% (阈值 {ACCOUNT_MAX_DRAWDOWN*100:.2f}%)，在 {idx_full[j]} 强制平仓（空头头寸）。"
                            )
                            break

            if not hit and not stopped and not breakeven:
                exit_price = close_arr[i_end] * (1 + slippage_rate)
                exit_j = i_end

            gross_ret = (entry_price - exit_price) / entry_price
            net_ret = gross_ret - 2 * fee_rate
            pnl_list.append(net_ret)

            # 更新账户权益与最大回撤监控
            equity *= (1.0 + net_ret)
            if equity > equity_peak:
                equity_peak = equity
            if equity_peak > 0:
                cur_dd = 1.0 - equity / equity_peak
                if cur_dd > max_dd_overall:
                    max_dd_overall = cur_dd

            if ENFORCE_NO_OVERLAP or COOLDOWN_BARS > 0:
                busy_until = exit_j + COOLDOWN_BARS

        else:
            # 无交易
            final_signals.append(0)
    actual_labels_df = create_trend_labels(test_df, **trend_config).dropna()
    pred_index = test_features_df.index[look_back : look_back + len(final_signals)]
    pred_series = pd.Series(final_signals, index=pred_index)
    results_df = pd.DataFrame(actual_labels_df["label"]).join(
        pred_series.to_frame("final_signal"), how="inner"
    )
    if results_df.empty or np.sum(results_df["final_signal"]) == 0:
        logger.warning("回测期间没有产生任何交易信号，无法计算胜率。")
        return
    y_test_actual = results_df["label"].values
    y_pred_final = results_df["final_signal"].values

    winning_trades_drawdown = []
    winning_signals_df = results_df[
        (results_df["final_signal"] == 1) & (results_df["label"] == 1)
    ]
    if not winning_signals_df.empty:
        logger.info("正在计算盈利信号在盈利前的最大回撤...")
        look_forward = trend_config["look_forward_steps"]
        test_df_copy = test_df.copy()
        test_df_copy["future_min_low"] = (
            test_df_copy["Low"].rolling(window=look_forward).min().shift(-look_forward)
        )
        winning_trades_details = winning_signals_df.join(
            test_df_copy[["Close", "future_min_low"]], how="inner"
        )
        winning_trades_details["drawdown_pct"] = (
            (winning_trades_details["Close"] - winning_trades_details["future_min_low"])
            / winning_trades_details["Close"]
        ) * 100
        winning_trades_drawdown = winning_trades_details["drawdown_pct"].tolist()

    print("\n--- [客观] 测试集回测评估结果 (仅ML模型信号) ---")
    print(f"总回测K线数: {len(y_pred_final)}")
    print(f"发出看涨信号总次数 (交易频率): {np.sum(y_pred_final)}")
    # 增加zero_division=0以防止在没有信号时报错
    print(
        f"精确率 (胜率): {precision_score(y_test_actual, y_pred_final, zero_division=0):.4f}"
    )
    print(
        f"召回率 (盈利机会捕捉率): {recall_score(y_test_actual, y_pred_final, zero_division=0):.4f}"
    )
    print("\n混淆矩阵 (TN, FP / FN, TP):")
    print(confusion_matrix(y_test_actual, y_pred_final))

    if winning_trades_drawdown:
        drawdown_array = np.array(winning_trades_drawdown)
        avg_drawdown = np.mean(drawdown_array)

        print("\n--- [新指标] 盈利交易在盈利前的最大回撤分析 ---")
        print(f"盈利的交易总数: {len(drawdown_array)}")
        print(f"平均回撤: {avg_drawdown:.4f}%")
        print(f"最大回撤 (最差情况): {np.max(drawdown_array):.4f}%")
        print(f"最小回撤 (最好情况): {np.min(drawdown_array):.4f}%")

        # --- <<< 新增：回撤分布分析 >>> ---
        # 计算回撤大于/小于平均回撤的交易百分比
        count_above_avg = np.sum(drawdown_array > avg_drawdown)
        count_below_or_equal_avg = len(drawdown_array) - count_above_avg

        percent_above_avg = (count_above_avg / len(drawdown_array)) * 100
        percent_below_or_equal_avg = (
            count_below_or_equal_avg / len(drawdown_array)
        ) * 100

        print(f"\n  回撤分布 (与平均值 {avg_drawdown:.4f}% 相比):")
        print(
            f"    - 大于平均回撤的交易占比: {percent_above_avg:.2f}% ({count_above_avg} 笔)"
        )
        print(
            f"    - 小于或等于平均回撤的交易占比: {percent_below_or_equal_avg:.2f}% ({count_below_or_equal_avg} 笔)"
        )

        # 计算不同回撤水平下的交易占比
        print("\n  回撤水平分布:")
        for threshold_pct in [0.1, 0.25, 0.5, 0.75]:
            count_below_threshold = np.sum(drawdown_array <= threshold_pct)
            percent_below_threshold = (
                count_below_threshold / len(drawdown_array)
            ) * 100
            print(
                f"    - 回撤 <= {threshold_pct:.2f}% 的交易占比: {percent_below_threshold:.2f}%"
            )

    else:
        print("\n--- [新指标] 盈利交易在盈利前的最大回撤分析 ---")
        print("本次回测没有产生任何盈利的交易，无法计算相关指标。")

    # --- 收益统计（已在预测循环中累积 pnl_list，并考虑风控与不重叠） ---

    if trade_count > 0:
        pnl_arr = np.array(pnl_list)
        print("\n--- [考虑交易成本与滑点] 策略收益概览 ---")
        print(f"交易次数: {trade_count}")
        print(f"平均单笔净收益: {np.mean(pnl_arr):.5f}")
        print(f"胜率(净收益>0): {np.mean(pnl_arr > 0):.4f}")
        print(f"累计净收益: {np.sum(pnl_arr):.4f}")
        # 输出账户层最大回撤监控结果
        if max_dd_overall > 0:
            print(f"账户层最大回撤(基于回测权益曲线): {max_dd_overall*100:.2f}% (阈值: {ACCOUNT_MAX_DRAWDOWN*100:.2f}%)")
            if max_dd_overall >= ACCOUNT_MAX_DRAWDOWN:
                logger.warning(
                    f"账户权益最大回撤已达到 {max_dd_overall*100:.2f}%，超过监控阈值 {ACCOUNT_MAX_DRAWDOWN*100:.2f}%。"
                )
    else:
        print("\n--- [考虑交易成本与滑点] 策略收益概览 ---")
        print("无交易，无收益统计。")


if __name__ == "__main__":
    if os.path.exists(DATA_CACHE_PATH):
        logger.info(f"从缓存加载数据: {DATA_CACHE_PATH}")
        raw_df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        raw_df = fetch_binance_klines(
            s=SYMBOL, i=INTERVAL, st=DATA_START_DATE, en=TEST_END
        )
        if not raw_df.empty:
            raw_df.to_csv(DATA_CACHE_PATH)
    if raw_df.empty:
        logger.error("数据为空，程序退出。")
        exit()
    train_df = raw_df[(raw_df.index >= TRAIN_START) & (raw_df.index < VALIDATION_START)]
    validation_df = raw_df[
        (raw_df.index >= VALIDATION_START) & (raw_df.index < TEST_START)
    ]
    test_df = raw_df[(raw_df.index >= TEST_START) & (raw_df.index <= TEST_END)]
    logger.info(
        f"数据集划分完成: {len(train_df)} 训练, {len(validation_df)} 验证, {len(test_df)} 测试。"
    )
    trained_model, trained_scaler, best_threshold = train_and_validate(
        train_df, validation_df, LOOK_BACK, TREND_CONFIG
    )
    if trained_model and best_threshold is not None:
        run_backtest_and_evaluate(
            test_df,
            trained_model,
            trained_scaler,
            LOOK_BACK,
            best_threshold,
            TREND_CONFIG,
        )
    else:
        logger.error("模型训练失败，跳过回测。")
