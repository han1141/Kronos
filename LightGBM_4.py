import numpy as np
import pandas as pd
import requests
import time
import os
import joblib
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from tqdm import tqdm
from scipy.signal import find_peaks

# Prefer the 'ta' library if available; avoid pandas_ta to prevent NumPy 2.0 import issues
try:
    import ta as ta_lib  # pip package 'ta'
    _HAVE_TA_LIB = True
except Exception:
    ta_lib = None
    _HAVE_TA_LIB = False

# 不再使用随机欠采样，避免破坏时序。使用类权重处理不平衡

# ================== CONFIG ==================
SYMBOL = "ETHUSDT"
INTERVAL = "15m"
DATA_START_DATE = "2017-01-01"
VALIDATION_START = "2024-01-01"
TEST_START = "2025-01-01"
TEST_END = "2025-11-06"
LOOK_BACK = 60
SWING_ORDER = 10
RANGE_ADX_PERIOD = 14
RANGE_ADX_THRESHOLD = 20.0
RANGE_CONFIRM_BARS = 3

# Scaling config
# - "global": single scaler fit on train, applied to val/test (simpler, may leak across train time)
# - "rolling": causal rolling min-max using only past W bars (recommended for deployment)
SCALING_MODE = "rolling"
ROLLING_SCALER_WINDOW = 500  # bars

# Dynamic ADX thresholding (optional). If enabled, use rolling quantile as threshold.
DYNAMIC_ADX = False
DYNAMIC_ADX_WINDOW_BARS = 96 * 90  # approx 90 days for 15m bars
DYNAMIC_ADX_QUANTILE = 0.30

# Labeling mode: use composite signals beyond ADX
LABEL_MODE = "composite"  # options: "adx", "composite"

# Composite label components and thresholds
COMPOSITE_USE_ADX = True
COMPOSITE_USE_CHOP = True
COMPOSITE_USE_DONCH_NO_BREAK = True
COMPOSITE_USE_BB_CENTER = True
COMPOSITE_MIN_VOTES = 2  # number of components that must agree on range

# CHOP settings
CHOP_PERIOD = 14
CHOP_DYNAMIC = True
CHOP_DYNAMIC_QUANTILE = 0.70
CHOP_STATIC_THRESHOLD = 61.8

# Donchian/BB settings
DONCHIAN_WINDOW = 20
BB_CENTER_TOLERANCE = 0.30  # fraction of band width from center

MODELS_DIR = "models"
DATA_DIR = "data"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

ARTIFACTS_PATH = f"{MODELS_DIR}/eth_trend_artifacts_{INTERVAL}.joblib"
DATA_CACHE_PATH = f"{DATA_DIR}/{SYMBOL.lower()}_{INTERVAL}_data.csv"


# ================== DATA FETCH ==================
def fetch_binance_klines(symbol, interval, start, end=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.to_datetime(start).timestamp() * 1000)
    end_ts = (
        int(pd.to_datetime(end).timestamp() * 1000) if end else int(time.time() * 1000)
    )
    data = []
    print("Fetching data from Binance...")
    pbar = tqdm()
    while start_ts < end_ts:
        try:
            r = requests.get(
                url,
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": start_ts,
                    "endTime": end_ts,
                    "limit": limit,
                },
            )
            r.raise_for_status()
            d = r.json()
            if not d:
                break
            data.extend(d)
            start_ts = d[-1][0] + 1
            pbar.update(1)
            time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    pbar.close()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_time",
            "qv",
            "trades",
            "tb_base",
            "tb_quote",
            "ignore",
        ],
    )
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


# ================== FEATURES ==================
def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-14)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f"RSI_{period}"
    return rsi


def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    macd.name = f"MACD_{fast}_{slow}_{signal}"
    macd_signal.name = f"MACDs_{fast}_{slow}_{signal}"
    macd_hist.name = f"MACDh_{fast}_{slow}_{signal}"
    return macd, macd_signal, macd_hist


def _compute_bbands(close: pd.Series, length: int = 20, std: float = 2.0):
    mavg = close.rolling(window=length).mean()
    sdev = close.rolling(window=length).std(ddof=0)
    upper = mavg + std * sdev
    lower = mavg - std * sdev
    mavg.name = f"BBM_{length}_{int(std)}"
    upper.name = f"BBU_{length}_{int(std)}"
    lower.name = f"BBL_{length}_{int(std)}"
    return mavg, upper, lower


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + 1e-14))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / (atr + 1e-14))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di).replace(0, np.nan)))
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    adx.name = f"ADX_{period}"
    return adx


def _rolling_vwap(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    pv = price * volume
    vwap = pv.rolling(window=window, min_periods=window).sum() / (
        volume.rolling(window=window, min_periods=window).sum() + 1e-14
    )
    vwap.name = f"VWAP_{window}"
    return vwap


def _compute_choppiness(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Choppiness Index (CHOP). Higher => more range-bound.
    CHOP = 100 * log10( sum(TR_n) / (HH_n - LL_n) ) / log10(n)
    Uses causal rolling sums with min_periods=period.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    denom = (hh - ll).replace(0, np.nan)
    chop = 100 * np.log10((tr_sum / (denom + 1e-14)).clip(lower=1e-14)) / np.log10(period)
    chop.name = f"CHOP_{period}"
    return chop


def feature_engineering(df):
    df = df.copy()

    if _HAVE_TA_LIB:
        # Use ta library implementations when available
        df["RSI_14"] = ta_lib.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd_ind = ta_lib.trend.MACD(df["Close"], window_fast=12, window_slow=26, window_sign=9)
        df["MACD_12_26_9"] = macd_ind.macd()
        df["MACDs_12_26_9"] = macd_ind.macd_signal()
        df["MACDh_12_26_9"] = macd_ind.macd_diff()
        bb = ta_lib.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BBL_20_2"] = bb.bollinger_lband()
        df["BBM_20_2"] = bb.bollinger_mavg()
        df["BBU_20_2"] = bb.bollinger_hband()
        adx_ind = ta_lib.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
        df["ADX_14"] = adx_ind.adx()
    else:
        # Fallback to pure pandas/numpy implementations
        df["RSI_14"] = _compute_rsi(df["Close"], 14)
        macd, macds, macdh = _compute_macd(df["Close"], 12, 26, 9)
        df[macd.name] = macd
        df[macds.name] = macds
        df[macdh.name] = macdh
        bbm, bbu, bbl = _compute_bbands(df["Close"], 20, 2)
        df[bbm.name] = bbm
        df[bbu.name] = bbu
        df[bbl.name] = bbl
        df["ADX_14"] = _compute_adx(df["High"], df["Low"], df["Close"], 14)

    # Additional hand-crafted features (causal, no future info)
    # - Multi-scale RSI captures different momentum horizons
    df["RSI_7"] = _compute_rsi(df["Close"], 7)
    df["RSI_28"] = _compute_rsi(df["Close"], 28)
    # - Choppiness Index
    df["CHOP_14"] = _compute_choppiness(df["High"], df["Low"], df["Close"], 14)
    df["CHOP_28"] = _compute_choppiness(df["High"], df["Low"], df["Close"], 28)
    # - Bollinger bandwidth (relative width)
    if {"BBU_20_2", "BBL_20_2", "BBM_20_2"}.issubset(df.columns):
        bb_width = (df["BBU_20_2"] - df["BBL_20_2"]) / (df["BBM_20_2"].abs() + 1e-14)
        df["BB_width_20_2"] = bb_width
    # - Volume features
    df["Vol_MA20"] = df["Volume"].rolling(window=20, min_periods=20).mean()
    df["Vol_MA_Ratio"] = df["Volume"] / (df["Vol_MA20"] + 1e-14)
    # - Donchian channel (20) position
    donch_hi = df["High"].rolling(window=20, min_periods=20).max()
    donch_lo = df["Low"].rolling(window=20, min_periods=20).min()
    df["Donch_pos_20"] = (df["Close"] - donch_lo) / ((donch_hi - donch_lo) + 1e-14)
    # - Rolling VWAP deviation (20)
    vwap20 = _rolling_vwap(df["Close"], df["Volume"], window=20)
    df["VWAP_dev_20"] = (df["Close"] - vwap20) / (vwap20.abs() + 1e-14)
    # - Realized volatility (multi-scale)
    ret = np.log(df["Close"]).diff()
    df["rv_5"] = ret.rolling(5, min_periods=5).std()
    df["rv_20"] = ret.rolling(20, min_periods=20).std()
    df["rv_60"] = ret.rolling(60, min_periods=60).std()

    df["volatility"] = (np.log(df["Close"] / df["Close"].shift(1))).rolling(20).std()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ================== LABEL: TREND STRUCTURE ==================
def create_trend_structure_labels(df, swing_order=5):
    df = df.copy()
    high_peaks_idx, _ = find_peaks(df["High"], distance=swing_order)
    low_peaks_idx, _ = find_peaks(-df["Low"], distance=swing_order)
    df["swing_high_point"] = np.nan
    df["swing_low_point"] = np.nan
    df.iloc[high_peaks_idx, df.columns.get_loc("swing_high_point")] = df.iloc[
        high_peaks_idx
    ]["High"]
    df.iloc[low_peaks_idx, df.columns.get_loc("swing_low_point")] = df.iloc[
        low_peaks_idx
    ]["Low"]
    df["swing_high"] = df["swing_high_point"].ffill()
    df["swing_low"] = df["swing_low_point"].ffill()
    df.drop(columns=["swing_high_point", "swing_low_point"], inplace=True)
    df.dropna(subset=["swing_high", "swing_low"], inplace=True)
    df["is_up"] = (
        (df["swing_high"] > df["swing_high"].shift(1))
        & (df["swing_low"] > df["swing_low"].shift(1))
    ).astype(int)
    df["is_down"] = (
        (df["swing_high"] < df["swing_high"].shift(1))
        & (df["swing_low"] < df["swing_low"].shift(1))
    ).astype(int)
    df["trend_label"] = (df["is_up"] - df["is_down"]) + 1
    df = df.dropna(subset=["trend_label"])
    return df["trend_label"]


# ================== LABEL: RANGE/CHOP ==================
def create_ranging_labels(
    df,
    adx_period: int = RANGE_ADX_PERIOD,
    adx_threshold: float = RANGE_ADX_THRESHOLD,
    confirm_bars: int = RANGE_CONFIRM_BARS,
):
    """
    Create binary labels indicating range-bound (1) vs trending (0) market.
    Label rules:
      - Compute ADX on OHLC using a causal EWM-based formulation (no future leak).
      - Base flag: ADX < threshold.
      - If dynamic thresholding is enabled, the threshold is the rolling quantile
        over the past window (bars), e.g., 30% quantile over ~90 days.
      - Confirmation: require 'confirm_bars' consecutive bars to satisfy the flag.
    Notes:
      - Labels reflect information available at time t.
      - For forecasting future regimes, shift labels as needed by the user.
    """
    df = df.copy()

    # 1) ADX-based range flag (low ADX -> range)
    adx = _compute_adx(df["High"], df["Low"], df["Close"], adx_period)
    if DYNAMIC_ADX:
        adx_thr = adx.rolling(
            window=DYNAMIC_ADX_WINDOW_BARS,
            min_periods=DYNAMIC_ADX_WINDOW_BARS,
        ).quantile(DYNAMIC_ADX_QUANTILE)
        adx_flag = adx < adx_thr
    else:
        adx_flag = adx < adx_threshold

    if LABEL_MODE == "adx":
        base_flag = adx_flag
    else:
        # 2) Choppiness high -> range
        chop = _compute_choppiness(df["High"], df["Low"], df["Close"], CHOP_PERIOD)
        if CHOP_DYNAMIC:
            chop_thr = chop.rolling(
                window=DYNAMIC_ADX_WINDOW_BARS,
                min_periods=DYNAMIC_ADX_WINDOW_BARS,
            ).quantile(CHOP_DYNAMIC_QUANTILE)
            chop_flag = chop > chop_thr
        else:
            chop_flag = chop > CHOP_STATIC_THRESHOLD

        # 3) No Donchian breakout at current bar -> range
        hh = df["High"].rolling(window=DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).max()
        ll = df["Low"].rolling(window=DONCHIAN_WINDOW, min_periods=DONCHIAN_WINDOW).min()
        up_break = df["Close"] >= hh
        dn_break = df["Close"] <= ll
        no_break_flag = ~(up_break | dn_break)

        # 4) Price near Bollinger center (20, 2) -> range
        mavg = df["Close"].rolling(window=20, min_periods=20).mean()
        sdev = df["Close"].rolling(window=20, min_periods=20).std(ddof=0)
        upper = mavg + 2.0 * sdev
        lower = mavg - 2.0 * sdev
        width = (upper - lower).replace(0, np.nan)
        bb_center_dist = (df["Close"] - mavg).abs() / (width + 1e-14)
        bb_center_flag = bb_center_dist <= BB_CENTER_TOLERANCE

        flags = []
        valids = []
        if COMPOSITE_USE_ADX:
            flags.append(adx_flag)
            valids.append(adx_flag.notna())
        if COMPOSITE_USE_CHOP:
            flags.append(chop_flag)
            valids.append(chop_flag.notna())
        if COMPOSITE_USE_DONCH_NO_BREAK:
            flags.append(no_break_flag)
            valids.append(no_break_flag.notna())
        if COMPOSITE_USE_BB_CENTER:
            flags.append(bb_center_flag)
            valids.append(bb_center_flag.notna())

        # Vote and require all components valid for a label at time t
        valid_mask = valids[0]
        for v in valids[1:]:
            valid_mask = valid_mask & v
        vote_count = None
        for f in flags:
            f_i = f.astype(int)
            vote_count = f_i if vote_count is None else (vote_count + f_i)
        vote_count = vote_count.astype(float)
        vote_count.loc[~valid_mask] = np.nan
        base_flag = vote_count >= COMPOSITE_MIN_VOTES

    # Confirmation by consecutive bars (causal). Ensure windows valid in the confirm span
    window_valid = base_flag.notna().rolling(window=confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars)
    window_range = base_flag.fillna(False).rolling(window=confirm_bars, min_periods=confirm_bars).sum().eq(confirm_bars)
    confirmed = (window_valid & window_range).astype(int)
    confirmed.name = "range_label"
    return confirmed.dropna()


# ================== SEQUENCE BUILDER ==================
def create_flattened_sequences(X, y, look_back):
    X_seq, y_seq = [], []
    for i in tqdm(range(len(X) - look_back), desc="Creating Sequences"):
        X_seq.append(X[i : i + look_back].flatten())
        y_seq.append(y[i + look_back])
    return np.array(X_seq), np.array(y_seq)


def _rolling_minmax_scale_df(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Causal rolling min-max scaling per feature using only past 'window' bars.
    Returns a DataFrame with the same index/columns; rows prior to 'window' will be NaN.
    """
    roll_min = df.rolling(window=window, min_periods=window).min()
    roll_max = df.rolling(window=window, min_periods=window).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    scaled = (df - roll_min) / (denom + 1e-14)
    return scaled


# ================== TRAIN ==================
def train_and_validate(train_df, val_df):
    print("\nStarting feature engineering...")
    X_train_df = feature_engineering(train_df)
    feature_cols = list(X_train_df.columns)  # Save column order
    X_val_df = feature_engineering(val_df).reindex(columns=feature_cols, fill_value=0)

    print("Creating labels (Ranging vs Trending)...")
    y_train = create_ranging_labels(train_df)
    y_val = create_ranging_labels(val_df)

    print("Aligning data...")
    X_train_aligned, y_train_aligned = X_train_df.align(y_train, join="inner", axis=0)
    X_val_aligned, y_val_aligned = X_val_df.align(y_val, join="inner", axis=0)

    # Scaling
    print("Scaling data...")
    scaler = None
    if SCALING_MODE == "global":
        # Fit one scaler on training only
        scaler = MinMaxScaler()
        X_train_scaled_df = pd.DataFrame(
            scaler.fit_transform(X_train_aligned.values),
            index=X_train_aligned.index,
            columns=X_train_aligned.columns,
        )
        X_val_scaled_df = pd.DataFrame(
            scaler.transform(X_val_aligned.values),
            index=X_val_aligned.index,
            columns=X_val_aligned.columns,
        )
    else:
        # Rolling causal scaling.
        # Train: scale on its own history.
        X_train_scaled_df = _rolling_minmax_scale_df(X_train_aligned, ROLLING_SCALER_WINDOW)
        # Val: include last (window-1) bars from train to bootstrap rolling window.
        bridge = X_train_aligned.tail(ROLLING_SCALER_WINDOW - 1)
        combined = pd.concat([bridge, X_val_aligned], axis=0)
        combined_scaled = _rolling_minmax_scale_df(combined, ROLLING_SCALER_WINDOW)
        X_val_scaled_df = combined_scaled.loc[X_val_aligned.index]

    # Drop rows without enough history (NaNs from rolling) and align with labels
    idx_train = (
        X_train_scaled_df.dropna(how="any").index.intersection(y_train_aligned.dropna().index)
    )
    X_train_scaled_df = X_train_scaled_df.loc[idx_train]
    y_train_np = y_train_aligned.loc[idx_train].values

    idx_val = (
        X_val_scaled_df.dropna(how="any").index.intersection(y_val_aligned.dropna().index)
    )
    X_val_scaled_df = X_val_scaled_df.loc[idx_val]
    y_val_np = y_val_aligned.loc[idx_val].values

    X_train_seq, y_train_seq = create_flattened_sequences(
        X_train_scaled_df.values, y_train_np, LOOK_BACK
    )
    X_val_seq, y_val_seq = create_flattened_sequences(
        X_val_scaled_df.values, y_val_np, LOOK_BACK
    )

    # Create stable flattened column names to keep feature names consistent
    flattened_columns = []
    for step in range(LOOK_BACK):  # earliest -> latest
        lag = LOOK_BACK - 1 - step  # distance to latest bar
        for col in feature_cols:
            flattened_columns.append(f"{col}_lag_{lag}")

    # Convert sequences to DataFrames with column names so model stores feature names
    X_train_seq_full_df = pd.DataFrame(X_train_seq, columns=flattened_columns)
    X_val_seq_full_df = pd.DataFrame(X_val_seq, columns=flattened_columns)

    # Drop constant/near-constant columns to improve splits and avoid 'no positive gain' messages
    nunique = X_train_seq_full_df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"移除恒定或近恒定特征列数量: {len(const_cols)}")
    keep_columns = [c for c in flattened_columns if c not in const_cols]
    keep_indices = [i for i, c in enumerate(flattened_columns) if c in keep_columns]

    X_train_seq_df = X_train_seq_full_df[keep_columns]
    X_val_seq_df = X_val_seq_full_df[keep_columns]

    print("\n训练集类别分布:")
    print(pd.Series(y_train_seq).value_counts(normalize=True).round(4))
    # 使用类权重替代欠采样，保持时序结构

    print("\nTraining LightGBM model (binary: range vs trend)...")
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        n_estimators=3000,
        max_depth=6,
        num_leaves=63,
        min_child_samples=50,
        min_sum_hessian_in_leaf=0.01,
        reg_alpha=0.1,
        reg_lambda=1.0,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )

    early_stopping_callback = lgb.early_stopping(stopping_rounds=100, verbose=False)

    # 直接在完整序列上训练（使用类权重处理不平衡）
    model.fit(
        X_train_seq_df,
        y_train_seq,
        eval_set=[(X_val_seq_df, y_val_seq)],
        callbacks=[early_stopping_callback],
    )

    print("\n✅ 验证集表现 (Ranging vs Trending)：")
    target_names = ["Trending (0)", "Ranging (1)"]
    print(
        classification_report(
            y_val_seq,
            model.predict(X_val_seq_df),
            target_names=target_names,
            digits=4,
        )
    )

    # Report top feature importances
    try:
        importances = pd.Series(model.feature_importances_, index=X_train_seq_df.columns)
        top_imp = importances.sort_values(ascending=False).head(20)
        print("\nTop-20 feature importances:")
        print(top_imp.round(2))
    except Exception:
        pass

    artifacts = {
        "model": model,
        "scaler": scaler,
        "scaling_mode": SCALING_MODE,
        "rolling_window": ROLLING_SCALER_WINDOW,
        "feature_columns": feature_cols,
        "flattened_columns": keep_columns,
        "flattened_keep_indices": keep_indices,
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"\n模型及相关组件已保存至 {ARTIFACTS_PATH}")
    return artifacts


# ================== BACKTEST ==================
def run_backtest(test_df, artifacts):
    print("\nRunning backtest...")
    model = artifacts["model"]
    scaler = artifacts.get("scaler")
    scaling_mode = artifacts.get("scaling_mode", "global")
    rolling_window = artifacts.get("rolling_window", ROLLING_SCALER_WINDOW)
    feature_cols = artifacts["feature_columns"]
    flattened_columns = artifacts.get("flattened_columns")
    if flattened_columns is None and hasattr(model, "feature_name_"):
        # Fallback to names stored inside the trained model (if any)
        flattened_columns = list(getattr(model, "feature_name_", [])) or None

    X_test_df = feature_engineering(test_df)
    X_test_aligned = X_test_df.reindex(columns=feature_cols, fill_value=0)
    final_idx = X_test_aligned.index[LOOK_BACK:]

    if scaling_mode == "global" and scaler is not None:
        X_test_scaled_df = pd.DataFrame(
            scaler.transform(X_test_aligned.values),
            index=X_test_aligned.index,
            columns=X_test_aligned.columns,
        )
    else:
        # Rolling scaling: include bridge from prior history if available
        # Try to load cached data for bridge; if not, just use internal window within test
        # Here we simply rely on test's own past because we are in backtest.
        X_test_scaled_df = _rolling_minmax_scale_df(X_test_aligned, rolling_window)

    X_test_scaled_df = X_test_scaled_df.dropna(how="any")
    preds = []

    scaled = X_test_scaled_df.values
    valid_index = X_test_scaled_df.index
    for i in tqdm(range(LOOK_BACK, len(scaled)), desc="Backtesting"):
        seq_full = scaled[i - LOOK_BACK : i].flatten().reshape(1, -1)
        keep_idx = artifacts.get("flattened_keep_indices")
        seq_np = seq_full[:, keep_idx] if keep_idx is not None else seq_full
        if flattened_columns is not None:
            seq_df = pd.DataFrame(seq_np, columns=flattened_columns)
            pred = model.predict(seq_df)[0]
        else:
            pred = model.predict(seq_np)[0]
        preds.append(pred)

    # Align predictions to valid index (accounting for rolling scaling drop and look_back)
    pred_index = valid_index[LOOK_BACK:]
    return pd.Series(preds, index=pred_index)


# ================== WALK-FORWARD CV ==================
def _fit_and_eval_once(train_df, val_df):
    # Feature engineering
    X_train_df = feature_engineering(train_df)
    feature_cols = list(X_train_df.columns)
    X_val_df = feature_engineering(val_df).reindex(columns=feature_cols, fill_value=0)

    # Labels
    y_train = create_ranging_labels(train_df)
    y_val = create_ranging_labels(val_df)

    # Align
    X_train_aligned, y_train_aligned = X_train_df.align(y_train, join="inner", axis=0)
    X_val_aligned, y_val_aligned = X_val_df.align(y_val, join="inner", axis=0)

    # Numpy
    X_train_np = X_train_aligned.values
    y_train_np = y_train_aligned.values
    X_val_np = X_val_aligned.values
    y_val_np = y_val_aligned.values

    # Scale
    if SCALING_MODE == "global":
        scaler = MinMaxScaler()
        X_train_scaled_df = pd.DataFrame(
            scaler.fit_transform(X_train_aligned.values),
            index=X_train_aligned.index,
            columns=X_train_aligned.columns,
        )
        X_val_scaled_df = pd.DataFrame(
            scaler.transform(X_val_aligned.values),
            index=X_val_aligned.index,
            columns=X_val_aligned.columns,
        )
    else:
        # Rolling causal scaling
        X_train_scaled_df = _rolling_minmax_scale_df(X_train_aligned, ROLLING_SCALER_WINDOW)
        bridge = X_train_aligned.tail(ROLLING_SCALER_WINDOW - 1)
        combined = pd.concat([bridge, X_val_aligned], axis=0)
        combined_scaled = _rolling_minmax_scale_df(combined, ROLLING_SCALER_WINDOW)
        X_val_scaled_df = combined_scaled.loc[X_val_aligned.index]

    # Drop NaNs due to insufficient history and align with labels
    idx_train = (
        X_train_scaled_df.dropna(how="any").index.intersection(y_train_aligned.dropna().index)
    )
    X_train_scaled_df = X_train_scaled_df.loc[idx_train]
    y_train_np = y_train_aligned.loc[idx_train].values
    idx_val = (
        X_val_scaled_df.dropna(how="any").index.intersection(y_val_aligned.dropna().index)
    )
    X_val_scaled_df = X_val_scaled_df.loc[idx_val]
    y_val_np = y_val_aligned.loc[idx_val].values

    # Sequences
    X_train_seq, y_train_seq = create_flattened_sequences(
        X_train_scaled_df.values, y_train_np, LOOK_BACK
    )
    X_val_seq, y_val_seq = create_flattened_sequences(
        X_val_scaled_df.values, y_val_np, LOOK_BACK
    )

    # Columns
    flattened_columns = []
    for step in range(LOOK_BACK):
        lag = LOOK_BACK - 1 - step
        for col in feature_cols:
            flattened_columns.append(f"{col}_lag_{lag}")
    X_train_seq_full_df = pd.DataFrame(X_train_seq, columns=flattened_columns)
    X_val_seq_full_df = pd.DataFrame(X_val_seq, columns=flattened_columns)

    # Drop constant columns
    nunique = X_train_seq_full_df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    keep_columns = [c for c in flattened_columns if c not in const_cols]
    X_train_seq_df = X_train_seq_full_df[keep_columns]
    X_val_seq_df = X_val_seq_full_df[keep_columns]

    # Model with stronger regularization
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        n_estimators=3000,
        max_depth=6,
        num_leaves=63,
        min_child_samples=50,
        min_sum_hessian_in_leaf=0.01,
        reg_alpha=0.1,
        reg_lambda=1.0,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train_seq_df,
        y_train_seq,
        eval_set=[(X_val_seq_df, y_val_seq)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    y_pred = model.predict(X_val_seq_df)

    metrics = {
        "accuracy": float(accuracy_score(y_val_seq, y_pred)),
        "precision": float(precision_score(y_val_seq, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val_seq, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val_seq, y_pred, zero_division=0)),
        "n_val": int(len(y_val_seq)),
    }
    return metrics


def run_walk_forward_cv(full_df, validation_start_ts):
    print("\n===== Walk-Forward Cross-Validation (WF-CV) =====")
    df = full_df.copy()
    trainable_df = df[df.index < validation_start_ts]
    if len(trainable_df) < 5000:
        print("训练期数据不足，跳过WF-CV。")
        return

    years = sorted(trainable_df.index.year.unique().tolist())
    if len(years) <= 2:
        # Not enough distinct years; fallback to 4 equal splits
        split_points = np.linspace(0.4, 0.9, 4)
        folds = []
        for p in split_points:
            cut_idx = int(len(trainable_df) * p)
            val_start = trainable_df.index[cut_idx]
            val_end = min(trainable_df.index[-1], val_start + pd.Timedelta(days=90))
            folds.append((trainable_df.index[0], val_start, val_end))
    else:
        # Yearly folds: each year as validation, prior years as training
        folds = []
        min_year = years[0]
        for y in years[1:]:  # ensure at least one prior year for training
            val_start = pd.Timestamp(f"{y}-01-01")
            val_end = pd.Timestamp(f"{y}-12-31")
            val_end = min(val_end, trainable_df.index.max())
            if val_start >= trainable_df.index.max():
                continue
            folds.append((pd.Timestamp(f"{y-1}-12-31"), val_start, val_end))

    results = []
    for (train_end, val_start, val_end) in folds:
        train_df = trainable_df[trainable_df.index < val_start]
        val_df = trainable_df[(trainable_df.index >= val_start) & (trainable_df.index <= val_end)]
        if len(train_df) < 2000 or len(val_df) < 1000:
            continue
        try:
            m = _fit_and_eval_once(train_df, val_df)
            print(
                f"Fold {val_start.date()}–{val_end.date()} | Acc={m['accuracy']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} N={m['n_val']}"
            )
            results.append(m)
        except Exception as e:
            print(f"WF-CV fold failed for {val_start.date()}–{val_end.date()}: {e}")
            continue

    if results:
        def agg(key):
            arr = np.array([r[key] for r in results], dtype=float)
            return float(arr.mean()), float(arr.std())

        acc_m, acc_s = agg("accuracy")
        p_m, p_s = agg("precision")
        r_m, r_s = agg("recall")
        f1_m, f1_s = agg("f1")
        print(
            f"WF-CV Summary | Acc={acc_m:.4f}±{acc_s:.4f} P={p_m:.4f}±{p_s:.4f} R={r_m:.4f}±{r_s:.4f} F1={f1_m:.4f}±{f1_s:.4f} (folds={len(results)})"
        )
    else:
        print("WF-CV 未产生成绩（数据或分割不足）。")

# ================== MAIN ==================
if __name__ == "__main__":
    if os.path.exists(DATA_CACHE_PATH):
        print(f"Loading data from cache: {DATA_CACHE_PATH}")
        df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        df = fetch_binance_klines(SYMBOL, INTERVAL, DATA_START_DATE, TEST_END)
        df.to_csv(DATA_CACHE_PATH)
        print(f"Data saved to cache: {DATA_CACHE_PATH}")

    train_df = df[df.index < VALIDATION_START]
    val_df = df[(df.index >= VALIDATION_START) & (df.index < TEST_START)]
    test_df = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

    print(f"训练集数据尺寸: {train_df.shape}")
    print(f"验证集数据尺寸: {val_df.shape}")
    print(f"测试集数据尺寸: {test_df.shape}")

    # 先进行 Walk-Forward Cross-Validation 评估泛化稳定性
    try:
        run_walk_forward_cv(df, pd.to_datetime(VALIDATION_START))
    except Exception as e:
        print(f"WF-CV 执行失败: {e}")

    artifacts = train_and_validate(train_df, val_df)
    pred = run_backtest(test_df, artifacts)

    print("\n📊 市场震荡识别分布：")
    pred_counts = pred.map({0: "Trending", 1: "Ranging"}).value_counts()
    print(pred_counts)
