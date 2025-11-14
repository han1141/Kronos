# -*- coding: utf-8 -*-
# File: gork8_8.py (Final Corrected Version)

# --- 1. 导入所有需要的库 ---
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import time
import requests
from datetime import datetime, timedelta
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import deque
from backtesting import Backtest, Strategy
from tqdm import tqdm

# --- 2. 日志与全局设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def set_chinese_font():
    try:
        font = next(
            f
            for f in [
                "PingFang SC",
                "Microsoft YaHei",
                "SimHei",
                "Heiti TC",
                "sans-serif",
            ]
            if f in [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        )
        plt.rcParams["font.sans-serif"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
        logger.info(f"成功设置中文字体: {font}")
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")


set_chinese_font()

# --- 3. 共享的函数和配置 ---
def _ema_series(s: pd.Series, length: int) -> pd.Series:
    return s.ewm(span=length, adjust=False).mean()


def _rsi_series(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use EMA-style smoothing: alpha = 2 / (length + 1)
    alpha = 2.0 / (length + 1.0)
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema_series(close, fast)
    ema_slow = _ema_series(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema_series(macd, signal)
    macd_hist = macd - macd_signal
    return macd, macd_hist, macd_signal


def _bbands_series(close: pd.Series, length: int = 20, std: float = 2.0):
    mid = close.rolling(window=length, min_periods=length).mean()
    dev = close.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + std * dev
    lower = mid - std * dev
    # Extra common Bollinger features
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    percent_b = (close - lower) / (upper - lower)
    return lower, mid, upper, bandwidth, percent_b


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Use EMA-style smoothing: alpha = 2 / (length + 1)
    alpha = 2.0 / (length + 1.0)
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    return atr


def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _atr_series(high, low, close, length)
    plus_di = 100 * (plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx, plus_di, minus_di


def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * volume.fillna(0)).cumsum()
    return obv


def _ema_np(values, length: int):
    return _ema_series(pd.Series(values), length).to_numpy()


def _atr_np(high, low, close, length: int):
    return _atr_series(pd.Series(high), pd.Series(low), pd.Series(close), length).to_numpy()

CONFIG = {
    "symbol": "ETHUSDT",
    "interval": "15m",
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-11-11",
    "data_lookback_days": 100,
    "feature_lookback": 60,
    "output_model_path": "models/eth_trend_artifacts_15m.joblib",
    "gbm2_dir": "models_gbm2",
    "initial_cash": 500_000,
    "commission": 0.00085,
    "spread": 0.0002,
    "show_plots": False,
    # Factor model inference options
    "use_selected_flattened_columns": True,
    "apply_macd_filter_to_score": True,
    "macd_filter_mode": "hist_pos_or_rising",  # strict | hist_pos | hist_pos_or_rising | recent_cross
    "macd_recent_cross_lookback": 3,
    "adx_gate_enabled": False,
    "adx_gate_threshold": 18.0,
}


def fetch_binance_klines(s, i, st, en=None, l=1000):
    url, cols = "https://api.binance.com/api/v3/klines", [
        "timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    sts = int(pd.to_datetime(st).timestamp() * 1000)
    ets = int(pd.to_datetime(en).timestamp() * 1000) if en is not None else None
    all_d = []
    max_ets = int(datetime.now().timestamp() * 1000) if ets is None else ets
    while sts < max_ets:
        p = {"symbol": s.upper(), "interval": i, "startTime": sts, "limit": l}
        if ets is not None:
            p["endTime"] = ets
        try:
            r = requests.get(url, params=p, timeout=15)
            r.raise_for_status()
            d = r.json()
            if not d:
                break
            all_d.extend(d)
            sts = d[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.error(f"获取 {s} 失败: {e}")
            time.sleep(5)
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d, columns=[*cols, "c1", "c2", "c3", "c4", "c5", "c6"])[
        cols
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    if en is not None:
        df = df[
            (df["timestamp"] >= pd.to_datetime(st))
            & (df["timestamp"] <= pd.to_datetime(en))
        ]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


# --- 特征工程 ---
def feature_engineering_for_factor_model(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close, vol = x["Close"], x.get("Volume", pd.Series(0.0, index=x.index))

    # RSI
    x["RSI_14"] = _rsi_series(close, length=14)

    # MACD (12, 26, 9)
    macd, macdh, macds = _macd_series(close, fast=12, slow=26, signal=9)
    x["MACD_12_26_9"] = macd
    x["MACDh_12_26_9"] = macdh
    x["MACDs_12_26_9"] = macds

    # Bollinger Bands (20, 2)
    bbl, bbm, bbu, bbb, bbp = _bbands_series(close, length=20, std=2)
    x["BBL_20_2.0"] = bbl
    x["BBM_20_2.0"] = bbm
    x["BBU_20_2.0"] = bbu
    x["BBB_20_2.0"] = bbb
    x["BBP_20_2.0"] = bbp

    # ADX & DIs
    adx, dmp, dmn = _adx_series(x["High"], x["Low"], close, length=14)
    x["ADX_14"] = adx
    x["DMP_14"] = dmp
    x["DMN_14"] = dmn

    # ATR
    x["ATRr_14"] = _atr_series(x["High"], x["Low"], close, length=14)
    x["ATR_14"] = x["ATRr_14"]

    # OBV
    x["OBV"] = _obv_series(close, vol)

    x["volatility_log_ret"] = (np.log(close / close.shift(1))).rolling(window=20).std()

    # Simple causal market structure
    swing_period = 20
    df_copy = x
    # Use only historical data: shift(1) before rolling to exclude current bar
    df_copy["swing_high_price"] = (
        df_copy["High"].shift(1).rolling(window=swing_period, min_periods=1).max()
    )
    df_copy["swing_low_price"] = (
        df_copy["Low"].shift(1).rolling(window=swing_period, min_periods=1).min()
    )
    is_higher_high = df_copy["swing_high_price"] > df_copy["swing_high_price"].shift(1)
    is_higher_low = df_copy["swing_low_price"] > df_copy["swing_low_price"].shift(1)
    is_lower_high = df_copy["swing_high_price"] < df_copy["swing_high_price"].shift(1)
    is_lower_low = df_copy["swing_low_price"] < df_copy["swing_low_price"].shift(1)
    x["market_structure"] = (is_higher_high & is_higher_low).astype(int) - (
        is_lower_high & is_lower_low
    ).astype(int)

    # Longer MACD (24, 52, 18)
    macd_l, macdh_l, macds_l = _macd_series(close, fast=24, slow=52, signal=18)
    x["MACD_long"] = macd_l
    x["MACDh_long"] = macdh_l
    x["MACDs_long"] = macds_l

    x["ret_1"] = close.pct_change(1)
    x["ret_4"] = close.pct_change(4)
    x["ret_16"] = close.pct_change(16)
    x["rsi_delta_1"] = x["RSI_14"].diff(1)
    x["macd_delta_1"] = x["MACDh_12_26_9"].diff(1)

    all_features = x.drop(
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "swing_high_price",
            "swing_low_price",
        ],
        errors="ignore",
    )
    return all_features.replace([np.inf, -np.inf], np.nan)


# --- 模型加载与预测 ---
_FACTOR_SCORE_ARTIFACTS_CACHE = None


def _load_factor_score_artifacts():
    global _FACTOR_SCORE_ARTIFACTS_CACHE
    if _FACTOR_SCORE_ARTIFACTS_CACHE:
        return _FACTOR_SCORE_ARTIFACTS_CACHE
    try:
        base_dir = CONFIG.get("gbm2_dir", "models_gbm2")
        sym_prefix = CONFIG["symbol"].replace("USDT", "").lower()

        def find_latest_file(keyword):
            files = [
                f
                for f in os.listdir(base_dir)
                if keyword in f and f.startswith(sym_prefix)
            ]
            if not files:
                logger.error(
                    f"在 {base_dir} 中未找到 {keyword} 文件 (前缀: {sym_prefix})"
                )
                return None
            return max([os.path.join(base_dir, f) for f in files], key=os.path.getmtime)

        # Prefer selected_flattened_columns if present and allowed
        keys = ["model", "scaler", "feature_columns", "flattened_columns"]
        if CONFIG.get("use_selected_flattened_columns", True):
            keys.insert(3, "selected_flattened_columns")
        paths = {k: find_latest_file(k) for k in keys}
        if any(v is None for v in paths.values()):
            return None

        loaded = {k: joblib.load(v) for k, v in paths.items()}
        flat_cols = (
            loaded.get("selected_flattened_columns")
            if CONFIG.get("use_selected_flattened_columns", True)
            and loaded.get("selected_flattened_columns") is not None
            else loaded.get("flattened_columns")
        )
        _FACTOR_SCORE_ARTIFACTS_CACHE = {
            "model": loaded["model"],
            "scaler": loaded["scaler"],
            "feature_columns": loaded["feature_columns"],
            "flattened_columns": flat_cols,
        }
        logger.info(f"✅ 已加载多因子打分模型: {os.path.basename(paths['model'])}")
        return _FACTOR_SCORE_ARTIFACTS_CACHE
    except Exception as e:
        logger.error(f"加载多因子模型失败: {e}", exc_info=True)
        return None


def _infer_lookback(flat_cols, feat_cols, fallback: int) -> int:
    try:
        return int(len(flat_cols) // len(feat_cols)) if feat_cols else fallback
    except:
        return fallback


def predict_factor_score_causally(
    full_data: pd.DataFrame, backtest_slice_index: pd.DatetimeIndex, arts: dict
) -> pd.Series:
    logger.info("以因果无偏差的方式生成多因子分数...")
    if arts is None or full_data.empty:
        return pd.Series(dtype=float)

    model, scaler, feat_cols, flat_cols = (
        arts["model"],
        arts["scaler"],
        arts["feature_columns"],
        arts["flattened_columns"],
    )
    lookback = _infer_lookback(flat_cols, feat_cols, CONFIG["feature_lookback"])
    all_features = feature_engineering_for_factor_model(full_data)
    all_features = all_features.reindex(columns=feat_cols).fillna(0)
    predictions = []

    def _macd_filter_ok(prefix_features: pd.DataFrame) -> bool:
        try:
            mode = str(CONFIG.get("macd_filter_mode", "hist_pos_or_rising"))
            recent_n = int(CONFIG.get("macd_recent_cross_lookback", 3))
            use_adx = bool(CONFIG.get("adx_gate_enabled", False))
            adx_thr = float(CONFIG.get("adx_gate_threshold", 18.0))
            macd = prefix_features["MACD_long"].iloc[-1]
            macds = prefix_features["MACDs_long"].iloc[-1]
            macdh = prefix_features["MACDh_long"].iloc[-1]
            if np.isnan(macd) or np.isnan(macds) or np.isnan(macdh):
                return False
            ok = False
            if mode == "strict":
                ok = (macd > macds) and (macd > 0)
            elif mode == "hist_pos":
                ok = macdh > 0
            elif mode == "hist_pos_or_rising":
                prev_macdh = prefix_features["MACDh_long"].iloc[-2] if len(prefix_features) > 1 else np.nan
                prev_macd = prefix_features["MACD_long"].iloc[-2] if len(prefix_features) > 1 else macd
                slope = macd - prev_macd
                ok = (macdh > 0) or ((macd > macds) and (slope > 0) and (not np.isnan(prev_macdh) and macdh >= prev_macdh))
            elif mode == "recent_cross":
                window = prefix_features.tail(recent_n + 1)
                crossed = False
                if len(window) >= 2:
                    prev = window.iloc[:-1]
                    crossed = ((prev["MACD_long"] <= prev["MACDs_long"]).any()) and (macd > macds)
                ok = crossed
            else:
                ok = (macd > macds)
            if use_adx and "ADX_14" in prefix_features.columns:
                adx = prefix_features["ADX_14"].iloc[-1]
                if np.isnan(adx) or adx < adx_thr:
                    return False
            return bool(ok)
        except Exception:
            return False

    for i in tqdm(range(len(backtest_slice_index)), desc="Causal Prediction"):
        current_ts = backtest_slice_index[i]
        historical_features = all_features.loc[:current_ts]

        if len(historical_features) < lookback:
            predictions.append(0.5)
            continue

        scaled_historical_features = scaler.transform(historical_features)
        last_window_scaled = scaled_historical_features[-lookback:]
        model_input = last_window_scaled.flatten().reshape(1, -1)
        in_df = pd.DataFrame(model_input, columns=flat_cols)

        try:
            proba = model.predict_proba(in_df)
            score = proba[0, 1]
        except Exception:
            score = model.predict(in_df)[0]
        # Optional MACD-based gating, aligned with training evaluation
        if CONFIG.get("apply_macd_filter_to_score", True):
            macd_ok = _macd_filter_ok(historical_features[[
                c for c in ["MACD_long", "MACDs_long", "MACDh_long", "ADX_14"] if c in historical_features.columns
            ]].dropna())
            if not macd_ok:
                score = 0.0
        predictions.append(np.clip(score, 0.0, 1.0))

    return pd.Series(predictions, index=backtest_slice_index, name="factor_score")


# --- 市场状态（多/空/中性）推断 ---
def compute_market_regime_series(full_data: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """
    使用与训练一致的 MACD 长周期逻辑在线推断市场状态：
    1 = 看多，只允许做多；-1 = 看空，只允许做空；0 = 中性，不开仓。
    只依赖已发生的数据（无前视）。
    """
    feats = feature_engineering_for_factor_model(full_data)
    # 只取需要的列并前缀对齐
    cols = [c for c in ["MACD_long", "MACDs_long", "MACDh_long", "ADX_14"] if c in feats.columns]
    feats = feats[cols].copy()
    feats = feats.reindex(index=index, method=None)
    macd = feats.get("MACD_long")
    macds = feats.get("MACDs_long")
    macdh = feats.get("MACDh_long")
    if macd is None or macds is None or macdh is None:
        # 若数据不足，返回全 0（中性）
        return pd.Series(0, index=index, dtype=int)

    prev_macd = macd.shift(1)
    prev_macdh = macdh.shift(1)
    slope = macd - prev_macd
    bullish = (macdh > 0) | ((macd > macds) & (slope > 0) & (macdh >= prev_macdh))
    bearish = (macdh < 0) | ((macd < macds) & (slope < 0) & (macdh <= prev_macdh))
    regime = pd.Series(0, index=index, dtype=int)
    regime[bullish & ~bearish] = 1
    regime[bearish & ~bullish] = -1
    return regime.fillna(0).astype(int)

# --- 策略与回测 ---
STRATEGY_PARAMS = {
    "tsl_enabled": True,
    "tsl_activation_profit_pct": 0.005,
    "tsl_trailing_atr_mult": 2.0,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "kelly_trade_history": 20,
    "tf_atr_period": 14,
    "tf_ema_fast_window": 20,
    "tf_ema_slow_window": 50,
    "tf_stop_loss_atr_multiplier": 2.0,
    # Stop-loss mode: "swing" (support/resistance) or "volatility"
    "sl_mode": "volatility",
    # Swing-based SL params
    "sl_swing_lookback": 20,
    "sl_buffer_atr_mult": 0.5,
    # Volatility-adaptive SL params
    "sl_vol_window": 50,
    "sl_vol_ratio_floor": 0.8,
    "sl_vol_ratio_cap": 2.5,
    "tf_risk_multiplier": 1.0,
    "daily_max_entries": 2,
    "score_threshold": 0.3101,
}


class UltimateStrategy(Strategy):
    precomp_regime = None
    precomp_score = None

    def init(self):
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)
        self.market_regime = self.I(lambda: self.precomp_regime, name="market_regime")
        self.factor_score = self.I(lambda: self.precomp_score, name="factor_score")
        self.tf_atr = self.I(
            _atr_np,
            self.data.High.s,
            self.data.Low.s,
            self.data.Close.s,
            self.tf_atr_period,
        )
        self.tf_ema_fast = self.I(
            _ema_np, self.data.Close.s, self.tf_ema_fast_window
        )
        self.tf_ema_slow = self.I(
            _ema_np, self.data.Close.s, self.tf_ema_slow_window
        )
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()
        self._atr_smooth = None  # for volatility-adaptive SL

    def _compute_stop_and_risk(self, direction: str, price: float):
        atr_curr = float(self.tf_atr[-1]) if np.isfinite(self.tf_atr[-1]) else 0.0
        if self.sl_mode == "swing":
            lb = int(self.sl_swing_lookback)
            if len(self.data) < max(lb + 1, 5) or atr_curr <= 0:
                # Fallback to volatility mode if insufficient history
                self.sl_mode = "volatility"
            else:
                buffer_ps = self.sl_buffer_atr_mult * atr_curr
                if direction == "long":
                    swing_low = float(np.nanmin(self.data.Low[-lb:]))
                    stop_price = swing_low - buffer_ps
                    risk_ps = max(0.0, price - stop_price)
                    return stop_price, risk_ps
                else:
                    swing_high = float(np.nanmax(self.data.High[-lb:]))
                    stop_price = swing_high + buffer_ps
                    risk_ps = max(0.0, stop_price - price)
                    return stop_price, risk_ps

        # Volatility-adaptive (default)
        # Maintain a smoothed ATR internally (EMA over sl_vol_window)
        if self._atr_smooth is None or not np.isfinite(self._atr_smooth):
            self._atr_smooth = atr_curr
        else:
            n = max(2, int(self.sl_vol_window))
            alpha = 2.0 / (n + 1.0)
            self._atr_smooth = (1 - alpha) * self._atr_smooth + alpha * atr_curr
        base_mult = float(self.tf_stop_loss_atr_multiplier)
        ratio = atr_curr / self._atr_smooth if self._atr_smooth and self._atr_smooth > 0 else 1.0
        ratio = float(np.clip(ratio, self.sl_vol_ratio_floor, self.sl_vol_ratio_cap))
        dyn_mult = base_mult * ratio
        risk_ps = max(0.0, atr_curr * dyn_mult)
        if direction == "long":
            stop_price = price - risk_ps
        else:
            stop_price = price + risk_ps
        return stop_price, risk_ps

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0:
            return 0
        risk_amount = self.equity * min(max(float(risk_pct), 0.0), self.max_risk_pct)
        units = risk_amount / rps
        if units * p > self.equity:
            units = (self.equity * 0.95) / p
        # --- FIXED: Cast the number of units to an integer before returning ---
        return int(max(0, units))

    def reset_trade_state(self):
        self.stop_loss_price, self.tsl_active = 0.0, False

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            if self.position.is_long and (
                price <= self.stop_loss_price
                or self.tf_ema_fast[-1] < self.tf_ema_slow[-1]
            ):
                self.position.close()
            elif self.position.is_short and (
                price >= self.stop_loss_price
                or self.tf_ema_fast[-1] > self.tf_ema_slow[-1]
            ):
                self.position.close()
            return

        # Directional gating: only long in bullish regime, only short in bearish regime
        regime_now = int(self.market_regime[-1]) if np.isfinite(self.market_regime[-1]) else 0
        current_score = self.factor_score[-1]

        # Removed EMA crossover as entry condition; entries rely on factor score only
        if regime_now == 1 and current_score >= self.score_threshold:
            stop_price, risk_ps = self._compute_stop_and_risk("long", price)
            # Risk scales with how far score exceeds threshold → [0, 1]
            conf = (current_score - self.score_threshold) / (1.0 - self.score_threshold)
            conf = max(0.0, min(1.0, float(conf)))
            risk_pct = min(self.max_risk_pct, self.tf_risk_multiplier * conf * self.max_risk_pct)
            size = self._calculate_position_size(price, risk_ps, risk_pct)
            if size > 0:
                self.stop_loss_price = stop_price
                self.buy(size=size)
        elif regime_now == -1 and (1.0 - current_score) >= self.score_threshold:
            stop_price, risk_ps = self._compute_stop_and_risk("short", price)
            # Risk scales with how far (1-score) exceeds threshold → [0, 1]
            conf = ((1.0 - current_score) - self.score_threshold) / (1.0 - self.score_threshold)
            conf = max(0.0, min(1.0, float(conf)))
            risk_pct = min(self.max_risk_pct, self.tf_risk_multiplier * conf * self.max_risk_pct)
            size = self._calculate_position_size(price, risk_ps, risk_pct)
            if size > 0:
                self.stop_loss_price = stop_price
                self.sell(size=size)


def run_backtest():
    logger.info("====== 进入回测模式 ======")
    data_fetch_start_date = (
        pd.to_datetime(CONFIG["backtest_start_date"])
        - timedelta(days=CONFIG["data_lookback_days"])
    ).strftime("%Y-%m-%d")
    raw_data = fetch_binance_klines(
        CONFIG["symbol"],
        CONFIG["interval"],
        data_fetch_start_date,
        CONFIG["backtest_end_date"],
    )
    if raw_data.empty:
        logger.error("数据获取失败，终止。")
        return

    data_slice = raw_data.loc[
        CONFIG["backtest_start_date"] : CONFIG["backtest_end_date"]
    ].copy()
    if data_slice.empty:
        logger.warning(f"在回测周期内无数据，跳过。")
        return

    print(f"\n{'='*80}\n正在回测品种: {CONFIG['symbol']}\n{'='*80}")
    # 依据 MACD 长周期在线推断市场状态（看多/看空/中性）
    precomp_regime_sr = compute_market_regime_series(raw_data, data_slice.index)
    precomp_regime = precomp_regime_sr.values

    precomp_score = pd.Series(0.5, index=data_slice.index, dtype=float)
    factor_arts = _load_factor_score_artifacts()
    if factor_arts:
        score_sr = predict_factor_score_causally(
            raw_data, data_slice.index, factor_arts
        )
        if not score_sr.empty:
            precomp_score = score_sr

    precomp_score_values = precomp_score.values

    if len(precomp_score_values) > 0:
        qs = (
            pd.Series(precomp_score_values)
            .quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            .values
        )
        print(
            "多因子分数分位: min={:.2f}, p10={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, p90={:.2f}, max={:.2f}".format(
                *qs
            )
        )
        print(
            f"分数均值={precomp_score_values.mean():.3f} | 阈值={STRATEGY_PARAMS['score_threshold']:.2f}"
        )

    # Ensure open trades are finalized at the end to avoid warnings
    try:
        bt = Backtest(
            data_slice,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            finalize_trades=True,
        )
    except TypeError:
        # Fallback for older backtesting versions without finalize_trades
        bt = Backtest(
            data_slice,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
        )
    stats = bt.run(precomp_regime=precomp_regime, precomp_score=precomp_score_values)

    print(f"\n{'-'*40}\n          {CONFIG['symbol']} 回测结果摘要\n{'-'*40}")
    print(stats)

    if CONFIG["show_plots"]:
        bt.plot()


if __name__ == "__main__":
    run_backtest()
