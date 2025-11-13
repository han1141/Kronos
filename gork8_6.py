# -*- coding: utf-8 -*-
# File: gork8_final.py (Single-File Version)

# --- 1. 导入所有需要的库 ---
import pandas as pd
import numpy as np
import ta
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
        plt.rcParams["font.sans-serif"], plt.rcParams["axes.unicode_minus"] = [
            font
        ], False
        logger.info(f"成功设置中文字体: {font}")
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")


set_chinese_font()

# --- 3. 共享的函数和配置 (训练和回测都会用到) ---

# 全局配置
CONFIG = {
    "symbol": "ETHUSDT",
    "interval": "15m",
    # 回测数据周期
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-11-11",
    "data_lookback_days": 100,
    # 模型相关
    "feature_lookback": 60,
    # 仅推理用的序列长度（训练功能已移除）
    "label_look_forward": 24,  # 保留占位，避免历史模型依赖出错
    "output_model_path": "models/eth_trend_artifacts_15m.joblib",
    # 追加：多因子打分模型所在目录（与市场状态模型不同用途）
    "gbm2_dir": "models_gbm2",
    # 回测相关
    "initial_cash": 500_000,
    "commission": 0.00085,
    "spread": 0.0002,
    "show_plots": False,
}


# 共享的数据获取函数
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
    while sts < ets:
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
    # 结果再按区间裁剪（双保险，避免超过 en 的数据进入）
    if en is not None:
        end_dt = pd.to_datetime(en)
        df = df[(df["timestamp"] >= pd.to_datetime(st)) & (df["timestamp"] <= end_dt)]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


# 共享的特征工程函数
def feature_engineering_for_regime_model(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["RSI_14"] = ta.momentum.RSIIndicator(x["Close"], window=14).rsi()
    macd = ta.trend.MACD(x["Close"], window_fast=12, window_slow=26, window_sign=9)
    x["MACD_12_26_9"] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(x["High"], x["Low"], x["Close"], window=14)
    x["ADX_14"] = adx.adx()
    bb = ta.volatility.BollingerBands(x["Close"], window=20, window_dev=2)
    x["BB_WIDTH_20"] = bb.bollinger_wband()
    x["ATR_14"] = ta.volatility.AverageTrueRange(
        x["High"], x["Low"], x["Close"], window=14
    ).average_true_range()
    # 去除所有4小时宏观相关特征

    def get_hurst(ts):
        if len(ts) < 100:
            return 0.5
        lags = range(2, 100)
        tau = [pd.Series(ts).diff(lag).std() for lag in lags]
        tau = [t for t in tau if t > 0]
        if len(tau) < 2:
            return 0.5
        lags = range(2, len(tau) + 2)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    x["hurst_100"] = x["Close"].rolling(window=100).apply(get_hurst, raw=False)
    return x.replace([np.inf, -np.inf], np.nan)


def feature_engineering_for_factor_model(df: pd.DataFrame,
                                         macd_long_fast: int = 24,
                                         macd_long_slow: int = 52,
                                         macd_long_signal: int = 9,
                                         atr_period: int = 14,
                                         rsi_period: int = 14,
                                         bb_period: int = 20,
                                         bb_std: float = 2.0,
                                         ema_4h_period: int = 20) -> pd.DataFrame:
    """为多因子打分模型计算特征（与 models_gbm2 一致或近似）。

    说明：
    - MACD_12_26_9/MACDh_12_26_9/MACDs_12_26_9: 标准 MACD 线/柱/信号线
    - BBL/BBM/BBU/BBB/BBP: 布林下/中/上/带宽/%B
    - MACD_long/MACDh_long/MACDs_long: 较长周期 MACD（默认 24/52/9）
    - volatility_log: log1p(ATR_14)
    - macd_cross_signal / macd_long_cross_signal: 近一根发生金叉为1，否则0
    - macd_cross_confirm: 当前 MACD > 信号线 为1，否则0
    - price_above_ema_4h: 当前收盘 > 4小时EMA(20)
    - adx_x_atr_norm: (ADX/100) × (ATR/Close)
    - rsi_x_hurst: (RSI/100) × Hurst100（裁剪0~1）
    - market_structure: 简单结构强弱（Close>前高则1，否则0）
    """
    x = df.copy()
    close = pd.Series(x["Close"])
    high = pd.Series(x["High"])
    low = pd.Series(x["Low"])
    vol = pd.Series(x["Volume"]) if "Volume" in x.columns else pd.Series(index=x.index, data=0.0)

    # 标准 MACD 12/26/9
    macd_std = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    x["MACD_12_26_9"] = macd_std.macd()
    x["MACDh_12_26_9"] = macd_std.macd_diff()
    x["MACDs_12_26_9"] = macd_std.macd_signal()

    # Bollinger Bands 20, 2.0
    bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=bb_std)
    x[f"BBL_{bb_period}_{bb_std}"] = bb.bollinger_lband()
    x[f"BBM_{bb_period}_{bb_std}"] = bb.bollinger_mavg()
    x[f"BBU_{bb_period}_{bb_std}"] = bb.bollinger_hband()
    x[f"BBB_{bb_period}_{bb_std}"] = bb.bollinger_wband()
    x[f"BBP_{bb_period}_{bb_std}"] = bb.bollinger_pband()

    # 长周期 MACD（默认 24/52/9，可按需覆盖）
    macd_l = ta.trend.MACD(close, window_fast=macd_long_fast, window_slow=macd_long_slow, window_sign=macd_long_signal)
    x["MACD_long"] = macd_l.macd()
    x["MACDh_long"] = macd_l.macd_diff()
    x["MACDs_long"] = macd_l.macd_signal()

    # 波动率相关
    atr = ta.volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()
    x["ATR_14"] = atr
    x["volatility_log"] = np.log1p(atr)

    # 4小时 EMA(20) 并上采样到15m
    try:
        close_4h = close.resample("4h").last()
        ema_4h = ta.trend.EMAIndicator(close_4h, window=ema_4h_period).ema_indicator()
        ema_4h_upsampled = ema_4h.reindex(close.index, method="ffill")
        x["price_above_ema_4h"] = (close > ema_4h_upsampled).astype(float)
    except Exception:
        x["price_above_ema_4h"] = np.nan

    # ADX × ATR% 与 RSI × Hurst
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    adx = adx_ind.adx()
    x["ADX_14"] = adx
    x["DMP_14"] = adx_ind.adx_pos()
    x["DMN_14"] = adx_ind.adx_neg()
    atr_pct = (atr / close).replace([np.inf, -np.inf], np.nan)
    x["adx_x_atr_norm"] = (adx / 100.0) * atr_pct

    rsi = ta.momentum.RSIIndicator(close, window=rsi_period).rsi()
    x["RSI_14"] = rsi
    # Hurst 与 regime 的估计函数复用
    def get_hurst(ts):
        if len(ts) < 100:
            return 0.5
        lags = range(2, 100)
        tau = [pd.Series(ts).diff(lag).std() for lag in lags]
        tau = [t for t in tau if t > 0]
        if len(tau) < 2:
            return 0.5
        lags = range(2, len(tau) + 2)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    hurst100 = close.rolling(window=100).apply(get_hurst, raw=False).clip(0.0, 1.0)
    x["rsi_x_hurst"] = (rsi / 100.0) * hurst100
    x["hurst"] = hurst100

    # MACD 金叉/确认信号（标准与长周期）
    macd_line, macd_sig = x["MACD_12_26_9"], x["MACDs_12_26_9"]
    macd_line_prev, macd_sig_prev = macd_line.shift(1), macd_sig.shift(1)
    x["macd_cross_signal"] = ((macd_line_prev <= macd_sig_prev) & (macd_line > macd_sig)).astype(float)
    x["macd_cross_confirm"] = (macd_line > macd_sig).astype(float)

    macd_l_line, macd_l_sig = x["MACD_long"], x["MACDs_long"]
    macd_l_line_prev, macd_l_sig_prev = macd_l_line.shift(1), macd_l_sig.shift(1)
    x["macd_long_cross_signal"] = ((macd_l_line_prev <= macd_l_sig_prev) & (macd_l_line > macd_l_sig)).astype(float)

    # 市场结构（简单版）：若当前最高价高于过去N(=20)根最高，视为强结构
    N = 20
    prev_high = high.shift(1).rolling(window=N).max()
    x["market_structure"] = (high > prev_high).astype(float)

    # OBV
    try:
        obv = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        x["OBV"] = obv
    except Exception:
        x["OBV"] = np.nan

    return x.replace([np.inf, -np.inf], np.nan)


# 已移除：4小时宏观趋势过滤相关函数


# --- 4. 训练模式相关函数 ---


    # 训练功能已移除：仅支持加载既有模型工件进行推理


# --- 5. 回测模式相关函数和类 ---

_TREND_ARTIFACTS_CACHE = None
_FACTOR_SCORE_ARTIFACTS_CACHE = None


def _load_trend_artifacts():
    global _TREND_ARTIFACTS_CACHE
    if _TREND_ARTIFACTS_CACHE:
        return _TREND_ARTIFACTS_CACHE
    artifacts_path = CONFIG.get("output_model_path")
    if not artifacts_path or not os.path.exists(artifacts_path):
        logger.warning(
            f"未找到市场状态模型文件: {artifacts_path}。请确认 models/eth_trend_artifacts_15m.joblib 是否存在。"
        )
        return None
    try:
        _TREND_ARTIFACTS_CACHE = joblib.load(artifacts_path)
        logger.info("✅ 已加载市场状态分类模型（单文件工件）。")
        return _TREND_ARTIFACTS_CACHE
    except Exception as e:
        logger.error(f"加载市场状态工件失败: {e}")
        return None


def _load_factor_score_artifacts():
    """加载多因子打分模型（GBM2，分散存储在目录内）。
    目录中应包含：model、scaler、feature_columns、flattened_columns 四个 .joblib 文件。
    """
    global _FACTOR_SCORE_ARTIFACTS_CACHE
    if _FACTOR_SCORE_ARTIFACTS_CACHE:
        return _FACTOR_SCORE_ARTIFACTS_CACHE
    try:
        base_dir = CONFIG.get("gbm2_dir", "models_gbm2")
        if not os.path.isdir(base_dir):
            logger.warning(f"未找到多因子模型目录: {base_dir}")
            return None
        files = [f for f in os.listdir(base_dir) if f.endswith(".joblib")]
        if not files:
            logger.warning(f"目录 {base_dir} 中没有 .joblib 文件。")
            return None
        sym = str(CONFIG.get("symbol", "")).upper()
        prefix = sym.replace("USDT", "").lower()
        def select_latest(kw: str):
            all_matches = [os.path.join(base_dir, f) for f in files if kw in f]
            if not all_matches:
                return None
            preferred = [p for p in all_matches if os.path.basename(p).lower().startswith(prefix)]
            choices = preferred if preferred else all_matches
            return max(choices, key=lambda p: os.path.getmtime(p)) if choices else None
        model_path = select_latest("model")
        scaler_path = select_latest("scaler")
        feat_path = select_latest("feature_columns")
        flat_path = select_latest("flattened_columns")
        missing = [("model", model_path), ("scaler", scaler_path), ("feature_columns", feat_path), ("flattened_columns", flat_path)]
        miss_keys = [k for k, v in missing if not v]
        if miss_keys:
            logger.warning(f"{base_dir} 缺少必要文件: {', '.join(miss_keys)}。无法加载多因子模型。")
            return None
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(feat_path)
        flattened_columns = joblib.load(flat_path)
        _FACTOR_SCORE_ARTIFACTS_CACHE = {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "flattened_columns": flattened_columns,
        }
        logger.info(
            "✅ 已加载多因子打分模型: "
            f"model={os.path.basename(model_path)}, scaler={os.path.basename(scaler_path)}"
        )
        return _FACTOR_SCORE_ARTIFACTS_CACHE
    except Exception as e:
        logger.error(f"加载多因子模型失败: {e}")
        return None


def _infer_lookback(flat_cols, feat_cols, fallback: int) -> int:
    try:
        if flat_cols is not None and feat_cols is not None and len(feat_cols) > 0:
            lb = int(len(flat_cols) // len(feat_cols))
            if lb > 0:
                return lb
    except Exception:
        pass
    return int(fallback)


def _transform_with_names_aware_scaler(scaler, Xdf: pd.DataFrame, feat_cols: list):
    """Transforms features using scaler while handling fit-with/without-names cases.
    - If scaler has feature_names_in_, align to those names and pass DataFrame.
    - Else, align to feat_cols and pass numpy array to avoid sklearn warnings.
    """
    try:
        if hasattr(scaler, "feature_names_in_") and getattr(scaler, "feature_names_in_", None) is not None:
            cols = list(getattr(scaler, "feature_names_in_"))
            # 若与传入feat_cols不一致，优先以scaler所存列名为准
            return scaler.transform(Xdf.reindex(columns=cols))
        # 无特征名：用既定顺序并传 numpy，避免警告
        return scaler.transform(Xdf.reindex(columns=feat_cols).values)
    except Exception:
        # 兜底
        return scaler.transform(Xdf.reindex(columns=feat_cols).values)


def _predict_trend_ranging_series(df_ohlcv: pd.DataFrame, arts: dict) -> pd.Series:
    if arts is None or df_ohlcv.empty:
        return pd.Series(dtype=float)
    model, scaler, feat_cols = arts["model"], arts["scaler"], arts["feature_columns"]
    flat_cols = arts.get("flattened_columns")

    Xdf = feature_engineering_for_regime_model(df_ohlcv)
    missing = [c for c in feat_cols if c not in Xdf.columns]
    if missing:
        for c in missing:
            Xdf[c] = 0
    Xdf = Xdf.reindex(columns=feat_cols, fill_value=0).fillna(0)

    # 使用带列名的DataFrame以保证列对齐
    X_scaled = _transform_with_names_aware_scaler(scaler, Xdf, feat_cols)
    lookback = _infer_lookback(flat_cols, feat_cols, CONFIG["feature_lookback"])
    if len(X_scaled) <= lookback:
        return pd.Series(dtype=float)

    # 批量构造所有滑窗样本，一次预测，显著提速
    n = len(X_scaled)
    rows = n - lookback
    if rows <= 0:
        return pd.Series(dtype=float)
    # 组装二维矩阵 (rows, lookback * n_features)
    samples = np.empty((rows, lookback * X_scaled.shape[1]), dtype=X_scaled.dtype)
    for i in range(lookback, n):
        samples[i - lookback] = X_scaled[i - lookback : i].reshape(-1)
    in_df = pd.DataFrame(samples, columns=flat_cols)
    pred_arr = model.predict(in_df)
    # 转成 int 标签
    preds = [int(p) for p in pred_arr]
    return pd.Series(preds, index=df_ohlcv.index[lookback:], name="trend_range_label")


def _predict_factor_score_series(df_ohlcv: pd.DataFrame, arts: dict) -> pd.Series:
    """用多因子打分模型进行滑窗打分，返回每根K线的分数（0~1）。
    约定：若模型为二分类，取类别1的概率作为多因子分数。
    """
    if arts is None or df_ohlcv.empty:
        return pd.Series(dtype=float)
    model, scaler, feat_cols = arts["model"], arts["scaler"], arts["feature_columns"]
    flat_cols = arts.get("flattened_columns")

    # 使用多因子专用特征工程
    Xdf = feature_engineering_for_factor_model(df_ohlcv)
    missing = [c for c in feat_cols if c not in Xdf.columns]
    if missing:
        # 调试：打印缺失特征数量与部分名称
        try:
            sample = ", ".join(missing[:10])
            logger.warning(f"多因子特征缺失 {len(missing)} 项（示例）：{sample}")
        except Exception:
            pass
        for c in missing:
            Xdf[c] = 0
    Xdf = Xdf.reindex(columns=feat_cols, fill_value=0).fillna(0)

    # 使用带列名的DataFrame以保证列对齐
    X_scaled = _transform_with_names_aware_scaler(scaler, Xdf, feat_cols)
    lookback = _infer_lookback(flat_cols, feat_cols, CONFIG["feature_lookback"])
    if len(X_scaled) <= lookback:
        return pd.Series(dtype=float)
    n = len(X_scaled)
    rows = n - lookback
    if rows <= 0:
        return pd.Series(dtype=float)
    samples = np.empty((rows, lookback * X_scaled.shape[1]), dtype=X_scaled.dtype)
    for i in range(lookback, n):
        samples[i - lookback] = X_scaled[i - lookback : i].reshape(-1)
    in_df = pd.DataFrame(samples, columns=flat_cols)
    # 兼容不同API：优先 predict_proba
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(in_df)
            if isinstance(proba, (list, np.ndarray)) and np.ndim(proba) == 2 and proba.shape[1] >= 2:
                score = proba[:, 1]
            else:
                # 单值回归或异常形状，直接使用预测值
                score = np.asarray(model.predict(in_df)).reshape(-1)
        else:
            pred = model.predict(in_df)
            score = np.asarray(pred).reshape(-1)
        # 将分数裁剪到[0,1]
        score = np.clip(score, 0.0, 1.0)
    except Exception:
        return pd.Series(dtype=float)
    return pd.Series(score, index=df_ohlcv.index[lookback:], name="factor_score")


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """仅进行必要的轻量清理，不再全局预先计算 market_regime 或 macro_trend。

    目标：避免在回测开始前就“知道”整个序列的状态，彻底消除前视偏差。
    后续由策略在 next() 中按步（仅用历史）因果计算需要的过滤/状态。
    """
    df = data_in.copy()
    logger.info(f"[{symbol}] 开始数据预处理（轻量清理，无全局状态计算）...")
    # 可按需添加其他与时间无关的数据清理步骤
    logger.info(f"[{symbol}] 数据预处理完成，行数: {len(df)}")
    return df


# 策略参数
STRATEGY_PARAMS = {
    "tsl_enabled": True,
    "tsl_activation_profit_pct": 0.005,
    "tsl_activation_atr_mult": 1.5,
    "tsl_trailing_atr_mult": 2.0,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "kelly_trade_history": 20,
    "tf_atr_period": 14,
    "tf_ema_fast_window": 20,
    "tf_ema_slow_window": 50,
    "tf_stop_loss_atr_multiplier": 2.0,
    "tf_risk_multiplier": 1.0,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mr_rsi_period": 14,
    "daily_max_entries": 2,
    # 多因子打分阈值：做多用 score，做空用 (1-score)
    "score_threshold": 0.55,
}
ASSET_SPECIFIC_OVERRIDES = {"ETHUSDT": {"strategy_class": "ETHStrategy"}}


# 策略类
class BaseAssetStrategy:
    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy


class ETHStrategy(BaseAssetStrategy):
    pass


STRATEGY_MAPPING = {"BaseAssetStrategy": BaseAssetStrategy, "ETHStrategy": ETHStrategy}


class UltimateStrategy(Strategy):
    symbol = None
    vol_weight = 1.0
    # 接收 Backtest.run(...) 传入的预计算序列参数（可选）
    precomp_regime = None
    precomp_score = None

    def init(self):
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)
        strategy_class_name = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get(
            "strategy_class", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(
            strategy_class_name, BaseAssetStrategy
        )(self)
        # 如果传入了预计算的序列，则用作指标，提升速度且保持因果性
        pre_regime = getattr(self, "precomp_regime", None)
        self._has_pre_regime = pre_regime is not None
        if self._has_pre_regime:
            self.market_regime = self.I(lambda arr=np.asarray(pre_regime): arr)
        # 预计算的多因子分数
        pre_score = getattr(self, "precomp_score", None)
        self._has_pre_score = pre_score is not None
        if self._has_pre_score:
            self.factor_score = self.I(lambda arr=np.asarray(pre_score): arr)

        # 否则使用按步因果计算作为后备方案（较慢）
        self.trend_artifacts = _load_trend_artifacts() if not self._has_pre_regime else None
        self.factor_artifacts = _load_factor_score_artifacts() if not self._has_pre_score else None
        self.feature_lookback = int(CONFIG.get("feature_lookback", 60))
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                pd.Series(self.data.High),
                pd.Series(self.data.Low),
                pd.Series(self.data.Close),
                self.tf_atr_period,
            ).average_true_range()
        )
        # 趋势交易用 EMA
        self.tf_ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(
                pd.Series(self.data.Close), window=self.tf_ema_fast_window
            ).ema_indicator()
        )
        self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(
                pd.Series(self.data.Close), window=self.tf_ema_slow_window
            ).ema_indicator()
        )
        bb = ta.volatility.BollingerBands(
            pd.Series(self.data.Close),
            window=self.mr_bb_period,
            window_dev=self.mr_bb_std,
        )
        self.mr_bb_upper = self.I(lambda: bb.bollinger_hband())
        self.mr_bb_middle = self.I(lambda: bb.bollinger_mavg())
        self.mr_bb_lower = self.I(lambda: bb.bollinger_lband())
        stoch_rsi = ta.momentum.StochRSIIndicator(
            pd.Series(self.data.Close), window=14, smooth1=3, smooth2=3
        )
        self.mr_stoch_rsi_k = self.I(lambda: stoch_rsi.stochrsi_k())
        self.mr_stoch_rsi_d = self.I(lambda: stoch_rsi.stochrsi_d())
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()

    # 已移除：4小时宏观趋势因果计算

    def _compute_market_regime_current(self) -> int:
        """因果计算：用训练好的市场状态模型，仅基于过去窗口预测当前状态。
        返回: 1(Trending) / -1(Ranging) / 0(Unknown)
        """
        arts = self.trend_artifacts
        if not arts:
            return 0
        try:
            lookback = self.feature_lookback
            idx = self.data.df.index
            df_now = pd.DataFrame({
                "Open": pd.Series(self.data.Open, index=idx),
                "High": pd.Series(self.data.High, index=idx),
                "Low": pd.Series(self.data.Low, index=idx),
                "Close": pd.Series(self.data.Close, index=idx),
                "Volume": pd.Series(self.data.Volume, index=idx) if hasattr(self.data, 'Volume') else pd.Series(0, index=idx),
            }).iloc[: len(idx)]
            if len(df_now) <= lookback:
                return 0
            Xdf = feature_engineering_for_regime_model(df_now)
            model, scaler, feat_cols = arts["model"], arts["scaler"], arts["feature_columns"]
            flat_cols = arts.get("flattened_columns")
            for c in feat_cols:
                if c not in Xdf.columns:
                    Xdf[c] = 0
            Xdf = Xdf.reindex(columns=feat_cols, fill_value=0).fillna(0)
            # 列名对齐 + 自适配 lookback
            X_scaled = _transform_with_names_aware_scaler(scaler, Xdf, feat_cols)
            lookback = _infer_lookback(flat_cols, feat_cols, lookback)
            seq = X_scaled[len(X_scaled) - lookback : len(X_scaled)].flatten().reshape(1, -1)
            in_df = pd.DataFrame(seq, columns=flat_cols)
            pred = int(model.predict(in_df)[0])
            # 训练时约定 0=Trending, 1=Ranging；映射为: Trending=1, Ranging=-1
            return -1 if pred == 1 else 1
        except Exception:
            return 0

    def _compute_factor_score_current(self) -> float:
        """因果计算多因子分数：仅用过去窗口，返回 0~1。"""
        arts = getattr(self, "factor_artifacts", None)
        if not arts:
            return 0.5
        try:
            lookback = self.feature_lookback
            idx = self.data.df.index
            df_now = pd.DataFrame({
                "Open": pd.Series(self.data.Open, index=idx),
                "High": pd.Series(self.data.High, index=idx),
                "Low": pd.Series(self.data.Low, index=idx),
                "Close": pd.Series(self.data.Close, index=idx),
                "Volume": pd.Series(self.data.Volume, index=idx) if hasattr(self.data, 'Volume') else pd.Series(0, index=idx),
            }).iloc[: len(idx)]
            if len(df_now) <= lookback:
                return 0.5
            Xdf = feature_engineering_for_factor_model(df_now)
            model, scaler, feat_cols = arts["model"], arts["scaler"], arts["feature_columns"]
            flat_cols = arts.get("flattened_columns")
            for c in feat_cols:
                if c not in Xdf.columns:
                    Xdf[c] = 0
            Xdf = Xdf.reindex(columns=feat_cols, fill_value=0).fillna(0)
            # 列名对齐 + 自适配 lookback
            X_scaled = _transform_with_names_aware_scaler(scaler, Xdf, feat_cols)
            lookback = _infer_lookback(flat_cols, feat_cols, lookback)
            seq = X_scaled[len(X_scaled) - lookback : len(X_scaled)].flatten().reshape(1, -1)
            in_df = pd.DataFrame(seq, columns=flat_cols)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(in_df)
                if isinstance(proba, (list, np.ndarray)) and np.ndim(proba) == 2 and proba.shape[1] >= 2:
                    score = float(proba[0, 1])
                else:
                    score = float(np.clip(model.predict(in_df)[0], 0.0, 1.0))
            else:
                score = float(np.clip(model.predict(in_df)[0], 0.0, 1.0))
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.5

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0:
            return 0
        risk_amount = self.equity * min(max(float(risk_pct), 0.0), self.max_risk_pct)
        units = risk_amount / rps
        if units * p > self.equity:
            units = (self.equity * 0.95) / p
        return max(1, int(units))

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        if avg_loss == 0:
            return self.default_risk_pct * self.vol_weight
        rr = avg_win / avg_loss if avg_loss else 0
        kelly = win_rate - (1 - win_rate) / rr if rr != 0 else 0
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)

    def _update_day_counter(self):
        try:
            cur_ts = self.data.df.index[-1]
        except Exception:
            return
        cur_day = cur_ts.date()
        if getattr(self, "_current_day", None) != cur_day:
            self._current_day = cur_day
            self._entries_today = 0

    def _can_enter_today(self) -> bool:
        self._update_day_counter()
        return getattr(self, "_entries_today", 0) < self.daily_max_entries

    def _mark_entered(self):
        self._update_day_counter()
        self._entries_today = getattr(self, "_entries_today", 0) + 1

    def manage_mean_reversion_exit(self, p):
        mid = float(self.mr_bb_middle[-1]) if np.isfinite(self.mr_bb_middle[-1]) else p
        if (self.position.is_long and (p >= mid or p <= self.stop_loss_price)) or (
            self.position.is_short and (p <= mid or p >= self.stop_loss_price)
        ):
            self.close_position("MR_Exit")

    def manage_trend_exit(self, p):
        # 趋势跟随退出与移动止损
        if not self.position:
            return
        atr = float(self.tf_atr[-1]) if np.isfinite(self.tf_atr[-1]) else 0.0
        fast_now = float(self.tf_ema_fast[-1]) if np.isfinite(self.tf_ema_fast[-1]) else p
        slow_now = float(self.tf_ema_slow[-1]) if np.isfinite(self.tf_ema_slow[-1]) else p
        fast_prev = (
            float(self.tf_ema_fast[-2])
            if len(self.tf_ema_fast) >= 2 and np.isfinite(self.tf_ema_fast[-2])
            else fast_now
        )
        slow_prev = (
            float(self.tf_ema_slow[-2])
            if len(self.tf_ema_slow) >= 2 and np.isfinite(self.tf_ema_slow[-2])
            else slow_now
        )

        # 移动止损激活与更新
        if self.tsl_enabled and atr > 0:
            current_price = float(self.data.Close[-1])
            entry_price = (
                float(self.trades[-1].entry_price)
                if len(self.trades) > 0 and hasattr(self.trades[-1], "entry_price")
                else current_price
            )
            if self.position.is_long:
                profit_pct = current_price / entry_price - 1.0
                if (not getattr(self, "tsl_active", False)) and profit_pct >= float(self.tsl_activation_profit_pct):
                    self.tsl_active = True
                if getattr(self, "tsl_active", False):
                    new_sl = current_price - atr * float(self.tsl_trailing_atr_mult)
                    self.stop_loss_price = max(self.stop_loss_price, new_sl)
            elif self.position.is_short:
                profit_pct = entry_price / current_price - 1.0
                if (not getattr(self, "tsl_active", False)) and profit_pct >= float(self.tsl_activation_profit_pct):
                    self.tsl_active = True
                if getattr(self, "tsl_active", False):
                    new_sl = current_price + atr * float(self.tsl_trailing_atr_mult)
                    self.stop_loss_price = min(self.stop_loss_price, new_sl)

        # 趋势反转或触发止损则退出
        if self.position.is_long:
            trend_reversal = (fast_prev >= slow_prev and fast_now < slow_now)
            if p <= self.stop_loss_price or trend_reversal:
                self.close_position("TF_Exit")
        elif self.position.is_short:
            trend_reversal = (fast_prev <= slow_prev and fast_now > slow_now)
            if p >= self.stop_loss_price or trend_reversal:
                self.close_position("TF_Exit")

    def close_position(self, reason: str):
        eq_before = self.equity
        self.position.close()
        try:
            self.recent_trade_returns.append(self.equity / eq_before - 1)
        except Exception:
            pass
        self.reset_trade_state()

    def reset_trade_state(self):
        self.active_sub_strategy = None
        self.stop_loss_price = 0.0
        self.tsl_active = False

    def next(self):
        price = self.data.Close[-1]
        # 获取当前市场状态（优先使用预计算序列，否则按步因果计算）
        if getattr(self, "_has_pre_regime", False):
            current_regime = int(self.market_regime[-1])
        else:
            current_regime = self._compute_market_regime_current()
        # 存储以便退出逻辑使用
        self.current_market_regime = current_regime
        # 多因子分数（0~1）
        if getattr(self, "_has_pre_score", False):
            current_score = float(self.factor_score[-1])
        else:
            current_score = self._compute_factor_score_current()
        self.current_factor_score = current_score
        # 已持仓：按趋势策略管理；旧 MR 持仓也能兼容退出
        if self.position:
            if self.active_sub_strategy == "TF":
                self.manage_trend_exit(price)
            else:
                self.manage_mean_reversion_exit(price)
            return

        # 仅在“趋势”市场交易；震荡不交易
        if current_regime != 1:
            return

        if len(self.tf_ema_fast) < 2 or len(self.tf_ema_slow) < 2:
            return

        fast_now = self.tf_ema_fast[-1]
        slow_now = self.tf_ema_slow[-1]
        fast_prev = self.tf_ema_fast[-2]
        slow_prev = self.tf_ema_slow[-2]

        # 去掉宏观过滤：仅用均线金叉/死叉作为入场信号
        long_signal = (fast_now > slow_now and fast_prev <= slow_prev)
        short_signal = (fast_now < slow_now and fast_prev >= slow_prev)

        # 多因子阈值过滤：做多用 score，做空用 1-score（若模型为“做多概率”）
        st = float(self.score_threshold)
        score_available = getattr(self, "_has_pre_score", False) or (self.factor_artifacts is not None)
        if score_available and np.isfinite(current_score):
            allow_long = current_score >= st
            allow_short = (1.0 - current_score) >= st
        else:
            allow_long = True
            allow_short = True

        if long_signal and allow_long and self._can_enter_today():
            atr = self.tf_atr[-1]
            risk_ps = atr * self.tf_stop_loss_atr_multiplier
            size = self._calculate_position_size(
                price, risk_ps, self._calculate_dynamic_risk() * self.tf_risk_multiplier
            )
            if size > 0:
                self.reset_trade_state()
                self.active_sub_strategy = "TF"
                self.buy(size=size)
                self.stop_loss_price = price - risk_ps
                self._mark_entered()
            return

        if short_signal and allow_short and self._can_enter_today():
            atr = self.tf_atr[-1]
            risk_ps = atr * self.tf_stop_loss_atr_multiplier
            size = self._calculate_position_size(
                price, risk_ps, self._calculate_dynamic_risk() * self.tf_risk_multiplier
            )
            if size > 0:
                self.reset_trade_state()
                self.active_sub_strategy = "TF"
                self.sell(size=size)
                self.stop_loss_price = price + risk_ps
                self._mark_entered()


# 回测主流程
def run_backtest():
    logger.info("====== 进入回测模式 ======")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    # 结束日期按“自然日”包含整天：若为 00:00，则扩展到 23:59:59
    raw_end_dt = pd.to_datetime(CONFIG["backtest_end_date"]) if CONFIG.get("backtest_end_date") else None
    if raw_end_dt is not None:
        backtest_end_dt = raw_end_dt
        if backtest_end_dt == backtest_end_dt.normalize():
            backtest_end_dt = backtest_end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        backtest_end_dt = None
    data_lookback = timedelta(days=CONFIG["data_lookback_days"])
    data_fetch_start_date = (backtest_start_dt - data_lookback).strftime("%Y-%m-%d")
    raw_data = {
        s: fetch_binance_klines(
            s,
            CONFIG["interval"],
            data_fetch_start_date,
            backtest_end_dt.strftime("%Y-%m-%d %H:%M:%S") if backtest_end_dt is not None else None,
        )
        for s in [CONFIG["symbol"]]
    }
    raw_data = {s: d for s, d in raw_data.items() if not d.empty}
    if not raw_data:
        logger.error("数据获取失败，终止。")
        return

    for symbol, data in raw_data.items():
        logger.info(f"为 {symbol} 预处理数据...")
        processed_full = preprocess_data_for_strategy(data, symbol)
        if backtest_end_dt is not None:
            data_slice = processed_full.loc[
                pd.to_datetime(CONFIG["backtest_start_date"]): backtest_end_dt
            ].copy()
        else:
            data_slice = processed_full.loc[pd.to_datetime(CONFIG["backtest_start_date"]):].copy()
        if data_slice.empty:
            logger.warning(f"{symbol} 在模型模式下无数据，跳过。")
            continue

        print(f"\n{'='*80}\n正在回测品种: {symbol}\n{'='*80}")

        # 预计算：市场状态 + 多因子分数（滑窗预测，不含未来）以提速
        precomp_regime = pd.Series(0, index=data_slice.index, dtype=int)
        precomp_score = pd.Series(np.nan, index=data_slice.index, dtype=float)
        artifacts = _load_trend_artifacts()
        if artifacts:
            try:
                pred_sr = _predict_trend_ranging_series(processed_full, artifacts)
                if not pred_sr.empty:
                    mapped = pd.Series(
                        np.where(pred_sr == 1, -1, 1), index=pred_sr.index, name="market_regime"
                    )
                    precomp_regime = mapped.reindex(data_slice.index).fillna(0).astype(int)
            except Exception as e:
                logger.error(f"[{symbol}] 批量预测市场状态失败，将使用 Unknown(0): {e}")
        # 预计算多因子分数
        factor_arts = _load_factor_score_artifacts()
        if factor_arts:
            try:
                score_sr = _predict_factor_score_series(processed_full, factor_arts)
                if not score_sr.empty:
                    precomp_score = score_sr.reindex(data_slice.index).astype(float)
            except Exception as e:
                logger.error(f"[{symbol}] 批量预测多因子分数失败，将在运行时计算: {e}")
        # 调试输出：多因子分数概览
        try:
            if precomp_score.notna().any():
                qs = precomp_score.dropna().quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).values
                print(
                    "多因子分数分位: min={:.2f}, p10={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, p90={:.2f}, max={:.2f}".format(
                        *qs
                    )
                )
                print(f"分数均值={precomp_score.dropna().mean():.3f} | 阈值={STRATEGY_PARAMS.get('score_threshold', 0.55):.2f}")
        except Exception:
            pass

        bt = Backtest(
            data_slice,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            finalize_trades=True,
        )
        stats = bt.run(
            symbol=symbol,
            precomp_regime=precomp_regime.values,
            precomp_score=precomp_score.values if precomp_score.notna().any() else None,
        )
        print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
        print(stats)

        # 附加汇总：分月收益、长/短PnL、手续费占比、单笔收益分布
        try:
            trades = stats.get("_trades", None)
            equity_curve = stats.get("_equity_curve", None)
            commissions = float(stats.get("Commissions [$]", 0.0))
            eq_final = float(stats.get("Equity Final [$]", np.nan))
            eq_start = float(CONFIG.get("initial_cash", np.nan))
            net_profit = eq_final - eq_start if np.isfinite(eq_final) and np.isfinite(eq_start) else np.nan
            gross_profit = (net_profit + commissions) if np.isfinite(net_profit) else np.nan

            print(f"\n{'-'*40}\n          {symbol} 附加汇总\n{'-'*40}")

            # 1) 分月收益
            if equity_curve is not None and not getattr(equity_curve, "empty", True):
                try:
                    ec = equity_curve.copy()
                    # 兼容不同列名，优先 'Equity'
                    eq_col = "Equity" if "Equity" in ec.columns else ec.columns[0]
                    ec = ec[[eq_col]].rename(columns={eq_col: "Equity"})
                    # Pandas 2.2+ deprecates alias "M"; use "ME" (month-end)
                    monthly = ec["Equity"].resample("ME").agg(["first", "last"])  # type: ignore
                    monthly["Return_%"] = (monthly["last"] / monthly["first"] - 1.0) * 100
                    print("分月收益 [%]:")
                    for idx, row in monthly.iterrows():
                        print(f"  {idx.strftime('%Y-%m')}: {row['Return_%']:.2f}%")
                except Exception as e:
                    print(f"分月收益统计失败: {e}")
            else:
                print("分月收益: 无 equity 曲线数据")

            # 2) 长/短侧 PnL 汇总
            if trades is not None and not trades.empty:
                try:
                    # 兼容列名
                    pnl_col = "PnL" if "PnL" in trades.columns else None
                    size_col = "Size" if "Size" in trades.columns else None
                    ret_col = "Return [%]" if "Return [%]" in trades.columns else None
                    if pnl_col and size_col:
                        long_pnl = trades.loc[trades[size_col] > 0, pnl_col].sum()
                        short_pnl = trades.loc[trades[size_col] < 0, pnl_col].sum()
                        print(f"长侧PnL: {long_pnl:.2f} | 短侧PnL: {short_pnl:.2f}")
                        print(f"长侧笔数: {(trades[size_col] > 0).sum()} | 短侧笔数: {(trades[size_col] < 0).sum()}")
                    else:
                        print("长/短PnL: 缺少列，跳过")

                    # 3) 手续费占比
                    if np.isfinite(gross_profit) and gross_profit != 0:
                        print(
                            f"手续费: ${commissions:,.2f} | 占毛收益比: {commissions / abs(gross_profit) * 100:.2f}% | 单笔均值: ${commissions / max(1, len(trades)):.2f}"
                        )
                    else:
                        print(f"手续费: ${commissions:,.2f}")

                    # 4) 单笔收益分布
                    if ret_col and ret_col in trades.columns:
                        r = trades[ret_col].dropna().astype(float)
                        if not r.empty:
                            qs = r.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).values
                            print(
                                "单笔收益[%] 分位: min={:.2f}, p10={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, p90={:.2f}, max={:.2f}".format(
                                    *qs
                                )
                            )
                            wins = (r > 0).sum()
                            losses = (r <= 0).sum()
                            print(
                                f"交易分布: 胜 {wins} / 负 {losses} | 胜均值 {r[r>0].mean():.2f}% | 负均值 {r[r<=0].mean():.2f}%"
                            )
                        else:
                            print("单笔收益分布: 无数据")
                    elif pnl_col and pnl_col in trades.columns:
                        r = trades[pnl_col].dropna().astype(float)
                        if not r.empty:
                            qs = r.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).values
                            print(
                                "单笔PnL[$] 分位: min={:.2f}, p10={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, p90={:.2f}, max={:.2f}".format(
                                    *qs
                                )
                            )
                            wins = (r > 0).sum()
                            losses = (r <= 0).sum()
                            print(
                                f"交易分布: 胜 {wins} / 负 {losses} | 胜均值 ${r[r>0].mean():.2f} | 负均值 ${r[r<=0].mean():.2f}"
                            )
                        else:
                            print("单笔收益分布: 无数据")
                except Exception as e:
                    print(f"长短PnL/分布统计失败: {e}")
            else:
                print("交易明细: 无 _trades 数据")
        except Exception as e:
            print(f"附加汇总生成失败: {e}")
        if CONFIG["show_plots"]:
            bt.plot()


# --- 6. 主程序入口 ---
if __name__ == "__main__":
    # 仅运行回测：模型训练功能已移除
    run_backtest()
