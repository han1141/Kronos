# --- 核心库导入 ---
import logging
import time
import warnings
import os
import re

# --- 数据分析和机器学习库 ---
import numpy as np
import pandas as pd
import requests
import ta
from backtesting import Backtest, Strategy
try:
    import joblib
    import tensorflow as tf
    ML_LIBS_INSTALLED = True
except Exception:
    ML_LIBS_INSTALLED = False

# 忽略一些常见的未来警告
warnings.simplefilter(action="ignore", category=FutureWarning)
# 过滤 sklearn 在列名不匹配时的提示（我们会传入DataFrame以根除，但此处兜底）
warnings.filterwarnings(
    "ignore",
    message=r".*X does not have valid feature names, but .* was fitted with feature names.*",
)
# 可选：压制 backtesting 的相对仓位保证金告警（我们已改为绝对手数，下行仅兜底）
warnings.filterwarnings(
    "ignore",
    message=r".*Broker canceled the relative-sized order due to insufficient margin.*",
)

# --- 1. 全局配置和策略参数 ---

# 回测配置
SYMBOL = "ETHUSDT"
INTERVAL = "15m"
START_DATE = "2025-01-01"
END_DATE = "2025-11-11"
INITIAL_CASH = 100_000
COMMISSION = 0.0006  # 币安费率: 0.06%

# 策略核心参数
STRATEGY_PARAMS = {
    # --- 市场状态检测参数 ---
    "regime_adx_period": 14,
    "regime_atr_period": 14,
    "regime_atr_slope_period": 6,
    "regime_rsi_period": 14,
    "regime_rsi_vol_period": 14,
    "regime_norm_period": 252,
    "regime_hurst_period": 80,
    "regime_score_weight_adx": 0.55,
    "regime_score_weight_atr": 0.3,
    "regime_score_weight_rsi": 0.1,
    "regime_score_weight_hurst": 0.05,
    "regime_score_threshold": 0.4,
    # --- 趋势跟随(TF)模块参数 ---
    "tf_donchian_period": 24,
    "tf_ema_fast_period": 21,
    "tf_ema_slow_period": 60,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.6,
    # --- 均值回归(MR)模块参数 ---
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_stop_loss_atr_multiplier": 1.5,
    # --- 多周期过滤和信号权重 ---
    "mtf_period": 40,
    "score_entry_threshold": 0.5,
    # 与 okx_req_1.py 一致的打分权重（含 ML）
    "score_weights_tf": {
        "breakout": 0.22,
        "momentum": 0.18,
        "mtf": 0.12,
        "ml": 0.23,
        "advanced_ml": 0.25,
    },
    # 风控（参考 okx_req_1）
    "tsl_enabled": True,
    "tsl_activation_profit_pct": 0.007,
    "tsl_activation_atr_mult": 1.8,
    "tsl_trailing_atr_mult": 2.2,
    "kelly_trade_history": 25,
    "default_risk_pct": 0.012,
    "max_risk_pct": 0.035,
    "mr_risk_multiplier": 0.5,
    # --- 迁移自 okx_req_1.py 的 ML 配置 ---
    "keras_model_path": "models/eth_trend_model_v1_15m.keras",
    "scaler_path": "models/eth_trend_scaler_v1_15m.joblib",
    "feature_columns_path": "models/feature_columns_15m.joblib",
    "keras_sequence_length": 60,
    # 日志控制
    "verbose_entry_log": False,
    "log_missing_ml_features": True,
    # ML 调试：输出推理上下文与异常栈（默认关闭以减少噪音）
    "ml_debug": False,
    # 每根K线打印 ML 分数（默认关闭）
    "ml_log_each_bar": False,
}

# --- 2. 日志系统设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# --- 3. 特征工程 & 原始策略逻辑 ---
# 这部分代码是从 `ultimate_strategy.py` 移植过来的，用于预先计算所有指标。


def compute_hurst(ts, max_lag=100):
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau = [
        np.std(np.subtract(ts[lag:], ts[:-lag]))
        for lag in lags
        if np.std(np.subtract(ts[lag:], ts[:-lag])) > 0
    ]
    if len(tau) < 2:
        return 0.5
    try:
        hurst = np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0]
        return max(0.0, min(1.0, hurst))
    except:
        return 0.5


def add_ml_features_ported(df: pd.DataFrame) -> pd.DataFrame:
    p = STRATEGY_PARAMS
    norm = lambda s: (
        (s - s.rolling(p["regime_norm_period"]).min())
        / (
            s.rolling(p["regime_norm_period"]).max()
            - s.rolling(p["regime_norm_period"]).min()
        )
    ).fillna(0.5)
    adx = ta.trend.ADXIndicator(df.High, df.Low, df.Close, p["regime_adx_period"]).adx()
    atr = ta.volatility.AverageTrueRange(
        df.High, df.Low, df.Close, p["regime_atr_period"]
    ).average_true_range()
    rsi = ta.momentum.RSIIndicator(df.Close, p["regime_rsi_period"]).rsi()
    bb = ta.volatility.BollingerBands(
        df.Close, window=p["mr_bb_period"], window_dev=p["mr_bb_std"]
    )
    df["regime_adx"] = adx  # 保留原始 ADX 值用于阈值过滤
    df["feature_adx_norm"] = norm(adx)
    df["feature_atr_slope_norm"] = norm(
        (atr - atr.shift(p["regime_atr_slope_period"]))
        / atr.shift(p["regime_atr_slope_period"])
    )
    df["feature_rsi_vol_norm"] = 1 - norm(rsi.rolling(p["regime_rsi_vol_period"]).std())
    df["feature_hurst"] = (
        df.Close.rolling(p["regime_hurst_period"])
        .apply(lambda x: compute_hurst(np.log(x + 1e-9)), raw=False)
        .fillna(0.5)
    )
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["feature_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    return df


def run_advanced_model_inference(df: pd.DataFrame) -> pd.DataFrame:
    """简化版高级ML信号：对 ai_filter_signal 做滚动均值，若无ML库则返回0。"""
    if not ML_LIBS_INSTALLED:
        df["advanced_ml_signal"] = 0.0
        return df
    base = df.get("ai_filter_signal", pd.Series(0, index=df.index))
    df["advanced_ml_signal"] = base.rolling(24).mean().fillna(0)
    return df


def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    """为Keras模型准备与 okx_req_1.py 对齐的技术特征。"""
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    df["EMA_8"] = ta.trend.EMAIndicator(close=close, window=8).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = (
        adx_indicator.adx(), adx_indicator.adx_pos(), adx_indicator.adx_neg()
    )
    atr_raw = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    df["ATRr_14"] = (atr_raw / close) * 100
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
    df["BBU_20_2.0"], df["BBM_20_2.0"], df["BBL_20_2.0"] = (
        bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
    )
    df["BBB_20_2.0"], df["BBP_20_2.0"] = (
        bb.bollinger_wband(), bb.bollinger_pband()
    )
    macd = ta.trend.MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = (
        macd.macd(), macd.macd_signal(), macd.macd_diff()
    )
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["volume_change_rate"] = volume.pct_change()
    return df


# --- ML 组件加载与批量推理预计算 ---

def _load_ml_components_from_params(p):
    if not ML_LIBS_INSTALLED:
        return None, None, None
    try:
        model_path = p.get("keras_model_path")
        scaler_path = p.get("scaler_path")
        feat_cols_path = p.get("feature_columns_path")
        if not (model_path and os.path.exists(model_path)):
            return None, None, None
        if not (scaler_path and os.path.exists(scaler_path)):
            return None, None, None
        if not (feat_cols_path and os.path.exists(feat_cols_path)):
            return None, None, None
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(feat_cols_path)
        if hasattr(feature_columns, "tolist"):
            feature_columns = feature_columns.tolist()
        return model, scaler, feature_columns
    except Exception:
        return None, None, None


def _sanitize_alias(col: str) -> str:
    alias = re.sub(r"[^0-9A-Za-z_]", "_", col)
    if alias and alias[0].isdigit():
        alias = f"f_{alias}"
    return alias


def precompute_keras_scores(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    以向量化/批量方式为整段数据预计算 Keras 置信度分数，并写入列 'ml_confidence_score'。
    若组件不可用或列缺失，则返回原 df（分数列填 0）。
    """
    model, scaler, feature_columns = _load_ml_components_from_params(p)
    if not all([model is not None, scaler is not None, feature_columns]):
        df["ml_confidence_score"] = 0.0
        return df

    seq_len = int(p.get("keras_sequence_length", 60))

    # 准备特征矩阵（按原始训练列名顺序）。支持非法标识符的别名回退。
    sequences = {}
    for col in feature_columns:
        s = df.get(col)
        if s is None:
            alias = _sanitize_alias(col)
            s = df.get(alias)
        if s is None:
            # 缺列则整体置 0
            df["ml_confidence_score"] = 0.0
            return df
        sequences[col] = s
    feat_df = pd.DataFrame(sequences)
    # 前向填充 + 0 兜底
    feat_df = feat_df.fillna(method="ffill").fillna(0)

    # 使用滑窗构造 [N, seq_len, n_features]
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        X = sliding_window_view(feat_df.values, window_shape=(seq_len, feat_df.shape[1]))
        # sliding_window_view 在二维数组上按 (rows, cols) 窗口返回形状 (n_rows-seq+1, 1, 1, seq, n_features)
        # 简化为 (n_samples, seq_len, n_features)
        X = X[:, 0, 0, :, :]
    except Exception:
        # 兼容实现：退化为循环（较慢，但只在环境不支持时）
        values = feat_df.values
        n = len(values)
        if n < seq_len:
            df["ml_confidence_score"] = 0.0
            return df
        X = np.stack([values[i - seq_len : i] for i in range(seq_len, n + 1)], axis=0)

    n_samples, _, n_features = X.shape
    if n_samples <= 0:
        df["ml_confidence_score"] = 0.0
        return df

    # 按批缩放与推理，避免内存峰值
    batch = int(p.get("ml_batch_size", 512))
    preds = np.zeros((n_samples,), dtype=float)
    for i in range(0, n_samples, batch):
        sl = slice(i, min(i + batch, n_samples))
        # 缩放：将 [B, T, F] 变平为 [B*T, F]，再 reshape 回去
        flat = X[sl].reshape(-1, n_features)
        # 使用带列名的DataFrame，避免 sklearn 的feature name 警告
        flat_df = pd.DataFrame(flat, columns=feature_columns)
        flat_scaled = scaler.transform(flat_df)
        X_scaled = flat_scaled.reshape(-1, seq_len, n_features)
        preds_batch = model.predict(X_scaled, verbose=0)
        preds[sl] = preds_batch.reshape(-1)

    # 映射到 [-1, 1]
    scores = (preds - 0.5) * 2.0
    # 对齐索引：从第 seq_len-1 根开始有分数
    out = pd.Series(0.0, index=df.index)
    out.iloc[seq_len - 1 :] = scores
    df["ml_confidence_score"] = out.values
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["regime_score"] = df["feature_regime_score"]
    df["trend_regime"] = np.where(
        df["regime_score"] > STRATEGY_PARAMS["regime_score_threshold"], "趋势", "震荡"
    )
    # 与 okx_req_1 对齐：计算年化波动率列，供 ML 特征使用
    try:
        vol = df["Close"].pct_change().rolling(24 * 7).std() * np.sqrt(24 * 365)
        df["volatility"] = vol
        # 可选：分档（不强依赖，仅保持一致）
        low_vol, high_vol = df["volatility"].quantile(0.33), df["volatility"].quantile(0.67)
        df["volatility_regime"] = pd.cut(
            df["volatility"],
            bins=[-np.inf, low_vol, high_vol, np.inf],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )
    except Exception:
        df["volatility"] = 0.0
    df["market_regime"] = np.where(df["trend_regime"] == "趋势", 1, -1)
    return df


class UltimateStrategyCalculator:
    """
    这个类仅用于在回测前，一次性地计算所有需要的技术指标。
    它不是一个可交易的策略，而是一个特征计算器。
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.data = df.copy()
        self.symbol = symbol
        # 将策略参数加载到实例中
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)

    def compute_all_features(self, trader, kline_interval: str):
        logger.debug(f"[{self.symbol}] 开始计算特征...")
        # 确保数据完整性
        self.data.dropna(
            subset=["Open", "High", "Low", "Close", "Volume"], inplace=True
        )
        if len(self.data) < self.tf_donchian_period:
            logger.warning(f"数据量过少({len(self.data)}条)，无法计算特征。")
            return

        # 计算所有需要的指标
        # 简单 AI 过滤信号（与 okx_req_1 对齐）
        rsi_filter = ta.momentum.RSIIndicator(self.data.Close, 14).rsi()
        self.data["ai_filter_signal"] = (
            ((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50)
        ).clip(-1, 1).fillna(0)

        self.data = run_advanced_model_inference(self.data)
        self.data = add_ml_features_ported(self.data)
        self.data = add_features_for_keras_model(self.data)
        self.data = add_market_regime_features(self.data)
        # 预计算 Keras ML 置信度，避免回测循环中重复 predict
        try:
            self.data = precompute_keras_scores(self.data, STRATEGY_PARAMS)
        except Exception:
            # 失败则回退为 0
            self.data["ml_confidence_score"] = 0.0

        # 多时间框架(MTF)信号
        # --- !!! 这里是修改点 !!! ---
        if (
            data_1d := trader.fetch_history_klines(
                self.symbol, bar="1d", limit=self.mtf_period + 50
            )
        ) is not None and not data_1d.empty:
            sma = ta.trend.SMAIndicator(
                data_1d["Close"], window=self.mtf_period
            ).sma_indicator()
            mtf_signal_1d = pd.Series(
                np.where(data_1d["Close"] > sma, 1, -1), index=data_1d.index
            )
            self.data["mtf_signal"] = mtf_signal_1d.reindex(
                self.data.index, method="ffill"
            ).fillna(0)
        else:
            self.data["mtf_signal"] = 0  # 如果获取失败，则为中性信号

        # 宏观趋势过滤
        df_4h = self.data["Close"].resample("4H").last().to_frame()
        df_4h["macro_ema"] = ta.trend.EMAIndicator(
            df_4h["Close"], window=50
        ).ema_indicator()
        df_4h["macro_trend"] = np.where(df_4h["Close"] > df_4h["macro_ema"], 1, -1)
        self.data["macro_trend_filter"] = (
            df_4h["macro_trend"].reindex(self.data.index, method="ffill").fillna(0)
        )

        # 趋势跟随(TF)指标
        self.data["tf_atr"] = ta.volatility.AverageTrueRange(
            self.data.High, self.data.Low, self.data.Close, self.tf_atr_period
        ).average_true_range()
        self.data["tf_donchian_h"] = (
            self.data.High.rolling(self.tf_donchian_period).max().shift(1)
        )
        self.data["tf_donchian_l"] = (
            self.data.Low.rolling(self.tf_donchian_period).min().shift(1)
        )
        self.data["tf_ema_fast"] = ta.trend.EMAIndicator(
            self.data.Close, self.tf_ema_fast_period
        ).ema_indicator()
        self.data["tf_ema_slow"] = ta.trend.EMAIndicator(
            self.data.Close, self.tf_ema_slow_period
        ).ema_indicator()

        # 均值回归(MR)指标
        bb = ta.volatility.BollingerBands(
            self.data.Close, self.mr_bb_period, self.mr_bb_std
        )
        self.data["mr_bb_upper"] = bb.bollinger_hband()
        self.data["mr_bb_lower"] = bb.bollinger_lband()
        self.data["mr_bb_mid"] = bb.bollinger_mavg()
        stoch_rsi = ta.momentum.StochRSIIndicator(
            self.data.Close, window=14, smooth1=3, smooth2=3
        )
        self.data["mr_stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        self.data["mr_stoch_rsi_d"] = stoch_rsi.stochrsi_d()

        # 清理数据
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.fillna(method="ffill", inplace=True)
        self.data.dropna(inplace=True)  # 删除计算后仍然存在的NaN行
        # 为回测访问兼容性添加安全别名列（将不合法标识符列名复制一份可访问的别名）
        def _alias(col: str) -> str:
            a = re.sub(r"[^0-9A-Za-z_]", "_", col)
            # 别名必须是有效标识符
            if not a or a[0].isdigit():
                a = f"f_{a}"
            return a
        new_cols = {}
        for c in list(self.data.columns):
            if not c.isidentifier():
                a = _alias(c)
                if a not in self.data.columns:
                    new_cols[a] = c
        for a, c in new_cols.items():
            self.data[a] = self.data[c]
        logger.debug(f"[{self.symbol}] 特征计算完成。")


# --- 4. 数据获取函数（带磁盘缓存） ---


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _time_key(s: str) -> str:
    try:
        ts = pd.to_datetime(s)
        return ts.strftime("%Y%m%d%H%M%S")
    except Exception:
        return re.sub(r"[^0-9A-Za-z]", "", str(s))


def _cache_path(symbol: str, interval: str, start_str: str, end_str: str | None,
                base_dir: str = "data/cache/binance") -> str:
    _ensure_dir(base_dir)
    s_key = _time_key(start_str)
    e_key = _time_key(end_str) if end_str is not None else "latest"
    fname = f"{symbol.upper()}__{interval}__{s_key}__{e_key}.csv"
    return os.path.join(base_dir, fname)


def fetch_binance_klines(
    symbol,
    interval,
    start_str,
    end_str=None,
    limit=1000,
    use_cache: bool = True,
    cache_dir: str = "data/cache/binance",
):
    if end_str is None:
        end_str = pd.to_datetime("now", utc=True).strftime("%Y-%m-%d %H:%M:%S")

    url = "https://api.binance.com/api/v3/klines"
    cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)
    all_data = []

    # 先尝试从缓存读取
    cache_file = _cache_path(symbol, interval, start_str, end_str, cache_dir)
    if use_cache and os.path.isfile(cache_file):
        try:
            logger.info(
                f"从缓存加载 {symbol} {interval} [{start_str} ~ {end_str}] -> {cache_file}"
            )
            df_cached = pd.read_csv(
                cache_file, parse_dates=["timestamp"], index_col="timestamp"
            ).sort_index()
            if set(["Open", "High", "Low", "Close", "Volume"]).issubset(
                df_cached.columns
            ):
                return df_cached
        except Exception as e:
            logger.warning(f"读取缓存失败，将从网络获取。原因: {e}")

    logger.info(f"正在从币安获取 {symbol} 从 {start_str} 到 {end_str} 的K线数据...")

    while start_ts < end_ts:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ts,
            "limit": limit,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_data.extend(data)
            start_ts = data[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.error(f"获取数据失败: {e}")
            time.sleep(5)

    if not all_data:
        logger.warning(f"未能获取到 {symbol} 的任何数据。")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[*cols, "c1", "c2", "c3", "c4", "c5", "c6"])[
        cols
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df.set_index("timestamp").sort_index()

    # 写入缓存
    if use_cache:
        try:
            _ensure_dir(os.path.dirname(cache_file))
            result.to_csv(cache_file)
            logger.info(f"已缓存到 {cache_file}")
        except Exception as e:
            logger.warning(f"写入缓存失败: {e}")

    logger.info(f"✅ 成功获取 {len(result)} 条 {symbol} 的K线数据")
    return result


# --- 5. Backtesting.py 策略集成 ---


class UltimateBacktestStrategy(Strategy):
    """
    这是与 `backtesting.py` 库兼容的策略封装类。
    """

    def init(self):
        # 将预先计算好的指标作为 `self.I` 注册，以便在图表中查看
        self.market_regime = self.I(
            lambda x: self.data.market_regime, self.data.Close, name="Market_Regime"
        )
        self.macro_trend_filter = self.I(
            lambda x: self.data.macro_trend_filter, self.data.Close, name="Macro_Trend"
        )
        self.tf_donchian_h = self.I(
            lambda x: self.data.tf_donchian_h, self.data.Close, name="Donchian_H"
        )
        self.tf_donchian_l = self.I(
            lambda x: self.data.tf_donchian_l, self.data.Close, name="Donchian_L"
        )
        self.mr_bb_upper = self.I(
            lambda x: self.data.mr_bb_upper, self.data.Close, name="BB_Upper"
        )
        self.mr_bb_lower = self.I(
            lambda x: self.data.mr_bb_lower, self.data.Close, name="BB_Lower"
        )

        # 存储策略参数
        self.p = STRATEGY_PARAMS
        self.score_threshold = self.p["score_entry_threshold"]
        self.sl_atr_mult_tf = self.p["tf_stop_loss_atr_multiplier"]
        self.sl_atr_mult_mr = self.p["mr_stop_loss_atr_multiplier"]
        # Kelly 历史与TSL占位（可扩展）
        self._recent_trade_returns = []
        self._tsl_active = False
        self._tsl_stop = None
        # ML 组件（可选）
        self.keras_model = None
        self.scaler = None
        self.feature_columns = None
        self._load_models()
        # 可选：在初始化时检查一次缺失的 ML 特征列
        if self.p.get("log_missing_ml_features", True):
            try:
                self._log_missing_ml_features_once()
            except Exception:
                pass

    def next(self):
        # 获取当前K线的价格和指标
        price = self.data.Close[-1]
        atr = self.data.tf_atr[-1]

        # --- 入场逻辑 ---
        if not self.position:
            action = None
            confidence = 1.0

            # 趋势跟随 (TF) 逻辑 - 与 okx_req_1 一致的入场判定（不做额外阈值预过滤）
            if self.market_regime[-1] == 1:  # 趋势市场
                score = self._calculate_entry_score()
                if self.macro_trend_filter[-1] == 1 and score > self.score_threshold:
                    action = "BUY_TF"
                    confidence = float(score)
                elif self.macro_trend_filter[-1] == -1 and score < -self.score_threshold:
                    action = "SELL_TF"
                    confidence = float(abs(score))

            # 均值回归 (MR) 逻辑
            elif self.market_regime[-1] == -1:  # 震荡市场
                mr_signal = self._define_mr_entry_signal()
                if self.macro_trend_filter[-1] == 1 and mr_signal == 1:
                    action = "BUY_MR"
                    confidence = 1.0
                elif self.macro_trend_filter[-1] == -1 and mr_signal == -1:
                    action = "SELL_MR"
                    confidence = 1.0

            # 如果有交易信号，则执行
            if action:
                # 动态风险与仓位（近似 okx_req_1 的Kelly方案：默认用 default_risk_pct）
                risk_multiplier = 1.0 if "TF" in action else self.p.get("mr_risk_multiplier", 0.5)
                base_risk = self._calculate_dynamic_risk()
                risk_pct = max(0.0, min(self.p.get("max_risk_pct", 0.035), base_risk * risk_multiplier * float(confidence)))
                sl_atr_mult = self.sl_atr_mult_tf if "TF" in action else self.sl_atr_mult_mr
                stop_loss_dist = atr * sl_atr_mult

                # 计算资金占比下单（相对大小），并限制不超过95%资金
                # 推导：为使止损风险≈ risk_pct * equity，有 size_frac = risk_pct * price / stop_loss_dist
                denom = max(1e-9, stop_loss_dist)
                size_frac = max(0.0, min(0.95, (risk_pct * price) / denom))
                if size_frac <= 0:
                    return

                if "BUY" in action:
                    self.buy(sl=price - stop_loss_dist, size=size_frac)
                elif "SELL" in action:
                    self.sell(sl=price + stop_loss_dist, size=size_frac)

                # 可选：记录一次分项打分快照（仅在 verbose_entry_log 为 True 时启用）
                if self.p.get("verbose_entry_log", False):
                    try:
                        w = self.p["score_weights_tf"]
                        ml_s = self.get_ml_confidence_score()
                        adv_s = float(getattr(self.data, "advanced_ml_signal")[-1]) if hasattr(self.data, "advanced_ml_signal") else 0.0
                        mo_s = 1 if self.data.tf_ema_fast[-1] > self.data.tf_ema_slow[-1] else -1
                        b_s = 1 if self.data.High[-1] > self.tf_donchian_h[-1] else (-1 if self.data.Low[-1] < self.tf_donchian_l[-1] else 0)
                        parts = {
                            "breakout": b_s * w.get("breakout", 0),
                            "momentum": mo_s * w.get("momentum", 0),
                            "mtf": float(self.data.mtf_signal[-1]) * w.get("mtf", 0),
                            "ml": ml_s * w.get("ml", 0),
                            "advanced_ml": adv_s * w.get("advanced_ml", 0),
                        }
                        logger.info(
                            f"{action} price={price:.4f} size_frac={size_frac:.4f} risk_pct={risk_pct:.4f} "
                            f"ml_raw={ml_s:.6f} adv_raw={adv_s:.6f} score_parts={parts} total={sum(parts.values()):.4f}"
                        )
                    except Exception:
                        pass

        # --- 出场逻辑 ---
        # 初始止损已在 self.buy/sell 中设置。
        # 这里可以添加更复杂的出场逻辑，例如均值回归的目标止盈。
        else:
            # 简单MR止盈
            if self.position.is_long and self.data.Close[-1] >= self.data.mr_bb_mid[-1]:
                self.position.close()
            elif self.position.is_short and self.data.Close[-1] <= self.data.mr_bb_mid[-1]:
                self.position.close()
            # 可选：移动止损（近似 okx TSL）
            if self.p.get("tsl_enabled", False):
                try:
                    entry = float(self.position.entry_price)
                    price = float(self.data.Close[-1])
                    atr = float(self.data.tf_atr[-1])
                    act_pct = float(self.p.get("tsl_activation_profit_pct", 0.007))
                    trail_mult = float(self.p.get("tsl_trailing_atr_mult", 2.2))
                    if self.position.is_long:
                        profit_pct = (price - entry) / max(1e-9, entry)
                        if profit_pct >= act_pct:
                            ts = price - atr * trail_mult
                            if price <= ts:
                                self.position.close()
                    else:
                        profit_pct = (entry - price) / max(1e-9, entry)
                        if profit_pct >= act_pct:
                            ts = price + atr * trail_mult
                            if price >= ts:
                                self.position.close()
                except Exception:
                    pass

    # 策略内部的辅助计算函数
    def _calculate_entry_score(self, return_parts: bool = False):
        # 与 okx_req_1 一致的打分：简单突破与EMA方向，不加额外阈值
        w = self.p["score_weights_tf"]
        last = self.data
        b_s = (
            1
            if last.High[-1] > self.tf_donchian_h[-1]
            else -1 if last.Low[-1] < self.tf_donchian_l[-1] else 0
        )
        mo_s = 1 if last.tf_ema_fast[-1] > last.tf_ema_slow[-1] else -1
        ml_score = self.get_ml_confidence_score()
        try:
            adv_ml_score = float(getattr(self.data, "advanced_ml_signal")[-1])
        except Exception:
            adv_ml_score = 0.0
        score = (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + last.mtf_signal[-1] * w.get("mtf", 0)
            + ml_score * w.get("ml", 0)
            + adv_ml_score * w.get("advanced_ml", 0)
        )
        if return_parts:
            direction = 1 if score >= 0 else -1
            return score, {"b_s": b_s, "mo_s": mo_s, "mtf": last.mtf_signal[-1], "dir": direction}
        return score

    def _load_models(self):
        if not ML_LIBS_INSTALLED:
            return
        try:
            model_path = self.p.get("keras_model_path")
            scaler_path = self.p.get("scaler_path")
            feat_cols_path = self.p.get("feature_columns_path")
            if not (model_path and os.path.exists(model_path)):
                return
            if not (scaler_path and os.path.exists(scaler_path)):
                return
            if not (feat_cols_path and os.path.exists(feat_cols_path)):
                return
            self.keras_model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(feat_cols_path)
            if hasattr(self.feature_columns, "tolist"):
                self.feature_columns = self.feature_columns.tolist()
            logger.info("已加载 Keras 模型与特征组件用于回测 ML 信号")
            if self.p.get("verbose_entry_log", False):
                try:
                    preview = self.feature_columns[:8]
                    logger.info(f"ML特征列数={len(self.feature_columns)} 预览={preview}")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"加载ML模型失败：{e}")
            self.keras_model = None
            self.scaler = None
            self.feature_columns = None

    def get_ml_confidence_score(self) -> float:
        # 优先使用预计算列，避免循环内预测
        try:
            return float(getattr(self.data, "ml_confidence_score")[-1])
        except Exception:
            pass
        if not ML_LIBS_INSTALLED:
            return 0.0
        if not all([self.keras_model is not None, self.scaler is not None, self.feature_columns]):
            return 0.0
        seq_len = int(self.p.get("keras_sequence_length", 60))
        # 收集序列并构造与训练时相同的列名DataFrame
        sequences = {}
        for col in self.feature_columns:
            # 先尝试原名（允许含点号的列名），不行再用别名
            series_like = None
            try:
                series_like = getattr(self.data, col)
            except AttributeError:
                alias = re.sub(r"[^0-9A-Za-z_]", "_", col)
                if alias and alias[0].isdigit():
                    alias = f"f_{alias}"
                try:
                    series_like = getattr(self.data, alias)
                except AttributeError:
                    if not getattr(self, "_ml_missing_logged", False) and self.p.get("verbose_entry_log", False):
                        logger.debug(f"ML特征缺失，跳过打分: {col} (alias={alias})")
                        self._ml_missing_logged = True
                    return 0.0
            arr = np.asarray(series_like)
            if len(arr) < seq_len:
                return 0.0
            sequences[col] = arr[-seq_len:]
        if not sequences:
            return 0.0
        try:
            df_seq = pd.DataFrame(sequences)
            # 保证列顺序与训练一致
            df_seq = df_seq[self.feature_columns]
            # 诊断 NaN（仅调试模式）
            if self.p.get("ml_debug", False):
                nan_total = int(df_seq.isna().sum().sum())
                logger.debug(
                    f"ML df_seq shape={df_seq.shape}, nan_total={nan_total} at {self.data.index[-1]}"
                )
            if df_seq.isna().any().any():
                df_seq = df_seq.fillna(method="ffill").fillna(0)
            scaled = self.scaler.transform(df_seq)
            x = np.expand_dims(scaled, axis=0)
            pred = self.keras_model.predict(x, verbose=0)
            score = float((pred[0][0] - 0.5) * 2.0)
            # 在每根K线计算后打印 ML 分数（仅在开关启用时）
            if self.p.get("ml_log_each_bar", False):
                try:
                    logger.info(f"Time: {self.data.index[-1]}, ML Score: {score:.4f}")
                except Exception:
                    pass
            return score
        except Exception as e:
            if self.p.get("ml_debug", False):
                # 打印更丰富的上下文信息与堆栈
                try:
                    shape = df_seq.shape if 'df_seq' in locals() else None
                    cols_ok = 'df_seq' in locals() and list(df_seq.columns)[:8]
                    logger.exception(
                        f"Keras 推理失败: {e}; df_seq_shape={shape}; cols_preview={cols_ok}"
                    )
                except Exception:
                    logger.exception(f"Keras 推理失败: {e}")
            else:
                # 非调试模式：仅第一次打印告警，后续静默，以免刷屏
                if not getattr(self, "_ml_exc_logged", False):
                    logger.warning(f"Keras 推理失败：{e}")
                    self._ml_exc_logged = True
            return 0.0

    def _calculate_dynamic_risk(self) -> float:
        # 简化版 Kelly：回测中默认使用固定风险百分比
        return float(self.p.get("default_risk_pct", 0.012))

    # --- 辅助：一次性输出缺失的 ML 特征列 ---
    def _sanitize_feature_name(self, col: str) -> str:
        alias = re.sub(r"[^0-9A-Za-z_]", "_", col)
        if alias and alias[0].isdigit():
            alias = f"f_{alias}"
        return alias

    def _log_missing_ml_features_once(self):
        if getattr(self, "_ml_features_logged", False):
            return
        self._ml_features_logged = True
        if not all([self.keras_model is not None, self.scaler is not None, self.feature_columns]):
            logger.info("ML未启用或特征列未加载，跳过特征完整性检查。")
            return
        missing, present = [], []
        for col in self.feature_columns:
            ok = True
            try:
                getattr(self.data, col)
            except AttributeError:
                alias = self._sanitize_feature_name(col)
                try:
                    getattr(self.data, alias)
                except AttributeError:
                    ok = False
            (present if ok else missing).append(col)
        if missing:
            logger.warning(
                f"ML特征缺失 {len(missing)}/{len(self.feature_columns)} 项（仅列出前10个）：{missing[:10]}"
            )
        else:
            logger.info("ML特征完整，无缺失。")

    def _define_mr_entry_signal(self):
        # 使用属性访问来获取当前和前一个 K 线的数据
        last_close, prev_close = self.data.Close[-1], self.data.Close[-2]
        last_k, last_d = self.data.mr_stoch_rsi_k[-1], self.data.mr_stoch_rsi_d[-1]
        prev_k, prev_d = self.data.mr_stoch_rsi_k[-2], self.data.mr_stoch_rsi_d[-2]
        bb_lower, bb_upper = (
            self.data.mr_bb_lower,
            self.data.mr_bb_upper,
        )  # 获取布林带上下轨

        # 定义买入信号
        if (
            prev_close < bb_lower[-2]
            and last_close > bb_lower[-1]
            and last_k > last_d
            and prev_k <= prev_d
            and last_k < 40
        ):
            return 1  # 买入信号

        # 定义卖出信号
        if (
            prev_close > bb_upper[-2]
            and last_close < bb_upper[-1]
            and last_k < last_d
            and prev_k >= prev_d
            and last_k > 60
        ):
            return -1  # 卖出信号

        return 0  # 无信号


# --- 6. 主执行程序 ---
if __name__ == "__main__":
    # 步骤 1: 获取历史数据
    data_df = fetch_binance_klines(SYMBOL, INTERVAL, START_DATE, END_DATE)

    if not data_df.empty:
        # 步骤 2: 预先计算所有策略指标
        logger.info("正在预计算所有策略指标...")

        # 创建一个模拟的'trader'对象，以满足特征计算函数的需求
        class MockTrader:
            def fetch_history_klines(self, symbol, bar, limit):
                # 为MTF指标获取真实日线数据
                return fetch_binance_klines(symbol, bar, START_DATE, END_DATE, limit)

        # 使用计算器类来生成带有所有指标的DataFrame
        calculator = UltimateStrategyCalculator(df=data_df.copy(), symbol=SYMBOL)
        calculator.compute_all_features(trader=MockTrader(), kline_interval=INTERVAL)
        augmented_df = calculator.data

        # 步骤 3: 设置并运行回测
        logger.info("设置并运行回测...")
        bt = Backtest(
            augmented_df,
            UltimateBacktestStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION,
            trade_on_close=True,
        )

        stats = bt.run()

        # 步骤 4: 打印结果并生成图表
        logger.info("回测完成，输出结果:")
        print(stats)
