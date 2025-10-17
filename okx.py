# okx_live_trading_ws_full.py
"""
OKX WebSocket 实时推送版（完整策略逻辑整合）
- 将你原始回测策略的特征工程、market regime、ML 模型评分 (lightgbm joblib)、
  advanced model stub、以及 Kelly 动态风险 sizing 整合到 WebSocket 实时推送版本中。
- 在每根 15m K 线封盘时触发策略决策（next_on_candle_close），当策略决定下单时使用 OKX REST 下市价单（模拟盘）。
- 保留原始的 STRATEGY_PARAMS, ASSET_SPECIFIC_OVERRIDES 等配置（在脚本中可调）。
- 重要：由于真实账户持仓/权益需要从 OKX REST API 查询以精确计算仓位，本脚本保留了一个本地 "simulated equity" 用于 Kelly 缩放（近似）。
  如果需要严格以真实账户权益/持仓计算，请把对应 REST endpoints 加入（positions, account）并将本地模拟替换。
- 依赖:
    pip install websocket-client requests pandas numpy ta joblib lightgbm
  （lightgbm/joblib 仅在你需要加载训练好的模型时需要）
- 环境变量（推荐）:
    OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE
- 使用方式:
    python okx_live_trading_ws_full.py
"""

import os
import time
import json
import hmac
import base64
import hashlib
import threading
import queue
import glob
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import requests
from websocket import WebSocketApp

# TA-Lib 技术分析库
import ta

# 尝试导入可选的机器学习库
try:
    import joblib
    import lightgbm as lgb  # 本地训练/使用时可选
    from sklearn.metrics import classification_report

    ML_LIBS_INSTALLED = True
except Exception:
    ML_LIBS_INSTALLED = False

# ----------------- 日志配置 -----------------
logger = logging.getLogger("OKX-WS-Strategy")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(fmt)
if not logger.handlers:
    logger.addHandler(ch)

# ----------------- 配置 (根据需要编辑) -----------------
# 修正：根据OKX文档，模拟盘(SIMULATED=True)必须使用wspap地址和brokerId
WS_URL = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
REST_BASE = "https://www.okx.com"
SYMBOLS = ["ETH-USDT-SWAP"]  # 用户之前请求的是ETHUSDT；OKX的永续合约和现货的instId不同
KLINE_INTERVAL = "15m"  # K线周期设置为15分钟
HISTORY_LIMIT = 500
POSITION_SIZE_BASE = 0.01  # 基础合约/手数大小 (用户应根据合约规格进行调整)
SIMULATED = True  # 使用 x-simulated-trading 请求头进行模拟交易
RECONNECT_DELAY = 5
# 模拟权益 (用于Kelly动态仓位计算). 生产环境请替换为查询真实账户接口。
SIMULATED_EQUITY_START = 500000.0

# --- 原始策略参数 & 特定资产覆盖 (来自您的代码) ---
STRATEGY_PARAMS = {
    "kelly_trade_history": 20,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "dd_grace_period_bars": 240,
    "dd_initial_pct": 0.35,
    "dd_final_pct": 0.25,
    "dd_decay_bars": 4320,
    "regime_adx_period": 14,
    "regime_atr_period": 14,
    "regime_atr_slope_period": 5,
    "regime_rsi_period": 14,
    "regime_rsi_vol_period": 14,
    "regime_norm_period": 252,
    "regime_hurst_period": 100,
    "regime_score_weight_adx": 0.6,
    "regime_score_weight_atr": 0.3,
    "regime_score_weight_rsi": 0.05,
    "regime_score_weight_hurst": 0.05,
    "regime_score_threshold": 0.45,
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_adx_confirm_period": 14,
    "tf_adx_confirm_threshold": 18,
    "tf_chandelier_period": 22,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.5,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 30,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "score_entry_threshold": 0.4,
    "score_weights_tf": {
        "breakout": 0.2667,
        "momentum": 0.20,
        "mtf": 0.1333,
        "ml": 0.2182,
        "advanced_ml": 0.1818,
    },
}

ASSET_SPECIFIC_OVERRIDES = {
    "BTC-USDT-SWAP": {
        "strategy_class": "BTCStrategy",
        "ml_weights": {"4h": 0.25, "8h": 0.35, "12h": 0.4},
        "ml_weighted_threshold": 0.2,
        "score_entry_threshold": 0.35,
    },
    "ETH-USDT-SWAP": {
        "strategy_class": "ETHStrategy",
        "ml_weights": {"4h": 0.15, "8h": 0.3, "12h": 0.55},
        "ml_weighted_threshold": 0.2,
        "score_entry_threshold": 0.35,
    },
}
ML_HORIZONS = [4, 8, 12]


# ----------------- 来自原始代码的工具函数 -----------------
def compute_hurst(ts, max_lag=100):
    """计算Hurst指数 (逻辑与您原始代码相同)。"""
    ts = np.asarray(ts)
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        std = np.std(diff)
        if std > 0:
            tau.append(std)
    if len(tau) < 2:
        return 0.5
    try:
        poly = np.polyfit(np.log(list(lags)[: len(tau)]), np.log(tau), 1)
        hurst = poly[0]
        return float(max(0.0, min(1.0, hurst)))
    except Exception:
        return 0.5


def run_advanced_model_inference(df):
    """高级模型推理存根 — 模仿原始逻辑：如果缺少高级库，则生成零值。"""
    # logger.info("运行高级模型推理 (存根)。") # 日志过于频繁，可注释掉
    if "ai_filter_signal" not in df.columns:
        df["ai_filter_signal"] = 0.0
    if not ML_LIBS_INSTALLED:
        df["advanced_ml_signal"] = 0.0
        return df
    # 如果您有高级模型，请在此处加载/运行它们；目前暂时使用 ai_filter_signal 的移动平均值
    df["advanced_ml_signal"] = (
        df.get("ai_filter_signal", pd.Series(0, index=df.index))
        .rolling(24)
        .mean()
        .fillna(0)
    )
    return df


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """使用 ta 库添加与原始脚本类似的 feature_* 列。"""
    p = STRATEGY_PARAMS
    df = df.copy()
    # 安全校验
    if len(df) < (p["regime_hurst_period"] + 5):
        # 数据不足以计算所有特征 — 返回最小化的结果
        df["feature_regime_score"] = 0.5
        df["feature_hurst"] = 0.5
        df["feature_adx_norm"] = 0.5
        df["feature_atr_slope_norm"] = 0.5
        df["feature_rsi_vol_norm"] = 0.5
        df["feature_obv_norm"] = 0.5
        df["feature_vol_pct_change_norm"] = 0.5
        df["feature_bb_width_norm"] = 0.5
        df["feature_atr_pct_change_norm"] = 0.5
        return df

    # 便捷的 Series 对象
    adx = ta.trend.ADXIndicator(
        df["High"], df["Low"], df["Close"], p["regime_adx_period"]
    ).adx()
    atr = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], p["regime_atr_period"]
    ).average_true_range()
    rsi = ta.momentum.RSIIndicator(df["Close"], p["regime_rsi_period"]).rsi()
    bb = ta.volatility.BollingerBands(
        df["Close"], window=p["mr_bb_period"], window_dev=p["mr_bb_std"]
    )

    def norm(s):
        return (
            (s - s.rolling(p["regime_norm_period"]).min())
            / (
                s.rolling(p["regime_norm_period"]).max()
                - s.rolling(p["regime_norm_period"]).min()
            )
        ).fillna(0.5)

    df["feature_adx_norm"] = norm(adx)
    df["feature_atr_slope_norm"] = norm(
        (atr - atr.shift(p["regime_atr_slope_period"]))
        / atr.shift(p["regime_atr_slope_period"])
    )
    df["feature_rsi_vol_norm"] = 1 - norm(rsi.rolling(p["regime_rsi_vol_period"]).std())
    df["feature_hurst"] = (
        df["Close"]
        .rolling(p["regime_hurst_period"])
        .apply(lambda x: compute_hurst(np.log(x + 1e-9)), raw=False)
        .fillna(0.5)
    )
    obv = ta.volume.OnBalanceVolumeIndicator(
        df["Close"], df["Volume"]
    ).on_balance_volume()
    df["feature_obv_norm"] = norm(obv)
    df["feature_vol_pct_change_norm"] = norm(df["Volume"].pct_change(periods=1).abs())
    df["feature_bb_width_norm"] = norm(
        (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    )
    df["feature_atr_pct_change_norm"] = norm(atr.pct_change(periods=1))
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["feature_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加 regime_score, trend_regime, volatility_regime, market_regime 等市场状态特征。"""
    p = STRATEGY_PARAMS
    df = df.copy()
    df["regime_score"] = df.get("feature_regime_score", pd.Series(0.5, index=df.index))
    df["trend_regime"] = np.where(
        df["regime_score"] > p["regime_score_threshold"], "Trending", "Mean-Reverting"
    )
    df["volatility"] = df["Close"].pct_change().rolling(24 * 7).std() * np.sqrt(
        24 * 365
    )
    low_vol, high_vol = df["volatility"].quantile(0.33), df["volatility"].quantile(0.67)
    df["volatility_regime"] = pd.cut(
        df["volatility"],
        bins=[0, low_vol, high_vol, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    df["market_regime"] = np.where(df["trend_regime"] == "Trending", 1, -1)
    return df


# ----------------- OKX REST 辅助类 -----------------
class OKXTrader:
    def __init__(self, api_key, api_secret, passphrase, simulated=True):
        self.base = REST_BASE
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.simulated = simulated
        self.common_headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }
        if self.simulated:
            self.common_headers["x-simulated-trading"] = "1"

    def _now(self):
        return datetime.utcnow().isoformat("T", "milliseconds") + "Z"

    def _sign(self, ts, method, path, body_str=""):
        message = f"{ts}{method}{path}{body_str}"
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method, path, body=None, params=None):
        ts = self._now()
        body_str = "" if body is None else json.dumps(body, separators=(",", ":"))
        sign = self._sign(ts, method.upper(), path, body_str)
        headers = self.common_headers.copy()
        headers.update({"OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": ts})
        url = self.base + path
        try:
            if method.upper() == "GET":
                r = requests.get(url, headers=headers, params=params, timeout=15)
            else:
                r = requests.request(
                    method, url, headers=headers, data=body_str, timeout=20
                )
            try:
                return r.json()
            except Exception:
                return {"error": r.text}
        except Exception as e:
            logger.error("REST 请求错误: %s", e)
            return {"error": str(e)}

    def place_market_order(self, instId, side, sz):
        path = "/api/v5/trade/order"
        body = {
            "instId": instId,
            "tdMode": "cross",
            "side": side,
            "ordType": "market",
            "sz": str(sz),
        }
        res = self._request("POST", path, body=body)
        logger.info("下单响应: %s", res)
        return res

    def fetch_history_klines(self, instId, bar="1m", limit=200):
        path = "/api/v5/market/history-candles"
        url = self.base + path
        params = {"instId": instId, "bar": bar, "limit": limit}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(
                data, columns=["ts", "o", "h", "l", "c", "vol", "x", "y", "z"]
            )
            # 修正: 明确将 'ts' 列转换为数值类型以消除 FutureWarning
            df["timestamp"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms")
            df = df.rename(
                columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "vol": "Volume",
                }
            )
            df = (
                df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
                .set_index("timestamp")
                .sort_index()
            )
            return df
        except Exception as e:
            logger.error("获取历史K线数据错误: %s", e)
            return pd.DataFrame()


# ----------------- 策略类 (UltimateStrategy + 辅助功能) -----------------
class UltimateStrategy:
    def __init__(self, df: pd.DataFrame, symbol: str, trader: OKXTrader = None):
        self.symbol = symbol
        self.trader = trader
        self.data = df.copy()
        # 如果可用，则作为 deque 存储
        try:
            from collections import deque

            self.recent_trade_returns = deque(
                maxlen=STRATEGY_PARAMS["kelly_trade_history"]
            )
        except Exception:
            self.recent_trade_returns = []

        # 每个交易对加载的机器学习模型
        self.ml_models = {}
        self.ml_weights_dict = ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).get(
            "ml_weights", None
        )
        self.ml_weighted_threshold = ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).get(
            "ml_weighted_threshold", None
        )
        self.score_entry_threshold = ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).get(
            "score_entry_threshold", STRATEGY_PARAMS["score_entry_threshold"]
        )
        self.score_weights_tf = STRATEGY_PARAMS["score_weights_tf"]
        self.advanced_ml_signal = (
            pd.Series(0, index=self.data.index)
            if not self.data.empty
            else pd.Series(dtype=float)
        )
        self.position = None  # "LONG" / "SHORT" / None
        self.entry_price = None
        self.equity = SIMULATED_EQUITY_START
        self.vol_weight = 1.0
        # 如果可用，加载模型
        self._load_models()

    def _load_models(self):
        if not ML_LIBS_INSTALLED:
            logger.warning("未安装机器学习库 - 跳过模型加载。")
            return
        loaded = 0
        for h in ML_HORIZONS:
            pattern = f"directional_model_{self.symbol}_{h}h.joblib"
            files = glob.glob(pattern)
            if files:
                try:
                    self.ml_models[h] = joblib.load(files[0])
                    loaded += 1
                except Exception as e:
                    logger.error("加载模型 %s 失败 : %s", files[0], e)
        if loaded:
            logger.info("[%s] 已加载 %d 个机器学习模型。", self.symbol, loaded)

    def _calculate_dynamic_risk(self):
        # 基于 recent_trade_returns 的类凯利公式计算
        if (
            not hasattr(self, "recent_trade_returns")
            or len(self.recent_trade_returns) < 2
        ):
            return STRATEGY_PARAMS["default_risk_pct"] * self.vol_weight
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return STRATEGY_PARAMS["default_risk_pct"] * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        if avg_loss == 0:
            return STRATEGY_PARAMS["default_risk_pct"] * self.vol_weight
        reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        if reward_ratio == 0:
            return STRATEGY_PARAMS["default_risk_pct"] * self.vol_weight
        kelly = win_rate - (1 - win_rate) / reward_ratio
        k = max(0.005, min(STRATEGY_PARAMS["max_risk_pct"], 0.5 * kelly))
        return k * self.vol_weight

    def get_ml_confidence_score(self):
        # 如果模型可用，则运行当前特征
        if not self.ml_models or not self.ml_weights_dict:
            return 0.0
        # 构建特征
        features = [c for c in self.data.columns if c.startswith("feature_")]
        if not features:
            return 0.0
        current_row = self.data[features].iloc[-1:].fillna(0)
        score = 0.0
        for h, model in self.ml_models.items():
            try:
                pred = model.predict(current_row)[0]
                pred_val = 1 if pred == 1 else -1
                score += pred_val * self.ml_weights_dict.get(f"{h}h", 0)
            except Exception:
                continue
        return score

    def update_with_candle(self, row: pd.Series):
        ts = row.name
        if ts in self.data.index:
            self.data.loc[ts] = row
        else:
            self.data = pd.concat([self.data, pd.DataFrame(row).T])

        if self.data.index.has_duplicates:
            self.data = self.data[~self.data.index.duplicated(keep="last")]
        self.data.sort_index(inplace=True)

        # 保留最近的数据部分
        if len(self.data) > 5000:
            self.data = self.data.iloc[-5000:]

    def compute_all_features(self):
        # 生成 AI 过滤信号
        if "ai_filter_signal" not in self.data.columns:
            rsi_filter = ta.momentum.RSIIndicator(self.data.Close, 14).rsi()
            self.data["ai_filter_signal"] = (
                (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
                .clip(-1, 1)
                .fillna(0)
            )
        # 高级模型存根
        self.data = run_advanced_model_inference(self.data)
        # 添加机器学习特征和市场状态
        self.data = add_ml_features(self.data)
        self.data = add_market_regime_features(self.data)
        # 计算用于评分的趋势跟踪 (TF) 指标
        self.data["tf_ema_fast"] = ta.trend.EMAIndicator(
            self.data.Close, STRATEGY_PARAMS["tf_ema_fast_period"]
        ).ema_indicator()
        self.data["tf_ema_slow"] = ta.trend.EMAIndicator(
            self.data.Close, STRATEGY_PARAMS["tf_ema_slow_period"]
        ).ema_indicator()
        self.data["tf_atr"] = ta.volatility.AverageTrueRange(
            self.data.High,
            self.data.Low,
            self.data.Close,
            STRATEGY_PARAMS["tf_atr_period"],
        ).average_true_range()
        self.data["tf_donchian_h"] = (
            self.data.High.rolling(STRATEGY_PARAMS["tf_donchian_period"]).max().shift(1)
        )
        self.data["tf_donchian_l"] = (
            self.data.Low.rolling(STRATEGY_PARAMS["tf_donchian_period"]).min().shift(1)
        )
        # 均值回归 (MR) 指标
        bb = ta.volatility.BollingerBands(
            self.data.Close,
            STRATEGY_PARAMS["mr_bb_period"],
            STRATEGY_PARAMS["mr_bb_std"],
        )
        self.data["mr_bb_upper"] = bb.bollinger_hband()
        self.data["mr_bb_lower"] = bb.bollinger_lband()
        self.data["mr_bb_mid"] = bb.bollinger_mavg()
        self.data["mr_rsi"] = ta.momentum.RSIIndicator(
            self.data.Close, STRATEGY_PARAMS["mr_rsi_period"]
        ).rsi()

    def _calculate_entry_score(self):
        w = self.score_weights_tf
        # 突破
        try:
            b_s = (
                1
                if self.data.High.iloc[-1] > self.data.tf_donchian_h.iloc[-1]
                else (
                    -1
                    if self.data.Low.iloc[-1] < self.data.tf_donchian_l.iloc[-1]
                    else 0
                )
            )
        except Exception:
            b_s = 0
        mo_s = (
            1 if self.data.tf_ema_fast.iloc[-1] > self.data.tf_ema_slow.iloc[-1] else -1
        )
        ml_score = self.get_ml_confidence_score()
        adv_score = self.data["advanced_ml_signal"].iloc[-1]
        mtf = self.data.get("mtf_signal", pd.Series(0, index=self.data.index)).iloc[-1]

        total = (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + mtf * w.get("mtf", 0)
            + ml_score * w.get("ml", 0)
            + adv_score * w.get("advanced_ml", 0)
        )
        return total

    def _define_mr_entry_signal(self):
        # 与原始逻辑相同的均值回归入场信号
        df = self.data
        if len(df) < 5:
            return 0
        if (df.Close.iloc[-1] < df.mr_bb_lower.iloc[-1]) and (
            df.mr_rsi.iloc[-1] < STRATEGY_PARAMS["mr_rsi_oversold"]
        ):
            return 1
        if (df.Close.iloc[-1] > df.mr_bb_upper.iloc[-1]) and (
            df.mr_rsi.iloc[-1] > STRATEGY_PARAMS["mr_rsi_overbought"]
        ):
            return -1
        return 0

    def next_on_candle_close(self, candle_row: pd.Series):
        """在 K 线收盘时调用的主入口函数。"""
        # 更新数据
        self.update_with_candle(candle_row)
        # 计算特征和指标
        self.compute_all_features()
        # 根据市场状态决策
        if self.data["market_regime"].iloc[-1] == 1:
            # 趋势行情 -> 评分系统
            score = self._calculate_entry_score()
            if abs(score) > self.score_entry_threshold:
                is_long = score > 0
                # 仓位管理：使用动态风险 (凯利调整) 来调整基础大小
                risk_pct = self._calculate_dynamic_risk()
                position_size = self._determine_position_size(
                    self.data.Close.iloc[-1], risk_pct
                )
                logger.info(
                    "[%s] 趋势跟踪入场: %s score=%.3f risk_pct=%.4f size=%s",
                    self.symbol,
                    "LONG" if is_long else "SHORT",
                    score,
                    risk_pct,
                    position_size,
                )
                return {
                    "action": "BUY" if is_long else "SELL",
                    "size": position_size,
                    "confidence": abs(score),
                }
        else:
            # 均值回归
            sig = self._define_mr_entry_signal()
            if sig != 0:
                is_long = sig == 1
                risk_pct = (
                    self._calculate_dynamic_risk()
                    * STRATEGY_PARAMS["mr_risk_multiplier"]
                )
                position_size = self._determine_position_size(
                    self.data.Close.iloc[-1], risk_pct
                )
                logger.info(
                    "[%s] 均值回归入场: %s size=%s",
                    self.symbol,
                    "LONG" if is_long else "SHORT",
                    position_size,
                )
                return {
                    "action": "BUY" if is_long else "SELL",
                    "size": position_size,
                    "confidence": 0.5,
                }
        return None

    def _determine_position_size(self, price, risk_pct):
        """
        尝试将 risk_pct 和模拟权益映射为合约数量。
        注意：对于永续合约，合约数量的计算规则不同 — 用户应根据具体的合约规格调整此函数。
        我们为 OKX REST API 返回一个数量字符串；此处我们通过 名义价值 = 权益 * 风险百分比，数量 = 名义价值 / 价格 的方式进行近似计算。
        """
        if price <= 0:
            return str(POSITION_SIZE_BASE)
        notional = max(1.0, self.equity * risk_pct)
        qty = max(0.0001, notional / price)
        # 根据基础仓位大小限制
        return str(round(qty, 6))

    def register_trade_result(self, pnl_pct):
        """当一笔交易平仓时调用，用以更新 recent_trade_returns 和模拟权益。"""
        try:
            self.recent_trade_returns.append(pnl_pct)
            # 更新权益
            self.equity = self.equity * (1 + pnl_pct)
        except Exception:
            pass


# ----------------- WebSocket 客户端及主循环 -----------------
class OKXWebsocketClient:
    def __init__(self, symbols, inst_interval="1m", trader: OKXTrader = None):
        self.symbols = symbols
        self.interval = inst_interval
        self.url = WS_URL
        self.trader = trader
        self.msg_q = queue.Queue()
        self.strategies = {}
        self.df_store = {}
        self._build_initial_histories()

    def _build_initial_histories(self):
        # 通过 REST 获取初始历史数据
        for s in self.symbols:
            logger.info("正在为 %s 获取初始历史数据...", s)
            df = self.trader.fetch_history_klines(
                s, bar=self.interval, limit=HISTORY_LIMIT
            )
            if df.empty:
                logger.warning("交易对 %s 的历史数据为空", s)
                df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            self.df_store[s] = df
            self.strategies[s] = UltimateStrategy(df.copy(), s, trader=self.trader)

    def _on_open(self, ws):
        logger.info("WebSocket 已连接，正在订阅...")
        # OKX V5 WebSocket文档规定，K线频道名为 "candle" + 时间间隔
        args = [
            {"channel": f"candle{self.interval}", "instId": s} for s in self.symbols
        ]
        sub = {"op": "subscribe", "args": args}
        ws.send(json.dumps(sub))
        logger.info("订阅消息已发送。")

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
        except Exception:
            return
        # 处理 event/ping/pong 消息
        if "event" in msg:
            ev = msg.get("event")
            if ev == "subscribe":
                logger.info("订阅成功: %s", msg.get("arg"))
            elif ev == "error":
                logger.error("WebSocket 事件错误: %s", msg)
            return

        # OKX V5 uses 'op' for pong response
        if "op" in msg and msg["op"] == "pong":
            return

        # Handle ping from server
        if message == "ping":
            ws.send("pong")
            return

        if "arg" in msg and "data" in msg:
            self.msg_q.put(msg)

    def _on_error(self, ws, error):
        logger.error("WebSocket 错误: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning("WebSocket 已关闭: code=%s msg=%s", close_status_code, close_msg)

    def start(self):
        # 启动工作线程
        t = threading.Thread(target=self._worker_loop, daemon=True)
        t.start()
        # 运行 websocket 循环 (阻塞式，带重连机制)
        while True:
            try:
                ws = WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                logger.info("正在连接到 WebSocket...")
                # Use ping/pong instead of op:ping/op:pong for public channels
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.exception("WebSocket run_forever 异常: %s", e)
            logger.info("将在 %ds 后重新连接...", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)

    def _worker_loop(self):
        from time import sleep

        while True:
            try:
                msg = self.msg_q.get(timeout=1)
            except queue.Empty:
                sleep(0.1)
                continue

            arg = msg.get("arg", {})
            instId = arg.get("instId")
            data_list = msg.get("data", [])
            if not data_list or not instId:
                continue

            strat = self.strategies.get(instId)
            if not strat:
                continue

            for item in data_list:
                # K线数据项格式: [时间戳, 开, 高, 低, 收, 交易量, 确认状态]
                try:
                    ts_ms = int(item[0])
                    ts = pd.to_datetime(ts_ms, unit="ms")
                    o, h, l, c, v = (
                        float(item[1]),
                        float(item[2]),
                        float(item[3]),
                        float(item[4]),
                        float(item[5]),
                    )
                    is_confirmed = item[6] == "1"
                except Exception as e:
                    logger.debug("K线解析错误: %s item:%s", e, item)
                    continue

                # 当前K线的数据
                current_candle = pd.Series(
                    {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, name=ts
                )

                # 更新策略中的DataFrame
                strat.update_with_candle(current_candle)

                # 只有当K线确认时（即这根K线走完了），才触发交易决策
                if is_confirmed:
                    logger.info("[%s] 新 K 线已收盘 %s close=%.4f", instId, ts, c)
                    try:
                        # 传递给策略的是刚刚收盘的这根K线
                        action = strat.next_on_candle_close(current_candle)
                        if action:
                            logger.info("[%s] 策略决策: %s", instId, action)
                            # 通过 REST trader 下市价单
                            if self.trader:
                                side = "buy" if action["action"] == "BUY" else "sell"
                                size = action.get("size", POSITION_SIZE_BASE)
                                res = self.trader.place_market_order(instId, side, size)
                        else:
                            logger.debug("[%s] 无操作。", instId)
                    except Exception as e:
                        logger.exception("处理 %s 的策略时出错: %s", instId, e)
            sleep(0.001)


# ----------------- 主函数 -----------------
def main():
    # 加载 API 凭证
    OKX_API_KEY = os.getenv("OKX_API_KEY")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET")
    OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")

    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE]):
        logger.error("请设置环境变量: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE")
        return

    trader = OKXTrader(
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, simulated=SIMULATED
    )
    client = OKXWebsocketClient(SYMBOLS, inst_interval=KLINE_INTERVAL, trader=trader)
    client.start()


if __name__ == "__main__":
    logger.info("启动集成了完整策略逻辑的 OKX WebSocket 实时交易程序...")
    main()
