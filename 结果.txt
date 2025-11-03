# okx_live_trading_v4.7_sizing_fix.py
"""
OKX REST API 轮询版 (V4.7 - 仓位计算修复版)
- 核心变更:
    - [!!! 关键缺陷修复 !!!] 修正了 `_determine_position_size` 函数中的一个致命逻辑错误。
        - 旧的逻辑在检测到根据风险计算的仓位所需保证金不足时，错误地将其“降级”为一个使用95%总权益和最大杠杆的“全仓”仓位。这是导致之前“几乎全仓买入”现象的根本原因。
        - 新的逻辑将其修改为最安全的“故障安全”模式：当检测到所需保证金过高时，程序将直接取消本次交易并打印警告日志，而不是开一个错误的巨大仓位。
    - [继承] 完整继承了 V4.6 版本的所有安全特性，包括仓位优先检查、紧急接管、熔断暂停和交易冷静期。此版本应是目前最稳定和安全的版本。
"""

# --- 核心库导入 ---
import os
import time
import json
import hmac
import base64
import hashlib
import logging
import math
import csv
import urllib.parse
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from collections import deque
import pandas as pd
import numpy as np
import requests
import ta
import warnings
from requests.models import PreparedRequest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    import joblib
    import tensorflow as tf

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False
    print("警告: tensorflow 或 joblib 未安装，机器学习相关功能将不可用。")


# --- 日志系统设置 ---
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    log_file_path = os.path.join(log_dir, "trading_bot.log")
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    trade_flow_logger = logging.getLogger("TradeFlow")
    trade_flow_logger.setLevel(logging.DEBUG)
    trade_flow_logger.propagate = False
    trade_flow_path = os.path.join(log_dir, "trades_flow.log")
    trade_flow_handler = RotatingFileHandler(
        trade_flow_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    trade_flow_handler.setLevel(logging.DEBUG)
    trade_flow_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    trade_flow_handler.setFormatter(trade_flow_formatter)
    trade_flow_logger.addHandler(trade_flow_handler)
    trade_flow_logger.addHandler(console_handler)
    return trade_flow_logger


def setup_csv_logger(name, log_file, fields):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, log_file)
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.INFO)
    logger_obj.propagate = False
    if logger_obj.hasHandlers():
        return logger_obj
    handler = logging.FileHandler(csv_path, mode="a", encoding="utf-8")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def emit_csv(record):
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(record.msg)

    handler.emit = emit_csv
    logger_obj.addHandler(handler)
    return logger_obj


# --- 全局配置 ---
REST_BASE = "https://www.okx.com"
SYMBOLS = ["ETH-USDT-SWAP"]
KLINE_INTERVAL = "15m"
DESIRED_LEVERAGE = "1"
HISTORY_LIMIT = 500
POLL_INTERVAL_SECONDS = 10
SIMULATED = True
SIMULATED_EQUITY_START = 500000.0
MAX_DAILY_DRAWDOWN_PCT = 0.03
MAX_CONSECUTIVE_LOSSES = 5
TRADING_PAUSE_HOURS = 4
AUDIT_INTERVAL_MINUTES = 15
COOL_DOWN_PERIOD_MINUTES = 1

# --- Keras模型文件路径配置 ---
KERAS_MODEL_PATH = "models/eth_trend_model_v1_15m.keras"
SCALER_PATH = "models/eth_trend_scaler_v1_15m.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns_15m.joblib"
KERAS_SEQUENCE_LENGTH = 60

# --- 策略参数 ---
STRATEGY_PARAMS = {
    "tsl_enabled": True,
    "tsl_activation_profit_pct": 0.5,
    "tsl_activation_atr_mult": 1.5,
    "tsl_trailing_atr_mult": 2.0,
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
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "score_entry_threshold": 0.4,
    "score_weights_tf": {
        "breakout": 0.25,
        "momentum": 0.18,
        "mtf": 0.10,
        "ml": 0.22,
        "advanced_ml": 0.25,
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETH-USDT-SWAP": {"score_entry_threshold": 0.45},
}


# --- 状态管理 & 辅助函数 ---
def save_trade_state(state):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    state_file = os.path.join(log_dir, "trade_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=4)
    logging.debug(f"已保存交易状态: {state}")


def load_trade_state():
    log_dir = "logs"
    state_file = os.path.join(log_dir, "trade_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def clear_trade_state():
    log_dir = "logs"
    state_file = os.path.join(log_dir, "trade_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        logging.info("交易状态文件已清除。")


def save_kelly_history(symbol, history_deque):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    history_file = os.path.join(log_dir, f"{symbol}_kelly_history.json")
    try:
        with open(history_file, "w") as f:
            json.dump(list(history_deque), f, indent=4)
        logging.debug(f"[{symbol}] 已保存凯利公式交易历史。")
    except Exception as e:
        logging.error(f"[{symbol}] 保存凯利历史时出错: {e}")


def load_kelly_history(symbol, maxlen):
    log_dir = "logs"
    history_file = os.path.join(log_dir, f"{symbol}_kelly_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_list = json.load(f)
                logging.info(
                    f"[{symbol}] 成功加载 {len(history_list)} 条凯利历史记录。"
                )
                return deque(history_list, maxlen=maxlen)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"[{symbol}] 加载凯利历史文件失败: {e}")
    return deque(maxlen=maxlen)


# --- 特征工程函数 ---
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
        return max(
            0.0, min(1.0, np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0])
        )
    except:
        return 0.5


def run_advanced_model_inference(df):
    if not ML_LIBS_INSTALLED:
        df["advanced_ml_signal"] = 0.0
        return df
    df["advanced_ml_signal"] = (
        df.get("ai_filter_signal", pd.Series(0, index=df.index))
        .rolling(24)
        .mean()
        .fillna(0)
    )
    return df


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
    df["feature_obv_norm"] = norm(
        ta.volume.OnBalanceVolumeIndicator(df.Close, df.Volume).on_balance_volume()
    )
    df["feature_vol_pct_change_norm"] = norm(df.Volume.pct_change(periods=1).abs())
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
    df["regime_score"] = df["feature_regime_score"]
    df["trend_regime"] = np.where(
        df["regime_score"] > STRATEGY_PARAMS["regime_score_threshold"],
        "Trending",
        "Mean-Reverting",
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


def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    df["EMA_8"] = ta.trend.EMAIndicator(close=close, window=8).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = (
        adx_indicator.adx(),
        adx_indicator.adx_pos(),
        adx_indicator.adx_neg(),
    )
    atr_raw = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    df["ATRr_14"] = (atr_raw / close) * 100
    bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
    df["BBU_20_2.0"], df["BBM_20_2.0"], df["BBL_20_2.0"] = (
        bb_indicator.bollinger_hband(),
        bb_indicator.bollinger_mavg(),
        bb_indicator.bollinger_lband(),
    )
    df["BBB_20_2.0"], df["BBP_20_2.0"] = (
        bb_indicator.bollinger_wband(),
        bb_indicator.bollinger_pband(),
    )
    macd_indicator = ta.trend.MACD(
        close=close, window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = (
        macd_indicator.macd(),
        macd_indicator.macd_signal(),
        macd_indicator.macd_diff(),
    )
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()
    df["volume_change_rate"] = volume.pct_change()
    return df


# --- OKX 交易接口类 ---
class OKXTrader:
    def __init__(self, api_key, api_secret, passphrase, simulated=True):
        self.base, self.api_key, self.api_secret, self.passphrase, self.simulated = (
            REST_BASE,
            api_key,
            api_secret,
            passphrase,
            simulated,
        )
        self.instrument_info = {}
        self.common_headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }
        if self.simulated:
            self.common_headers["x-simulated-trading"] = "1"
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _now(self):
        return datetime.utcnow().isoformat("T", "milliseconds") + "Z"

    def _sign(self, ts, method, path, body_str=""):
        message = f"{ts}{method}{path}{body_str}"
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method, path, body=None, params=None, max_retries=7):
        ts = self._now()
        body_str = "" if body is None else json.dumps(body)
        prepped = PreparedRequest()
        prepped.prepare_url(self.base + path, params)
        path_for_signing = urllib.parse.urlparse(prepped.url).path
        if query := urllib.parse.urlparse(prepped.url).query:
            path_for_signing += "?" + query
        path_for_signing = path_for_signing.replace(self.base, "")
        sign = self._sign(ts, method.upper(), path_for_signing, body_str)
        headers = self.common_headers.copy()
        headers.update({"OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": ts})
        url, wait_time = self.base + path, 1.0
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.request(
                    method,
                    url,
                    headers=headers,
                    data=body_str,
                    params=params,
                    timeout=(5, 15),
                )
                if r.status_code == 404 and "cancel-algo-order" in path:
                    return {
                        "code": "0",
                        "data": [{"sCode": "0"}],
                        "msg": "Cancelled (already gone)",
                    }
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"HTTP请求错误 (尝试 {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return {
                        "code": "-1",
                        "msg": f"Max retries exceeded with error: {e}",
                    }
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 30)
        return {"code": "-1", "msg": "Max retries exceeded with network errors"}

    def place_trigger_order(
        self, instId, side, sz, trigger_price, order_type, posSide="net"
    ):
        path, price_str = "/api/v5/trade/order-algo", f"{trigger_price:.8f}".rstrip(
            "0"
        ).rstrip(".")
        body = {
            "instId": instId,
            "tdMode": "cross",
            "side": side,
            "posSide": posSide,
            "ordType": "trigger",
            "sz": str(sz),
            "triggerPx": price_str,
            "orderPx": "-1",
        }
        return self._request("POST", path, body=body)

    def fetch_open_algo_orders(self, instId):
        path, params = "/api/v5/trade/orders-algo-pending", {
            "instType": "SWAP",
            "instId": instId,
            "ordType": "trigger",
        }
        res = self._request("GET", path, params=params)
        return res.get("data", []) if res and res.get("code") == "0" else []

    def cancel_algo_orders(self, instId, algoIds):
        if not algoIds:
            return True
        path, body = "/api/v5/trade/cancel-algo-order", [
            {"instId": instId, "algoId": str(aid)} for aid in algoIds
        ]
        return (res := self._request("POST", path, body=body)) and res.get(
            "code"
        ) == "0"

    def set_leverage(self, instId, lever, mgnMode="cross", posSide=None):
        path, body = "/api/v5/account/set-leverage", {
            "instId": instId,
            "lever": str(lever),
            "mgnMode": mgnMode,
        }
        if posSide:
            body["posSide"] = posSide
        return self._request("POST", path, body=body).get("code") == "0"

    def fetch_account_balance(self, ccy="USDT"):
        path, params = "/api/v5/account/balance", {"ccy": ccy}
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            if (data := res.get("data", [])) and "details" in data[0]:
                for detail in data[0]["details"]:
                    if detail.get("ccy") == ccy:
                        return float(detail.get("eq", 0))
        return None

    def fetch_current_position(self, instId):
        path, params = "/api/v5/account/positions", {"instId": instId}
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            for pos_data in res.get("data", []):
                if float(pos_data.get("pos", "0")) != 0:
                    return pos_data
            return None
        logging.error(
            f"查询仓位失败 for {instId}: {res.get('msg') if res else 'Unknown error'}"
        )
        return False

    def fetch_instrument_details(self, instId, instType="SWAP"):
        if instId in self.instrument_info:
            return self.instrument_info[instId]
        path, params = "/api/v5/public/instruments", {
            "instType": instType,
            "instId": instId,
        }
        if (
            (res := self._request("GET", path, params=params))
            and res.get("code") == "0"
            and res.get("data")
        ):
            instrument_data = res["data"][0]
            self.instrument_info[instId] = instrument_data
            return instrument_data
        return None

    def place_market_order(self, instId, side, sz):
        path, body = "/api/v5/trade/order", {
            "instId": instId,
            "tdMode": "cross",
            "side": side,
            "ordType": "market",
            "sz": str(sz),
        }
        if res := self._request("POST", path, body=body):
            trade_logger.info(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": instId,
                    "action": f"{side.upper()}",
                    "size": sz,
                    "response": json.dumps(res),
                }
            )
        return res

    def fetch_history_klines(self, instId, bar="1m", limit=200):
        path, params = "/api/v5/market/history-candles", {
            "instId": instId,
            "bar": bar,
            "limit": limit,
        }
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            if not (data := res.get("data", [])):
                return pd.DataFrame()
            df = pd.DataFrame(
                data,
                columns=[
                    "ts",
                    "o",
                    "h",
                    "l",
                    "c",
                    "vol",
                    "volCcy",
                    "volCcyQuote",
                    "confirm",
                ],
            )
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
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col])
            return (
                df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
                .set_index("timestamp")
                .sort_index()
            )
        return None

    def fetch_ticker_price(self, instId):
        path, params = "/api/v5/market/ticker", {"instId": instId}
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            if data := res.get("data", []):
                return float(data[0].get("last"))
        return None


# --- 策略核心类 ---
class UltimateStrategy:
    def __init__(
        self, df, symbol, trader=None, pos_logger=None, instrument_details=None
    ):
        self.symbol, self.trader, self.data, self.position = (
            symbol,
            trader,
            df.copy(),
            None,
        )
        self.pos_logger, self.instrument_details = pos_logger, instrument_details
        logging.info(f"[{self.symbol}] 策略初始化...")
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)
        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(symbol, {})
        self.score_entry_threshold = asset_overrides.get(
            "score_entry_threshold", self.score_entry_threshold
        )
        self.recent_trade_returns = load_kelly_history(
            self.symbol, self.kelly_trade_history
        )
        self.keras_model, self.scaler, self.feature_columns = None, None, None
        self.equity, self.consecutive_losses, self.trading_paused_until = (
            SIMULATED_EQUITY_START,
            0,
            None,
        )
        self._load_models()

    @property
    def is_trading_paused(self):
        if self.trading_paused_until and datetime.utcnow() < self.trading_paused_until:
            return True
        self.trading_paused_until = None
        return False

    def register_loss(self):
        self.consecutive_losses += 1
        trade_flow_logger.warning(
            f"[{self.symbol}] 录得亏损，连亏: {self.consecutive_losses}"
        )
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.trading_paused_until = datetime.utcnow() + pd.Timedelta(
                hours=TRADING_PAUSE_HOURS
            )
            trade_flow_logger.critical(
                f"[{self.symbol}] 达到最大连亏({MAX_CONSECUTIVE_LOSSES})！暂停交易{TRADING_PAUSE_HOURS}小时。"
            )

    def register_win(self):
        if self.consecutive_losses > 0:
            trade_flow_logger.info(f"[{self.symbol}] 录得盈利，连亏计数重置。")
            self.consecutive_losses = 0

    def _load_models(self):
        if not ML_LIBS_INSTALLED:
            return
        try:
            if not os.path.exists(KERAS_MODEL_PATH):
                raise FileNotFoundError(f"Keras模型未找到: {KERAS_MODEL_PATH}")
            self.keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler未找到: {SCALER_PATH}")
            self.scaler = joblib.load(SCALER_PATH)
            if not os.path.exists(FEATURE_COLUMNS_PATH):
                raise FileNotFoundError(f"特征列文件未找到: {FEATURE_COLUMNS_PATH}")
            self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
            if hasattr(self.feature_columns, "tolist"):
                self.feature_columns = self.feature_columns.tolist()
            logging.info(f"[{self.symbol}] 成功加载所有ML模型及组件。")
        except Exception as e:
            logging.error(f"加载ML模型时出错: {e}")
            self.keras_model, self.scaler, self.feature_columns = None, None, None

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct
        wins, losses = [r for r in self.recent_trade_returns if r > 0], [
            r for r in self.recent_trade_returns if r < 0
        ]
        if not wins or not losses:
            return self.default_risk_pct
        win_rate, avg_win, avg_loss = (
            len(wins) / len(self.recent_trade_returns),
            sum(wins) / len(wins),
            abs(sum(losses) / len(losses)),
        )
        if avg_loss == 0 or (reward_ratio := avg_win / avg_loss) == 0:
            return self.default_risk_pct
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5), self.max_risk_pct)

    def get_ml_confidence_score(self):
        if not all([self.keras_model, self.scaler, self.feature_columns]):
            return 0.0
        if len(self.data) < KERAS_SEQUENCE_LENGTH:
            return 0.0
        if missing_cols := [
            col for col in self.feature_columns if col not in self.data.columns
        ]:
            logging.warning(f"缺少模型特征: {missing_cols}，ML评分为0。")
            return 0.0
        try:
            latest_sequence_df = (
                self.data[self.feature_columns]
                .iloc[-KERAS_SEQUENCE_LENGTH:]
                .copy()
                .fillna(method="ffill")
                .fillna(0)
            )
            scaled_sequence = self.scaler.transform(latest_sequence_df)
            input_for_model = np.expand_dims(scaled_sequence, axis=0)
            prediction = self.keras_model.predict(input_for_model, verbose=0)
            return float((prediction[0][0] - 0.5) * 2.0)
        except Exception as e:
            logging.error(f"Keras模型预测出错: {e}", exc_info=True)
            return 0.0

    def update_with_candle(self, row: pd.Series):
        self.data.loc[row.name] = row
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)
        if len(self.data) > 5000:
            self.data = self.data.iloc[-5000:]

    def compute_all_features(self, trader):
        logging.debug(f"[{self.symbol}] 开始数据健康检查和特征计算...")
        self.data.index = pd.to_datetime(self.data.index)
        if self.data.empty:
            logging.error(f"[{self.symbol}] 数据集为空，无法计算特征。")
            return
        resample_freq = KLINE_INTERVAL.replace("m", "T")
        self.data = self.data.resample(resample_freq).ffill()
        self.data.dropna(
            subset=["Open", "High", "Low", "Close", "Volume"], inplace=True
        )
        if len(self.data) < STRATEGY_PARAMS.get("tf_donchian_period", 30):
            logging.warning(
                f"[{self.symbol}] 数据修复后，有效数据量过少({len(self.data)}条)，跳过本轮特征计算。"
            )
            return
        rsi_filter = ta.momentum.RSIIndicator(self.data.Close, 14).rsi()
        self.data["ai_filter_signal"] = (
            (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
            .clip(-1, 1)
            .fillna(0)
        )
        self.data = run_advanced_model_inference(self.data)
        self.data = add_ml_features_ported(self.data)
        self.data = add_features_for_keras_model(self.data)
        self.data = add_market_regime_features(self.data)
        if (
            data_1d := trader.fetch_history_klines(
                self.symbol, bar="1D", limit=self.mtf_period + 10
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
            self.data["mtf_signal"] = 0
        df_4h = self.data["Close"].resample("4H").last().to_frame()
        df_4h["macro_ema"] = ta.trend.EMAIndicator(
            df_4h["Close"], window=50
        ).ema_indicator()
        df_4h["macro_trend"] = np.where(df_4h["Close"] > df_4h["macro_ema"], 1, -1)
        self.data["macro_trend_filter"] = (
            df_4h["macro_trend"].reindex(self.data.index, method="ffill").fillna(0)
        )
        self.data["tf_atr"] = ta.volatility.AverageTrueRange(
            self.data.High, self.data.Low, self.data.Close, self.tf_atr_period
        ).average_true_range()
        self.data["tf_donchian_h"], self.data["tf_donchian_l"] = self.data.High.rolling(
            self.tf_donchian_period
        ).max().shift(1), self.data.Low.rolling(self.tf_donchian_period).min().shift(1)
        self.data["tf_ema_fast"], self.data["tf_ema_slow"] = (
            ta.trend.EMAIndicator(
                self.data.Close, self.tf_ema_fast_period
            ).ema_indicator(),
            ta.trend.EMAIndicator(
                self.data.Close, self.tf_ema_slow_period
            ).ema_indicator(),
        )
        bb = ta.volatility.BollingerBands(
            self.data.Close, self.mr_bb_period, self.mr_bb_std
        )
        self.data["mr_bb_upper"], self.data["mr_bb_lower"], self.data["mr_bb_mid"] = (
            bb.bollinger_hband(),
            bb.bollinger_lband(),
            bb.bollinger_mavg(),
        )
        stoch_rsi = ta.momentum.StochRSIIndicator(
            self.data.Close, window=14, smooth1=3, smooth2=3
        )
        self.data["mr_stoch_rsi_k"], self.data["mr_stoch_rsi_d"] = (
            stoch_rsi.stochrsi_k(),
            stoch_rsi.stochrsi_d(),
        )
        self.data.fillna(method="ffill", inplace=True)
        self.data.fillna(method="bfill", inplace=True)
        logging.debug(f"[{self.symbol}] 特征计算完成。")

    def _calculate_entry_score(self):
        w, last = self.score_weights_tf, self.data.iloc[-1]
        b_s = (
            1
            if last.High > last.tf_donchian_h
            else -1 if last.Low < last.tf_donchian_l else 0
        )
        mo_s = 1 if last.tf_ema_fast > last.tf_ema_slow else -1
        return (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + last.get("mtf_signal", 0) * w.get("mtf", 0)
            + self.get_ml_confidence_score() * w.get("ml", 0)
            + last.get("advanced_ml_signal", 0) * w.get("advanced_ml", 0)
        )

    def _define_mr_entry_signal(self):
        if len(self.data) < 3:
            return 0
        last, prev = self.data.iloc[-1], self.data.iloc[-2]
        if (
            prev.Close < prev.mr_bb_lower
            and last.Close > last.mr_bb_lower
            and last.mr_stoch_rsi_k > last.mr_stoch_rsi_d
            and prev.mr_stoch_rsi_k <= prev.mr_stoch_rsi_d
            and last.mr_stoch_rsi_k < 40
        ):
            return 1
        if (
            prev.Close > prev.mr_bb_upper
            and last.Close < last.mr_bb_upper
            and last.mr_stoch_rsi_k < last.mr_stoch_rsi_d
            and prev.mr_stoch_rsi_k >= prev.mr_stoch_rsi_d
            and last.mr_stoch_rsi_k > 60
        ):
            return -1
        return 0

    def next_on_candle_close(self):
        last = self.data.iloc[-1]
        if pd.isna(last.get("market_regime")) or pd.isna(
            last.get("macro_trend_filter")
        ):
            return None
        action_to_take = None
        if last.macro_trend_filter == 1:
            if last.market_regime == 1:
                if (
                    score := self._calculate_entry_score()
                ) > self.score_entry_threshold:
                    action_to_take = {
                        "action": "BUY",
                        "sub_strategy": "TF",
                        "confidence": score,
                    }
            elif self._define_mr_entry_signal() == 1:
                action_to_take = {
                    "action": "BUY",
                    "sub_strategy": "MR",
                    "confidence": 1.0,
                }
        elif last.macro_trend_filter == -1:
            if last.market_regime == 1:
                if (
                    score := self._calculate_entry_score()
                ) < -self.score_entry_threshold:
                    action_to_take = {
                        "action": "SELL",
                        "sub_strategy": "TF",
                        "confidence": abs(score),
                    }
            elif self._define_mr_entry_signal() == -1:
                action_to_take = {
                    "action": "SELL",
                    "sub_strategy": "MR",
                    "confidence": 1.0,
                }
        if action_to_take:
            risk_multiplier = (
                1.0
                if action_to_take["sub_strategy"] == "TF"
                else self.mr_risk_multiplier
            )
            risk_pct = (
                self._calculate_dynamic_risk()
                * action_to_take["confidence"]
                * risk_multiplier
            )
            sl_atr_mult = (
                self.tf_stop_loss_atr_multiplier
                if action_to_take["sub_strategy"] == "TF"
                else self.mr_stop_loss_atr_multiplier
            )
            if (
                size_and_sl := self._determine_position_size(
                    last.Close, risk_pct, sl_atr_mult, action_to_take["action"] == "BUY"
                )
            ) and float(size_and_sl[0]) > 0:
                action_to_take["size"], action_to_take["stop_loss_price"] = size_and_sl
                return action_to_take
        return None

    def _adjust_size_to_lot_size(self, size):
        if not self.instrument_details or not (
            lot_sz_str := self.instrument_details.get("lotSz")
        ):
            return str(size)
        try:
            lot_sz = float(lot_sz_str)
            adjusted_size = math.floor(float(size) / lot_sz) * lot_sz
            decimals = len(lot_sz_str.split(".")[1]) if "." in lot_sz_str else 0
            return f"{adjusted_size:.{decimals}f}"
        except (ValueError, TypeError):
            return str(size)

    def _determine_position_size(self, price, risk_pct, sl_atr_mult, is_long):
        atr, ct_val = self.data["tf_atr"].iloc[-1], float(
            self.instrument_details.get("ctVal", 1)
        )
        risk_per_unit = atr * sl_atr_mult
        if price <= 0 or pd.isna(risk_per_unit) or risk_per_unit <= 0 or ct_val <= 0:
            return "0", 0.0
        risk_amount_dollars = self.equity * risk_pct
        units = risk_amount_dollars / (risk_per_unit * ct_val)
        margin_needed = (units * ct_val * price) / int(DESIRED_LEVERAGE)

        # <<< [核心修正 V4.7] --- START --- >>>
        if margin_needed > self.equity * 0.95:
            trade_flow_logger.warning(
                f"[{self.symbol}] 仓位计算警告：根据风险计算出的仓位所需保证金过高({margin_needed:.2f} USDT)，超过总权益的95%。"
                f"为保证安全，本次交易将被跳过。"
            )
            return "0", 0.0  # 直接返回0，放弃交易
        # <<< [核心修正 V4.7] --- END --- >>>

        stop_loss_price = price - risk_per_unit if is_long else price + risk_per_unit
        return self._adjust_size_to_lot_size(units), stop_loss_price

    def register_trade_result(self, pnl_pct):
        self.recent_trade_returns.append(pnl_pct)
        save_kelly_history(self.symbol, self.recent_trade_returns)
        self.register_loss() if pnl_pct < 0 else self.register_win()

    def set_position(self, new_position):
        if self.position != new_position:
            self.position = new_position
            if self.pos_logger:
                self.pos_logger.info(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": self.symbol,
                        "position": self.position,
                    }
                )


# --- Main Logic and Helper Functions ---
def manage_position_entry(
    trader: OKXTrader, symbol: str, strategy: UltimateStrategy, action: dict
):
    side = "buy" if action["action"] == "BUY" else "sell"
    res = trader.place_market_order(symbol, side, action["size"])
    if res and res.get("code") == "0" and res.get("data")[0].get("sCode") == "0":
        pos_data = None
        for _ in range(8):
            time.sleep(2.5)
            pos_data = trader.fetch_current_position(symbol)
            if pos_data and pos_data is not False:
                break
        if pos_data:
            intended_size, actual_size = float(action["size"]), abs(
                float(pos_data.get("pos", "0"))
            )
            if actual_size >= intended_size * 0.9:
                trade_flow_logger.info(
                    f"[{symbol}] 仓位建立已确认。意图: {intended_size}, 实际: {actual_size}"
                )
                new_pos_side = "LONG" if float(pos_data.get("pos")) > 0 else "SHORT"
                strategy.set_position(new_pos_side)
                close_side = "sell" if new_pos_side == "LONG" else "buy"
                sl_res = trader.place_trigger_order(
                    symbol,
                    close_side,
                    actual_size,
                    action["stop_loss_price"],
                    "Stop-Loss",
                )
                sl_id = (
                    sl_res["data"][0]["algoId"]
                    if sl_res
                    and sl_res.get("code") == "0"
                    and sl_res["data"][0]["sCode"] == "0"
                    else None
                )
                trade_state = {
                    "entry_price": float(pos_data.get("avgPx")),
                    "initial_stop_price": action["stop_loss_price"],
                    "current_stop_price": action["stop_loss_price"],
                    "current_stop_id": sl_id,
                    "sub_strategy": action["sub_strategy"],
                    "trailing_stop_active": False,
                    "highest_high_in_trade": float(pos_data.get("avgPx")),
                    "lowest_low_in_trade": float(pos_data.get("avgPx")),
                }
                save_trade_state(trade_state)
                trade_flow_logger.info(f"[{symbol}] 已成功保存交易状态: {trade_state}")
            else:
                trade_flow_logger.critical(
                    f"[{symbol}] [!!严重错误!!] 开仓验证失败！成交量不足！"
                )
        else:
            logging.error(f"[{symbol}] 下单后未能查询到仓位建立！")
    else:
        logging.error(f"[{symbol}] 下单请求失败，响应: {json.dumps(res)}")


def check_for_exit_signal(
    trader: OKXTrader,
    symbol: str,
    strategy: UltimateStrategy,
    pos_data: dict,
    current_price: float,
):
    if not (trade_state := load_trade_state()):
        return
    pos_side, pos_size = (
        "LONG" if float(pos_data.get("pos", "0")) > 0 else "SHORT"
    ), abs(float(pos_data.get("pos", "0")))
    if pos_side == "LONG":
        trade_state["highest_high_in_trade"] = max(
            trade_state.get("highest_high_in_trade", current_price), current_price
        )
    else:
        trade_state["lowest_low_in_trade"] = min(
            trade_state.get("lowest_low_in_trade", current_price), current_price
        )
    save_trade_state(trade_state)
    exit_condition_met, exit_reason = False, ""
    if trade_state.get("sub_strategy") == "TF":
        atr = strategy.data["tf_atr"].iloc[-1]
        if pd.isna(atr):
            return
        if pos_side == "LONG":
            if current_price < (
                lvl := trade_state["highest_high_in_trade"]
                - atr * strategy.tf_chandelier_atr_multiplier
            ):
                exit_condition_met, exit_reason = (
                    True,
                    f"Chandelier Exit (L) at {current_price:.4f} < {lvl:.4f}",
                )
        else:
            if current_price > (
                lvl := trade_state["lowest_low_in_trade"]
                + atr * strategy.tf_chandelier_atr_multiplier
            ):
                exit_condition_met, exit_reason = (
                    True,
                    f"Chandelier Exit (S) at {current_price:.4f} > {lvl:.4f}",
                )
    elif trade_state.get("sub_strategy") == "MR":
        bb_mid = strategy.data["mr_bb_mid"].iloc[-1]
        if pd.isna(bb_mid):
            return
        if pos_side == "LONG" and current_price >= bb_mid:
            exit_condition_met, exit_reason = (
                True,
                f"MR Exit (L) at {current_price:.4f} >= BB_mid {bb_mid:.4f}",
            )
        elif pos_side == "SHORT" and current_price <= bb_mid:
            exit_condition_met, exit_reason = (
                True,
                f"MR Exit (S) at {current_price:.4f} <= BB_mid {bb_mid:.4f}",
            )
    if exit_condition_met:
        trade_flow_logger.info(f"[{symbol}] [策略出场信号] {exit_reason}. 市价平仓...")
        close_side = "sell" if pos_side == "LONG" else "buy"
        trader.place_market_order(symbol, close_side, pos_size)
        if open_algos := trader.fetch_open_algo_orders(symbol):
            trader.cancel_algo_orders(symbol, [o["algoId"] for o in open_algos])


def manage_trailing_stop(
    trader: OKXTrader,
    symbol: str,
    strategy: UltimateStrategy,
    current_price: float,
    pos_data: dict,
):
    if not strategy.tsl_enabled or not (trade_state := load_trade_state()):
        return
    pos_side, entry_price, current_stop_price, is_active = (
        ("LONG" if float(pos_data.get("pos", "0")) > 0 else "SHORT"),
        float(trade_state["entry_price"]),
        float(trade_state["current_stop_price"]),
        trade_state["trailing_stop_active"],
    )
    atr = strategy.data["tf_atr"].iloc[-1]
    if pd.isna(atr):
        logging.warning(f"[{symbol}] TSL 检查跳过，因为ATR值为NaN。")
        return
    if not is_active:
        pnl_pct = (
            ((current_price - entry_price) / entry_price) * 100
            if pos_side == "LONG"
            else ((entry_price - current_price) / entry_price) * 100
        )
        profit_pct_condition_met = pnl_pct >= strategy.tsl_activation_profit_pct
        activation_distance = atr * strategy.tsl_activation_atr_mult
        price_move_condition_met = (
            pos_side == "LONG" and current_price >= entry_price + activation_distance
        ) or (
            pos_side == "SHORT" and current_price <= entry_price - activation_distance
        )
        if profit_pct_condition_met or price_move_condition_met:
            trade_state["trailing_stop_active"] = is_active = True
            trade_flow_logger.info(
                f"[{symbol}] 移动止损已激活! 原因: {'利润' if profit_pct_condition_met else '价格移动'}达到阈值"
            )
            save_trade_state(trade_state)
    if is_active:
        trailing_distance = atr * strategy.tsl_trailing_atr_mult
        new_stop_price = None
        potential_stop = 0.0
        if pos_side == "LONG":
            potential_stop = current_price - trailing_distance
            if potential_stop > current_stop_price:
                new_stop_price = potential_stop
        else:
            potential_stop = current_price + trailing_distance
            if potential_stop < current_stop_price:
                new_stop_price = potential_stop
        trade_flow_logger.debug(
            f"[{symbol}] TSL_DEBUG | Side: {pos_side}, Active: {is_active}, CurPx: {current_price:.4f}, ATR: {atr:.4f}, TrailDist: {trailing_distance:.4f}, PotentialSL: {potential_stop:.4f}, CurrentSL: {current_stop_price:.4f}, NewSL: {f'{new_stop_price:.4f}' if new_stop_price is not None else 'None'}"
        )
        if new_stop_price is not None:
            trade_flow_logger.info(
                f"[{symbol}] 调整移动止损: 旧={current_stop_price:.4f}, 新={new_stop_price:.4f}"
            )
            pos_size, close_side, old_stop_id = (
                abs(float(pos_data.get("pos"))),
                "sell" if pos_side == "LONG" else "buy",
                trade_state.get("current_stop_id"),
            )
            if (
                (
                    res := trader.place_trigger_order(
                        symbol,
                        close_side,
                        pos_size,
                        new_stop_price,
                        "Trailing-Stop-Update",
                    )
                )
                and res.get("code") == "0"
                and res["data"][0]["sCode"] == "0"
            ):
                new_stop_id = res["data"][0]["algoId"]
                trade_flow_logger.info(
                    f"[{symbol}] 新移动止损单设置成功, Algo ID: {new_stop_id}"
                )
                if old_stop_id:
                    trader.cancel_algo_orders(symbol, [old_stop_id])
                    trade_flow_logger.info(
                        f"[{symbol}] 旧止损单 (ID: {old_stop_id}) 已取消。"
                    )
                trade_state["current_stop_price"], trade_state["current_stop_id"] = (
                    new_stop_price,
                    new_stop_id,
                )
                save_trade_state(trade_state)
            else:
                trade_flow_logger.error(f"[{symbol}] 调整移动止损失败！响应: {res}")


def main():
    global trade_flow_logger, trade_logger, position_logger
    trade_flow_logger, trade_logger, position_logger = (
        setup_logging(),
        setup_csv_logger(
            "trade_logger",
            "trades.csv",
            ["timestamp", "symbol", "action", "size", "response"],
        ),
        setup_csv_logger(
            "position_logger", "positions.csv", ["timestamp", "symbol", "position"]
        ),
    )
    logging.info("启动 OKX REST API 轮询交易程序 (V4.7 仓位计算修复版)...")
    if not all(
        (
            OKX_API_KEY := os.getenv("OKX_API_KEY"),
            OKX_API_SECRET := os.getenv("OKX_API_SECRET"),
            OKX_API_PASSPHRASE := os.getenv("OKX_API_PASSPHRASE"),
        )
    ):
        logging.error("请设置环境变量: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE")
        return
    trader = OKXTrader(
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, simulated=SIMULATED
    )
    initial_equity = trader.fetch_account_balance("USDT") or SIMULATED_EQUITY_START
    logging.info(f"初始账户权益为: {initial_equity:.2f} USDT")

    strategies = {}
    for symbol in SYMBOLS:
        trader.set_leverage(symbol, DESIRED_LEVERAGE, mgnMode="cross", posSide="net")
        if not (instrument_details := trader.fetch_instrument_details(symbol)):
            logging.error(f"无法为 {symbol} 获取合约详细信息，跳过。")
            continue
        if (
            initial_df := trader.fetch_history_klines(
                symbol, bar=KLINE_INTERVAL, limit=HISTORY_LIMIT
            )
        ) is None or initial_df.empty:
            logging.error(f"无法为 {symbol} 获取初始数据，退出。")
            return
        strategy = UltimateStrategy(
            initial_df.copy(),
            symbol,
            trader,
            pos_logger=position_logger,
            instrument_details=instrument_details,
        )
        strategy.equity = initial_equity
        strategies[symbol] = strategy
        strategies[symbol].cool_down_until = None
        clear_trade_state()
        if initial_position_data := trader.fetch_current_position(symbol):
            trade_flow_logger.warning(f"[{symbol}] 检测到已有仓位，启动智能接管流程...")
            pos_side = (
                "LONG" if float(initial_position_data.get("pos", "0")) > 0 else "SHORT"
            )
            pos_size = abs(float(initial_position_data.get("pos", "0")))
            entry_price = float(initial_position_data.get("avgPx", "0"))
            strategy.set_position(pos_side)
            trade_flow_logger.info(f"[{symbol}] [接管] 正在取消所有旧的挂单...")
            for i in range(3):
                if not (open_algos := trader.fetch_open_algo_orders(symbol)):
                    break
                trader.cancel_algo_orders(
                    symbol, [order["algoId"] for order in open_algos]
                )
                time.sleep(1)
            trade_flow_logger.info(f"[{symbol}] [接管] 旧挂单已全部取消。")
            strategy.compute_all_features(trader)
            atr = strategy.data["tf_atr"].iloc[-1]
            if pd.isna(atr):
                trade_flow_logger.critical(
                    f"[{symbol}] [接管失败] 无法计算ATR，不能为已有仓位设置保护性止损！请手动处理！"
                )
                continue
            sl_atr_mult = strategy.tf_stop_loss_atr_multiplier
            stop_loss_price = (
                entry_price - (atr * sl_atr_mult)
                if pos_side == "LONG"
                else entry_price + (atr * sl_atr_mult)
            )
            close_side = "sell" if pos_side == "LONG" else "buy"
            sl_res = trader.place_trigger_order(
                symbol, close_side, pos_size, stop_loss_price, "Stop-Loss (Takeover)"
            )
            sl_id = (
                sl_res["data"][0]["algoId"]
                if sl_res
                and sl_res.get("code") == "0"
                and sl_res["data"][0]["sCode"] == "0"
                else None
            )
            if not sl_id:
                trade_flow_logger.critical(
                    f"[{symbol}] [接管失败] 无法设置新的保护性止损单！请手动处理！响应: {sl_res}"
                )
                continue
            trade_flow_logger.info(
                f"[{symbol}] [接管] 已成功设置新的保护性止损单，ID: {sl_id}, 价格: {stop_loss_price:.4f}"
            )
            trade_state = {
                "entry_price": entry_price,
                "initial_stop_price": stop_loss_price,
                "current_stop_price": stop_loss_price,
                "current_stop_id": sl_id,
                "sub_strategy": "TF",
                "trailing_stop_active": False,
                "highest_high_in_trade": entry_price,
                "lowest_low_in_trade": entry_price,
            }
            save_trade_state(trade_state)
            trade_flow_logger.info(
                f"[{symbol}] [接管完成] 已重建交易状态并默认采用趋势跟踪(TF)退出逻辑。"
            )
        logging.info(
            f"策略 {symbol} 初始化成功，最新K線時間: {strategy.data.index[-1]}"
        )

    last_audit_time = datetime.utcnow()
    is_globally_paused = False
    last_dd_warning_time = {}

    while True:
        try:
            if is_globally_paused:
                trade_flow_logger.critical(
                    "程序因触发全局熔断而处于暂停状态。请手动检查账户并重启程序以恢复交易。"
                )
                time.sleep(60)
                continue

            current_equity = trader.fetch_account_balance("USDT")

            for symbol in SYMBOLS:
                if symbol not in strategies:
                    continue
                strategy = strategies[symbol]

                real_position_data = trader.fetch_current_position(symbol)

                if real_position_data and strategy.position is None:
                    trade_flow_logger.critical(
                        f"[{symbol}] !!! 紧急情况 !!! 检测到未被程序管理的仓位（幽灵仓位），立即启动紧急接管！"
                    )
                    pos_side = (
                        "LONG"
                        if float(real_position_data.get("pos", "0")) > 0
                        else "SHORT"
                    )
                    pos_size = abs(float(real_position_data.get("pos", "0")))
                    entry_price = float(real_position_data.get("avgPx", "0"))
                    strategy.set_position(pos_side)
                    trade_flow_logger.info(
                        f"[{symbol}] [紧急接管] 正在取消所有旧的挂单..."
                    )
                    if open_algos := trader.fetch_open_algo_orders(symbol):
                        trader.cancel_algo_orders(
                            symbol, [order["algoId"] for order in open_algos]
                        )
                    trade_flow_logger.info(f"[{symbol}] [紧急接管] 旧挂单已取消。")
                    fresh_history_df = trader.fetch_history_klines(
                        symbol, bar=KLINE_INTERVAL, limit=HISTORY_LIMIT
                    )
                    if fresh_history_df is not None and not fresh_history_df.empty:
                        strategy.data = fresh_history_df
                        strategy.compute_all_features(trader)
                    atr = strategy.data["tf_atr"].iloc[-1]
                    if pd.isna(atr):
                        trade_flow_logger.critical(
                            f"[{symbol}] [紧急接管失败] 无法计算ATR，不能为幽灵仓位设置保护性止损！请立即手动处理！"
                        )
                        is_globally_paused = True
                        continue
                    sl_atr_mult = strategy.tf_stop_loss_atr_multiplier
                    stop_loss_price = (
                        entry_price - (atr * sl_atr_mult)
                        if pos_side == "LONG"
                        else entry_price + (atr * sl_atr_mult)
                    )
                    close_side = "sell" if pos_side == "LONG" else "buy"
                    sl_res = trader.place_trigger_order(
                        symbol,
                        close_side,
                        pos_size,
                        stop_loss_price,
                        "Stop-Loss (Emergency Takeover)",
                    )
                    sl_id = (
                        sl_res["data"][0]["algoId"]
                        if sl_res
                        and sl_res.get("code") == "0"
                        and sl_res["data"][0]["sCode"] == "0"
                        else None
                    )
                    if not sl_id:
                        trade_flow_logger.critical(
                            f"[{symbol}] [紧急接管失败] 无法设置新的保护性止损单！请立即手动处理！"
                        )
                        is_globally_paused = True
                        continue
                    trade_state = {
                        "entry_price": entry_price,
                        "initial_stop_price": stop_loss_price,
                        "current_stop_price": stop_loss_price,
                        "current_stop_id": sl_id,
                        "sub_strategy": "TF",
                        "trailing_stop_active": False,
                        "highest_high_in_trade": entry_price,
                        "lowest_low_in_trade": entry_price,
                    }
                    save_trade_state(trade_state)
                    trade_flow_logger.info(
                        f"[{symbol}] [紧急接管完成] 已为幽灵仓位设置止损保护。"
                    )

                if current_equity:
                    strategy.equity = current_equity
                    if current_equity < initial_equity * (1 - MAX_DAILY_DRAWDOWN_PCT):
                        now = datetime.utcnow()
                        if symbol not in last_dd_warning_time or (
                            now
                            - last_dd_warning_time.get(symbol, now - timedelta(hours=1))
                        ) > timedelta(minutes=5):
                            trade_flow_logger.critical(
                                f"全局熔断！当前权益 {current_equity:.2f} USDT 已低于最大回撤限制。程序将暂停所有交易活动！"
                            )
                            last_dd_warning_time[symbol] = now
                        is_globally_paused = True
                        continue

                if (
                    latest_candles_df := trader.fetch_history_klines(
                        symbol, bar=KLINE_INTERVAL, limit=2
                    )
                ) is None or latest_candles_df.empty:
                    logging.warning(f"无法获取 {symbol} 的最新K线数据。")
                    continue
                latest_candle = latest_candles_df.iloc[-1]
                if latest_candle.name > strategy.data.index[-1]:
                    logging.info(
                        f"[{symbol}] 新K線 {latest_candle.name} C={latest_candle.Close:.4f} | 仓位: {strategy.position}"
                    )
                    strategy.update_with_candle(latest_candle)
                    strategy.compute_all_features(trader)
                    atr_val = (
                        strategy.data["tf_atr"].iloc[-1]
                        if not strategy.data.empty and "tf_atr" in strategy.data.columns
                        else "N/A"
                    )
                    logging.info(
                        f"[{symbol}] 指标已基于新K线重算。最新ATR: {atr_val if pd.notna(atr_val) else 'NaN'}"
                    )
                    if strategy.position is None:
                        if (
                            strategy.cool_down_until
                            and datetime.utcnow() < strategy.cool_down_until
                        ):
                            trade_flow_logger.info(
                                f"[{symbol}] 策略处于交易后冷静期，暂停开仓至 {strategy.cool_down_until.strftime('%Y-%m-%d %H:%M:%S')} UTC。"
                            )
                            continue
                        if strategy.is_trading_paused:
                            logging.warning(f"[{symbol}] 策略因连续亏损暂停交易中...")
                            continue
                        if action := strategy.next_on_candle_close():
                            trade_flow_logger.info(f"[{symbol}] 策略决策: {action}")
                            manage_position_entry(trader, symbol, strategy, action)

                # 注意：这里的 real_position_data 可能是循环开始时获取的，我们再次获取以确保最新
                current_position_data = trader.fetch_current_position(symbol)
                if current_position_data:
                    if strategy.position is not None:
                        if current_price := trader.fetch_ticker_price(symbol):
                            check_for_exit_signal(
                                trader,
                                symbol,
                                strategy,
                                current_position_data,
                                current_price,
                            )
                            manage_trailing_stop(
                                trader,
                                symbol,
                                strategy,
                                current_position_data,
                                current_price,
                            )
                        else:
                            logging.warning(f"[{symbol}] 无法获取价格，跳过退出检查。")
                elif strategy.position is not None:
                    trade_flow_logger.info(f"[{symbol}] 检测到仓位已平仓。")
                    strategy.cool_down_until = datetime.utcnow() + timedelta(
                        minutes=COOL_DOWN_PERIOD_MINUTES
                    )
                    trade_flow_logger.warning(
                        f"[{symbol}] 交易结束，启动 {COOL_DOWN_PERIOD_MINUTES} 分钟冷静期。"
                    )
                    if (
                        closed_trade_state := load_trade_state()
                    ) and "entry_price" in closed_trade_state:
                        entry_price = closed_trade_state["entry_price"]
                        exit_price = latest_candle.Close
                        pnl_pct = (
                            (
                                (
                                    (exit_price - entry_price) / entry_price
                                    if strategy.position == "LONG"
                                    else (entry_price - exit_price) / entry_price
                                )
                            )
                            if entry_price != 0
                            else 0.0
                        )
                        trade_flow_logger.info(
                            f"[{symbol}] 交易结束: 入场={entry_price:.4f}, 出场≈{exit_price:.4f}, PnL %≈{pnl_pct:.4%}"
                        )
                        strategy.register_trade_result(pnl_pct)
                    strategy.set_position(None)
                    clear_trade_state()
                    trade_flow_logger.info(f"[{symbol}] 策略状态已重置。")

                current_price = trader.fetch_ticker_price(symbol)
                price_str = f"{current_price:.4f}" if current_price else "N/A"
                equity_str = f"{strategy.equity:.2f}" if strategy.equity else "N/A"
                pos_str = strategy.position if strategy.position is not None else "无"
                logging.info(
                    f"[{symbol}] 等待新K線... 仓位: {pos_str} | 权益: {equity_str} | 价格: {price_str} | K線: {strategy.data.index[-1]}"
                )

            if datetime.utcnow() - last_audit_time >= pd.Timedelta(
                minutes=AUDIT_INTERVAL_MINUTES
            ):
                last_audit_time = datetime.utcnow()

            logging.info(
                f"所有交易对检查完毕，将在 {POLL_INTERVAL_SECONDS} 秒后再次轮询..."
            )
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("程序被手动中断...")
            break
        except Exception as e:
            logging.exception(f"主循环发生未知错误: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)

    logging.info("程序正在退出。")
    clear_trade_state()


if __name__ == "__main__":
    main()
