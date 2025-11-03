# okx_live_trading_v4.19_timeout_compat_fix.py
"""
OKX WebSocket API 订阅版 (V4.19 - 最终兼容性修正版)
- 核心变更:
    - [!!! 关键崩溃修复 !!!] 解决了 `TypeError` on `socket_timeout` 的问题。
        - 移除 `run_forever` 中的 `socket_timeout` 参数，因为它仅在最新版的 websocket-client 中可用。
        - 改用 `websocket.setdefaulttimeout(30)` 这一全局设置，它兼容所有版本，并同样能解决网络连接超时问题。
    - [继承] 此版本是建立在之前所有修复（双连接架构、正确的频道订阅、稳定的 SSL 连接）之上的最终完整版本。
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
import ssl
import threading
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
import websocket
import certifi  # <-- 必需: pip install certifi

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
    root_logger.setLevel(logging.INFO)
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
    trade_flow_logger.setLevel(logging.INFO)
    trade_flow_logger.propagate = False
    trade_flow_path = os.path.join(log_dir, "trades_flow.log")
    trade_flow_handler = RotatingFileHandler(
        trade_flow_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    trade_flow_handler.setLevel(logging.INFO)
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
REST_BASE, SYMBOLS, KLINE_INTERVAL = "https://www.okx.com", ["ETH-USDT-SWAP"], "15m"
DESIRED_LEVERAGE, HISTORY_LIMIT, POLL_INTERVAL_SECONDS = "1", 500, 5
SIMULATED, SIMULATED_EQUITY_START = True, 500000.0
MAX_DAILY_DRAWDOWN_PCT, MAX_CONSECUTIVE_LOSSES = 0.03, 5
TRADING_PAUSE_HOURS, COOL_DOWN_PERIOD_MINUTES = 4, 1
KERAS_MODEL_PATH = "models/eth_trend_model_v1_15m.keras"
SCALER_PATH, FEATURE_COLUMNS_PATH = (
    "models/eth_trend_scaler_v1_15m.joblib",
    "models/feature_columns_15m.joblib",
)
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
ASSET_SPECIFIC_OVERRIDES = {"ETH-USDT-SWAP": {"score_entry_threshold": 0.45}}


# --- 状态管理 & 辅助函数 ---
def save_trade_state(state):
    with open(os.path.join("logs", "trade_state.json"), "w") as f:
        json.dump(state, f, indent=4)
    logging.debug(f"已保存交易状态: {state}")


def load_trade_state():
    state_file = os.path.join("logs", "trade_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None


def clear_trade_state():
    state_file = os.path.join("logs", "trade_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        logging.info("交易状态文件已清除。")


def save_kelly_history(symbol, history_deque):
    with open(os.path.join("logs", f"{symbol}_kelly_history.json"), "w") as f:
        json.dump(list(history_deque), f, indent=4)
    logging.debug(f"[{symbol}] 已保存凯利公式交易历史。")


def load_kelly_history(symbol, maxlen):
    history_file = os.path.join("logs", f"{symbol}_kelly_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                return deque(json.load(f), maxlen=maxlen)
        except (json.JSONDecodeError, TypeError):
            pass
    return deque(maxlen=maxlen)


# --- 特征工程函数 ---
def compute_hurst(ts, max_lag=100):
    if len(ts) < 10:
        return 0.5
    lags, tau = range(2, min(max_lag, len(ts) // 2 + 1)), []
    for lag in lags:
        std_dev = np.std(np.subtract(ts[lag:], ts[:-lag]))
        if std_dev > 0:
            tau.append(std_dev)
    if len(tau) < 2:
        return 0.5
    try:
        return max(
            0.0, min(1.0, np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0])
        )
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
    atr = ta.volatility.AverageTrueRange(
        df.High, df.Low, df.Close, p["regime_atr_period"]
    ).average_true_range()
    df["feature_adx_norm"] = norm(
        ta.trend.ADXIndicator(df.High, df.Low, df.Close, p["regime_adx_period"]).adx()
    )
    df["feature_atr_slope_norm"] = norm(
        (atr - atr.shift(p["regime_atr_slope_period"]))
        / atr.shift(p["regime_atr_slope_period"])
    )
    df["feature_rsi_vol_norm"] = 1 - norm(
        ta.momentum.RSIIndicator(df.Close, p["regime_rsi_period"])
        .rsi()
        .rolling(p["regime_rsi_vol_period"])
        .std()
    )
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


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["regime_score"] = df["feature_regime_score"]
    df["trend_regime"] = np.where(
        df["regime_score"] > STRATEGY_PARAMS["regime_score_threshold"],
        "Trending",
        "Mean-Reverting",
    )
    df["market_regime"] = np.where(df["trend_regime"] == "Trending", 1, -1)
    return df


def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    if not ML_LIBS_INSTALLED:
        return df
    close = df["Close"]
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], close, 14)
    bb = ta.volatility.BollingerBands(close, 20, 2.0)
    macd = ta.trend.MACD(close, 12, 26, 9)
    df["EMA_8"] = ta.trend.EMAIndicator(close=close, window=8).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    df["BBP_20_2.0"] = bb.bollinger_pband()
    df["MACDh_12_26_9"] = macd.macd_diff()
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
        self.instrument_info, self.common_headers = {}, {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }
        if self.simulated:
            self.common_headers["x-simulated-trading"] = "1"
        self.session = requests.Session()
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=5,
                    backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504],
                    allowed_methods=["GET", "POST"],
                )
            ),
        )

    def _now(self):
        return datetime.utcnow().isoformat("T", "milliseconds") + "Z"

    def _sign(self, ts, method, path, body_str=""):
        mac = hmac.new(
            self.api_secret.encode(),
            f"{ts}{method}{path}{body_str}".encode(),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method, path, body=None, params=None, max_retries=7):
        ts, body_str = self._now(), "" if body is None else json.dumps(body)
        prepped = PreparedRequest()
        prepped.prepare_url(self.base + path, params)
        path_for_signing = urllib.parse.urlparse(prepped.url).path
        if query := urllib.parse.urlparse(prepped.url).query:
            path_for_signing += "?" + query
        headers = self.common_headers.copy()
        headers.update(
            {
                "OK-ACCESS-SIGN": self._sign(
                    ts,
                    method.upper(),
                    path_for_signing.replace(self.base, ""),
                    body_str,
                ),
                "OK-ACCESS-TIMESTAMP": ts,
            }
        )
        for attempt in range(max_retries):
            try:
                r = self.session.request(
                    method,
                    self.base + path,
                    headers=headers,
                    data=body_str,
                    params=params,
                    timeout=(5, 15),
                )
                if r.status_code == 404 and "cancel-algo-order" in path:
                    return {"code": "0", "data": [{"sCode": "0"}]}
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"HTTP请求错误 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return {"code": "-1", "msg": f"Max retries exceeded: {e}"}
                time.sleep(min(2**attempt, 30))
        return {"code": "-1", "msg": "Max retries exceeded"}

    def place_trigger_order(self, instId, side, sz, trigger_price, posSide="net"):
        body = {
            "instId": instId,
            "tdMode": "cross",
            "side": side,
            "posSide": posSide,
            "ordType": "trigger",
            "sz": str(sz),
            "triggerPx": f"{trigger_price:.8f}".rstrip("0").rstrip("."),
            "orderPx": "-1",
        }
        return self._request("POST", "/api/v5/trade/order-algo", body=body)

    def fetch_open_algo_orders(self, instId):
        res = self._request(
            "GET",
            "/api/v5/trade/orders-algo-pending",
            params={"instType": "SWAP", "instId": instId, "ordType": "trigger"},
        )
        return res.get("data", []) if res and res.get("code") == "0" else []

    def cancel_algo_orders(self, instId, algoIds):
        if not algoIds:
            return True
        return (
            res := self._request(
                "POST",
                "/api/v5/trade/cancel-algo-order",
                body=[{"instId": instId, "algoId": str(aid)} for aid in algoIds],
            )
        ) and res.get("code") == "0"

    def set_leverage(self, instId, lever, mgnMode="cross", posSide="net"):
        return (
            self._request(
                "POST",
                "/api/v5/account/set-leverage",
                body={
                    "instId": instId,
                    "lever": str(lever),
                    "mgnMode": mgnMode,
                    "posSide": posSide,
                },
            ).get("code")
            == "0"
        )

    def fetch_account_balance(self, ccy="USDT"):
        if (
            res := self._request("GET", "/api/v5/account/balance", params={"ccy": ccy})
        ) and res.get("code") == "0":
            if (data := res.get("data", [])) and "details" in data[0]:
                return next(
                    (
                        float(d.get("eq", 0))
                        for d in data[0]["details"]
                        if d.get("ccy") == ccy
                    ),
                    None,
                )
        return None

    def fetch_current_position(self, instId):
        if (
            res := self._request(
                "GET", "/api/v5/account/positions", params={"instId": instId}
            )
        ) and res.get("code") == "0":
            return next(
                (pos for pos in res.get("data", []) if float(pos.get("pos", "0")) != 0),
                None,
            )
        return False

    def fetch_instrument_details(self, instId):
        if instId in self.instrument_info:
            return self.instrument_info[instId]
        if (
            (
                res := self._request(
                    "GET",
                    "/api/v5/public/instruments",
                    params={"instType": "SWAP", "instId": instId},
                )
            )
            and res.get("code") == "0"
            and res.get("data")
        ):
            return self.instrument_info.setdefault(instId, res["data"][0])
        return None

    def place_market_order(self, instId, side, sz):
        res = self._request(
            "POST",
            "/api/v5/trade/order",
            body={
                "instId": instId,
                "tdMode": "cross",
                "side": side,
                "ordType": "market",
                "sz": str(sz),
            },
        )
        if res:
            trade_logger.info(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "symbol": instId,
                    "action": side.upper(),
                    "size": sz,
                    "response": json.dumps(res),
                }
            )
        return res

    def fetch_history_klines(self, instId, bar="1m", limit=200):
        if (
            res := self._request(
                "GET",
                "/api/v5/market/history-candles",
                params={"instId": instId, "bar": bar, "limit": limit},
            )
        ) and res.get("code") == "0":
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
            for col in ["o", "h", "l", "c", "vol"]:
                df[col] = pd.to_numeric(df[col])
            df = df.rename(
                columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "vol": "Volume",
                }
            )
            df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
            return df.set_index("timestamp").sort_index()
        return None

    def fetch_ticker_price(self, instId):
        if (
            (
                res := self._request(
                    "GET", "/api/v5/market/ticker", params={"instId": instId}
                )
            )
            and res.get("code") == "0"
            and res.get("data")
        ):
            return float(res["data"][0].get("last"))
        return None


# --- [ 新增 ] 公共数据 WebSocket 管理器 ---
class OKXPublicWebSocketManager:
    def __init__(self, symbols, strategies, trader):
        self._WS_BASE = (
            "wss://ws.okx.com:8443/ws/v5/public"
            if not SIMULATED
            else "wss://ws.okx.com:8443/ws/v5/public?brokerId=9999"
        )
        self.symbols, self.strategies, self.trader = symbols, strategies, trader
        self.should_run, self.ws_thread, self.ws_app = True, None, None

    def _on_open(self, ws):
        logging.info("Public WebSocket connection opened. Subscribing to channels...")
        self._subscribe_to_channels(ws)

    def _on_message(self, ws, message):
        if message == "pong":
            return logging.debug("Public WS received pong.")
        data = json.loads(message)

        if "event" in data:
            if data.get("event") == "error":
                logging.error(
                    f"Public WS Error: {data.get('msg')} (Code: {data.get('code')})"
                )
            elif data.get("event") == "subscribe":
                logging.info(
                    f"Public WS subscribed to: {data.get('arg', {}).get('channel')}"
                )
            return

        if "arg" in data and "data" in data:
            channel, instId = data["arg"]["channel"], data["arg"]["instId"]
            if (
                channel == f"candles{KLINE_INTERVAL}"
            ):  # Note: OKX channel name is plural
                self._handle_candle_data(instId, data["data"])

    def _subscribe_to_channels(self, ws):
        # [!!! V4.19 修正 !!!] 恢复为正确的频道名称 `candles{bar}` (复数)
        subscriptions = [
            {"channel": f"candles{KLINE_INTERVAL}", "instId": symbol}
            for symbol in self.symbols
        ]
        payload = {"op": "subscribe", "args": subscriptions}
        ws.send(json.dumps(payload))
        logging.info(f"Public WS subscription sent: {json.dumps(payload)}")

    def _handle_candle_data(self, symbol, candle_data_list):
        if symbol not in self.strategies:
            return
        for ts, o, h, l, c, vol, _, _, confirm in candle_data_list:
            if confirm == "1":
                strategy, candle_ts = self.strategies[symbol], pd.to_datetime(
                    int(ts), unit="ms"
                )
                if candle_ts <= strategy.data.index[-1]:
                    continue
                trade_flow_logger.info(f"[{symbol}] 新K线收盘: {candle_ts} | C={c}")
                new_candle = pd.Series(
                    {
                        "Open": float(o),
                        "High": float(h),
                        "Low": float(l),
                        "Close": float(c),
                        "Volume": float(vol),
                    },
                    name=candle_ts,
                )
                strategy.update_with_candle(new_candle)
                strategy.compute_all_features(self.trader)
                if (
                    strategy.position is None
                    and not (
                        strategy.cool_down_until
                        and datetime.utcnow() < strategy.cool_down_until
                    )
                    and not strategy.is_trading_paused
                ):
                    if action := strategy.next_on_candle_close():
                        trade_flow_logger.info(f"[{symbol}] 策略决策: {action}")
                        manage_position_entry(self.trader, symbol, strategy, action)

    def _on_error(self, ws, error):
        logging.error(f"Public WS Error: {error}")

    def _on_close(self, ws, status, msg):
        logging.warning("Public WS connection closed.")

    def connect(self):
        def run_ws():
            reconnect_wait = 5
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            sslopt = {"ssl_context": ssl_context}

            while self.should_run:
                logging.info("Attempting to connect to Public WebSocket...")
                self.ws_app = websocket.WebSocketApp(
                    self._WS_BASE,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws_app.run_forever(
                    ping_interval=20, ping_timeout=10, sslopt=sslopt
                )
                if not self.should_run:
                    break
                logging.error(
                    f"Public WS disconnected. Reconnecting in {reconnect_wait}s..."
                )
                time.sleep(reconnect_wait)
                reconnect_wait = min(reconnect_wait * 2, 60)

        self.ws_thread = threading.Thread(target=run_ws)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def close(self):
        logging.info("Closing Public WebSocket connection...")
        self.should_run = False
        if self.ws_app:
            self.ws_app.close()


# --- [ 重构 ] 私有数据 WebSocket 管理器 ---
class OKXPrivateWebSocketManager:
    def __init__(
        self, api_key, api_secret, passphrase, strategies, trader, initial_equity
    ):
        self._WS_BASE = (
            "wss://ws.okx.com:8443/ws/v5/private"
            if not SIMULATED
            else "wss://ws.okx.com:8443/ws/v5/private?brokerId=9999"
        )
        self.api_key, self.api_secret, self.passphrase = api_key, api_secret, passphrase
        self.strategies, self.trader = strategies, trader
        self.should_run, self.is_globally_paused = True, False
        self.initial_equity = initial_equity
        self.ws_thread, self.ws_app = None, None

    def _get_auth_args(self):
        ts = str(int(time.time()))
        message = ts + "GET" + "/users/self/verify"
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        return [
            {
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": ts,
                "sign": base64.b64encode(mac.digest()).decode(),
            }
        ]

    def _on_open(self, ws):
        logging.info("Private WebSocket connection opened. Authenticating...")
        ws.send(json.dumps({"op": "login", "args": self._get_auth_args()}))

    def _on_message(self, ws, message):
        if message == "pong":
            return logging.debug("Private WS received pong.")
        data = json.loads(message)

        if "event" in data:
            event, msg, code = data.get("event"), data.get("msg", ""), data.get("code")
            if event == "login" and code == "0":
                logging.info("Private WS authenticated. Subscribing...")
                self._subscribe_to_channels(ws)
            elif event == "subscribe":
                logging.info(
                    f"Private WS subscribed to: {data.get('arg', {}).get('channel')}"
                )
            elif event == "error":
                logging.error(f"Private WS Error: {msg} (Code: {code})")
            return

        if "arg" in data and "data" in data:
            channel = data["arg"]["channel"]
            if channel == "positions":
                self._handle_position_data(data["data"])
            elif channel == "account":
                self._handle_account_data(data["data"])

    def _subscribe_to_channels(self, ws):
        subscriptions = [
            {"channel": "account", "ccy": "USDT"},
            {"channel": "positions", "instType": "SWAP"},
        ]
        payload = {"op": "subscribe", "args": subscriptions}
        ws.send(json.dumps(payload))
        logging.info(f"Private WS subscription sent: {json.dumps(payload)}")

    def _handle_position_data(self, position_data_list):
        active_symbols = {
            pos.get("instId")
            for pos in position_data_list
            if float(pos.get("pos", "0")) != 0
        }
        for symbol, strategy in self.strategies.items():
            if strategy.position is not None and symbol not in active_symbols:
                trade_flow_logger.info(f"[{symbol}] WebSocket确认仓位已平仓。")
                self._handle_position_closure(symbol)

    def _handle_position_closure(self, symbol):
        strategy = self.strategies[symbol]
        strategy.cool_down_until = datetime.utcnow() + timedelta(
            minutes=COOL_DOWN_PERIOD_MINUTES
        )
        trade_flow_logger.warning(
            f"[{symbol}] 交易结束，启动 {COOL_DOWN_PERIOD_MINUTES} 分钟冷静期。"
        )
        if (trade_state := load_trade_state()) and "entry_price" in trade_state:
            entry_price, pos_side = trade_state["entry_price"], strategy.position
            exit_price = (
                self.trader.fetch_ticker_price(symbol)
                or strategy.data["Close"].iloc[-1]
            )
            if entry_price != 0:
                pnl_pct = (
                    (exit_price - entry_price) / entry_price
                    if pos_side == "LONG"
                    else (entry_price - exit_price) / entry_price
                )
                trade_flow_logger.info(
                    f"[{symbol}] 交易结果: 入场={entry_price:.4f}, 出场≈{exit_price:.4f}, PnL %≈{pnl_pct:.4%}"
                )
                strategy.register_trade_result(pnl_pct)
        strategy.set_position(None)
        clear_trade_state()
        trade_flow_logger.info(f"[{symbol}] 策略状态已重置。")

    def _handle_account_data(self, account_data_list):
        for acc in account_data_list:
            if "details" in acc:
                for detail in acc["details"]:
                    if detail.get("ccy") == "USDT":
                        equity = float(detail.get("eq", 0))
                        for strategy in self.strategies.values():
                            strategy.equity = equity
                        if (
                            not self.is_globally_paused
                            and equity
                            < self.initial_equity * (1 - MAX_DAILY_DRAWDOWN_PCT)
                        ):
                            self.is_globally_paused = True
                            trade_flow_logger.critical(
                                f"全局熔断！当前权益 {equity:.2f} USDT 已低于最大回撤限制。"
                            )
                        return

    def _on_error(self, ws, error):
        logging.error(f"Private WS Error: {error}")

    def _on_close(self, ws, status, msg):
        logging.warning("Private WS connection closed.")

    def connect(self):
        def run_ws():
            reconnect_wait = 5
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            sslopt = {"ssl_context": ssl_context}

            while self.should_run:
                logging.info("Attempting to connect to Private WebSocket...")
                self.ws_app = websocket.WebSocketApp(
                    self._WS_BASE,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self.ws_app.run_forever(
                    ping_interval=20, ping_timeout=10, sslopt=sslopt
                )
                if not self.should_run:
                    break
                logging.error(
                    f"Private WS disconnected. Reconnecting in {reconnect_wait}s..."
                )
                time.sleep(reconnect_wait)
                reconnect_wait = min(reconnect_wait * 2, 60)

        self.ws_thread = threading.Thread(target=run_ws)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def close(self):
        logging.info("Closing Private WebSocket connection...")
        self.should_run = False
        if self.ws_app:
            self.ws_app.close()


# --- 策略核心类 ---
class UltimateStrategy:
    def __init__(self, df, symbol, trader, pos_logger, instrument_details):
        self.symbol, self.trader, self.data, self.position = (
            symbol,
            trader,
            df.copy(),
            None,
        )
        self.pos_logger, self.instrument_details = pos_logger, instrument_details
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)
        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(symbol, {})
        self.score_entry_threshold = asset_overrides.get(
            "score_entry_threshold", self.score_entry_threshold
        )
        self.recent_trade_returns = load_kelly_history(symbol, self.kelly_trade_history)
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
            self.keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
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
            np.mean(wins),
            abs(np.mean(losses)),
        )
        if avg_loss == 0 or (reward_ratio := avg_win / avg_loss) == 0:
            return self.default_risk_pct
        return min(
            max(0.005, (win_rate - (1 - win_rate) / reward_ratio) * 0.5),
            self.max_risk_pct,
        )

    def get_ml_confidence_score(self):
        if not all([self.keras_model, self.scaler, self.feature_columns]):
            return 0.0
        if len(self.data) < KERAS_SEQUENCE_LENGTH:
            return 0.0
        try:
            features = (
                self.data[self.feature_columns]
                .iloc[-KERAS_SEQUENCE_LENGTH:]
                .copy()
                .fillna(method="ffill")
                .fillna(0)
            )
            scaled = self.scaler.transform(features)
            pred = self.keras_model.predict(np.expand_dims(scaled, axis=0), verbose=0)
            return float((pred[0][0] - 0.5) * 2.0)
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
        if len(self.data) < self.tf_donchian_period:
            return
        self.data = add_ml_features_ported(self.data)
        self.data = add_market_regime_features(self.data)
        self.data = add_features_for_keras_model(self.data)
        if (
            data_1d := trader.fetch_history_klines(
                self.symbol, bar="1D", limit=self.mtf_period + 10
            )
        ) is not None and not data_1d.empty:
            sma = ta.trend.SMAIndicator(
                data_1d["Close"], self.mtf_period
            ).sma_indicator()
            self.data["mtf_signal"] = (
                pd.Series(np.where(data_1d["Close"] > sma, 1, -1), index=data_1d.index)
                .reindex(self.data.index, method="ffill")
                .fillna(0)
            )
        else:
            self.data["mtf_signal"] = 0
        df_4h = self.data["Close"].resample("4H").last().to_frame()
        df_4h["macro_ema"] = ta.trend.EMAIndicator(df_4h["Close"], 50).ema_indicator()
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
        self.data["mr_bb_mid"] = bb.bollinger_mavg()
        self.data.fillna(method="ffill", inplace=True)
        self.data.fillna(method="bfill", inplace=True)

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
        )

    def next_on_candle_close(self):
        last = self.data.iloc[-1]
        if pd.isna(last.get("market_regime")) or pd.isna(
            last.get("macro_trend_filter")
        ):
            return None
        action = None
        if (
            last.macro_trend_filter == 1
            and last.market_regime == 1
            and (score := self._calculate_entry_score()) > self.score_entry_threshold
        ):
            action = {"action": "BUY", "sub_strategy": "TF", "confidence": score}
        elif (
            last.macro_trend_filter == -1
            and last.market_regime == 1
            and (score := self._calculate_entry_score()) < -self.score_entry_threshold
        ):
            action = {"action": "SELL", "sub_strategy": "TF", "confidence": abs(score)}
        if action:
            risk_pct = self._calculate_dynamic_risk() * action["confidence"]
            if (
                size_and_sl := self._determine_position_size(
                    last.Close,
                    risk_pct,
                    self.tf_stop_loss_atr_multiplier,
                    action["action"] == "BUY",
                )
            ) and float(size_and_sl[0]) > 0:
                action["size"], action["stop_loss_price"] = size_and_sl
                return action
        return None

    def _adjust_size_to_lot_size(self, size):
        if not (details := self.instrument_details) or not (
            lot_sz_str := details.get("lotSz")
        ):
            return str(size)
        try:
            lot_sz = float(lot_sz_str)
            adjusted_size = math.floor(float(size) / lot_sz) * lot_sz
            return f"{adjusted_size:.{len(lot_sz_str.split('.')[1]) if '.' in lot_sz_str else 0}f}"
        except:
            return str(size)

    def _determine_position_size(self, price, risk_pct, sl_atr_mult, is_long):
        atr, ct_val = self.data["tf_atr"].iloc[-1], float(
            self.instrument_details.get("ctVal", 1)
        )
        if (
            price <= 0
            or pd.isna(atr)
            or (risk_per_unit := atr * sl_atr_mult) <= 0
            or ct_val <= 0
        ):
            return "0", 0.0
        units = (self.equity * risk_pct) / (risk_per_unit * ct_val)
        if (units * ct_val * price) / int(DESIRED_LEVERAGE) > self.equity * 0.95:
            trade_flow_logger.warning(
                f"[{self.symbol}] 仓位计算警告：所需保证金过高。跳过交易。"
            )
            return "0", 0.0
        return self._adjust_size_to_lot_size(units), (
            price - risk_per_unit if is_long else price + risk_per_unit
        )

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
                        "position": new_position,
                    }
                )


# --- 主逻辑 & 辅助函数 ---
def manage_position_entry(
    trader: OKXTrader, symbol: str, strategy: UltimateStrategy, action: dict
):
    res = trader.place_market_order(
        symbol, "buy" if action["action"] == "BUY" else "sell", action["size"]
    )
    if res and res.get("code") == "0" and res.get("data")[0].get("sCode") == "0":
        time.sleep(5)
        if pos_data := trader.fetch_current_position(symbol):
            actual_size, avgPx = abs(float(pos_data.get("pos", "0"))), float(
                pos_data.get("avgPx")
            )
            if actual_size >= float(action["size"]) * 0.9:
                pos_side = "LONG" if float(pos_data.get("pos")) > 0 else "SHORT"
                strategy.set_position(pos_side)
                sl_res = trader.place_trigger_order(
                    symbol,
                    "sell" if pos_side == "LONG" else "buy",
                    actual_size,
                    action["stop_loss_price"],
                )
                sl_id = (
                    sl_res.get("data", [{}])[0].get("algoId")
                    if sl_res.get("code") == "0"
                    else None
                )
                trade_state = {
                    "entry_price": avgPx,
                    "current_stop_price": action["stop_loss_price"],
                    "current_stop_id": sl_id,
                    "trailing_stop_active": False,
                    "highest_high_in_trade": avgPx,
                    "lowest_low_in_trade": avgPx,
                }
                save_trade_state(trade_state)
                trade_flow_logger.info(f"[{symbol}] 仓位建立并成功保存状态。")
            else:
                trade_flow_logger.critical(f"[{symbol}] 开仓验证失败！成交量不足！")
        else:
            logging.error(f"[{symbol}] 下单后未能查询到仓位！")
    else:
        logging.error(f"[{symbol}] 下单请求失败: {res}")


def check_for_exit_signal(
    trader: OKXTrader, strategy: UltimateStrategy, pos_data: dict, current_price: float
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
    atr = strategy.data["tf_atr"].iloc[-1]
    if pd.isna(atr):
        return
    if (
        pos_side == "LONG"
        and current_price
        < (
            lvl := trade_state["highest_high_in_trade"]
            - atr * strategy.tf_chandelier_atr_multiplier
        )
    ) or (
        pos_side == "SHORT"
        and current_price
        > (
            lvl := trade_state["lowest_low_in_trade"]
            + atr * strategy.tf_chandelier_atr_multiplier
        )
    ):
        trade_flow_logger.info(
            f"[{strategy.symbol}] Chandelier Exit triggered at {current_price:.4f}. 市价平仓..."
        )
        trader.place_market_order(
            strategy.symbol, "sell" if pos_side == "LONG" else "buy", pos_size
        )
        if open_algos := trader.fetch_open_algo_orders(strategy.symbol):
            trader.cancel_algo_orders(
                strategy.symbol, [o["algoId"] for o in open_algos]
            )


def manage_trailing_stop(
    trader: OKXTrader, strategy: UltimateStrategy, current_price: float, pos_data: dict
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
        return
    if not is_active:
        pnl_pct = (
            ((current_price - entry_price) / entry_price)
            if pos_side == "LONG"
            else ((entry_price - current_price) / entry_price)
        )
        if (
            pnl_pct >= strategy.tsl_activation_profit_pct
            or abs(current_price - entry_price)
            >= atr * strategy.tsl_activation_atr_mult
        ):
            trade_state["trailing_stop_active"] = is_active = True
            trade_flow_logger.info(f"[{strategy.symbol}] 移动止损已激活!")
            save_trade_state(trade_state)
    if is_active:
        new_stop_price = None
        if (
            pos_side == "LONG"
            and (potential_stop := current_price - atr * strategy.tsl_trailing_atr_mult)
            > current_stop_price
        ):
            new_stop_price = potential_stop
        elif (
            pos_side == "SHORT"
            and (potential_stop := current_price + atr * strategy.tsl_trailing_atr_mult)
            < current_stop_price
        ):
            new_stop_price = potential_stop
        if new_stop_price is not None:
            trade_flow_logger.info(
                f"[{strategy.symbol}] 调整移动止损: 旧={current_stop_price:.4f}, 新={new_stop_price:.4f}"
            )
            pos_size, old_stop_id = abs(float(pos_data.get("pos"))), trade_state.get(
                "current_stop_id"
            )
            res = trader.place_trigger_order(
                strategy.symbol,
                "sell" if pos_side == "LONG" else "buy",
                pos_size,
                new_stop_price,
            )
            if res and res.get("code") == "0" and res["data"][0]["sCode"] == "0":
                if old_stop_id:
                    trader.cancel_algo_orders(strategy.symbol, [old_stop_id])
                trade_state.update(
                    {
                        "current_stop_price": new_stop_price,
                        "current_stop_id": res["data"][0]["algoId"],
                    }
                )
                save_trade_state(trade_state)
            else:
                trade_flow_logger.error(f"[{strategy.symbol}] 调整移动止损失败: {res}")


# --- 主函数 Main ---
def main():
    global trade_flow_logger, trade_logger, position_logger, __version__
    __version__ = "4.19"
    trade_flow_logger = setup_logging()
    trade_logger = setup_csv_logger(
        "trade_logger",
        "trades.csv",
        ["timestamp", "symbol", "action", "size", "response"],
    )
    position_logger = setup_csv_logger(
        "position_logger", "positions.csv", ["timestamp", "symbol", "position"]
    )

    logging.info(f"启动 OKX WebSocket API 交易程序 (V{__version__})...")

    # [!!! V4.19 修正 !!!] 增加全局超时设置
    websocket.setdefaulttimeout(30)

    OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE = (
        os.getenv("OKX_API_KEY"),
        os.getenv("OKX_API_SECRET"),
        os.getenv("OKX_API_PASSPHRASE"),
    )
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE]):
        return logging.error(
            "请设置环境变量: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE"
        )

    trader = OKXTrader(
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, simulated=SIMULATED
    )
    initial_equity = trader.fetch_account_balance("USDT") or SIMULATED_EQUITY_START
    logging.info(f"初始账户权益为: {initial_equity:.2f} USDT")

    strategies = {}
    for symbol in SYMBOLS:
        details = trader.fetch_instrument_details(symbol)
        if not details:
            logging.error(f"无法为 {symbol} 获取合约信息。")
            continue
        if (
            df := trader.fetch_history_klines(
                symbol, bar=KLINE_INTERVAL, limit=HISTORY_LIMIT
            )
        ) is None or df.empty:
            return logging.error(f"无法为 {symbol} 获取初始数据。")
        strategy = UltimateStrategy(df, symbol, trader, position_logger, details)
        strategy.equity = initial_equity
        strategies[symbol] = strategy
        strategy.compute_all_features(trader)
        logging.info(f"策略 {symbol} 初始化完成。最新K线: {strategy.data.index[-1]}")

    clear_trade_state()
    for symbol, strategy in strategies.items():
        if pos_data := trader.fetch_current_position(symbol):
            trade_flow_logger.warning(f"[{symbol}] 检测到已有仓位，启动智能接管流程...")
            strategy.set_position(
                "LONG" if float(pos_data.get("pos", "0")) > 0 else "SHORT"
            )

    # --- 启动双 WebSocket 连接 ---
    public_ws = OKXPublicWebSocketManager(SYMBOLS, strategies, trader)
    public_ws.connect()

    private_ws = OKXPrivateWebSocketManager(
        OKX_API_KEY,
        OKX_API_SECRET,
        OKX_API_PASSPHRASE,
        strategies,
        trader,
        initial_equity,
    )
    private_ws.connect()

    logging.info("双 WebSocket Managers 已启动。机器人上线。按 Ctrl+C 停止。")

    try:
        while True:
            for symbol, strategy in strategies.items():
                if (
                    strategy.position
                    and (price := trader.fetch_ticker_price(symbol))
                    and (pos := trader.fetch_current_position(symbol))
                ):
                    check_for_exit_signal(trader, strategy, pos, price)
                    manage_trailing_stop(trader, strategy, price, pos)
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logging.info("程序被手动中断...")
    except Exception as e:
        logging.exception(f"主线程发生未知错误: {e}")
    finally:
        public_ws.close()
        private_ws.close()
        clear_trade_state()
        logging.info("程序已退出。")


if __name__ == "__main__":
    main()
