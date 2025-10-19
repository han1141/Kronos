# okx_ws.py (Final Corrected Version V3.11)
"""
OKX WebSocket & REST 混合高频交易机器人框架 (V3.11 - Training_Fix)
- 特性:
    - [V3.11] 修复了模型训练函数中因变量引用顺序不当导致的 'UnboundLocalError'。
    - [V3.10] 策略智能化升级: 引入自适应因子归一化（Z-score）和动态入场阈值（滚动百分位）。
    - [V3.9] 新增功能: 策略逻辑完全重构，增加独立的做空信号判断，实现了双向交易能力。
    - [V3.8] 修复了WebSocket私有频道登录时的编程错误。
    - [V3.7] 新增功能: 程序启动时动态获取账户初始资金。
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
import threading
import urllib.parse
from datetime import datetime
from logging.handlers import RotatingFileHandler
from requests.models import PreparedRequest
from collections import deque
from scipy.stats import norm  # V3.10 新增：用于Z-score的CDF转换

import pandas as pd
import numpy as np
import requests
import ta
import warnings
import websocket
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.simplefilter(action="ignore", category=FutureWarning)


# --- 日志系统 ---
def setup_logging():
    """配置日志系统，同时输出到控制台和文件"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    log_file_path = os.path.join(log_dir, "hft_ml_bot.log")
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
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
    return logging.getLogger("HFT-ML-Flow")


# --- 全局配置 (V3.10 - 自适应版本) ---
REST_BASE = "https://www.okx.com"
SYMBOL = "ETH-USDT-SWAP"
KLINE_INTERVAL = "1m"
DESIRED_LEVERAGE = "10"
HISTORY_LIMIT = 300
SIMULATED = True
if SIMULATED:
    WS_PUBLIC_URL = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
    WS_PRIVATE_URL = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"
else:
    WS_PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"
    WS_PRIVATE_URL = "wss://ws.okx.com:8443/ws/v5/private"

STRATEGY_PARAMS = {
    "volume_avg_period": 21,
    "volatility_period": 14,
    "obi_levels": 10,
    "obi_threshold": 0.6,
    "sl_lookback": 1,
    "tp_rr_ratio": 2.0,
    "risk_per_trade_pct": 0.01,
    "allow_shorting": True,
    "weights": {"volume": 0.25, "volatility": 0.10, "obi": 0.25, "xgb": 0.40},
    "z_score_period": 120,
    "dynamic_threshold_period": 240,
    "threshold_percentile": 90,
    "min_signal_history": 50,
    "backup_fixed_threshold": 0.65,
}


# --- 辅助函数：交易状态管理 ---
def save_trade_state(state):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    state_file = os.path.join(log_dir, "trade_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f)


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


# --- OKX 交易接口类 ---
class OKXTrader:
    def __init__(self, api_key, api_secret, passphrase, simulated=True):
        self.base = REST_BASE
        self.api_key, self.api_secret, self.passphrase = api_key, api_secret, passphrase
        self.simulated = simulated
        self.instrument_info = {}
        self.common_headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }

    def _now(self):
        return datetime.utcnow().isoformat("T", "milliseconds") + "Z"

    def _sign(self, ts, method, path, body_str=""):
        message = f"{ts}{method}{path}{body_str}"
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method, path, body=None, params=None, max_retries=3):
        ts, body_str = self._now(), "" if body is None else json.dumps(body)
        prepped = PreparedRequest()
        prepped.prepare_url(self.base + path, params)
        path_for_signing = urllib.parse.urlparse(prepped.url).path
        if urllib.parse.urlparse(prepped.url).query:
            path_for_signing += "?" + urllib.parse.urlparse(prepped.url).query
        path_for_signing = path_for_signing.replace(self.base, "")
        sign = self._sign(ts, method.upper(), path_for_signing, body_str)
        headers = self.common_headers.copy()
        if self.simulated:
            headers["x-simulated-trading"] = "1"
        headers.update({"OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": ts})
        url = self.base + path
        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    r = requests.get(url, headers=headers, params=params, timeout=10)
                else:
                    r = requests.post(url, headers=headers, data=body_str, timeout=10)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"发生REST请求错误 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt + 1 == max_retries:
                    return {"error": str(e)}
                time.sleep(2)
        return None

    def set_leverage(self, instId, lever, mgnMode="cross"):
        path = "/api/v5/account/set-leverage"
        body = {
            "instId": instId,
            "lever": str(lever),
            "mgnMode": mgnMode,
            "posSide": "net",
        }
        res = self._request("POST", path, body=body)
        return res and res.get("code") == "0"

    def fetch_account_balance(self, ccy="USDT"):
        path = "/api/v5/account/balance"
        params = {"ccy": ccy}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0" and res.get("data"):
            details = res["data"][0].get("details", [])
            if details:
                return float(details[0].get("eq", 0))
        logging.error(f"获取账户余额失败: {res}")
        return None

    def fetch_instrument_details(self, instId, instType="SWAP"):
        if instId in self.instrument_info:
            return self.instrument_info[instId]
        path = "/api/v5/public/instruments"
        params = {"instType": instType, "instId": instId}
        try:
            headers = {"x-simulated-trading": "1"} if self.simulated else {}
            res = requests.get(
                self.base + path, params=params, headers=headers, timeout=10
            )
            res.raise_for_status()
            j = res.json()
            if j.get("code") == "0" and j.get("data"):
                self.instrument_info[instId] = j["data"][0]
                return j["data"][0]
        except Exception as e:
            logging.error(f"获取合约信息时发生网络错误 for {instId}: {e}")
        return None

    def place_market_order(self, instId, side, sz, posSide="net"):
        path = "/api/v5/trade/order"
        body = {
            "instId": instId,
            "tdMode": "cross",
            "side": side,
            "posSide": posSide,
            "ordType": "market",
            "sz": str(sz),
        }
        return self._request("POST", path, body=body)

    def fetch_history_klines(self, instId, bar="1m", limit=200):
        path = "/api/v5/market/history-candles"
        params = {"instId": instId, "bar": bar, "limit": str(limit)}
        try:
            headers = {"x-simulated-trading": "1"} if self.simulated else {}
            r = requests.get(
                self.base + path, params=params, headers=headers, timeout=10
            )
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            if not data:
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
        except Exception as e:
            logging.error(f"获取历史K线数据错误: {e}")
            return None

    def fetch_current_position(self, instId):
        path = "/api/v5/account/positions"
        params = {"instId": instId}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0" and res.get("data"):
            for pos in res["data"]:
                if pos.get("instId") == instId and float(pos.get("pos", "0")) != 0:
                    return pos
        return None


# --- 模型与特征工程 ---
def create_features(df):
    df["feature_rsi"] = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["feature_atr_pct"] = (
        ta.volatility.AverageTrueRange(
            df.High, df.Low, df.Close, 14
        ).average_true_range()
        / df.Close
    )
    df["feature_volume_ma_ratio"] = df.Volume / df.Volume.rolling(21).mean()
    df["target"] = (df.Close.shift(-1) > df.Close).astype(int)
    return df.dropna()


def train_and_save_model(symbol, trader):
    """[V3.11修复] 修正了UnboundLocalError的bug"""
    logging.info("开始模型训练流程...")
    df_train = trader.fetch_history_klines(symbol, bar="1m", limit=1000)
    if df_train is None or df_train.empty or len(df_train) < 200:
        logging.error("用于训练的数据不足或获取失败。")
        return False
    df_featured = create_features(df_train)
    features = [col for col in df_featured.columns if "feature_" in col]
    X, y = df_featured[features], df_featured["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
    )
    logging.info("正在训练XGBoost模型...")
    model.fit(X_train, y_train)

    # 修复：将单行赋值拆分为两行
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"模型在测试集上的准确率: {accuracy:.4f}")
    model.save_model("xgb_model.json")
    logging.info("✅ XGBoost模型已成功训练并保存为 xgb_model.json")
    return True


# --- 策略逻辑类 (V3.10 - 自适应版本) ---
class HFStrategy:
    def __init__(self, symbol: str, initial_df: pd.DataFrame):
        self.symbol, self.params = symbol, STRATEGY_PARAMS
        self.data = initial_df.copy()
        self.last_candle_ts = self.data.index[-1] if not self.data.empty else None
        self.current_obi = 0.5
        self.model = xgb.XGBClassifier()
        self.model.load_model("xgb_model.json")
        self.recent_scores = deque(maxlen=self.params["dynamic_threshold_period"])
        self.compute_all_features()
        logging.info(f"[{self.symbol}] 高级自适应策略初始化完成。")

    def _normalize_z_score(self, series: pd.Series, period: int) -> pd.Series:
        mean = series.rolling(window=period, min_periods=max(10, period // 4)).mean()
        std = series.rolling(window=period, min_periods=max(10, period // 4)).std()
        std.replace(0, np.nan, inplace=True)
        z_scores = (series - mean) / std
        return pd.Series(norm.cdf(z_scores), index=series.index).fillna(0.5)

    def compute_all_features(self):
        required_length = self.params["z_score_period"] + 5
        if len(self.data) < required_length:
            logging.debug(
                f"数据量不足({len(self.data)})以计算Z-score因子(需要{required_length})"
            )
            return
        self.data["volume_ma"] = self.data.Volume.rolling(
            self.params["volume_avg_period"]
        ).mean()
        self.data["atr"] = ta.volatility.AverageTrueRange(
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.params["volatility_period"],
        ).average_true_range()
        self.data["raw_vol_factor"] = self.data.Volume / self.data.volume_ma
        self.data["raw_atr_factor"] = (
            self.data.atr
            / self.data.atr.rolling(self.params["volatility_period"]).mean()
        )
        z_period = self.params["z_score_period"]
        self.data["volume_factor"] = self._normalize_z_score(
            self.data["raw_vol_factor"], z_period
        )
        self.data["volatility_factor"] = self._normalize_z_score(
            self.data["raw_atr_factor"], z_period
        )
        self.data = create_features(self.data.copy())

    def update_order_book(self, asks, bids):
        ask_levels, bid_levels = min(len(asks), self.params["obi_levels"]), min(
            len(bids), self.params["obi_levels"]
        )
        ask_volume, bid_volume = sum(
            [float(level[1]) for level in asks[:ask_levels]]
        ), sum([float(level[1]) for level in bids[:bid_levels]])
        total_volume = ask_volume + bid_volume
        if total_volume == 0:
            self.current_obi = 0.5
            return
        self.current_obi = bid_volume / total_volume

    def on_new_candle(self, new_candle_df: pd.DataFrame):
        new_candle, candle_dt = new_candle_df.iloc[0], new_candle_df.iloc[0].name
        if self.last_candle_ts is not None and candle_dt <= self.last_candle_ts:
            return None
        logging.info(
            f"[{self.symbol}] 新1M K线获取: {candle_dt} | C={new_candle.Close}"
        )
        self.data = pd.concat([self.data, new_candle_df])
        if len(self.data) > HISTORY_LIMIT * 2:
            self.data = self.data.iloc[-(HISTORY_LIMIT * 2) :]
        self.compute_all_features()
        self.last_candle_ts = candle_dt
        return self.check_entry_signal()

    def check_entry_signal(self):
        required_cols = ["volume_factor", "volatility_factor"]
        if not all(col in self.data.columns for col in required_cols):
            return None
        last = self.data.iloc[-1]
        if pd.isna(last.volume_factor) or pd.isna(last.volatility_factor):
            return None
        volume_factor, volatility_factor = last.volume_factor, last.volatility_factor
        obi_buy_factor = min(self.current_obi / self.params["obi_threshold"], 1.0)
        obi_sell_factor = min(
            (1 - self.current_obi) / (1 - self.params["obi_threshold"]), 1.0
        )
        features = [col for col in self.data.columns if "feature_" in col]
        current_features = (
            self.data[features].iloc[-1:].drop(columns=["target"], errors="ignore")
        )
        if current_features.isnull().values.any():
            return None
        xgb_probs = self.model.predict_proba(current_features)[0]
        xgb_long_factor, xgb_short_factor = xgb_probs[1], xgb_probs[0]
        weights = self.params["weights"]
        long_score = (
            volume_factor * weights["volume"]
            + volatility_factor * weights["volatility"]
            + obi_buy_factor * weights["obi"]
            + xgb_long_factor * weights["xgb"]
        )
        short_score = (
            volume_factor * weights["volume"]
            + volatility_factor * weights["volatility"]
            + obi_sell_factor * weights["obi"]
            + xgb_short_factor * weights["xgb"]
        )
        self.recent_scores.append(long_score)
        if self.params.get("allow_shorting", False):
            self.recent_scores.append(short_score)
        if len(self.recent_scores) < self.params["min_signal_history"]:
            dynamic_threshold = self.params["backup_fixed_threshold"]
        else:
            dynamic_threshold = np.percentile(
                self.recent_scores, self.params["threshold_percentile"]
            )
        logging.info(
            f"[{self.symbol}] 信号评分: [多]分={long_score:.4f} | [空]分={short_score:.4f} | 动态阈值={dynamic_threshold:.4f}"
        )
        if long_score > dynamic_threshold:
            logging.info(
                f"[{self.symbol}] ✅ 做多信号确认！得分 {long_score:.4f} > 动态阈值 {dynamic_threshold:.4f}"
            )
            entry_price = last.Close
            sl_price = self.data.Low.iloc[-1 - self.params["sl_lookback"] : -1].min()
            if sl_price >= entry_price:
                logging.warning(
                    f"信号过滤 (多头): 止损价({sl_price})高于入场价({entry_price})"
                )
                return None
            tp_price = (
                entry_price + (entry_price - sl_price) * self.params["tp_rr_ratio"]
            )
            return {"side": "buy", "sl_price": sl_price, "tp_price": tp_price}
        elif short_score > dynamic_threshold and self.params.get(
            "allow_shorting", False
        ):
            logging.info(
                f"[{self.symbol}] ✅ 做空信号确认！得分 {short_score:.4f} > 动态阈值 {dynamic_threshold:.4f}"
            )
            entry_price = last.Close
            sl_price = self.data.High.iloc[-1 - self.params["sl_lookback"] : -1].max()
            if sl_price <= entry_price:
                logging.warning(
                    f"信号过滤 (空头): 止损价({sl_price})低于入场价({entry_price})"
                )
                return None
            tp_price = (
                entry_price - (sl_price - entry_price) * self.params["tp_rr_ratio"]
            )
            return {"side": "sell", "sl_price": sl_price, "tp_price": tp_price}
        return None


# --- WebSocket 与主程序 ---
class OKXWebSocketManager:
    def __init__(self, api_key, api_secret, passphrase):
        self.api_key, self.api_secret, self.passphrase = api_key, api_secret, passphrase
        self.should_reconnect = True
        (
            self.public_ws,
            self.private_ws,
            self.heartbeat_thread,
            self.on_message_callback,
        ) = (None, None, None, None)

    def start(self, on_message_callback):
        self.on_message_callback = on_message_callback
        threading.Thread(target=self.run_public, daemon=True).start()
        threading.Thread(target=self.run_private, daemon=True).start()
        self._start_heartbeat()

    def _start_heartbeat(self):
        self.heartbeat_thread = threading.Thread(
            target=self._send_heartbeat, daemon=True
        )
        self.heartbeat_thread.start()

    def _send_heartbeat(self):
        while self.should_reconnect:
            try:
                if (
                    self.public_ws
                    and self.public_ws.sock
                    and self.public_ws.sock.connected
                ):
                    self.public_ws.send("ping")
                if (
                    self.private_ws
                    and self.private_ws.sock
                    and self.private_ws.sock.connected
                ):
                    self.private_ws.send("ping")
            except Exception as e:
                logging.warning(f"发送心跳时出错: {e}")
            time.sleep(25)

    def run_public(self):
        while self.should_reconnect:
            logging.info("正在连接到 OKX 公共 WebSocket...")
            self.public_ws = websocket.WebSocketApp(
                WS_PUBLIC_URL,
                on_open=self.on_public_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            self.public_ws.run_forever()
            time.sleep(5)

    def run_private(self):
        while self.should_reconnect:
            logging.info("正在连接到 OKX 私有 WebSocket...")
            self.private_ws = websocket.WebSocketApp(
                WS_PRIVATE_URL,
                on_open=self.on_private_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            self.private_ws.run_forever()
            time.sleep(5)

    def on_public_open(self, ws):
        logging.info("公共 WebSocket 连接成功！正在订阅公共频道...")
        ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {"channel": "tickers", "instId": SYMBOL},
                        {"channel": "books", "instId": SYMBOL},
                    ],
                }
            )
        )

    def on_private_open(self, ws):
        logging.info("私有 WebSocket 连接成功！正在进行认证...")
        ts = str(int(time.time()))
        message = ts + "GET" + "/users/self/verify"
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode()
        ws.send(
            json.dumps(
                {
                    "op": "login",
                    "args": [
                        {
                            "apiKey": self.api_key,
                            "passphrase": self.passphrase,
                            "timestamp": ts,
                            "sign": sign,
                        }
                    ],
                }
            )
        )

    def on_error(self, ws, error):
        logging.error(f"WebSocket 错误: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.warning(f"WebSocket 连接已关闭。")

    def on_message(self, ws, message):
        if self.on_message_callback:
            self.on_message_callback(message, ws)

    def stop(self):
        self.should_reconnect = False
        if self.public_ws:
            self.public_ws.close()
        if self.private_ws:
            self.private_ws.close()


class TradingBot:
    def __init__(self, trader: OKXTrader, strategy: HFStrategy):
        self.trader, self.strategy = trader, strategy
        self.bot_state = {"position": None, "current_price": None}
        self.equity = None

    def on_ws_message(self, message, ws_instance):
        if message == "pong":
            return
        try:
            msg_json = json.loads(message)
            if "event" in msg_json:
                if msg_json["event"] == "login" and msg_json["code"] == "0":
                    logging.info("WebSocket 认证成功！正在订阅私有频道...")
                    ws_instance.send(
                        json.dumps(
                            {
                                "op": "subscribe",
                                "args": [
                                    {
                                        "channel": "positions",
                                        "instType": "SWAP",
                                        "instId": self.strategy.symbol,
                                    }
                                ],
                            }
                        )
                    )
                elif msg_json["event"] == "error":
                    logging.error(f"WebSocket API 错误: {msg_json.get('msg')}")
                elif msg_json["event"] == "subscribe":
                    logging.info(f"成功订阅频道: {msg_json.get('arg')}")
                return
            if "arg" in msg_json and "data" in msg_json and msg_json["data"]:
                channel, data = msg_json["arg"]["channel"], msg_json["data"][0]
                if channel == "tickers":
                    self.bot_state["current_price"] = float(data["last"])
                elif channel == "positions":
                    pos_size = float(data.get("pos", "0"))
                    if pos_size != 0:
                        self.bot_state["position"] = "LONG" if pos_size > 0 else "SHORT"
                    else:
                        if self.bot_state["position"] is not None:
                            logging.info(f"[{self.strategy.symbol}] 仓位已平仓。")
                            clear_trade_state()
                        self.bot_state["position"] = None
                elif channel == "books":
                    self.strategy.update_order_book(data["asks"], data["bids"])
        except Exception as e:
            logging.exception(f"处理消息时发生致命错误: {message}")

    def calculate_position_size(self, entry_price, sl_price):
        instrument = self.trader.fetch_instrument_details(self.strategy.symbol)
        if not instrument or not entry_price or not sl_price:
            return "0"
        if self.equity is None:
            logging.error("账户权益未知，无法计算头寸大小。")
            return "0"
        risk_per_unit, risk_amount = (
            abs(entry_price - sl_price),
            self.equity * STRATEGY_PARAMS["risk_per_trade_pct"],
        )
        if risk_per_unit == 0:
            return "0"
        size_in_coin, lot_sz_str = risk_amount / risk_per_unit, instrument.get(
            "lotSz", "1"
        )
        lot_sz = float(lot_sz_str)
        if size_in_coin < lot_sz:
            logging.warning(
                f"计算出的理论仓位 {size_in_coin:.6f} 小于最小下单单位 {lot_sz}。无法下单。"
            )
            return "0"
        precision = len(lot_sz_str.split(".")[1]) if "." in lot_sz_str else 0
        final_size = math.floor(size_in_coin / lot_sz) * lot_sz
        if final_size == 0:
            logging.warning("计算出的仓位大小向下取整后为0，无法下单。")
            return "0"
        return f"{final_size:.{precision}f}"

    def execute_trade_signal(self, signal):
        logging.info(f"[{self.strategy.symbol}] 收到交易信号: {signal}")
        current_price = self.bot_state["current_price"]
        if current_price is None:
            logging.error("无法获取当前价格，无法下单。")
            return
        size = self.calculate_position_size(current_price, signal["sl_price"])
        if float(size) == 0:
            return
        res = self.trader.place_market_order(self.strategy.symbol, signal["side"], size)
        logging.info(f"[{self.strategy.symbol}] 开仓下单响应: {res}")
        if res and res.get("code") == "0" and res.get("data")[0].get("sCode") == "0":
            logging.info("下单成功，等待成交信息以保存状态...")
            time.sleep(3)
            pos_data = self.trader.fetch_current_position(self.strategy.symbol)
            if not pos_data:
                logging.error("无法获取成交后的仓位信息！")
                return
            entry_price, pos_size = float(pos_data.get("avgPx")), float(
                pos_data.get("pos")
            )
            trade_state = {
                "side": "LONG" if signal["side"] == "buy" else "SHORT",
                "entry_price": entry_price,
                "size": abs(pos_size),
                "sl_price": signal["sl_price"],
                "tp_price": signal["tp_price"],
            }
            save_trade_state(trade_state)
            logging.info(
                f"[{self.strategy.symbol}] 仓位已建立并保存状态: {trade_state}"
            )

    def manage_position(self):
        if (
            self.bot_state["position"] is None
            or self.bot_state["current_price"] is None
        ):
            return
        trade_state = load_trade_state()
        if not trade_state:
            return
        price = self.bot_state["current_price"]
        side, sl_price, tp_price, size = (
            trade_state["side"],
            trade_state["sl_price"],
            trade_state["tp_price"],
            trade_state["size"],
        )
        close_side, reason = None, ""
        if side == "LONG":
            if price <= sl_price:
                close_side, reason = "sell", "止损"
            elif price >= tp_price:
                close_side, reason = "sell", "止盈"
        elif side == "SHORT":
            if price >= sl_price:
                close_side, reason = "buy", "止损"
            elif price <= tp_price:
                close_side, reason = "buy", "止盈"
        if close_side:
            logging.info(f"[{self.strategy.symbol}] {reason}触发 at {price:.4f}")
            res = self.trader.place_market_order(
                self.strategy.symbol, close_side, str(size)
            )
            logging.info(f"[{self.strategy.symbol}] 平仓响应: {res}")
            clear_trade_state()
            self.bot_state["position"] = None

    def run(self):
        self.ws_manager = OKXWebSocketManager(
            self.trader.api_key, self.trader.api_secret, self.trader.passphrase
        )
        self.ws_manager.start(on_message_callback=self.on_ws_message)
        clear_trade_state()
        last_check_minute, last_equity_check_time = -1, 0
        try:
            while True:
                self.manage_position()
                now, current_time = datetime.utcnow(), time.time()
                if current_time - last_equity_check_time > 60:
                    current_equity = self.trader.fetch_account_balance("USDT")
                    if current_equity is not None:
                        self.equity = current_equity
                        logging.info(f"账户权益已更新: {self.equity:.2f} USDT")
                    last_equity_check_time = current_time
                if now.second == 5 and now.minute != last_check_minute:
                    last_check_minute = now.minute
                    if self.bot_state["position"] is None:
                        latest_candle_df = self.trader.fetch_history_klines(
                            self.strategy.symbol, bar=KLINE_INTERVAL, limit=1
                        )
                        if latest_candle_df is not None and not latest_candle_df.empty:
                            signal = self.strategy.on_new_candle(latest_candle_df)
                            if signal:
                                self.execute_trade_signal(signal)
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("程序被手动中断...")
        finally:
            self.ws_manager.stop()
            logging.info("程序已安全退出。")


def main():
    hf_logger = setup_logging()
    hf_logger.info("启动 OKX WebSocket 高频交易程序 (V3.11 - Training_Fix)...")
    OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE = (
        os.getenv("OKX_API_KEY"),
        os.getenv("OKX_API_SECRET"),
        os.getenv("OKX_API_PASSPHRASE"),
    )
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE]):
        hf_logger.error(
            "请设置环境变量: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE"
        )
        return
    trader = OKXTrader(
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, simulated=SIMULATED
    )
    if not os.path.exists("xgb_model.json"):
        hf_logger.warning(
            "未找到 XGBoost 模型文件 (xgb_model.json)。正在启动一次性训练流程..."
        )
        if not train_and_save_model(SYMBOL, trader):
            hf_logger.error("模型训练失败，程序退出。")
            return
    hf_logger.info("正在查询初始账户余额 (USDT)...")
    initial_equity = trader.fetch_account_balance("USDT")
    if SIMULATED and (initial_equity is None or initial_equity == 0):
        hf_logger.warning(
            "无法获取模拟盘账户余额或余额为0，将使用默认启动资金 10000 USDT。"
        )
        initial_equity = 10000.0
    elif initial_equity is None:
        hf_logger.error("无法获取账户余额，程序退出。请检查API密钥权限或网络连接。")
        return
    else:
        hf_logger.info(f"查询成功，当前账户权益为: {initial_equity:.2f} USDT")
    hf_logger.info(f"正在为 {SYMBOL} 设置全仓杠杆为 {DESIRED_LEVERAGE}x...")
    trader.set_leverage(SYMBOL, DESIRED_LEVERAGE)
    hf_logger.info(f"正在为 {SYMBOL} 获取初始历史数据 (需要 {HISTORY_LIMIT} 条)...")
    initial_df = trader.fetch_history_klines(
        SYMBOL, bar=KLINE_INTERVAL, limit=HISTORY_LIMIT
    )
    if initial_df is None or len(initial_df) < STRATEGY_PARAMS["z_score_period"] + 5:
        hf_logger.error(
            f"为 {SYMBOL} 获取的初始数据不足({len(initial_df)})，程序退出。"
        )
        return
    strategy = HFStrategy(SYMBOL, initial_df)
    bot = TradingBot(trader, strategy)
    bot.equity = initial_equity
    bot.run()


if __name__ == "__main__":
    main()
xa