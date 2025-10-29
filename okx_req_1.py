# okx_live_trading_v3.5_kelly_persistence.py
"""
OKX REST API 轮询版 (V3.5 凯利持久化版)
- 核心变更:
    - [!!! 关键功能 !!!] 实现凯利公式交易历史持久化：
        - 新增 'save_kelly_history' 和 'load_kelly_history' 函数，用于将各交易对的盈亏历史写入和读出 JSON 文件。
        - 策略初始化时会自动加载历史记录，使凯利公式在程序重启后能立即基于历史数据进行风险计算，解决了“重启后退化为默认值”的问题。
        - 每笔交易结果产生后，会自动更新并保存历史文件，确保数据始终为最新。
    - [!!! 关键修复 !!!] 修复模型输入形状错误：重写了 'get_ml_confidence_score' 函数，
                       现在会正确地截取最近60根K线数据，并将其重塑为 (1, 60, num_features)
                       的3D形状，以完全匹配LSTM模型的输入要求。
    - [!!! 关键修复 !!!] 补全缺失特征：在 'add_ml_features' 函数中增加了 'EMA_8' 的计算。
                       该特征在训练时被隐式加入，是导致模型警告和评分为0的直接原因。
    - [!!! 关键修复 !!!] 修复Pandas Index错误：在加载 feature_columns.joblib 文件后，
                       强制将其转换为Python列表，从根本上解决了 "The truth value of a Index is ambiguous" 的错误。
    - [关键对齐] 特征工程重构：完全采用与模型训练脚本一致的特征计算逻辑，确保了模型在实盘与训练环境中的数据一致性。
    - [模型升级] 集成了指定的 Keras 神经网络模型 (eth_trend_model_v1.keras)。
    - [关键健壮性] (继承) 保留了下单后的成交量验证机制和状态审计功能。
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
from datetime import datetime
from logging.handlers import RotatingFileHandler
from collections import deque
import pandas as pd
import numpy as np
import requests
import ta  # 技术分析指标库
import warnings
from requests.models import PreparedRequest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 忽略 pandas 未来版本可能产生的警告
warnings.simplefilter(action="ignore", category=FutureWarning)

# 尝试导入机器学习相关的库，如果未安装则设置标志位
try:
    import joblib
    import tensorflow as tf

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False
    print("警告: tensorflow 或 joblib 未安装，机器学习相关功能将不可用。")


# --- 日志系统设置 ---
def setup_logging():
    """配置日志系统，分为主日志和交易流日志。"""
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
    trade_flow_logger.setLevel(logging.INFO)
    trade_flow_logger.propagate = False
    trade_flow_path = os.path.join(log_dir, "trades_flow.log")
    trade_flow_handler = RotatingFileHandler(
        trade_flow_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
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
DESIRED_LEVERAGE = "10"
HISTORY_LIMIT = 500
POLL_INTERVAL_SECONDS = 15
SIMULATED = True
SIMULATED_EQUITY_START = 500000.0
TSL_ENABLED = True
TSL_ACTIVATION_PROFIT_PCT = 1.0
TSL_ACTIVATION_ATR_MULT = 1.5
TSL_TRAILING_ATR_MULT = 2.0
MAX_DAILY_DRAWDOWN_PCT = 0.03
MAX_CONSECUTIVE_LOSSES = 5
TRADING_PAUSE_HOURS = 4
AUDIT_INTERVAL_MINUTES = 15

STRATEGY_PARAMS = {
    "sl_atr_multiplier": 2.5,
    "tp_rr_ratio": 1.5,
    "kelly_trade_history": 20,
    "default_risk_pct": 0.1,
    "max_risk_pct": 0.04,
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_atr_period": 14,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_risk_multiplier": 0.5,
    "score_entry_threshold": 0.4,
    "score_weights_tf": {
        "breakout": 0.20,
        "momentum": 0.12,
        "mtf": 0.28,
        "ml": 0.20,
        "advanced_ml": 0.20,
    },
}
ASSET_SPECIFIC_OVERRIDES = {
    "ETH-USDT-SWAP": {
        "score_entry_threshold": 0.35,
    }
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


# --- [新增代码] 凯利历史持久化函数 ---
def save_kelly_history(symbol, history_deque):
    """保存指定交易对的凯利公式历史交易记录。"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    history_file = os.path.join(log_dir, f"{symbol}_kelly_history.json")
    try:
        with open(history_file, "w") as f:
            # 将 deque 转换为 list 进行 JSON 序列化
            json.dump(list(history_deque), f, indent=4)
        logging.debug(f"[{symbol}] 已保存凯利公式交易历史。")
    except Exception as e:
        logging.error(f"[{symbol}] 保存凯利历史时出错: {e}")


def load_kelly_history(symbol, maxlen):
    """加载指定交易对的凯利公式历史交易记录。"""
    log_dir = "logs"
    history_file = os.path.join(log_dir, f"{symbol}_kelly_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history_list = json.load(f)
                # 使用加载的历史记录初始化 deque，deque 会自动处理 maxlen
                logging.info(
                    f"[{symbol}] 成功加载 {len(history_list)} 条凯利历史记录。"
                )
                return deque(history_list, maxlen=maxlen)
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"[{symbol}] 加载凯利历史文件失败，将使用空历史记录: {e}")
    # 如果文件不存在或加载失败，返回一个空的 deque
    return deque(maxlen=maxlen)


# --- [新增代码结束] ---


def run_advanced_model_inference(df):
    if "ai_filter_signal" not in df.columns:
        df["ai_filter_signal"] = 0.0
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


def _calculate_tr(df):
    """辅助函数：计算真实波幅(True Range)"""
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1, skipna=False)
    return tr


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """为DataFrame计算并添加与模型训练脚本完全一致的特征。"""
    logging.debug("--- 开始为模型计算特征指标 (逻辑源自训练脚本) ---")
    df_out = df.copy()

    # 1. RSI (Relative Strength Index)
    close_delta = df_out["Close"].diff()
    gain = close_delta.where(close_delta > 0, 0)
    loss = -close_delta.where(close_delta < 0, 0)
    avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
    avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df_out["RSI_14"] = 100 - (100 / (1 + rs))

    # 2. MACD (Moving Average Convergence Divergence)
    ema_fast = df_out["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df_out["Close"].ewm(span=26, adjust=False).mean()
    df_out["MACD_12_26_9"] = ema_fast - ema_slow
    # 【关键修复】将列名中的 'S' 和 'H' 修改为小写，以匹配模型要求
    df_out["MACDs_12_26_9"] = df_out["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
    df_out["MACDh_12_26_9"] = df_out["MACD_12_26_9"] - df_out["MACDs_12_26_9"]

    # 3. Bollinger Bands
    sma_20 = df_out["Close"].rolling(window=20).mean()
    std_20 = df_out["Close"].rolling(window=20).std()
    df_out["BBM_20_2.0"] = sma_20
    df_out["BBU_20_2.0"] = sma_20 + (std_20 * 2)
    df_out["BBL_20_2.0"] = sma_20 - (std_20 * 2)
    # 【关键补充】计算模型需要的 BBB 和 BBP 指标
    # BBB (Bollinger Band Width)
    df_out["BBB_20_2.0"] = (df_out["BBU_20_2.0"] - df_out["BBL_20_2.0"]) / np.where(
        df_out["BBM_20_2.0"] == 0, 1, df_out["BBM_20_2.0"]
    )
    # BBP (Bollinger Band Percentage)
    band_width = df_out["BBU_20_2.0"] - df_out["BBL_20_2.0"]
    df_out["BBP_20_2.0"] = (df_out["Close"] - df_out["BBL_20_2.0"]) / np.where(
        band_width == 0, 1, band_width
    )

    # 4. ADX (Average Directional Movement Index)
    tr = _calculate_tr(df_out)
    atr = tr.ewm(com=14 - 1, min_periods=14).mean()
    up_move = df_out["High"].diff()
    down_move = -df_out["Low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    plus_di = 100 * (plus_dm.ewm(com=14 - 1, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(com=14 - 1, min_periods=14).mean() / atr)
    dx = 100 * (
        np.abs(plus_di - minus_di)
        / np.where(plus_di + minus_di == 0, 1, plus_di + minus_di)
    )
    df_out["ADX_14"] = dx.ewm(com=14 - 1, min_periods=14).mean()
    df_out["DMP_14"] = plus_di
    df_out["DMN_14"] = minus_di

    # 5. ATR (Average True Range)
    df_out["ATRr_14"] = atr

    # 6. OBV (On-Balance Volume)
    df_out["OBV"] = (
        (np.sign(df_out["Close"].diff()) * df_out["Volume"]).fillna(0).cumsum()
    )

    # 7. 自定义指标
    df_out["volatility"] = (
        np.log(df_out["Close"] / df_out["Close"].shift(1)).rolling(window=20).std()
    )
    df_out["volume_change_rate"] = df_out["Volume"].pct_change()

    # 8. EMA_8 (Exponential Moving Average)
    df_out["EMA_8"] = df_out["Close"].ewm(span=8, adjust=False).mean()

    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_out


# --- OKX 交易接口类 ---
class OKXTrader:
    """封装了与 OKX API 交互的所有方法。"""

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
        if urllib.parse.urlparse(prepped.url).query:
            path_for_signing += "?" + urllib.parse.urlparse(prepped.url).query
        path_for_signing = path_for_signing.replace(self.base, "")
        sign = self._sign(ts, method.upper(), path_for_signing, body_str)
        headers = self.common_headers.copy()
        headers.update({"OK-ACCESS-SIGN": sign, "OK-ACCESS-TIMESTAMP": ts})
        url = self.base + path
        wait_time = 1.0
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
        res = self._request("POST", path, body=body)
        return res and res.get("code") == "0"

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
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
            if data and "details" in data[0]:
                for detail in data[0]["details"]:
                    if detail.get("ccy") == ccy:
                        return float(detail.get("eq", 0))
        return None

    def fetch_current_position(self, instId):
        path, params = "/api/v5/account/positions", {"instId": instId}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            for pos_data in res.get("data", []):
                if float(pos_data.get("pos", "0")) != 0:
                    return pos_data
            return None
        logging.error(
            f"查詢倉位失敗 for {instId}: {res.get('msg') if res else 'Unknown error'}"
        )
        return False

    def fetch_instrument_details(self, instId, instType="SWAP"):
        if instId in self.instrument_info:
            return self.instrument_info[instId]
        path, params = "/api/v5/public/instruments", {
            "instType": instType,
            "instId": instId,
        }
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0" and res.get("data"):
            self.instrument_info[instId] = res["data"][0]
            return res["data"][0]
        return None

    def place_market_order(self, instId, side, sz):
        path, body = "/api/v5/trade/order", {
            "instId": instId,
            "tdMode": "cross",
            "side": side,
            "ordType": "market",
            "sz": str(sz),
        }
        res = self._request("POST", path, body=body)
        if res:
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
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
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
        return None

    def fetch_ticker_price(self, instId):
        path, params = "/api/v5/market/ticker", {"instId": instId}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
            if data:
                return float(data[0].get("last"))
        return None


# --- 策略核心类 ---
class UltimateStrategy:
    """包含了所有交易逻辑、信号生成和风险管理的策略类。"""

    def __init__(
        self, df, symbol, trader=None, pos_logger=None, instrument_details=None
    ):
        self.symbol, self.trader = symbol, trader
        self.data, self.position = df.copy(), None
        self.pos_logger, self.instrument_details = pos_logger, instrument_details
        logging.info(f"[{self.symbol}] 策略初始化...")

        # --- [代码修改] 从文件加载历史交易记录，以实现持久化 ---
        self.recent_trade_returns = load_kelly_history(
            self.symbol, STRATEGY_PARAMS["kelly_trade_history"]
        )
        # --- [修改结束] ---

        self.keras_model, self.scaler, self.feature_columns = None, None, None
        self.score_entry_threshold = ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).get(
            "score_entry_threshold", STRATEGY_PARAMS["score_entry_threshold"]
        )
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
            f"[{self.symbol}] 錄得一次虧損，当前連續虧損次数: {self.consecutive_losses}"
        )
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.trading_paused_until = datetime.utcnow() + pd.Timedelta(
                hours=TRADING_PAUSE_HOURS
            )
            trade_flow_logger.critical(
                f"[{self.symbol}] 已達到最大連續虧損次数 ({MAX_CONSECUTIVE_LOSSES})！該交易對將暫停交易 {TRADING_PAUSE_HOURS} 小時，直到 {self.trading_paused_until} UTC"
            )

    def register_win(self):
        if self.consecutive_losses > 0:
            trade_flow_logger.info(
                f"[{self.symbol}] 錄得一次盈利，連續虧損計數已重置。"
            )
            self.consecutive_losses = 0

    def _load_models(self):
        if not ML_LIBS_INSTALLED:
            return
        model_path, scaler_path, features_path = (
            "models/eth_trend_model_v1.keras",
            "models/eth_trend_scaler_v1.joblib",
            "models/feature_columns.joblib",
        )
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Keras 模型文件未找到: {model_path}")
            self.keras_model = tf.keras.models.load_model(model_path)
            logging.info(f"[{self.symbol}] 已成功加载 Keras 模型: {model_path}")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler 文件未找到: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            logging.info(f"[{self.symbol}] 已成功加载 Scaler: {scaler_path}")
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"特征列文件未找到: {features_path}")
            feature_cols_loaded = joblib.load(features_path)
            if hasattr(feature_cols_loaded, "tolist"):
                self.feature_columns = feature_cols_loaded.tolist()
            else:
                self.feature_columns = feature_cols_loaded
            logging.info(f"[{self.symbol}] 已成功加载特征列: {features_path}")
        except Exception as e:
            logging.error(f"加载机器学习模型或相关文件时出错: {e}")
            self.keras_model, self.scaler, self.feature_columns = None, None, None

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < 2:
            return STRATEGY_PARAMS["default_risk_pct"]
        wins, losses = [r for r in self.recent_trade_returns if r > 0], [
            r for r in self.recent_trade_returns if r < 0
        ]
        if not wins or not losses:
            return STRATEGY_PARAMS["default_risk_pct"]
        win_rate, avg_win, avg_loss = (
            len(wins) / len(self.recent_trade_returns),
            sum(wins) / len(wins),
            abs(sum(losses) / len(losses)),
        )
        reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        if reward_ratio == 0:
            return STRATEGY_PARAMS["default_risk_pct"]
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return max(0.005, min(STRATEGY_PARAMS["max_risk_pct"], 0.5 * kelly))

    def get_ml_confidence_score(self):
        """使用加载的 Keras 模型计算并返回综合置信度评分。"""
        if not all([self.keras_model, self.scaler, self.feature_columns]):
            logging.warning("Keras 模型或其组件未加载，ML 评分为 0。")
            return 0.0

        LOOK_BACK = 60
        if len(self.data) < LOOK_BACK:
            logging.warning(
                f"数据不足 (需要 {LOOK_BACK} 条, 实际 {len(self.data)} 条)，无法创建预测序列。ML 评分为 0。"
            )
            return 0.0

        missing_cols = [
            col for col in self.feature_columns if col not in self.data.columns
        ]
        if missing_cols:
            logging.warning(f"缺少模型所需的特征列: {missing_cols}，ML 评分为 0。")
            return 0.0

        try:
            latest_sequence_df = (
                self.data[self.feature_columns].iloc[-LOOK_BACK:].copy()
            )
            latest_sequence_df.fillna(method="ffill", inplace=True)
            latest_sequence_df.fillna(0, inplace=True)
            scaled_sequence = self.scaler.transform(latest_sequence_df)
            input_for_model = np.expand_dims(scaled_sequence, axis=0)
            prediction = self.keras_model.predict(input_for_model, verbose=0)
            up_probability = prediction[0][0]
            score = (up_probability - 0.5) * 2.0
            return float(score)
        except Exception as e:
            logging.error(f"使用 Keras 模型进行预测时出错: {e}", exc_info=True)
            return 0.0

    def update_with_candle(self, row: pd.Series):
        self.data.loc[row.name] = row
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)
        if len(self.data) > 5000:
            self.data = self.data.iloc[-5000:]

    def compute_all_features(self):
        if "ai_filter_signal" not in self.data.columns:
            rsi_filter = ta.momentum.RSIIndicator(self.data.Close, 14).rsi()
            self.data["ai_filter_signal"] = (
                (((rsi_filter.rolling(3).mean() - rsi_filter.rolling(10).mean()) / 50))
                .clip(-1, 1)
                .fillna(0)
            )
        self.data = run_advanced_model_inference(self.data)
        self.data = add_ml_features(self.data)
        if "ATRr_14" in self.data.columns:
            self.data["tf_atr"] = self.data["ATRr_14"]
        else:
            self.data["tf_atr"] = ta.volatility.AverageTrueRange(
                self.data.High,
                self.data.Low,
                self.data.Close,
                STRATEGY_PARAMS["tf_atr_period"],
            ).average_true_range()
        if "BBU_20_2.0" in self.data.columns:
            self.data["mr_bb_upper"], self.data["mr_bb_lower"] = (
                self.data["BBU_20_2.0"],
                self.data["BBL_20_2.0"],
            )
        else:
            bb = ta.volatility.BollingerBands(
                self.data.Close,
                STRATEGY_PARAMS["mr_bb_period"],
                STRATEGY_PARAMS["mr_bb_std"],
            )
            self.data["mr_bb_upper"], self.data["mr_bb_lower"] = (
                bb.bollinger_hband(),
                bb.bollinger_lband(),
            )
        if "ADX_14" in self.data.columns:
            self.data["market_regime"] = np.where(self.data["ADX_14"] > 25, 1, -1)
        else:
            self.data["market_regime"] = 0
        self.data["tf_ema_fast"] = ta.trend.EMAIndicator(
            self.data.Close, STRATEGY_PARAMS["tf_ema_fast_period"]
        ).ema_indicator()
        self.data["tf_ema_slow"] = ta.trend.EMAIndicator(
            self.data.Close, STRATEGY_PARAMS["tf_ema_slow_period"]
        ).ema_indicator()
        self.data["tf_donchian_h"] = (
            self.data.High.rolling(STRATEGY_PARAMS["tf_donchian_period"]).max().shift(1)
        )
        self.data["tf_donchian_l"] = (
            self.data.Low.rolling(STRATEGY_PARAMS["tf_donchian_period"]).min().shift(1)
        )
        stoch_rsi = ta.momentum.StochRSIIndicator(
            self.data.Close, window=14, smooth1=3, smooth2=3
        )
        self.data["mr_stoch_rsi_k"], self.data["mr_stoch_rsi_d"] = (
            stoch_rsi.stochrsi_k(),
            stoch_rsi.stochrsi_d(),
        )

    def _calculate_entry_score(self):
        w, last = STRATEGY_PARAMS["score_weights_tf"], self.data.iloc[-1]
        try:
            b_s = (
                1
                if last.High > last.tf_donchian_h
                else -1 if last.Low < last.tf_donchian_l else 0
            )
        except (AttributeError, KeyError):
            b_s = 0
        mo_s, ml_score = (
            1 if last.tf_ema_fast > last.tf_ema_slow else -1
        ), self.get_ml_confidence_score()
        adv_score, mtf_score = last.get("advanced_ml_signal", 0), last.get(
            "mtf_signal", 0
        )
        return (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + mtf_score * w.get("mtf", 0)
            + ml_score * w.get("ml", 0)
            + adv_score * w.get("advanced_ml", 0)
        )

    def _define_mr_entry_signal(self):
        if (
            len(self.data) < 5
            or "mr_stoch_rsi_k" not in self.data.columns
            or self.data["mr_stoch_rsi_k"].isnull().all()
        ):
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
        self.compute_all_features()
        last = self.data.iloc[-1]
        if pd.isna(last.get("market_regime")):
            return None, 0.0
        market_status = "趨勢市" if last.market_regime == 1 else "震盪市"
        logging.info(
            f"[{self.symbol}] 市場狀態判斷: {market_status} (ADX_14: {last.get('ADX_14', 0):.2f})"
        )
        action_to_take, score = None, 0.0
        if last.market_regime == 1:
            score = self._calculate_entry_score()
            if abs(score) > self.score_entry_threshold:
                is_long = score > 0
                if (is_long and self.position == "LONG") or (
                    not is_long and self.position == "SHORT"
                ):
                    return None, score
                risk_pct = self._calculate_dynamic_risk()
                position_size = self._determine_position_size(last.Close, risk_pct)
                if float(position_size) > 0:
                    action_to_take = {
                        "action": "BUY" if is_long else "SELL",
                        "size": position_size,
                    }
        else:
            sig = self._define_mr_entry_signal()
            if sig != 0:
                is_long = sig == 1
                if (is_long and self.position == "LONG") or (
                    not is_long and self.position == "SHORT"
                ):
                    return None, float(sig)
                score = float(sig)
                risk_pct = (
                    self._calculate_dynamic_risk()
                    * STRATEGY_PARAMS["mr_risk_multiplier"]
                )
                position_size = self._determine_position_size(last.Close, risk_pct)
                if float(position_size) > 0:
                    action_to_take = {
                        "action": "BUY" if is_long else "SELL",
                        "size": position_size,
                    }
        return action_to_take, score

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

    def _determine_position_size(self, price, risk_pct):
        if price <= 0:
            return "0"
        notional = self.equity * risk_pct
        return self._adjust_size_to_lot_size(notional / price) if price > 0 else "0"

    def register_trade_result(self, pnl_pct):
        self.recent_trade_returns.append(pnl_pct)
        # --- [代码修改] 每次更新后，将新的交易历史保存到文件 ---
        save_kelly_history(self.symbol, self.recent_trade_returns)
        # --- [修改结束] ---
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
def manage_position_exit_orders(
    trader: OKXTrader,
    symbol: str,
    pos_data: dict,
    strategy: UltimateStrategy,
    exit_mode: str,
):
    entry_px, pos_size, pos_side = (
        float(pos_data.get("avgPx")),
        abs(float(pos_data.get("pos"))),
        "LONG" if float(pos_data.get("pos")) > 0 else "SHORT",
    )
    strategy.compute_all_features()
    atr = strategy.data["tf_atr"].iloc[-1]
    if pd.isna(atr):
        trade_flow_logger.error(f"[{symbol}] ATR計算失敗，無法設置出場訂單。")
        return False
    close_side, sl_mult = "sell" if pos_side == "LONG" else "buy", STRATEGY_PARAMS[
        "sl_atr_multiplier"
    ]
    sl_price = (
        entry_px - (atr * sl_mult) if pos_side == "LONG" else entry_px + (atr * sl_mult)
    )
    trade_flow_logger.info(
        f"[{symbol}] 為 {pos_side} 倉位 (均價: {entry_px:.4f}) 設置止損單... 訂單類型: {close_side.upper()}, 觸發價: {sl_price:.4f}"
    )
    sl_res = trader.place_trigger_order(
        symbol, close_side, pos_size, sl_price, "Stop-Loss"
    )
    sl_id = None
    if sl_res and sl_res.get("code") == "0" and sl_res["data"][0]["sCode"] == "0":
        sl_id = sl_res["data"][0]["algoId"]
        trade_flow_logger.info(f"[{symbol}] 止損單設置成功, Algo ID: {sl_id}")
    else:
        trade_flow_logger.error(f"[{symbol}] 設置止損單失敗！響應: {sl_res}")
        return False
    if exit_mode == "fixed_target":
        tp_rr = STRATEGY_PARAMS["tp_rr_ratio"]
        tp_price = (
            entry_px + (atr * sl_mult * tp_rr)
            if pos_side == "LONG"
            else entry_px - (atr * sl_mult * tp_rr)
        )
        trade_flow_logger.info(
            f"[{symbol}] (震盪市) 設置固定止盈單... 訂單類型: {close_side.upper()}, 觸發價: {tp_price:.4f}"
        )
        trader.place_trigger_order(
            symbol, close_side, pos_size, tp_price, "Take-Profit"
        )
    if exit_mode == "trailing_stop":
        trade_state = {
            "entry_price": entry_px,
            "initial_stop_price": sl_price,
            "current_stop_price": sl_price,
            "current_stop_id": sl_id,
            "trailing_stop_active": False,
            "exit_mode": "trailing_stop",
        }
        save_trade_state(trade_state)
        trade_flow_logger.info(f"[{symbol}] 已成功保存交易狀態，準備進行移動止損。")
    return True


def manage_trailing_stop(
    trader: OKXTrader,
    symbol: str,
    strategy: UltimateStrategy,
    current_price: float,
    pos_data: dict,
):
    if not TSL_ENABLED:
        return
    trade_state = load_trade_state()
    if not trade_state or trade_state.get("exit_mode") != "trailing_stop":
        return
    pos_side, entry_price, current_stop_price, is_active = (
        "LONG" if float(pos_data.get("pos", "0")) > 0 else "SHORT",
        float(trade_state["entry_price"]),
        float(trade_state["current_stop_price"]),
        trade_state["trailing_stop_active"],
    )
    if not is_active:
        unrealized_pnl_pct = 0
        if entry_price > 0:
            unrealized_pnl_pct = (
                ((current_price - entry_price) / entry_price) * 100
                if pos_side == "LONG"
                else ((entry_price - current_price) / entry_price) * 100
            )
        profit_pct_condition_met = unrealized_pnl_pct >= TSL_ACTIVATION_PROFIT_PCT
        atr = strategy.data["tf_atr"].iloc[-1]
        activation_distance = atr * TSL_ACTIVATION_ATR_MULT
        price_move_condition_met = (
            pos_side == "LONG" and current_price >= entry_price + activation_distance
        ) or (
            pos_side == "SHORT" and current_price <= entry_price - activation_distance
        )
        if profit_pct_condition_met or price_move_condition_met:
            trade_state["trailing_stop_active"] = True
            is_active = True
            reason = (
                "利潤達到閾值" if profit_pct_condition_met else "價格移動達到ATR閾值"
            )
            trade_flow_logger.info(f"[{symbol}] 移動止損已激活! 原因: {reason}")
            save_trade_state(trade_state)
    if is_active:
        atr = strategy.data["tf_atr"].iloc[-1]
        trailing_distance = atr * TSL_TRAILING_ATR_MULT
        new_stop_price = None
        if pos_side == "LONG":
            potential_stop = current_price - trailing_distance
            if potential_stop > current_stop_price:
                new_stop_price = potential_stop
        else:
            potential_stop = current_price + trailing_distance
            if potential_stop < current_stop_price:
                new_stop_price = potential_stop
        if new_stop_price is not None:
            trade_flow_logger.info(
                f"[{symbol}] 調整移動止損: 舊={current_stop_price:.4f}, 新={new_stop_price:.4f}"
            )
            pos_size, close_side, old_stop_id = (
                abs(float(pos_data.get("pos"))),
                "sell" if pos_side == "LONG" else "buy",
                trade_state.get("current_stop_id"),
            )
            res = trader.place_trigger_order(
                symbol, close_side, pos_size, new_stop_price, "Trailing-Stop-Update"
            )
            if res and res.get("code") == "0" and res["data"][0]["sCode"] == "0":
                new_stop_id = res["data"][0]["algoId"]
                trade_flow_logger.info(
                    f"[{symbol}] 新移動止損單設置成功, Algo ID: {new_stop_id}"
                )
                if old_stop_id:
                    trader.cancel_algo_orders(symbol, [old_stop_id])
                    trade_flow_logger.info(
                        f"[{symbol}] 舊止損單 (ID: {old_stop_id}) 已取消。"
                    )
                (
                    trade_state["current_stop_price"],
                    trade_state["current_stop_id"],
                ) = (new_stop_price, new_stop_id)
                save_trade_state(trade_state)
            else:
                trade_flow_logger.error(f"[{symbol}] 調整移動止損失敗！響應: {res}")


def main():
    global trade_flow_logger, trade_logger
    trade_flow_logger = setup_logging()
    position_logger = setup_csv_logger(
        "position_logger", "positions.csv", ["timestamp", "symbol", "position"]
    )
    trade_logger = setup_csv_logger(
        "trade_logger",
        "trades.csv",
        ["timestamp", "symbol", "action", "size", "response"],
    )
    logging.info("啟動 OKX REST API 輪詢交易程序 (V3.5 凯利持久化版)...")
    OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE = (
        os.getenv("OKX_API_KEY"),
        os.getenv("OKX_API_SECRET"),
        os.getenv("OKX_API_PASSPHRASE"),
    )
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE]):
        logging.error("请设置环境变量: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE")
        return
    trader = OKXTrader(
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, simulated=SIMULATED
    )
    initial_equity = trader.fetch_account_balance("USDT")
    if initial_equity is None:
        logging.error("无法获取初始账户余额，将使用默认启动资金。")
        initial_equity = SIMULATED_EQUITY_START
    else:
        logging.info(f"查询成功，初始账户权益为: {initial_equity:.2f} USDT")
    strategies = {}
    for symbol in SYMBOLS:
        trader.set_leverage(symbol, DESIRED_LEVERAGE, mgnMode="cross", posSide="net")
        instrument_details = trader.fetch_instrument_details(symbol)
        if not instrument_details:
            logging.error(f"无法为 {symbol} 获获取合约详细信息，跳过。")
            continue
        initial_df = trader.fetch_history_klines(
            symbol, bar=KLINE_INTERVAL, limit=HISTORY_LIMIT
        )
        if initial_df is None or initial_df.empty:
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
        initial_position_data = trader.fetch_current_position(symbol)
        clear_trade_state()
        if initial_position_data and initial_position_data is not False:
            trade_flow_logger.warning(f"[{symbol}] 检测到已有仓位，尝试智能接管...")
            pos_side = (
                "LONG" if float(initial_position_data.get("pos", "0")) > 0 else "SHORT"
            )
            strategy.set_position(pos_side)
            for i in range(3):
                open_algos = trader.fetch_open_algo_orders(symbol)
                if not open_algos:
                    break
                trader.cancel_algo_orders(
                    symbol, [order["algoId"] for order in open_algos]
                )
                time.sleep(1)
            manage_position_exit_orders(
                trader, symbol, initial_position_data, strategy, "trailing_stop"
            )
        logging.info(
            f"策略 {symbol} 初始化成功，最新K線時間: {strategy.data.index[-1]}"
        )
    last_audit_time = datetime.utcnow()
    while True:
        try:
            current_equity = trader.fetch_account_balance("USDT")
            if current_equity is not None and current_equity < initial_equity * (
                1 - MAX_DAILY_DRAWDOWN_PCT
            ):
                trade_flow_logger.critical(
                    f"全局熔断！当前权益 {current_equity:.2f} USDT 已低于最大回撤限制。停止！"
                )
                break
            for symbol in SYMBOLS:
                if symbol not in strategies:
                    continue
                strategy = strategies[symbol]
                if current_equity is not None:
                    strategy.equity = current_equity
                latest_candles_df = trader.fetch_history_klines(
                    symbol, bar=KLINE_INTERVAL, limit=2
                )
                if latest_candles_df is None or latest_candles_df.empty:
                    logging.warning(f"无法获取 {symbol} 的最新K线数据。")
                    continue
                latest_candle = latest_candles_df.iloc[-1]
                last_known_ts = strategy.data.index[-1]
                if latest_candle.name > last_known_ts:
                    logging.info(
                        f"[{symbol}] 新K線 {latest_candle.name} close={latest_candle.Close:.4f} | 倉位: {strategy.position}"
                    )
                    strategy.update_with_candle(latest_candle)
                    if strategy.position is None:
                        if strategy.is_trading_paused:
                            logging.warning(f"[{symbol}] 策略因连续亏损暂停交易中...")
                            continue
                        action, score = strategy.next_on_candle_close()
                        if action:
                            exit_mode = (
                                "trailing_stop"
                                if strategy.data.iloc[-1]["market_regime"] == 1
                                else "fixed_target"
                            )
                            trade_flow_logger.info(
                                f"[{symbol}] 策略決策: {action} | 觸發權重值: {score:.4f} | 出場模式: {exit_mode}"
                            )
                            side = "buy" if action["action"] == "BUY" else "sell"
                            res = trader.place_market_order(
                                symbol, side, action["size"]
                            )
                            if (
                                res
                                and res.get("code") == "0"
                                and res.get("data")[0].get("sCode") == "0"
                            ):
                                pos_data = None
                                for i in range(8):
                                    time.sleep(2.5)
                                    pos_data = trader.fetch_current_position(symbol)
                                    if pos_data and pos_data is not False:
                                        break
                                if pos_data:
                                    intended_size, actual_size = float(
                                        action["size"]
                                    ), abs(float(pos_data.get("pos", "0")))
                                    if actual_size >= intended_size * 0.9:
                                        trade_flow_logger.info(
                                            f"[{symbol}] 仓位建立已确认并验证通过。意图: {intended_size}, 实际: {actual_size}"
                                        )
                                        new_pos_side = (
                                            "LONG"
                                            if float(pos_data.get("pos")) > 0
                                            else "SHORT"
                                        )
                                        strategy.set_position(new_pos_side)
                                        manage_position_exit_orders(
                                            trader,
                                            symbol,
                                            pos_data,
                                            strategy,
                                            exit_mode,
                                        )
                                    else:
                                        trade_flow_logger.critical(
                                            f"[{symbol}] [!!严重错误!!] 开仓验证失败！成交量严重不足！意图开仓: {intended_size}, 实际成交: {actual_size}。策略状态将不被更新。"
                                        )
                                else:
                                    logging.error(
                                        f"[{symbol}] 下单请求已发送，但在多次尝试后仍未能查询到仓位建立！"
                                    )
                            else:
                                logging.error(
                                    f"[{symbol}] 下单请求失败或被交易所拒绝，响应: {json.dumps(res)}"
                                )
                real_position_data = trader.fetch_current_position(symbol)
                if (
                    strategy.position is not None
                    and real_position_data
                    and real_position_data is not False
                ):
                    current_price = trader.fetch_ticker_price(symbol)
                    if current_price:
                        manage_trailing_stop(
                            trader,
                            symbol,
                            strategy,
                            current_price,
                            real_position_data,
                        )
                    else:
                        logging.warning(
                            f"[{symbol}] 無法獲取當前價格，跳過移動止損檢查。"
                        )
                elif strategy.position is not None and real_position_data is None:
                    trade_flow_logger.info(f"[{symbol}] 檢測到倉位已被平倉。")
                    closed_trade_state = load_trade_state()
                    if closed_trade_state and "entry_price" in closed_trade_state:
                        entry_price = closed_trade_state["entry_price"]
                        exit_price = strategy.data.iloc[-1].Close
                        pnl_pct = (
                            (
                                (exit_price - entry_price) / entry_price
                                if strategy.position == "LONG"
                                else (entry_price - exit_price) / entry_price
                            )
                            if entry_price != 0
                            else 0.0
                        )
                        trade_flow_logger.info(
                            f"[{symbol}] 交易结束: 入场价={entry_price:.4f}, 出场价(约)={exit_price:.4f}, PnL %={pnl_pct:.4%}"
                        )
                        strategy.register_trade_result(pnl_pct)
                    strategy.set_position(None)
                    clear_trade_state()
                    trade_flow_logger.info(f"[{symbol}] 策略狀態已重置。")
                current_price = trader.fetch_ticker_price(symbol)
                price_str, equity_str = (
                    f"{current_price:.4f}" if current_price is not None else "N/A"
                ), (f"{strategy.equity:.2f}" if strategy.equity is not None else "N/A")
                logging.info(
                    f"[{symbol}] 等待新K線... 倉位: {strategy.position} | 權益: {equity_str} USDT | 價格: {price_str} | K線: {strategy.data.index[-1]}"
                )
            if datetime.utcnow() - last_audit_time >= pd.Timedelta(
                minutes=AUDIT_INTERVAL_MINUTES
            ):
                last_audit_time = datetime.utcnow()
                trade_flow_logger.info(
                    f"--- [状态审计开始 (每 {AUDIT_INTERVAL_MINUTES} 分钟)] ---"
                )
                for symbol in SYMBOLS:
                    if symbol not in strategies:
                        continue
                    strategy = strategies[symbol]
                    real_position = trader.fetch_current_position(symbol)
                    if real_position is False:
                        logging.warning(f"[{symbol}] [审计跳过] 无法获取仓位。")
                        continue
                    if strategy.position is None and real_position is not None:
                        trade_flow_logger.critical(
                            f"[{symbol}] [审计发现] 状态不一致！策略无仓位，但交易所存在仓位。需手动干预！"
                        )
                    elif strategy.position is not None and real_position is None:
                        trade_flow_logger.warning(
                            f"[{symbol}] [审计发现] 状态不一致！策略有仓位，但交易所无。自动重置。"
                        )
                        strategy.set_position(None)
                        clear_trade_state()
                    elif real_position is not None:
                        real_pos_side = (
                            "LONG"
                            if float(real_position.get("pos", "0")) > 0
                            else "SHORT"
                        )
                        if strategy.position != real_pos_side:
                            trade_flow_logger.critical(
                                f"[{symbol}] [审计发现] 状态严重不一致！策略: {strategy.position} vs 交易所: {real_pos_side}。"
                            )
                            trade_flow_logger.warning(
                                f"[{symbol}] 進入緊急風控模式：將清空所有倉位和掛單..."
                            )
                            try:
                                open_algos = trader.fetch_open_algo_orders(symbol)
                                if open_algos:
                                    trader.cancel_algo_orders(
                                        symbol, [o["algoId"] for o in open_algos]
                                    )
                                pos_size = abs(float(real_position.get("pos", "0")))
                                close_side = (
                                    "buy" if real_pos_side == "SHORT" else "sell"
                                )
                                trader.place_market_order(symbol, close_side, pos_size)
                            except Exception as e:
                                trade_flow_logger.error(
                                    f"[{symbol}] [風控] 緊急平仓/撤单失败: {e}"
                                )
                            strategy.set_position(None)
                            clear_trade_state()
                            trade_flow_logger.info(
                                f"[{symbol}] [風控] 策略狀態已強制重置。"
                            )
                trade_flow_logger.info("--- [状态审计结束] ---")
            logging.info(
                f"所有交易對檢查完畢，將在 {POLL_INTERVAL_SECONDS} 秒後再次輪詢..."
            )
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("程序被手動中斷...")
            break
        except Exception as e:
            logging.exception(f"主循環發生未知錯誤: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
    logging.info("程序正在退出。")
    clear_trade_state()


if __name__ == "__main__":
    main()
