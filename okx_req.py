# okx_live_trading_rest_polling_v3.1.py
"""
OKX REST API 輪詢版 (V3.1 盈利後追蹤止損版)
- 核心變更:
    - [功能增強] 新增可配置的盈利百分比，作為移動停損的啟動條件之一。
    - [代碼重構] 新增 manage_trailing_stop 函式，將 "移動停損的激活與調整" 邏輯從主循環中完全剝離，使其更清晰、更易於維護。
    - [提升健壯性] 移動停損調整時，採用 "先下新單，再撤舊單" 的安全策略，防止在調整過程中倉位短暫失去保護。
- 繼承了 V3 的一致性與可讀性優化。

- V3.1.1 (風控增強版):
    - [風險控制] 增強狀態審計功能。當檢測到策略與交易所狀態嚴重不一致時，不再僅打印日誌，
                 而是自動執行緊急平倉、取消所有掛單，並重置策略狀態，以防止意外虧損。
"""

# --- 核心庫導入 (無變化) ---
import os, time, json, hmac, base64, hashlib, glob, logging, math, csv, urllib.parse
from datetime import datetime, time as dt_time, timedelta
from logging.handlers import RotatingFileHandler
import pandas as pd, numpy as np, requests, ta, warnings
from requests.models import PreparedRequest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.simplefilter(action="ignore", category=FutureWarning)
try:
    import joblib, lightgbm as lgb

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False


# --- 日志系統設置 (無變化) ---
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


# --- 全局配置 (無變化) ---
REST_BASE = "https://www.okx.com"
SYMBOLS = ["ETH-USDT-SWAP"]
KLINE_INTERVAL = "5m"
DESIRED_LEVERAGE = "20"
HISTORY_LIMIT = 500
POLL_INTERVAL_SECONDS = 20
SIMULATED = True
SIMULATED_EQUITY_START = 500000.0
SCHEDULED_EXIT_ENABLED = False
EXIT_TIME_UTC = "20:00"

# 移動停損 (Trailing Stop Loss) 相關設定
TSL_ENABLED = True  # 是否啟用移動停損功能
TSL_ACTIVATION_PROFIT_PCT = 1.0  # 盈利達到 1.0% 時，激活移動停損 (新功能)
TSL_ACTIVATION_ATR_MULT = 1.5  # 價格朝有利方向移動 N 倍 ATR 時，也激活移動停損
TSL_TRAILING_ATR_MULT = 2.0  # 激活後，停損點距離當前價格 N 倍 ATR

MAX_DAILY_DRAWDOWN_PCT = 0.05
MAX_CONSECUTIVE_LOSSES = 5
TRADING_PAUSE_HOURS = 4
AUDIT_INTERVAL_MINUTES = 15
STRATEGY_PARAMS = {
    "sl_atr_multiplier": 2.5,
    "tp_rr_ratio": 1.5,
    "kelly_trade_history": 20,
    "default_risk_pct": 30.0,
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
    "regime_score_threshold": 0.4,
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_atr_period": 14,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
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
    "ETH-USDT-SWAP": {
        "strategy_class": "ETHStrategy",
        "ml_weights": {"4h": 0.15, "8h": 0.3, "12h": 0.55},
        "ml_weighted_threshold": 0.2,
        "score_entry_threshold": 0.40,
    }
}
ML_HORIZONS = [4, 8, 12]


# --- 狀態管理 & 輔助函數 (無變化) ---
def save_trade_state(state):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    state_file = os.path.join(log_dir, "trade_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=4)
    logging.debug(f"Saved trade state: {state}")


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
        logging.info("Trade state file cleared.")


def compute_hurst(ts, max_lag=100):
    ts = np.asarray(ts)
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau = [
        np.std(ts[lag:] - ts[:-lag]) for lag in lags if np.std(ts[lag:] - ts[:-lag]) > 0
    ]
    if len(tau) < 2:
        return 0.5
    try:
        poly = np.polyfit(np.log(list(lags)[: len(tau)]), np.log(tau), 1)
        return float(max(0.0, min(1.0, poly[0])))
    except Exception:
        return 0.5


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


def add_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    p = STRATEGY_PARAMS
    df = df.copy()
    if len(df) < p["regime_hurst_period"] + 5:
        return df
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
        roll_max = s.rolling(p["regime_norm_period"]).max()
        roll_min = s.rolling(p["regime_norm_period"]).min()
        return ((s - roll_min) / (roll_max - roll_min)).fillna(0.5)

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


# --- OKX 交易接口類 (無變化) ---
class OKXTrader:
    def __init__(self, api_key, api_secret, passphrase, simulated=True):
        self.base = REST_BASE
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
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
                if method.upper() == "GET":
                    r = self.session.get(
                        url, headers=headers, params=params, timeout=(5, 15)
                    )
                else:
                    r = self.session.post(
                        url, headers=headers, data=body_str, timeout=(5, 15)
                    )
                if r.status_code == 404 and "cancel-algo-order" in path:
                    return {
                        "code": "0",
                        "data": [{"sCode": "0"}],
                        "msg": "Cancelled (already gone)",
                    }
                r.raise_for_status()
                return r.json()
            except (
                requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:
                logging.warning(
                    f"網路/SSL錯誤 (嘗試 {attempt}/{max_retries}): {e}。等待 {wait_time:.1f}s 後重試..."
                )
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 30)
            except requests.exceptions.RequestException as e:
                logging.error(f"HTTP請求錯誤 (嘗試 {attempt}/{max_retries}): {e}")
                if e.response is not None and 400 <= e.response.status_code < 500:
                    return {"code": str(e.response.status_code), "msg": str(e)}
                time.sleep(wait_time)
                wait_time = min(wait_time * 2, 30)
        logging.error(f"最終請求失敗: {method} {path}")
        return {"code": "-1", "msg": "Max retries exceeded with network errors"}

    def place_trigger_order(
        self, instId, side, sz, trigger_price, order_type, posSide="net"
    ):
        path = "/api/v5/trade/order-algo"
        price_str = f"{trigger_price:.8f}".rstrip("0").rstrip(".")
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
        path = "/api/v5/trade/orders-algo-pending"
        params = {"instType": "SWAP", "instId": instId, "ordType": "trigger"}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            return res.get("data", [])
        return []

    def cancel_algo_orders(self, instId, algoIds):
        if not algoIds:
            return True
        path = "/api/v5/trade/cancel-algo-order"
        body = [{"instId": instId, "algoId": str(aid)} for aid in algoIds]
        res = self._request("POST", path, body=body)
        return res and res.get("code") == "0"

    def set_leverage(self, instId, lever, mgnMode="cross", posSide=None):
        path = "/api/v5/account/set-leverage"
        body = {"instId": instId, "lever": str(lever), "mgnMode": mgnMode}
        if posSide:
            body["posSide"] = posSide
        return self._request("POST", path, body=body).get("code") == "0"

    def fetch_account_balance(self, ccy="USDT"):
        path = "/api/v5/account/balance"
        params = {"ccy": ccy}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
            if data and "details" in data[0]:
                for detail in data[0]["details"]:
                    if detail.get("ccy") == ccy:
                        return float(detail.get("eq", 0))
        return None

    def fetch_current_position(self, instId):
        path = "/api/v5/account/positions"
        params = {"instId": instId}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
            if data:
                for pos_data in data:
                    if float(pos_data.get("pos", "0")) != 0:
                        return pos_data
            return None
        else:
            logging.error(
                f"查詢倉位失敗 for {instId}: {res.get('msg') if res else 'Unknown error'}"
            )
            return False

    def fetch_instrument_details(self, instId, instType="SWAP"):
        if instId in self.instrument_info:
            return self.instrument_info[instId]
        path = "/api/v5/public/instruments"
        params = {"instType": instType, "instId": instId}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0" and res.get("data"):
            self.instrument_info[instId] = res["data"][0]
            return res["data"][0]
        return None

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
        path = "/api/v5/market/history-candles"
        params = {"instId": instId, "bar": bar, "limit": limit}
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
        path = "/api/v5/market/ticker"
        params = {"instId": instId}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
            if data:
                return float(data[0].get("last"))
        return None


# --- 策略核心類 (無變化) ---
class UltimateStrategy:
    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        trader: OKXTrader = None,
        pos_logger=None,
        instrument_details=None,
    ):
        self.symbol = symbol
        self.trader = trader
        self.data = df.copy()
        self.position = None
        self.pos_logger = pos_logger
        self.instrument_details = instrument_details
        logging.info(f"[{self.symbol}] 策略初始化...")
        from collections import deque

        self.recent_trade_returns = deque(maxlen=STRATEGY_PARAMS["kelly_trade_history"])
        self.ml_models = {}
        self.score_entry_threshold = ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).get(
            "score_entry_threshold", STRATEGY_PARAMS["score_entry_threshold"]
        )
        self.equity = SIMULATED_EQUITY_START
        self.consecutive_losses = 0
        self.trading_paused_until = None
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
            self.trading_paused_until = datetime.utcnow() + timedelta(
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
        for h in ML_HORIZONS:
            files = glob.glob(f"directional_model_{self.symbol}_{h}h.joblib")
            if files:
                try:
                    self.ml_models[h] = joblib.load(files[0])
                    logging.info(f"[{self.symbol}] Loaded model: {files[0]}")
                except Exception as e:
                    logging.error(f"Failed to load model {files[0]}: {e}")

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < 2:
            return STRATEGY_PARAMS["default_risk_pct"]
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return STRATEGY_PARAMS["default_risk_pct"]
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        if reward_ratio == 0:
            return STRATEGY_PARAMS["default_risk_pct"]
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return max(0.005, min(STRATEGY_PARAMS["max_risk_pct"], 0.5 * kelly))

    def get_ml_confidence_score(self):
        if not self.ml_models:
            return 0.0
        features = [c for c in self.data.columns if c.startswith("feature_")]
        if not features:
            return 0.0
        current_row = self.data[features].iloc[-1:].fillna(0)
        score = 0.0
        ml_weights = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get("ml_weights", {})
        for h, model in self.ml_models.items():
            try:
                pred = model.predict(current_row)[0]
                pred_val = 1 if pred == 1 else -1
                score += pred_val * ml_weights.get(f"{h}h", 0)
            except Exception:
                continue
        return score

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
        self.data = add_market_regime_features(self.data)
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
        self.data["tf_atr"] = ta.volatility.AverageTrueRange(
            self.data.High,
            self.data.Low,
            self.data.Close,
            STRATEGY_PARAMS["tf_atr_period"],
        ).average_true_range()
        bb = ta.volatility.BollingerBands(
            self.data.Close,
            STRATEGY_PARAMS["mr_bb_period"],
            STRATEGY_PARAMS["mr_bb_std"],
        )
        self.data["mr_bb_upper"] = bb.bollinger_hband()
        self.data["mr_bb_lower"] = bb.bollinger_lband()
        stoch_rsi = ta.momentum.StochRSIIndicator(
            self.data.Close, window=14, smooth1=3, smooth2=3
        )
        self.data["mr_stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        self.data["mr_stoch_rsi_d"] = stoch_rsi.stochrsi_d()

    def _calculate_entry_score(self):
        w = STRATEGY_PARAMS["score_weights_tf"]
        last = self.data.iloc[-1]
        try:
            b_s = (
                1
                if last.High > last.tf_donchian_h
                else -1 if last.Low < last.tf_donchian_l else 0
            )
        except (AttributeError, KeyError):
            b_s = 0
        mo_s = 1 if last.tf_ema_fast > last.tf_ema_slow else -1
        ml_score = self.get_ml_confidence_score()
        adv_score = last.get("advanced_ml_signal", 0)
        mtf_score = last.get("mtf_signal", 0)
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
        long_reentry = prev.Close < prev.mr_bb_lower and last.Close > last.mr_bb_lower
        stoch_long_confirm = (
            last.mr_stoch_rsi_k > last.mr_stoch_rsi_d
            and prev.mr_stoch_rsi_k <= prev.mr_stoch_rsi_d
            and last.mr_stoch_rsi_k < 40
        )
        if long_reentry and stoch_long_confirm:
            return 1
        short_reentry = prev.Close > prev.mr_bb_upper and last.Close < last.mr_bb_upper
        stoch_short_confirm = (
            last.mr_stoch_rsi_k < last.mr_stoch_rsi_d
            and prev.mr_stoch_rsi_k >= prev.mr_stoch_rsi_d
            and last.mr_stoch_rsi_k > 60
        )
        if short_reentry and stoch_short_confirm:
            return -1
        return 0

    def next_on_candle_close(self):
        self.compute_all_features()
        last = self.data.iloc[-1]
        if pd.isna(last.get("market_regime")):
            return None, 0.0
        market_status = "趨勢市" if last.market_regime == 1 else "震盪市"
        logging.info(
            f"[{self.symbol}] 市場狀態判斷: {market_status} (Regime Score: {last.get('regime_score', 0):.3f})"
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
        if not self.instrument_details:
            return str(size)
        lot_sz_str = self.instrument_details.get("lotSz")
        if not lot_sz_str:
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
        calculated_size = notional / price if price > 0 else 0
        if calculated_size <= 0:
            return "0"
        return self._adjust_size_to_lot_size(calculated_size)

    def register_trade_result(self, pnl_pct):
        self.recent_trade_returns.append(pnl_pct)
        if pnl_pct < 0:
            self.register_loss()
        else:
            self.register_win()

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


# --- 輔助函數 (無變化) ---
def manage_position_exit_orders(
    trader: OKXTrader,
    symbol: str,
    pos_data: dict,
    strategy: UltimateStrategy,
    exit_mode: str,
):
    """
    為一個已確認的倉位，集中管理其止損和止盈訂單的設置。
    """
    entry_px = float(pos_data.get("avgPx"))
    pos_size = abs(float(pos_data.get("pos")))
    pos_side = "LONG" if float(pos_data.get("pos")) > 0 else "SHORT"

    strategy.compute_all_features()
    atr = strategy.data["tf_atr"].iloc[-1]
    if pd.isna(atr):
        trade_flow_logger.error(f"[{symbol}] ATR計算失敗，無法設置出場訂單。")
        return False

    close_side = "sell" if pos_side == "LONG" else "buy"
    sl_mult = STRATEGY_PARAMS["sl_atr_multiplier"]

    # --- 1. 設置止損單 (所有模式都需要) ---
    sl_price = (
        entry_px - (atr * sl_mult) if pos_side == "LONG" else entry_px + (atr * sl_mult)
    )
    trade_flow_logger.info(
        f"[{symbol}] 為 {pos_side} 倉位 (均價: {entry_px:.4f}) 設置止損單... "
        f"訂單類型: {close_side.upper()}, 觸發價: {sl_price:.4f}"
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

    # --- 2. 如果是震盪市，額外設置固定止盈單 ---
    if exit_mode == "fixed_target":
        tp_rr = STRATEGY_PARAMS["tp_rr_ratio"]
        tp_price = (
            entry_px + (atr * sl_mult * tp_rr)
            if pos_side == "LONG"
            else entry_px - (atr * sl_mult * tp_rr)
        )
        trade_flow_logger.info(
            f"[{symbol}] (震盪市) 設置固定止盈單... "
            f"訂單類型: {close_side.upper()}, 觸發價: {tp_price:.4f}"
        )
        trader.place_trigger_order(
            symbol, close_side, pos_size, tp_price, "Take-Profit"
        )

    # --- 3. 如果是趨勢市，保存狀態用於後續的移動止損 ---
    if exit_mode == "trailing_stop":
        trade_state = {
            "entry_price": entry_px,
            "initial_stop_price": sl_price,
            "current_stop_price": sl_price,
            "current_stop_id": sl_id,
            "trailing_stop_active": False,  # 初始為未激活
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
    """
    在持有倉位期間，管理移動停損的激活和調整。
    """
    if not TSL_ENABLED:
        return

    trade_state = load_trade_state()
    if not trade_state or trade_state.get("exit_mode") != "trailing_stop":
        return

    pos_side = "LONG" if float(pos_data.get("pos", "0")) > 0 else "SHORT"
    entry_price = float(trade_state["entry_price"])
    current_stop_price = float(trade_state["current_stop_price"])
    is_active = trade_state["trailing_stop_active"]

    # --- 1. 激活邏輯：檢查是否應啟動移動停損 ---
    if not is_active:
        unrealized_pnl_pct = 0
        if entry_price > 0:
            if pos_side == "LONG":
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100

        profit_pct_condition_met = unrealized_pnl_pct >= TSL_ACTIVATION_PROFIT_PCT

        atr = strategy.data["tf_atr"].iloc[-1]
        activation_distance = atr * TSL_ACTIVATION_ATR_MULT
        price_move_condition_met = False
        if pos_side == "LONG" and current_price >= entry_price + activation_distance:
            price_move_condition_met = True
        elif pos_side == "SHORT" and current_price <= entry_price - activation_distance:
            price_move_condition_met = True

        if profit_pct_condition_met or price_move_condition_met:
            trade_state["trailing_stop_active"] = True
            is_active = True
            reason = (
                "利潤達到閾值" if profit_pct_condition_met else "價格移動達到ATR閾值"
            )
            trade_flow_logger.info(f"[{symbol}] 移動止損已激活! 原因: {reason}")
            save_trade_state(trade_state)

    # --- 2. 調整邏輯：如果已激活，則跟蹤價格調整停損 ---
    if is_active:
        atr = strategy.data["tf_atr"].iloc[-1]
        trailing_distance = atr * TSL_TRAILING_ATR_MULT
        new_stop_price = None

        if pos_side == "LONG":
            potential_stop = current_price - trailing_distance
            if potential_stop > current_stop_price:  # 只上移，不上移
                new_stop_price = potential_stop
        else:  # SHORT
            potential_stop = current_price + trailing_distance
            if potential_stop < current_stop_price:  # 只下移，不上移
                new_stop_price = potential_stop

        if new_stop_price is not None:
            trade_flow_logger.info(
                f"[{symbol}] 調整移動止損: 舊={current_stop_price:.4f}, 新={new_stop_price:.4f}"
            )
            pos_size = abs(float(pos_data.get("pos")))
            close_side = "sell" if pos_side == "LONG" else "buy"
            old_stop_id = trade_state.get("current_stop_id")

            # 安全操作：先下新單
            res = trader.place_trigger_order(
                symbol, close_side, pos_size, new_stop_price, "Trailing-Stop-Update"
            )
            if res and res.get("code") == "0" and res["data"][0]["sCode"] == "0":
                new_stop_id = res["data"][0]["algoId"]
                trade_flow_logger.info(
                    f"[{symbol}] 新移動止損單設置成功, Algo ID: {new_stop_id}"
                )

                # 新單成功後，再取消舊單
                if old_stop_id:
                    trader.cancel_algo_orders(symbol, [old_stop_id])
                    trade_flow_logger.info(
                        f"[{symbol}] 舊止損單 (ID: {old_stop_id}) 已取消。"
                    )

                # 更新並保存狀態
                trade_state["current_stop_price"] = new_stop_price
                trade_state["current_stop_id"] = new_stop_id
                save_trade_state(trade_state)
            else:
                trade_flow_logger.error(f"[{symbol}] 調整移動止損失敗！響應: {res}")


# --- 主程序入口 ---
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

    logging.info("啟動 OKX REST API 輪詢交易程序 (V3.1.1 風控增強版)...")
    OKX_API_KEY = os.getenv("OKX_API_KEY")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET")
    OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE]):
        logging.error("請設置環境變數: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE")
        return

    trader = OKXTrader(
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, simulated=SIMULATED
    )

    initial_equity = trader.fetch_account_balance("USDT")
    if initial_equity is None:
        logging.error("無法獲取初始賬戶餘額，將使用默認啟動資金。")
        initial_equity = SIMULATED_EQUITY_START
    else:
        logging.info(f"查詢成功，初始賬戶權益為: {initial_equity:.2f} USDT")

    strategies = {}
    for symbol in SYMBOLS:
        trader.set_leverage(symbol, DESIRED_LEVERAGE, mgnMode="cross", posSide="net")
        instrument_details = trader.fetch_instrument_details(symbol)
        if not instrument_details:
            logging.error(f"無法為 {symbol} 獲取合約詳細資訊，跳過。")
            continue
        initial_df = trader.fetch_history_klines(
            symbol, bar=KLINE_INTERVAL, limit=HISTORY_LIMIT
        )
        if initial_df is None or initial_df.empty:
            logging.error(f"無法為 {symbol} 獲取初始數據，退出。")
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
            trade_flow_logger.warning(f"[{symbol}] 檢測到已有仓位，尝试智能接管...")
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
                    logging.warning(f"無法獲取 {symbol} 的最新K線數據。")
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
                                for i in range(5):
                                    time.sleep(2)
                                    pos_data = trader.fetch_current_position(symbol)
                                    if pos_data and pos_data is not False:
                                        trade_flow_logger.info(
                                            f"[{symbol}] 仓位建立已确认。"
                                        )
                                        break

                                if pos_data:
                                    new_pos_side = (
                                        "LONG"
                                        if float(pos_data.get("pos", "0")) > 0
                                        else "SHORT"
                                    )
                                    strategy.set_position(new_pos_side)
                                    manage_position_exit_orders(
                                        trader, symbol, pos_data, strategy, exit_mode
                                    )
                                else:
                                    logging.error(
                                        f"[{symbol}] 下單后未能确认仓位建立！"
                                    )
                            else:
                                logging.error(
                                    f"[{symbol}] 下單可能失敗，響應: {json.dumps(res)}"
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
                            trader, symbol, strategy, current_price, real_position_data
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
                                (
                                    (exit_price - entry_price)
                                    if strategy.position == "LONG"
                                    else (entry_price - exit_price)
                                )
                                / entry_price
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
                price_str = (
                    f"{current_price:.4f}" if current_price is not None else "N/A"
                )
                equity_str = (
                    f"{strategy.equity:.2f}" if strategy.equity is not None else "N/A"
                )
                logging.info(
                    f"[{symbol}] 等待新K線... 倉位: {strategy.position} | 權益: {equity_str} USDT | 價格: {price_str} | K線: {strategy.data.index[-1]}"
                )

            # 周期性状态审计
            if datetime.utcnow() - last_audit_time >= timedelta(
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
                    # --- [MODIFIED CODE BLOCK START] ---
                    # 這是本次修改的核心部分
                    elif real_position is not None:
                        real_pos_side = (
                            "LONG"
                            if float(real_position.get("pos", "0")) > 0
                            else "SHORT"
                        )
                        if strategy.position != real_pos_side:
                            # 檢測到嚴重不一致，啟動風控
                            trade_flow_logger.critical(
                                f"[{symbol}] [审计发现] 状态严重不一致！策略: {strategy.position} vs 交易所: {real_pos_side}。"
                            )
                            trade_flow_logger.warning(
                                f"[{symbol}] 進入緊急風控模式：將清空所有倉位和掛單..."
                            )

                            # 1. 取消所有現有掛單
                            try:
                                open_algos = trader.fetch_open_algo_orders(symbol)
                                if open_algos:
                                    algo_ids = [order["algoId"] for order in open_algos]
                                    trader.cancel_algo_orders(symbol, algo_ids)
                                    trade_flow_logger.info(
                                        f"[{symbol}] [風控] 已取消 {len(algo_ids)} 個掛單。"
                                    )
                            except Exception as e:
                                trade_flow_logger.error(
                                    f"[{symbol}] [風控] 緊急取消掛單失敗: {e}"
                                )

                            # 2. 以市價單平掉交易所的意外倉位
                            try:
                                pos_size = abs(float(real_position.get("pos", "0")))
                                # 如果實際是空頭(SHORT)，則需要買入(buy)來平倉
                                close_side = (
                                    "buy" if real_pos_side == "SHORT" else "sell"
                                )
                                trade_flow_logger.info(
                                    f"[{symbol}] [風控] 正在以市價單平掉 {real_pos_side} 倉位，數量: {pos_size}..."
                                )
                                res = trader.place_market_order(
                                    symbol, close_side, pos_size
                                )
                                trade_flow_logger.info(
                                    f"[{symbol}] [風控] 平倉指令已發送, 響應: {res}"
                                )
                            except Exception as e:
                                trade_flow_logger.error(
                                    f"[{symbol}] [風控] 緊急平倉失敗: {e}"
                                )

                            # 3. 重置策略的內部狀態
                            strategy.set_position(None)
                            clear_trade_state()
                            trade_flow_logger.info(
                                f"[{symbol}] [風控] 策略狀態已強制重置。"
                            )
                    # --- [MODIFIED CODE BLOCK END] ---

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
