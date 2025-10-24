# okx_live_trading_rest_polling_v3.1_fixed_and_commented.py
"""
OKX REST API 轮询版 (V3.1.1 盈利后追蹤止损 & 关键修复版)
- 核心变更:
    - [!!! 关键修复 !!!] 增加了下单后的成交量验证机制。现在程序会检查实际成交数量是否与意图开仓数量基本一致，
                       如果不一致（例如部分成交），将记录严重错误并停止后续操作（如设置止损），从根本上解决了策略状态与交易所状态脱节的问题。
    - [功能增强] 新增可配置的盈利百分比，作为移动停損的啟動條件之一。
    - [代碼重構] 新增 manage_trailing_stop 函式，將 "移動停損的激活與調整" 邏輯從主循環中完全剝離，使其更清晰、更易於維護。
    - [提升健壯性] 移動停損調整時，採用 "先下新單，再撤舊單" 的安全策略，防止在調整過程中倉位短暫失去保護。
    - [风险控制] (继承自V3.1.1) 增强状态审计功能。当检测到策略与交易所状态严重不一致时，自动执行紧急平仓、取消所有挂单，并重置策略状态。
"""

# --- 核心库导入 ---
import os
import time
import json
import hmac
import base64
import hashlib
import glob
import logging
import math
import csv
import urllib.parse
from datetime import datetime, time as dt_time, timedelta
from logging.handlers import RotatingFileHandler
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
    import lightgbm as lgb

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False
    print("警告: lightgbm 或 joblib 未安装，机器学习相关功能将不可用。")


# --- 日志系统设置 ---
def setup_logging():
    """配置日志系统，分为主日志和交易流日志。"""
    log_dir = "logs"  # 日志文件存放目录
    os.makedirs(log_dir, exist_ok=True)

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 设置最低日志级别

    # 清理已存在的处理器，防止重复记录
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 主日志文件处理器 (记录所有 DEBUG 及以上级别信息)
    log_file_path = os.path.join(log_dir, "trading_bot.log")
    # 使用 RotatingFileHandler 实现日志文件按大小轮转
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

    # 控制台处理器 (在屏幕上打印 INFO 及以上级别信息，便于实时监控)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 交易流日志记录器 (专门记录关键的交易决策和流程)
    trade_flow_logger = logging.getLogger("TradeFlow")
    trade_flow_logger.setLevel(logging.INFO)
    trade_flow_logger.propagate = False  # 防止向根日志记录器传递，避免重复记录
    trade_flow_path = os.path.join(log_dir, "trades_flow.log")
    trade_flow_handler = RotatingFileHandler(
        trade_flow_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    trade_flow_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    trade_flow_handler.setFormatter(trade_flow_formatter)
    trade_flow_logger.addHandler(trade_flow_handler)
    trade_flow_logger.addHandler(console_handler)  # 交易流信息也同时打印到控制台

    return trade_flow_logger


def setup_csv_logger(name, log_file, fields):
    """配置一个将日志记录为 CSV 格式的记录器，用于数据分析。"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, log_file)
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.INFO)
    logger_obj.propagate = False

    if logger_obj.hasHandlers():
        return logger_obj

    handler = logging.FileHandler(csv_path, mode="a", encoding="utf-8")

    # 如果文件是空的，先写入CSV表头
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    # 自定义 emit 方法来写入 CSV 行
    def emit_csv(record):
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(record.msg)

    handler.emit = emit_csv
    logger_obj.addHandler(handler)
    return logger_obj


# --- 全局配置 ---
# API 和交易对设置
REST_BASE = "https://www.okx.com"  # OKX API 地址
SYMBOLS = ["ETH-USDT-SWAP"]  # 要交易的合约列表
KLINE_INTERVAL = "15m"  # K线周期
DESIRED_LEVERAGE = "20"  # 期望设置的杠杆倍数
HISTORY_LIMIT = 500  # 每次启动时加载的历史K线数量
POLL_INTERVAL_SECONDS = 20  # 每次轮询之间的时间间隔（秒）

# 模拟交易与实盘交易切换
SIMULATED = True  # True: 使用OKX模拟盘, False: 使用实盘
SIMULATED_EQUITY_START = 500000.0  # 模拟盘的初始资金 (如果API查询失败则使用)

# 定时退出功能 (可选)
SCHEDULED_EXIT_ENABLED = False  # 是否启用定时退出所有仓位功能
EXIT_TIME_UTC = "20:00"  # 定时退出的UTC时间

# 移動停損 (Trailing Stop Loss) 相關設定
TSL_ENABLED = True  # 是否启用移动止损功能
TSL_ACTIVATION_PROFIT_PCT = 1.0  # 盈利达到 1.0% 时，激活移动止损 (新功能)
TSL_ACTIVATION_ATR_MULT = 1.5  # 价格朝有利方向移动 N 倍 ATR 时，也激活移动止损
TSL_TRAILING_ATR_MULT = 2.0  # 激活后，止损点距离当前价格 N 倍 ATR

# 全局风险控制
MAX_DAILY_DRAWDOWN_PCT = 0.05  # 最大日回撤百分比，触发全局熔断
MAX_CONSECUTIVE_LOSSES = 5  # 单个交易对最大连续亏损次数
TRADING_PAUSE_HOURS = 4  # 达到最大连亏后暂停交易的小时数
AUDIT_INTERVAL_MINUTES = 15  # 状态审计的间隔分钟数

# 策略核心参数
STRATEGY_PARAMS = {
    "sl_atr_multiplier": 2.5,  # 初始止损的ATR倍数
    "tp_rr_ratio": 1.5,  # 震荡市中止盈的风险回报比
    "kelly_trade_history": 20,  # 计算凯利公式使用的历史交易次数
    "default_risk_pct": 50.0,  # 默认的风险百分比 (凯利公式无法计算时使用)
    "max_risk_pct": 0.04,  # 凯利公式计算出的最大风险上限
    "regime_adx_period": 14,  # 市场状态判断 - ADX周期
    "regime_atr_period": 14,  # 市场状态判断 - ATR周期
    "regime_atr_slope_period": 5,  # 市场状态判断 - ATR斜率周期
    "regime_rsi_period": 14,  # 市场状态判断 - RSI周期
    "regime_rsi_vol_period": 14,  # 市场状态判断 - RSI波动周期
    "regime_norm_period": 252,  # 归一化周期
    "regime_hurst_period": 100,  # 赫斯特指数周期
    "regime_score_weight_adx": 0.6,  # 市场状态评分权重 - ADX
    "regime_score_weight_atr": 0.3,  # 市场状态评分权重 - ATR
    "regime_score_weight_rsi": 0.05,  # 市场状态评分权重 - RSI
    "regime_score_weight_hurst": 0.05,  # 市场状态评分权重 - Hurst
    "regime_score_threshold": 0.4,  # 市场状态评分阈值 (高于此值为趋势市)
    "tf_donchian_period": 30,  # 趋势跟踪 - 唐奇安通道周期
    "tf_ema_fast_period": 20,  # 趋势跟踪 - 快速EMA周期
    "tf_ema_slow_period": 75,  # 趋势跟踪 - 慢速EMA周期
    "tf_atr_period": 14,  # 趋势跟踪 - ATR周期
    "mr_bb_period": 20,  # 均值回归 - 布林带周期
    "mr_bb_std": 2.0,  # 均值回归 - 布林带标准差
    "mr_risk_multiplier": 0.5,  # 均值回归 - 风险乘数
    "mtf_period": 50,  # 多时间框架分析周期
    "score_entry_threshold": 0.4,  # 趋势市入场信号的最终评分阈值
    "score_weights_tf": {  # 趋势市信号评分权重
        "breakout": 0.2667,
        "momentum": 0.20,
        "mtf": 0.1333,
        "ml": 0.2182,
        "advanced_ml": 0.1818,
    },
}
# 特定交易对的参数覆盖 (例如，为不同币种设置不同参数)
ASSET_SPECIFIC_OVERRIDES = {
    "ETH-USDT-SWAP": {
        "strategy_class": "ETHStrategy",
        "ml_weights": {"4h": 0.15, "8h": 0.3, "12h": 0.55},
        "ml_weighted_threshold": 0.2,
        "score_entry_threshold": 0.35,
    }
}
# 机器学习模型的时间范围
ML_HORIZONS = [4, 8, 12]


# --- 状态管理 & 辅助函数 ---
def save_trade_state(state):
    """将当前交易状态（如止损ID）保存到JSON文件中，以便程序重启后恢复。"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    state_file = os.path.join(log_dir, "trade_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=4)
    logging.debug(f"已保存交易状态: {state}")


def load_trade_state():
    """从JSON文件中加载交易状态。"""
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
    """清除交易状态文件，通常在交易结束后调用。"""
    log_dir = "logs"
    state_file = os.path.join(log_dir, "trade_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        logging.info("交易状态文件已清除。")


def compute_hurst(ts, max_lag=100):
    """计算赫斯特指数(Hurst Exponent)，用于判断时间序列的趋势性。"""
    ts = np.asarray(ts)
    if len(ts) < 10:
        return 0.5  # 数据太少，返回中性值
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
    """运行高级机器学习模型进行推理（占位符）。"""
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
    """为DataFrame计算并添加用于机器学习模型的特征。"""
    p = STRATEGY_PARAMS
    df = df.copy()
    if len(df) < p["regime_hurst_period"] + 5:
        return df

    # 计算各种技术指标
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
    obv = ta.volume.OnBalanceVolumeIndicator(
        df["Close"], df["Volume"]
    ).on_balance_volume()

    # 定义一个归一化函数
    def norm(s):
        roll_max = s.rolling(p["regime_norm_period"]).max()
        roll_min = s.rolling(p["regime_norm_period"]).min()
        return ((s - roll_min) / (roll_max - roll_min)).fillna(0.5)

    # 计算并添加归一化后的特征
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
    df["feature_obv_norm"] = norm(obv)
    df["feature_vol_pct_change_norm"] = norm(df["Volume"].pct_change(periods=1).abs())
    df["feature_bb_width_norm"] = norm(
        (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    )
    df["feature_atr_pct_change_norm"] = norm(atr.pct_change(periods=1))

    # 综合所有特征计算市场状态评分
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["feature_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    return df


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """根据市场状态评分，为DataFrame添加市场状态的分类标签。"""
    p = STRATEGY_PARAMS
    df = df.copy()
    df["regime_score"] = df.get("feature_regime_score", pd.Series(0.5, index=df.index))
    # 根据阈值判断是趋势市还是震荡市
    df["trend_regime"] = np.where(
        df["regime_score"] > p["regime_score_threshold"], "Trending", "Mean-Reverting"
    )
    # 计算年化波动率
    df["volatility"] = df["Close"].pct_change().rolling(24 * 7).std() * np.sqrt(
        24 * 365
    )
    # 根据波动率分位数划分为低、中、高三个等级
    low_vol, high_vol = df["volatility"].quantile(0.33), df["volatility"].quantile(0.67)
    df["volatility_regime"] = pd.cut(
        df["volatility"],
        bins=[0, low_vol, high_vol, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
    # 最终的市场状态，1代表趋势，-1代表震荡
    df["market_regime"] = np.where(df["trend_regime"] == "Trending", 1, -1)
    return df


# --- OKX 交易接口类 ---
class OKXTrader:
    """封装了与 OKX API 交互的所有方法。"""

    def __init__(self, api_key, api_secret, passphrase, simulated=True):
        self.base = REST_BASE
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.simulated = simulated
        self.instrument_info = {}  # 缓存合约信息

        # 统一的请求头
        self.common_headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }
        if self.simulated:
            self.common_headers["x-simulated-trading"] = "1"

        # 使用 requests.Session 来保持连接和配置
        self.session = requests.Session()
        # 配置重试逻辑，应对临时的网络问题或服务器错误
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _now(self):
        """获取 ISO 8601 格式的 UTC 时间字符串。"""
        return datetime.utcnow().isoformat("T", "milliseconds") + "Z"

    def _sign(self, ts, method, path, body_str=""):
        """生成API请求签名。"""
        message = f"{ts}{method}{path}{body_str}"
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method, path, body=None, params=None, max_retries=7):
        """核心请求函数，包含签名、请求发送和错误处理。"""
        ts = self._now()
        body_str = "" if body is None else json.dumps(body)

        # 准备用于签名的URL路径（包含查询参数）
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

        # 带重试的请求循环
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

                # 特殊处理：如果取消一个已经不存在的订单，API会返回404，这里将其视为成功
                if r.status_code == 404 and "cancel-algo-order" in path:
                    return {
                        "code": "0",
                        "data": [{"sCode": "0"}],
                        "msg": "Cancelled (already gone)",
                    }

                r.raise_for_status()  # 如果是HTTP错误状态码，则抛出异常
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

        logging.error(f"最终请求失败: {method} {path}")
        return {"code": "-1", "msg": "Max retries exceeded with network errors"}

    # --- 以下是封装好的各类API接口调用方法 ---

    def place_trigger_order(
        self, instId, side, sz, trigger_price, order_type, posSide="net"
    ):
        """下达计划委托订单（止盈止损）。"""
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
            "orderPx": "-1",  # -1代表以市价执行
        }
        return self._request("POST", path, body=body)

    def fetch_open_algo_orders(self, instId):
        """获取当前所有未触发的计划委托订单。"""
        path = "/api/v5/trade/orders-algo-pending"
        params = {"instType": "SWAP", "instId": instId, "ordType": "trigger"}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            return res.get("data", [])
        return []

    def cancel_algo_orders(self, instId, algoIds):
        """批量取消计划委托订单。"""
        if not algoIds:
            return True
        path = "/api/v5/trade/cancel-algo-order"
        body = [{"instId": instId, "algoId": str(aid)} for aid in algoIds]
        res = self._request("POST", path, body=body)
        return res and res.get("code") == "0"

    def set_leverage(self, instId, lever, mgnMode="cross", posSide=None):
        """设置杠杆倍数。"""
        path = "/api/v5/account/set-leverage"
        body = {"instId": instId, "lever": str(lever), "mgnMode": mgnMode}
        if posSide:
            body["posSide"] = posSide
        return self._request("POST", path, body=body).get("code") == "0"

    def fetch_account_balance(self, ccy="USDT"):
        """获取账户权益。"""
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
        """获取指定合约的当前持仓信息。"""
        path = "/api/v5/account/positions"
        params = {"instId": instId}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
            if data:
                # 遍历返回的仓位数据，因为可能包含不同保证金模式的仓位
                for pos_data in data:
                    if float(pos_data.get("pos", "0")) != 0:
                        return pos_data  # 返回第一个非空的仓位信息
            return None  # 没有仓位
        else:
            logging.error(
                f"查詢倉位失敗 for {instId}: {res.get('msg') if res else 'Unknown error'}"
            )
            return False  # False 代表查询失败

    def fetch_instrument_details(self, instId, instType="SWAP"):
        """获取合约的详细信息，如最小下单量(lotSz)。"""
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
        """下达市价单。"""
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
            # 记录下单操作到CSV日志
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
        """获取历史K线数据。"""
        path = "/api/v5/market/history-candles"
        params = {"instId": instId, "bar": bar, "limit": limit}
        res = self._request("GET", path, params=params)
        if res and res.get("code") == "0":
            data = res.get("data", [])
            if not data:
                return pd.DataFrame()
            # 将API返回的列表转换为Pandas DataFrame
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
        """获取最新成交价。"""
        path = "/api/v5/market/ticker"
        params = {"instId": instId}
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
        self.position = None  # 当前策略认为的持仓状态 ("LONG", "SHORT", or None)
        self.pos_logger = pos_logger
        self.instrument_details = instrument_details
        logging.info(f"[{self.symbol}] 策略初始化...")
        from collections import deque

        self.recent_trade_returns = deque(
            maxlen=STRATEGY_PARAMS["kelly_trade_history"]
        )  # 用于计算凯利公式
        self.ml_models = {}
        self.score_entry_threshold = ASSET_SPECIFIC_OVERRIDES.get(symbol, {}).get(
            "score_entry_threshold", STRATEGY_PARAMS["score_entry_threshold"]
        )
        self.equity = SIMULATED_EQUITY_START
        self.consecutive_losses = 0  # 连亏计数器
        self.trading_paused_until = None  # 交易暂停截止时间
        self._load_models()

    @property
    def is_trading_paused(self):
        """检查当前策略是否处于暂停交易状态。"""
        if self.trading_paused_until and datetime.utcnow() < self.trading_paused_until:
            return True
        self.trading_paused_until = None  # 恢复交易
        return False

    def register_loss(self):
        """记录一次亏损，并检查是否触发连亏熔断。"""
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
        """记录一次盈利，并重置连亏计数器。"""
        if self.consecutive_losses > 0:
            trade_flow_logger.info(
                f"[{self.symbol}] 錄得一次盈利，連續虧損計數已重置。"
            )
            self.consecutive_losses = 0

    def _load_models(self):
        """从文件加载预训练的机器学习模型。"""
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
        """使用凯利公式动态计算本次交易的风险百分比。"""
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
        return max(
            0.005, min(STRATEGY_PARAMS["max_risk_pct"], 0.5 * kelly)
        )  # 使用半凯利并设置上下限

    def get_ml_confidence_score(self):
        """获取机器学习模型的综合置信度评分。"""
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
        """用新的K线数据更新策略内部的DataFrame。"""
        self.data.loc[row.name] = row
        if not self.data.index.is_monotonic_increasing:
            self.data.sort_index(inplace=True)
        if len(self.data) > 5000:  # 内存管理，防止DataFrame无限增大
            self.data = self.data.iloc[-5000:]

    def compute_all_features(self):
        """计算所有需要的技术指标和特征。"""
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
        """计算趋势市的入场综合评分。"""
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
        """定义震荡市的入场信号。"""
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
            return 1  # 做多信号
        short_reentry = prev.Close > prev.mr_bb_upper and last.Close < last.mr_bb_upper
        stoch_short_confirm = (
            last.mr_stoch_rsi_k < last.mr_stoch_rsi_d
            and prev.mr_stoch_rsi_k >= prev.mr_stoch_rsi_d
            and last.mr_stoch_rsi_k > 60
        )
        if short_reentry and stoch_short_confirm:
            return -1  # 做空信号
        return 0  # 无信号

    def next_on_candle_close(self):
        """在每根K线收盘时调用，执行策略决策。"""
        self.compute_all_features()
        last = self.data.iloc[-1]

        if pd.isna(last.get("market_regime")):
            return None, 0.0  # 数据不足，无法判断

        market_status = "趨勢市" if last.market_regime == 1 else "震盪市"
        logging.info(
            f"[{self.symbol}] 市場狀態判斷: {market_status} (Regime Score: {last.get('regime_score', 0):.3f})"
        )

        action_to_take, score = None, 0.0

        # 根据不同的市场状态，采用不同的策略
        if last.market_regime == 1:  # 趋势市
            score = self._calculate_entry_score()
            if abs(score) > self.score_entry_threshold:
                is_long = score > 0
                if (is_long and self.position == "LONG") or (
                    not is_long and self.position == "SHORT"
                ):
                    return None, score  # 已经持有同向仓位
                risk_pct = self._calculate_dynamic_risk()
                position_size = self._determine_position_size(last.Close, risk_pct)
                if float(position_size) > 0:
                    action_to_take = {
                        "action": "BUY" if is_long else "SELL",
                        "size": position_size,
                    }
        else:  # 震荡市
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
                )  # 震荡市降低风险
                position_size = self._determine_position_size(last.Close, risk_pct)
                if float(position_size) > 0:
                    action_to_take = {
                        "action": "BUY" if is_long else "SELL",
                        "size": position_size,
                    }

        return action_to_take, score

    def _adjust_size_to_lot_size(self, size):
        """根据合约的最小下单量(lotSz)，调整仓位大小。"""
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
        """根据权益、风险百分比和价格，计算仓位大小。"""
        if price <= 0:
            return "0"
        notional = self.equity * risk_pct
        calculated_size = notional / price if price > 0 else 0
        if calculated_size <= 0:
            return "0"
        return self._adjust_size_to_lot_size(calculated_size)

    def register_trade_result(self, pnl_pct):
        """注册一笔交易的结果，用于后续的风险管理。"""
        self.recent_trade_returns.append(pnl_pct)
        if pnl_pct < 0:
            self.register_loss()
        else:
            self.register_win()

    def set_position(self, new_position):
        """更新策略内部的持仓状态。"""
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


# --- 辅助函数 ---
def manage_position_exit_orders(
    trader: OKXTrader,
    symbol: str,
    pos_data: dict,
    strategy: UltimateStrategy,
    exit_mode: str,
):
    """
    为一个已确认的仓位，集中管理其止损和止盈订单的设置。
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
    在持有仓位期间，管理移动止损的激活和调整。
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

    # --- 1. 激活邏輯：检查是否应启动移动止损 ---
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

    # --- 2. 調整邏輯：如果已激活，则跟踪价格调整止损 ---
    if is_active:
        atr = strategy.data["tf_atr"].iloc[-1]
        trailing_distance = atr * TSL_TRAILING_ATR_MULT
        new_stop_price = None

        if pos_side == "LONG":
            potential_stop = current_price - trailing_distance
            if potential_stop > current_stop_price:  # 只上移，不反向移动
                new_stop_price = potential_stop
        else:  # SHORT
            potential_stop = current_price + trailing_distance
            if potential_stop < current_stop_price:  # 只下移，不反向移动
                new_stop_price = potential_stop

        if new_stop_price is not None:
            trade_flow_logger.info(
                f"[{symbol}] 調整移動止損: 舊={current_stop_price:.4f}, 新={new_stop_price:.4f}"
            )
            pos_size = abs(float(pos_data.get("pos")))
            close_side = "sell" if pos_side == "LONG" else "buy"
            old_stop_id = trade_state.get("current_stop_id")

            # 安全操作：先下新单
            res = trader.place_trigger_order(
                symbol, close_side, pos_size, new_stop_price, "Trailing-Stop-Update"
            )
            if res and res.get("code") == "0" and res["data"][0]["sCode"] == "0":
                new_stop_id = res["data"][0]["algoId"]
                trade_flow_logger.info(
                    f"[{symbol}] 新移動止損單設置成功, Algo ID: {new_stop_id}"
                )

                # 新单成功后，再取消旧单
                if old_stop_id:
                    trader.cancel_algo_orders(symbol, [old_stop_id])
                    trade_flow_logger.info(
                        f"[{symbol}] 舊止損單 (ID: {old_stop_id}) 已取消。"
                    )

                # 更新并保存状态
                trade_state["current_stop_price"] = new_stop_price
                trade_state["current_stop_id"] = new_stop_id
                save_trade_state(trade_state)
            else:
                trade_flow_logger.error(f"[{symbol}] 調整移動止損失敗！響應: {res}")


# --- 主程序入口 ---
def main():
    global trade_flow_logger, trade_logger
    # 初始化日志记录器
    trade_flow_logger = setup_logging()
    position_logger = setup_csv_logger(
        "position_logger", "positions.csv", ["timestamp", "symbol", "position"]
    )
    trade_logger = setup_csv_logger(
        "trade_logger",
        "trades.csv",
        ["timestamp", "symbol", "action", "size", "response"],
    )

    logging.info("啟動 OKX REST API 輪詢交易程序 (V3.1.1 关键修复版)...")
    # 从环境变量加载 API Keys
    OKX_API_KEY = os.getenv("OKX_API_KEY")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET")
    OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE")
    if not all([OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE]):
        logging.error("请设置环境变量: OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE")
        return

    # 实例化交易接口
    trader = OKXTrader(
        OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE, simulated=SIMULATED
    )

    # 获取初始账户权益
    initial_equity = trader.fetch_account_balance("USDT")
    if initial_equity is None:
        logging.error("无法获取初始账户余额，将使用默认启动资金。")
        initial_equity = SIMULATED_EQUITY_START
    else:
        logging.info(f"查询成功，初始账户权益为: {initial_equity:.2f} USDT")

    # 为每个交易对初始化策略对象
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

        # 程序启动时检查是否已有仓位，如果有，则尝试接管
        initial_position_data = trader.fetch_current_position(symbol)
        clear_trade_state()  # 启动时总是清除旧状态
        if initial_position_data and initial_position_data is not False:
            trade_flow_logger.warning(f"[{symbol}] 检测到已有仓位，尝试智能接管...")
            pos_side = (
                "LONG" if float(initial_position_data.get("pos", "0")) > 0 else "SHORT"
            )
            strategy.set_position(pos_side)

            # 先取消所有旧的挂单
            for i in range(3):
                open_algos = trader.fetch_open_algo_orders(symbol)
                if not open_algos:
                    break
                trader.cancel_algo_orders(
                    symbol, [order["algoId"] for order in open_algos]
                )
                time.sleep(1)

            # 为已存在的仓位设置新的止损
            manage_position_exit_orders(
                trader, symbol, initial_position_data, strategy, "trailing_stop"
            )

        logging.info(
            f"策略 {symbol} 初始化成功，最新K線時間: {strategy.data.index[-1]}"
        )

    last_audit_time = datetime.utcnow()
    # --- 主循环开始 ---
    while True:
        try:
            # 全局熔断检查
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

                # 获取最新K线
                latest_candles_df = trader.fetch_history_klines(
                    symbol, bar=KLINE_INTERVAL, limit=2
                )
                if latest_candles_df is None or latest_candles_df.empty:
                    logging.warning(f"无法获取 {symbol} 的最新K线数据。")
                    continue

                latest_candle = latest_candles_df.iloc[-1]
                last_known_ts = strategy.data.index[-1]

                # 如果有新的K线生成
                if latest_candle.name > last_known_ts:
                    logging.info(
                        f"[{symbol}] 新K線 {latest_candle.name} close={latest_candle.Close:.4f} | 倉位: {strategy.position}"
                    )
                    strategy.update_with_candle(latest_candle)

                    # 如果当前无仓位，则检查入场信号
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

                            # 下市价单
                            res = trader.place_market_order(
                                symbol, side, action["size"]
                            )

                            # 检查下单请求是否成功
                            if (
                                res
                                and res.get("code") == "0"
                                and res.get("data")[0].get("sCode") == "0"
                            ):
                                pos_data = None
                                # 增加延迟和重试次数以确保交易所数据更新
                                for i in range(8):
                                    time.sleep(2.5)  # 延长等待时间
                                    pos_data = trader.fetch_current_position(symbol)
                                    if pos_data and pos_data is not False:
                                        break

                                # --- [!!! 核心修复逻辑开始 !!!] ---
                                # 验证仓位是否真实建立，并且成交数量是否符合预期
                                if pos_data:
                                    intended_size = float(action["size"])
                                    actual_size = abs(float(pos_data.get("pos", "0")))

                                    # 验证实际成交量是否达到预期的90%以上，以容忍轻微的滑点或未完全成交
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

                                        # 基于真实的仓位数据(pos_data)来设置止盈止损
                                        manage_position_exit_orders(
                                            trader,
                                            symbol,
                                            pos_data,
                                            strategy,
                                            exit_mode,
                                        )
                                    else:
                                        # 如果成交量严重不足，这是一个严重问题，必须阻止后续操作
                                        trade_flow_logger.critical(
                                            f"[{symbol}] [!!严重错误!!] 开仓验证失败！成交量严重不足！"
                                            f"意图开仓: {intended_size}, 实际成交: {actual_size}。"
                                            "策略状态将不被更新，以防止状态错乱。强烈建议检查并手动处理该笔残余仓位！"
                                        )
                                        # (可选高级功能) 在这里可以增加逻辑，立即市价平掉这个残余的小仓位，实现全自动修复。
                                else:
                                    logging.error(
                                        f"[{symbol}] 下单请求已发送，但在多次尝试后仍未能查询到仓位建立！"
                                    )
                                # --- [!!! 核心修复逻辑结束 !!!] ---
                            else:
                                logging.error(
                                    f"[{symbol}] 下单请求失败或被交易所拒绝，响应: {json.dumps(res)}"
                                )

                # 获取交易所的真实仓位数据
                real_position_data = trader.fetch_current_position(symbol)

                # 如果策略有仓位，且交易所也有仓位，则检查移动止损
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

                # 如果策略认为有仓位，但交易所实际没有仓位了，说明仓位被平掉了
                elif strategy.position is not None and real_position_data is None:
                    trade_flow_logger.info(f"[{symbol}] 檢測到倉位已被平倉。")
                    closed_trade_state = load_trade_state()
                    if closed_trade_state and "entry_price" in closed_trade_state:
                        entry_price = closed_trade_state["entry_price"]
                        # 注意：这里的exit_price是K线收盘价，是一个估算值，用于日志记录
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

                    # 重置状态
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

            # --- 周期性状态审计 ---
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

                    if real_position is False:  # 查询失败
                        logging.warning(f"[{symbol}] [审计跳过] 无法获取仓位。")
                        continue

                    # 检查状态不一致的情况
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
                            # 启动风控，紧急平仓
                            trade_flow_logger.critical(
                                f"[{symbol}] [审计发现] 状态严重不一致！策略: {strategy.position} vs 交易所: {real_pos_side}。"
                            )
                            trade_flow_logger.warning(
                                f"[{symbol}] 進入緊急風控模式：將清空所有倉位和掛單..."
                            )

                            # 1. 取消所有掛單
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

                            # 2. 市價平掉意外倉位
                            try:
                                pos_size = abs(float(real_position.get("pos", "0")))
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

                            # 3. 重置策略狀態
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
