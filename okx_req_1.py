# okx_live_trading_v4.9_defensive_cancel_fix_no_hurst.py
"""
OKX REST API 轮询版 (V4.9 - 防御性取消修复版 - 无Hurst版)
- 核心变更:
    - [!!! 终极安全修复 !!!] 针对API响应不可靠问题，实现了“防御性取消”机制。
        - 重写了 `cancel_algo_orders` 函数，不再信任API的单次成功返回。
        - 新逻辑是“发送取消指令 -> 等待 -> 循环验证”，程序会主动在15秒的窗口期内，多次查询交易所的未成交订单列表，直到确认目标订单真正消失为止。
        - 如果在超时后订单依然存在，函数将返回失败，并记录严重错误日志。
        - 此项修改将应用于所有取消操作（启动接管、移动止损、交易后清扫），从根本上保证程序状态与交易所后台状态的强一致性。
    - [移除] 移除了赫斯特指数(Hurst Exponent)相关的所有计算和依赖，简化了市场状态判断逻辑，并降低了计算复杂性。
    - [继承] 完整继承了 V4.8 的所有安全特性。此版本是为应对极端不可靠的API行为而设计的最终健壮版本。

调试说明:
- 主要观察点：当程序执行取消操作时（特别是启动接管），观察日志中新增的 "[取消验证]" 相关条目，确认其验证逻辑是否按预期执行。
"""

# --- 核心库导入 ---
# 【调试】系统和工具库
import os  # 文件系统操作，用于日志目录创建和状态文件管理
import time  # 时间控制，用于轮询间隔和重试延迟
import json  # JSON序列化，用于状态持久化和API通信
import hmac  # HMAC签名，用于OKX API认证
import base64  # Base64编码，用于API签名
import hashlib  # 哈希算法，用于API签名
import logging  # 日志系统，用于调试和监控
import math  # 数学运算，用于仓位大小计算
import csv  # CSV文件操作，用于交易记录
import urllib.parse  # URL解析，用于API请求构建

# 【调试】时间和数据结构
from datetime import datetime, timedelta  # 时间处理，用于时间戳和时间间隔计算
from logging.handlers import RotatingFileHandler  # 日志轮转，防止日志文件过大
from collections import deque  # 双端队列，用于Kelly公式历史记录管理

# 【调试】数据分析和机器学习
import pandas as pd  # 数据分析，用于K线数据处理和技术指标计算
import numpy as np  # 数值计算，用于数学运算和数组操作
import requests  # HTTP请求，用于OKX API调用
import ta  # 技术分析，用于计算各种技术指标
import warnings  # 警告控制，用于抑制不必要的警告信息

# 【调试】HTTP请求优化
from requests.models import PreparedRequest  # 请求预处理，用于URL构建
from requests.adapters import HTTPAdapter  # HTTP适配器，用于连接池管理
from urllib3.util.retry import Retry  # 重试机制，用于网络错误恢复

warnings.simplefilter(action="ignore", category=FutureWarning)

# 【调试】机器学习库检查 - 如果导入失败，ML功能将被禁用
try:
    import joblib  # 模型序列化，用于加载预训练的scaler和特征列
    import tensorflow as tf  # 深度学习，用于Keras模型推理

    ML_LIBS_INSTALLED = True
    print("✓ 机器学习库加载成功：tensorflow 和 joblib 可用")
except ImportError:
    ML_LIBS_INSTALLED = False
    print("⚠️ 警告: tensorflow 或 joblib 未安装，机器学习相关功能将不可用。")


# --- 日志系统设置 ---
# 【调试】主日志系统设置 - 用于记录程序运行状态和错误信息
def setup_logging():
    """
    设置双重日志系统：文件日志(DEBUG级别) + 控制台日志(INFO级别)
    调试要点：
    - 文件日志包含详细的调试信息，包括函数名和行号
    - 控制台日志只显示重要信息，避免刷屏
    - 使用轮转日志防止文件过大
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # 【调试】确保日志目录存在

    # 【调试】配置根日志器 - 所有日志的基础
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 设置最低级别为DEBUG
    if root_logger.hasHandlers():
        root_logger.handlers.clear()  # 清除已有处理器，避免重复日志

    # 【调试】文件日志处理器 - 记录详细调试信息
    log_file_path = os.path.join(log_dir, "trading_bot.log")
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )  # 10MB轮转，保留5个备份文件
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )  # 包含模块名和行号，便于定位问题
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 【调试】控制台日志处理器 - 显示重要信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 只显示INFO及以上级别
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )  # 简化格式，避免控制台信息过多
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 【调试】交易流程专用日志器 - 记录交易决策和执行过程
    trade_flow_logger = logging.getLogger("TradeFlow")
    trade_flow_logger.setLevel(logging.DEBUG)
    trade_flow_logger.propagate = False  # 不传播到根日志器，避免重复记录
    trade_flow_path = os.path.join(log_dir, "trades_flow.log")
    trade_flow_handler = RotatingFileHandler(
        trade_flow_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )  # 5MB轮转，专门记录交易流程
    trade_flow_handler.setLevel(logging.DEBUG)
    trade_flow_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    trade_flow_handler.setFormatter(trade_flow_formatter)
    trade_flow_logger.addHandler(trade_flow_handler)
    trade_flow_logger.addHandler(console_handler)  # 同时输出到控制台
    return trade_flow_logger


# 【调试】CSV格式日志器 - 用于结构化数据记录（交易记录、仓位变化等）
def setup_csv_logger(name, log_file, fields):
    """
    创建CSV格式的日志记录器，用于记录结构化数据
    调试要点：
    - 自动创建CSV头部
    - 使用字典格式记录，便于后续分析
    - 适用于交易记录、仓位变化等结构化数据
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, log_file)
    logger_obj = logging.getLogger(name)
    logger_obj.setLevel(logging.INFO)
    logger_obj.propagate = False  # 不传播到其他日志器

    if logger_obj.hasHandlers():
        return logger_obj  # 避免重复创建处理器

    handler = logging.FileHandler(csv_path, mode="a", encoding="utf-8")

    # 【调试】如果文件不存在或为空，创建CSV头部
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)  # 写入字段名作为头部

    # 【调试】自定义emit函数，将日志记录写入CSV格式
    def emit_csv(record):
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow(record.msg)  # record.msg应该是一个字典

    handler.emit = emit_csv
    logger_obj.addHandler(handler)
    return logger_obj


# --- 全局配置 ---
# 【调试】OKX API和交易配置
REST_BASE = "https://www.okx.com"  # OKX REST API基础URL
SYMBOLS = ["ETH-USDT-SWAP"]  # 交易标的列表，当前只交易ETH永续合约
KLINE_INTERVAL = "15m"  # K线周期，15分钟线用于策略计算
DESIRED_LEVERAGE = "1"  # 杠杆倍数，1倍杠杆（现货模式）
HISTORY_LIMIT = 500  # 历史K线获取数量，用于技术指标计算
POLL_INTERVAL_SECONDS = 10  # 主循环轮询间隔（秒）

# 【调试】模拟交易和风控配置
SIMULATED = True  # 是否启用模拟交易模式
SIMULATED_EQUITY_START = 500000.0  # 模拟交易初始资金（USDT）
MAX_DAILY_DRAWDOWN_PCT = 0.03  # 最大日回撤百分比（3%），触发熔断
MAX_CONSECUTIVE_LOSSES = 5  # 最大连续亏损次数，触发暂停交易
TRADING_PAUSE_HOURS = 4  # 连续亏损后暂停交易时间（小时）
AUDIT_INTERVAL_MINUTES = 15  # 审计检查间隔（分钟）
COOL_DOWN_PERIOD_SECONDS = 10  # 交易后冷静期（秒）

# --- Keras模型文件路径配置 ---
# 【调试】机器学习模型文件路径 - 用于AI信号生成
KERAS_MODEL_PATH = "models/eth_trend_model_v1_15m.keras"  # Keras模型文件
SCALER_PATH = "models/eth_trend_scaler_v1_15m.joblib"  # 特征缩放器
FEATURE_COLUMNS_PATH = "models/feature_columns_15m.joblib"  # 特征列定义
KERAS_SEQUENCE_LENGTH = 60  # 模型输入序列长度（60个时间步）

# --- 策略参数 ---
# 【调试】策略核心参数配置 - 所有策略行为的控制中心
STRATEGY_PARAMS = {
    # --- 核心风控参数 ---
    # 【调试】移动止损(TSL)配置 - 用于保护利润和控制风险
    "tsl_enabled": True,  # 是否启用移动止损
    "tsl_activation_profit_pct": 0.007,  # 移动止损激活利润阈值(0.7%) - 调试：过低会频繁激活
    "tsl_activation_atr_mult": 1.8,  # 移动止损激活距离(ATR倍数) - 调试：过小会过早激活
    "tsl_trailing_atr_mult": 2.2,  # 移动止损追踪距离(ATR倍数) - 调试：过小会过早止损
    # 【调试】Kelly公式风险管理 - 动态调整仓位大小
    "kelly_trade_history": 25,  # Kelly计算历史交易样本数 - 调试：样本过少会不稳定
    "default_risk_pct": 0.012,  # 默认风险百分比(1.2%) - 调试：基础仓位大小
    "max_risk_pct": 0.035,  # 最大风险百分比(3.5%) - 调试：风险上限保护
    # --- 回撤动态风险调整 ---
    # 【调试】回撤期间风险调整参数 - 在回撤时降低风险暴露
    "dd_grace_period_bars": 240,  # 回撤宽限期(K线数量)
    "dd_initial_pct": 0.35,  # 回撤初始风险调整比例
    "dd_final_pct": 0.25,  # 回撤最终风险调整比例
    "dd_decay_bars": 4320,  # 风险调整衰减周期
    # 【调试】逆势交易风险折扣 - 允许逆势但降低风险
    "counter_trend_risk_factor": 0.4,  # 逆势方向风险系数(0~1)，数值越小逆势仓位越轻
    # --- 市场状态检测参数 ---
    # 【调试】市场regime识别 - 区分趋势市场和震荡市场
    "regime_adx_period": 14,  # ADX周期 - 趋势强度指标
    "regime_atr_period": 14,  # ATR周期 - 波动率指标
    "regime_atr_slope_period": 6,  # ATR斜率计算周期 - 调试：波动率变化速度
    "regime_rsi_period": 14,  # RSI周期 - 超买超卖指标
    "regime_rsi_vol_period": 14,  # RSI波动率计算周期
    "regime_norm_period": 252,  # 归一化周期 - 调试：用于指标标准化
    # 【调试】市场regime评分权重 - 综合评估市场状态
    "regime_score_weight_adx": 0.60,  # ADX权重 - 趋势强度的重要性
    "regime_score_weight_atr": 0.25,  # ATR权重 - 波动率的重要性
    "regime_score_weight_rsi": 0.15,  # RSI权重 - 震荡识别的重要性
    "regime_score_threshold": 0.4,  # 趋势判定阈值 - 调试：过低会误判震荡为趋势
    # --- 宏观趋势过滤参数 ---
    # 【调试】4小时级别EMA用于宏观趋势过滤（相对15m为中期趋势）
    "macro_trend_ema_period_4h": 30,  # 4H EMA周期，默认30根4H K线（约5天），比50更偏中期
    # --- 趋势跟随(TF)模块参数 ---
    # 【调试】趋势跟随策略配置 - 用于捕捉趋势行情
    "tf_donchian_period": 24,  # 唐奇安通道周期 - 调试：突破信号敏感度
    "tf_ema_fast_period": 21,  # 快速EMA周期 - 调试：趋势确认速度
    "tf_ema_slow_period": 60,  # 慢速EMA周期 - 调试：趋势过滤强度
    "tf_adx_confirm_period": 14,  # ADX确认周期
    "tf_adx_confirm_threshold": 20,  # ADX确认阈值 - 调试：趋势强度要求
    "tf_chandelier_period": 22,  # 吊灯止损周期
    "tf_chandelier_atr_multiplier": 3.0,  # 吊灯止损ATR倍数 - 调试：出场信号敏感度
    "tf_atr_period": 14,  # ATR计算周期
    "tf_stop_loss_atr_multiplier": 2.6,  # 止损ATR倍数 - 调试：初始止损距离
    # --- 均值回归(MR)模块参数 ---
    # 【调试】均值回归策略配置 - 用于震荡行情
    "mr_bb_period": 20,  # 布林带周期
    "mr_bb_std": 2.0,  # 布林带标准差倍数
    "mr_stop_loss_atr_multiplier": 1.5,  # 均值回归止损ATR倍数 - 调试：比TF更紧
    "mr_risk_multiplier": 0.5,  # 均值回归风险倍数 - 调试：降低仓位
    # --- 多周期过滤参数 ---
    # 【调试】多时间框架分析 - 提高信号质量
    "mtf_period": 40,  # 多周期SMA周期 - 调试：日线趋势确认
    "score_entry_threshold": 0.5,  # 入场评分阈值 - 调试：信号质量要求
    # --- 信号权重配置 ---
    # 【调试】各类信号的重要性权重 - 综合信号评分
    "score_weights_tf": {
        "breakout": 0.22,  # 突破信号权重 - 调试：价格突破的重要性
        "momentum": 0.18,  # 动量信号权重 - 调试：趋势确认的重要性
        "mtf": 0.12,  # 多周期信号权重 - 调试：长期趋势的重要性
        "ml": 0.23,  # 机器学习信号权重 - 调试：AI预测的重要性
        "advanced_ml": 0.25,  # 高级ML信号权重 - 调试：复杂模型的重要性
    },
}

# 【调试】资产特定参数覆盖 - 针对不同交易对的个性化配置
ASSET_SPECIFIC_OVERRIDES = {
    "ETH-USDT-SWAP": {"score_entry_threshold": 0.45},  # ETH的入场阈值稍低
}


# --- 状态管理 & 辅助函数 ---
# 【调试】交易状态持久化 - 确保程序重启后能恢复交易状态
def save_trade_state(state):
    """
    保存当前交易状态到JSON文件
    调试要点：
    - 包含入场价格、止损价格、止损单ID等关键信息
    - 程序重启后可以恢复交易状态
    - 文件损坏时会导致状态丢失
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    state_file = os.path.join(log_dir, "trade_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=4)
    logging.debug(f"已保存交易状态: {state}")


def load_trade_state():
    """
    从JSON文件加载交易状态
    调试要点：
    - 返回None表示无活跃交易或文件损坏
    - 用于程序启动时恢复交易状态
    - JSON解析错误会返回None
    """
    log_dir = "logs"
    state_file = os.path.join(log_dir, "trade_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error("交易状态文件损坏，返回None")
            return None
    return None


def clear_trade_state():
    """
    清除交易状态文件
    调试要点：
    - 在交易结束时调用
    - 确保下次启动时不会误读旧状态
    """
    log_dir = "logs"
    state_file = os.path.join(log_dir, "trade_state.json")
    if os.path.exists(state_file):
        os.remove(state_file)
        logging.info("交易状态文件已清除。")


# 【调试】Kelly公式历史记录管理 - 用于动态风险计算
def save_kelly_history(symbol, history_deque):
    """
    保存Kelly公式计算用的历史交易记录
    调试要点：
    - 记录每笔交易的盈亏百分比
    - 用于计算胜率和盈亏比
    - 影响下次交易的仓位大小
    """
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
    """
    加载Kelly公式历史记录
    调试要点：
    - maxlen限制历史记录数量，避免过度拟合
    - 历史记录影响风险计算的准确性
    - 文件不存在时返回空队列
    """
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
# 【调试】高级模型推理 - 简化版ML信号生成
def run_advanced_model_inference(df):
    """
    运行高级模型推理，生成advanced_ml_signal
    调试要点：
    - 如果ML库未安装，返回0信号
    - 使用ai_filter_signal的24周期移动平均作为简化信号
    - 实际应用中可替换为复杂的ML模型
    """
    if not ML_LIBS_INSTALLED:
        df["advanced_ml_signal"] = 0.0  # 【调试】ML库未安装，返回中性信号
        return df

    # 【调试】使用简化的信号生成逻辑
    df["advanced_ml_signal"] = (
        df.get("ai_filter_signal", pd.Series(0, index=df.index))
        .rolling(24)  # 【调试】24周期平滑
        .mean()
        .fillna(0)
    )
    return df


# 【调试】机器学习特征工程 - 为ML模型准备标准化特征
def add_ml_features_ported(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加机器学习特征，用于市场regime识别
    调试要点：
    - 所有特征都经过归一化处理，范围[0,1]
    - norm函数使用滚动窗口归一化，避免未来数据泄露
    - 特征包括趋势、波动率、动量等多个维度
    """
    p = STRATEGY_PARAMS

    # 【调试】归一化函数 - 将指标标准化到[0,1]范围
    norm = lambda s: (
        (s - s.rolling(p["regime_norm_period"]).min())
        / (
            s.rolling(p["regime_norm_period"]).max()
            - s.rolling(p["regime_norm_period"]).min()
        )
    ).fillna(
        0.5
    )  # 【调试】无法计算时填充0.5（中性值）

    # 【调试】计算基础技术指标
    adx = ta.trend.ADXIndicator(df.High, df.Low, df.Close, p["regime_adx_period"]).adx()
    atr = ta.volatility.AverageTrueRange(
        df.High, df.Low, df.Close, p["regime_atr_period"]
    ).average_true_range()
    rsi = ta.momentum.RSIIndicator(df.Close, p["regime_rsi_period"]).rsi()
    bb = ta.volatility.BollingerBands(
        df.Close, window=p["mr_bb_period"], window_dev=p["mr_bb_std"]
    )

    # 【调试】生成归一化特征
    df["feature_adx_norm"] = norm(adx)  # 【调试】趋势强度特征
    df["feature_atr_slope_norm"] = norm(
        (atr - atr.shift(p["regime_atr_slope_period"]))
        / atr.shift(p["regime_atr_slope_period"])
    )  # 【调试】ATR变化率特征，衡量波动率变化
    df["feature_rsi_vol_norm"] = 1 - norm(
        rsi.rolling(p["regime_rsi_vol_period"]).std()
    )  # 【调试】RSI稳定性特征
    df["feature_obv_norm"] = norm(
        ta.volume.OnBalanceVolumeIndicator(df.Close, df.Volume).on_balance_volume()
    )  # 【调试】成交量特征
    df["feature_vol_pct_change_norm"] = norm(
        df.Volume.pct_change(periods=1).abs()
    )  # 【调试】成交量变化特征
    df["feature_bb_width_norm"] = norm(
        (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    )  # 【调试】布林带宽度特征，衡量波动率
    df["feature_atr_pct_change_norm"] = norm(
        atr.pct_change(periods=1)
    )  # 【调试】ATR变化特征

    # 【调试】综合regime评分 - 加权组合各个特征
    df["feature_regime_score"] = (
        df["feature_adx_norm"] * p["regime_score_weight_adx"]
        + df["feature_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["feature_rsi_vol_norm"] * p["regime_score_weight_rsi"]
    )
    return df


# 【调试】市场regime特征 - 识别市场状态（趋势vs震荡）
def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加市场regime识别特征
    调试要点：
    - regime_score > threshold 为趋势市场，否则为震荡市场
    - volatility_regime 将波动率分为三档：低、中、高
    - market_regime 为数值化的市场状态：1=趋势，-1=震荡
    """
    df["regime_score"] = df["feature_regime_score"]

    # 【调试】趋势vs震荡判定
    df["trend_regime"] = np.where(
        df["regime_score"] > STRATEGY_PARAMS["regime_score_threshold"],
        "趋势",  # 【调试】趋势市场
        "震荡",  # 【调试】震荡市场
    )

    # 【调试】波动率regime计算 - 年化波动率
    df["volatility"] = df["Close"].pct_change().rolling(24 * 7).std() * np.sqrt(
        24 * 365  # 【调试】年化处理
    )

    # 【调试】波动率分档 - 使用33%和67%分位数
    low_vol, high_vol = df["volatility"].quantile(0.33), df["volatility"].quantile(0.67)
    df["volatility_regime"] = pd.cut(
        df["volatility"],
        bins=[0, low_vol, high_vol, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    # 【调试】数值化市场regime - 便于后续计算
    df["market_regime"] = np.where(df["trend_regime"] == "趋势", 1, -1)
    return df


# 【调试】Keras模型特征 - 为深度学习模型准备特征
def add_features_for_keras_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    为Keras模型添加技术指标特征
    调试要点：
    - 特征名称必须与训练时一致
    - 所有特征都是常用的技术指标
    - ATRr_14是相对ATR（百分比形式）
    """
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

    # 【调试】趋势指标
    df["EMA_8"] = ta.trend.EMAIndicator(close=close, window=8).ema_indicator()

    # 【调试】动量指标
    df["RSI_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    # 【调试】ADX系列指标 - 趋势强度和方向
    adx_indicator = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = (
        adx_indicator.adx(),  # 【调试】趋势强度
        adx_indicator.adx_pos(),  # 【调试】正向动量
        adx_indicator.adx_neg(),  # 【调试】负向动量
    )

    # 【调试】波动率指标 - 相对ATR
    atr_raw = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()
    df["ATRr_14"] = (atr_raw / close) * 100  # 【调试】转换为百分比形式

    # 【调试】布林带指标
    bb_indicator = ta.volatility.BollingerBands(close=close, window=20, window_dev=2.0)
    df["BBU_20_2.0"], df["BBM_20_2.0"], df["BBL_20_2.0"] = (
        bb_indicator.bollinger_hband(),  # 【调试】上轨
        bb_indicator.bollinger_mavg(),  # 【调试】中轨
        bb_indicator.bollinger_lband(),  # 【调试】下轨
    )
    df["BBB_20_2.0"], df["BBP_20_2.0"] = (
        bb_indicator.bollinger_wband(),  # 【调试】带宽
        bb_indicator.bollinger_pband(),  # 【调试】位置百分比
    )

    # 【调试】MACD指标
    macd_indicator = ta.trend.MACD(
        close=close, window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = (
        macd_indicator.macd(),  # 【调试】MACD线
        macd_indicator.macd_signal(),  # 【调试】信号线
        macd_indicator.macd_diff(),  # 【调试】柱状图
    )

    # 【调试】成交量指标
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
        close=close, volume=volume
    ).on_balance_volume()  # 【调试】能量潮指标
    df["volume_change_rate"] = volume.pct_change()  # 【调试】成交量变化率

    return df


# --- OKX 交易接口类 ---
# 【调试】OKX交易接口封装 - 处理所有与OKX API的交互
class OKXTrader:
    """
    OKX交易接口类，封装所有API调用
    调试要点：
    - 所有API请求都经过签名验证
    - 支持模拟交易和实盘交易
    - 内置重试机制和错误处理
    - 缓存合约信息以提高性能
    """

    def __init__(self, api_key, api_secret, passphrase, simulated=True):
        """
        初始化OKX交易接口
        调试要点：
        - simulated=True时启用模拟交易
        - 设置HTTP重试策略，处理网络错误
        - 缓存合约信息，避免重复查询
        """
        self.base, self.api_key, self.api_secret, self.passphrase, self.simulated = (
            REST_BASE,
            api_key,
            api_secret,
            passphrase,
            simulated,
        )
        self.instrument_info = {}  # 【调试】合约信息缓存
        self.common_headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
        }
        if self.simulated:
            self.common_headers["x-simulated-trading"] = "1"  # 【调试】模拟交易标识

        # 【调试】配置HTTP会话和重试策略
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],  # 【调试】服务器错误时重试
            allowed_methods=["GET", "POST"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _now(self):
        """
        获取当前UTC时间戳
        调试要点：
        - 返回ISO格式时间戳，用于API签名
        - 时间精度到毫秒
        """
        return datetime.utcnow().isoformat("T", "milliseconds") + "Z"

    def _sign(self, ts, method, path, body_str=""):
        """
        生成API签名
        调试要点：
        - 使用HMAC-SHA256算法
        - 签名字符串格式：timestamp + method + path + body
        - 签名错误会导致API调用失败
        """
        message = f"{ts}{method}{path}{body_str}"
        mac = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _request(self, method, path, body=None, params=None, max_retries=7):
        """
        通用API请求方法
        调试要点：
        - 自动处理签名和认证
        - 内置重试机制，指数退避
        - 特殊处理404错误（订单已不存在）
        - 超时设置：连接5秒，读取15秒
        """
        ts = self._now()
        body_str = "" if body is None else json.dumps(body)

        # 【调试】构建完整URL用于签名
        prepped = PreparedRequest()
        prepped.prepare_url(self.base + path, params)
        path_for_signing = urllib.parse.urlparse(prepped.url).path
        if query := urllib.parse.urlparse(prepped.url).query:
            path_for_signing += "?" + query
        path_for_signing = path_for_signing.replace(self.base, "")

        # 【调试】生成签名和请求头
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
                    timeout=(5, 15),  # 【调试】连接超时5秒，读取超时15秒
                )

                # 【调试】特殊处理：取消不存在的订单返回成功
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
                wait_time = min(wait_time * 2, 30)  # 【调试】指数退避，最大30秒
        return {"code": "-1", "msg": "Max retries exceeded with network errors"}

    def place_trigger_order(
        self, instId, side, sz, trigger_price, order_type, posSide="net"
    ):
        """
        下触发单（止损单、移动止损等）
        调试要点：
        - trigger_price是触发价格
        - orderPx="-1"表示市价单
        - 返回algoId用于后续管理
        """
        path, price_str = "/api/v5/trade/order-algo", f"{trigger_price:.8f}".rstrip(
            "0"
        ).rstrip(
            "."
        )  # 【调试】格式化价格，去除多余的0
        body = {
            "instId": instId,
            "tdMode": "cross",  # 【调试】全仓模式
            "side": side,
            "posSide": posSide,
            "ordType": "trigger",  # 【调试】触发单类型
            "sz": str(sz),
            "triggerPx": price_str,
            "orderPx": "-1",  # 【调试】-1表示市价单
        }
        return self._request("POST", path, body=body)

    def fetch_open_algo_orders(self, instId):
        path = "/api/v5/trade/orders-algo-pending"
        ord_types = ["trigger", "conditional", "oco"]
        all_orders = []

        for t in ord_types:
            params = {
                "instType": "SWAP",
                "instId": instId,
                "ordType": t,
            }
            res = self._request("GET", path, params=params)
            if res and res.get("code") == "0" and res.get("data"):
                all_orders.extend(res["data"])

        return all_orders

    def cancel_algo_orders(self, instId, algoIds):
        if not algoIds:
            return True

        # 1. 统一转字符串
        algoIds = [str(aid) for aid in algoIds]

        # 2. 发送取消（保留原始实现）
        path, body = "/api/v5/trade/cancel-algos", [
            {"instId": instId, "algoId": aid} for aid in algoIds
        ]
        res = self._request("POST", path, body=body)

        # 3. 验证循环 – 关键改进
        max_wait = 30  # 延长到 30 s（可配置）
        check_interval = 3  # 每 3 s 查询一次，降低频率压力
        start = time.time()
        remaining = set(algoIds)

        while time.time() - start < max_wait and remaining:
            trade_flow_logger.info(
                f"[{instId}] [取消验证] 剩余 {len(remaining)} 个订单待确认消失..."
            )
            pending = self.fetch_open_algo_orders(instId)
            if pending is None:
                time.sleep(check_interval)
                continue

            # ✅ 只保留仍在 active 状态的订单
            pending_ids = {o["algoId"] for o in pending if o.get("state") == "live"}

            remaining = remaining.intersection(pending_ids)

            if not remaining:
                trade_flow_logger.info(f"[{instId}] [取消验证] 全部订单已确认消失")
                return True

            time.sleep(check_interval)

        # 4. 超时后仍未消失 → 记录并返回 False（触发上层安全退出）
        trade_flow_logger.critical(
            f"[{instId}] [取消验证] 超时！仍存订单: {list(remaining)}"
        )
        return False

    def set_leverage(self, instId, lever, mgnMode="cross", posSide=None):
        """
        设置杠杆倍数
        调试要点：
        - 必须在开仓前设置
        - cross模式为全仓，isolated为逐仓
        - 返回True表示设置成功
        """
        path, body = "/api/v5/account/set-leverage", {
            "instId": instId,
            "lever": str(lever),
            "mgnMode": mgnMode,  # 【调试】保证金模式
        }
        if posSide:
            body["posSide"] = posSide
        return self._request("POST", path, body=body).get("code") == "0"

    def fetch_account_balance(self, ccy="USDT"):
        """
        查询账户余额
        调试要点：
        - 返回指定币种的权益余额
        - 用于风险管理和仓位计算
        - 返回None表示查询失败
        """
        path, params = "/api/v5/account/balance", {"ccy": ccy}
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            if (data := res.get("data", [])) and "details" in data[0]:
                for detail in data[0]["details"]:
                    if detail.get("ccy") == ccy:
                        return float(detail.get("eq", 0))  # 【调试】eq为权益余额
        return None

    def fetch_current_position(self, instId):
        """
        查询当前仓位
        调试要点：
        - 返回None表示无仓位
        - 返回False表示查询失败
        - pos字段为仓位大小，正数为多头，负数为空头
        """
        path, params = "/api/v5/account/positions", {"instId": instId}
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            for pos_data in res.get("data", []):
                if float(pos_data.get("pos", "0")) != 0:  # 【调试】过滤零仓位
                    return pos_data
            return None  # 【调试】无仓位
        logging.error(
            f"查询仓位失败 for {instId}: {res.get('msg') if res else 'Unknown error'}"
        )
        return False  # 【调试】查询失败

    def fetch_instrument_details(self, instId, instType="SWAP"):
        """
        查询合约详细信息
        调试要点：
        - 缓存结果，避免重复查询
        - 包含lotSz（最小下单量）、ctVal（合约面值）等信息
        - 用于仓位大小计算和订单格式化
        """
        if instId in self.instrument_info:
            return self.instrument_info[instId]  # 【调试】返回缓存结果

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
            self.instrument_info[instId] = instrument_data  # 【调试】缓存结果
            return instrument_data
        return None

    def place_market_order(self, instId, side, sz):
        """
        下市价单
        调试要点：
        - 立即成交，用于开仓和平仓
        - 自动记录到交易日志
        - 返回订单信息，包含ordId
        """
        path, body = "/api/v5/trade/order", {
            "instId": instId,
            "tdMode": "cross",  # 【调试】全仓模式
            "side": side,
            "ordType": "market",  # 【调试】市价单
            "sz": str(sz),
        }
        if res := self._request("POST", path, body=body):
            # 【调试】记录交易到CSV日志
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
        """
        获取历史K线数据
        调试要点：
        - 返回pandas DataFrame格式
        - 时间戳已转换为datetime索引
        - 数据按时间排序，最新的在最后
        - 返回None表示获取失败
        """
        path, params = "/api/v5/market/history-candles", {
            "instId": instId,
            "bar": bar,  # 【调试】K线周期：1m, 5m, 15m, 1H, 1D等
            "limit": limit,
        }
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            if not (data := res.get("data", [])):
                return pd.DataFrame()  # 【调试】无数据返回空DataFrame

            # 【调试】构建DataFrame
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

            # 【调试】时间戳转换和列重命名
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

            # 【调试】数据类型转换
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col])

            return (
                df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
                .set_index("timestamp")
                .sort_index()  # 【调试】按时间排序
            )
        return None

    def fetch_ticker_price(self, instId):
        """
        获取最新价格
        调试要点：
        - 返回最新成交价
        - 用于实时价格监控和止损计算
        - 返回None表示获取失败
        """
        path, params = "/api/v5/market/ticker", {"instId": instId}
        if (res := self._request("GET", path, params=params)) and res.get(
            "code"
        ) == "0":
            if data := res.get("data", []):
                return float(data[0].get("last"))  # 【调试】last为最新成交价
        return None


# --- 策略核心类 ---
# 【调试】终极策略类 - 集成多种策略和风险管理
class UltimateStrategy:
    """
    终极交易策略类，集成趋势跟随、均值回归、机器学习等多种策略
    调试要点：
    - 动态风险管理基于Kelly公式
    - 支持多种退出机制：移动止损、吊灯止损、策略信号
    - 集成机器学习模型进行信号增强
    - 完整的状态管理和持久化
    """

    def __init__(
        self, df, symbol, trader=None, pos_logger=None, instrument_details=None
    ):
        """
        初始化策略实例
        调试要点：
        - 加载历史交易记录用于Kelly公式计算
        - 动态加载策略参数，支持资产特定覆盖
        - 尝试加载ML模型，失败时禁用ML功能
        """
        self.symbol, self.trader, self.data, self.position = (
            symbol,
            trader,
            df.copy(),  # 【调试】复制数据，避免修改原始数据
            None,  # 【调试】当前仓位状态：None/LONG/SHORT
        )
        self.pos_logger, self.instrument_details = pos_logger, instrument_details
        logging.info(f"[{self.symbol}] 策略初始化...")

        # 【调试】动态加载策略参数
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)

        # 【调试】应用资产特定参数覆盖
        asset_overrides = ASSET_SPECIFIC_OVERRIDES.get(symbol, {})
        self.score_entry_threshold = asset_overrides.get(
            "score_entry_threshold", self.score_entry_threshold
        )

        # 【调试】加载Kelly公式历史记录
        self.recent_trade_returns = load_kelly_history(
            self.symbol, self.kelly_trade_history
        )

        # 【调试】初始化ML组件
        self.keras_model, self.scaler, self.feature_columns = None, None, None
        self.equity, self.consecutive_losses, self.trading_paused_until = (
            SIMULATED_EQUITY_START,
            0,
            None,
        )
        self._load_models()

    @property
    def is_trading_paused(self):
        """
        检查是否因连续亏损而暂停交易
        调试要点：
        - 连续亏损达到阈值时自动暂停
        - 暂停期过后自动恢复
        - 用于风险控制
        """
        if self.trading_paused_until and datetime.utcnow() < self.trading_paused_until:
            return True
        self.trading_paused_until = None  # 【调试】暂停期结束，清除标记
        return False

    def register_loss(self):
        """
        注册亏损交易
        调试要点：
        - 累计连续亏损次数
        - 达到阈值时触发交易暂停
        - 用于保护资金安全
        """
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
        """
        注册盈利交易
        调试要点：
        - 重置连续亏损计数
        - 恢复正常交易状态
        """
        if self.consecutive_losses > 0:
            trade_flow_logger.info(f"[{self.symbol}] 录得盈利，连亏计数重置。")
            self.consecutive_losses = 0

    def _load_models(self):
        """
        加载机器学习模型组件
        调试要点：
        - 依次加载Keras模型、Scaler、特征列定义
        - 任何组件加载失败都会禁用ML功能
        - 模型路径在全局配置中定义
        """
        if not ML_LIBS_INSTALLED:
            return  # 【调试】ML库未安装，跳过加载

        try:
            # 【调试】加载Keras模型
            if not os.path.exists(KERAS_MODEL_PATH):
                raise FileNotFoundError(f"Keras模型未找到: {KERAS_MODEL_PATH}")
            self.keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)

            # 【调试】加载特征缩放器
            if not os.path.exists(SCALER_PATH):
                raise FileNotFoundError(f"Scaler未找到: {SCALER_PATH}")
            self.scaler = joblib.load(SCALER_PATH)

            # 【调试】加载特征列定义
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
        """
        基于Kelly公式计算动态风险百分比
        调试要点：
        - 需要足够的历史交易记录
        - 计算胜率、平均盈利、平均亏损
        - Kelly值乘以0.5作为保守调整
        - 限制在最小和最大风险范围内
        """
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct  # 【调试】历史记录不足，使用默认值

        # 【调试】分离盈利和亏损交易
        wins, losses = [r for r in self.recent_trade_returns if r > 0], [
            r for r in self.recent_trade_returns if r < 0
        ]

        if not wins or not losses:
            return self.default_risk_pct  # 【调试】缺少盈利或亏损记录

        # 【调试】计算Kelly公式所需参数
        win_rate, avg_win, avg_loss = (
            len(wins) / len(self.recent_trade_returns),  # 胜率
            sum(wins) / len(wins),  # 平均盈利
            abs(sum(losses) / len(losses)),  # 平均亏损（绝对值）
        )

        if avg_loss == 0 or (reward_ratio := avg_win / avg_loss) == 0:
            return self.default_risk_pct  # 【调试】避免除零错误

        # 【调试】Kelly公式：f = p - q/b，其中p=胜率，q=败率，b=盈亏比
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(
            max(0.005, kelly * 0.5), self.max_risk_pct
        )  # 【调试】保守调整并限制范围

    def get_ml_confidence_score(self):
        """
        获取机器学习模型的置信度评分
        调试要点：
        - 检查模型组件完整性
        - 确保有足够的序列长度
        - 处理缺失特征的情况
        - 将模型输出转换为[-1, 1]范围
        """
        if not all([self.keras_model, self.scaler, self.feature_columns]):
            return 0.0  # 【调试】模型组件不完整

        if len(self.data) < KERAS_SEQUENCE_LENGTH:
            return 0.0  # 【调试】数据长度不足

        # 【调试】检查特征完整性
        if missing_cols := [
            col for col in self.feature_columns if col not in self.data.columns
        ]:
            logging.warning(f"缺少模型特征: {missing_cols}，ML评分为0。")
            return 0.0

        try:
            # 【调试】准备模型输入序列
            latest_sequence_df = (
                self.data[self.feature_columns]
                .iloc[-KERAS_SEQUENCE_LENGTH:]  # 取最后N个时间步
                .copy()
                .fillna(method="ffill")  # 前向填充
                .fillna(0)  # 剩余NaN填充为0
            )

            # 【调试】特征缩放和模型预测
            scaled_sequence = self.scaler.transform(latest_sequence_df)
            input_for_model = np.expand_dims(scaled_sequence, axis=0)  # 添加batch维度
            prediction = self.keras_model.predict(input_for_model, verbose=0)

            # 【调试】将[0,1]输出转换为[-1,1]范围
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
            df_4h["Close"], window=self.macro_trend_ema_period_4h
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

        # <<< [核心修复] START >>>
        # 在填充NaN之前，将所有可能由“除以零”等运算产生的无穷大值(inf)替换为NaN
        # 这样它们就能被后续的填充逻辑统一处理，防止模型预测时出错
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # <<< [核心修复] END >>>

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
        # 1) 先仅根据市场regime和信号确定方向与子策略，不再用macro_trend_filter做硬性方向限制
        if last.market_regime == 1:
            # 趋势市：使用趋势跟随评分决定多空
            score = self._calculate_entry_score()
            if score > self.score_entry_threshold:
                action_to_take = {
                    "action": "BUY",
                    "sub_strategy": "TF",
                    "confidence": score,
                }
            elif score < -self.score_entry_threshold:
                action_to_take = {
                    "action": "SELL",
                    "sub_strategy": "TF",
                    "confidence": abs(score),
                }
        else:
            # 震荡市：使用均值回归信号决定多空
            mr_signal = self._define_mr_entry_signal()
            if mr_signal == 1:
                action_to_take = {
                    "action": "BUY",
                    "sub_strategy": "MR",
                    "confidence": 1.0,
                }
            elif mr_signal == -1:
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
            # 2) 使用4H宏观趋势仅对逆势方向做风险折扣，而不是禁止逆势交易
            macro_dir = last.macro_trend_filter
            trade_dir = 1 if action_to_take["action"] == "BUY" else -1
            if macro_dir in (1, -1) and macro_dir != trade_dir:
                risk_pct *= getattr(self, "counter_trend_risk_factor", 0.4)
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

            # --- V4.8 核心修复：先取消，后下单 ---

            # 步骤 1: 取消旧的止损单
            if old_stop_id:
                trade_flow_logger.info(
                    f"[{symbol}] [TSL] 步骤1: 正在取消旧止损单 ID: {old_stop_id}..."
                )
                cancel_success = trader.cancel_algo_orders(symbol, [old_stop_id])
                if not cancel_success:
                    trade_flow_logger.error(
                        f"[{symbol}] [TSL] 严重警告：取消旧止损单失败！本次移动操作中止，以避免产生幽灵仓位。"
                    )
                    return  # 中止本次移动，等待下一次轮询

                time.sleep(0.5)  # 可选增加一个短暂延时，给交易所后台多一点处理时间
                trade_flow_logger.info(f"[{symbol}] [TSL] 步骤1完成: 旧止损单已取消。")

            # 步骤 2: 下达新的止损单
            trade_flow_logger.info(
                f"[{symbol}] [TSL] 步骤2: 正在设置新止损单，价格: {new_stop_price:.4f}..."
            )
            res = trader.place_trigger_order(
                symbol,
                close_side,
                pos_size,
                new_stop_price,
                "Trailing-Stop-Update",
            )

            if res and res.get("code") == "0" and res["data"][0]["sCode"] == "0":
                new_stop_id = res["data"][0]["algoId"]
                trade_flow_logger.info(
                    f"[{symbol}] [TSL] 步骤2完成: 新移动止损单设置成功, Algo ID: {new_stop_id}"
                )

                # 步骤 3: 更新本地状态
                trade_state["current_stop_price"], trade_state["current_stop_id"] = (
                    new_stop_price,
                    new_stop_id,
                )
                save_trade_state(trade_state)
            else:
                # 这是一个新的风险点：如果取消成功但下单失败，仓位会短暂失去保护
                trade_flow_logger.critical(
                    f"[{symbol}] [TSL] !!! 紧急情况 !!! 旧止损已取消，但新止损下单失败！响应: {res}。仓位暂时无保护，将在下一轮循环中尝试重新设置止损！"
                )
                # 清除本地ID，以便下次循环能重新设置
                trade_state["current_stop_id"] = None
                save_trade_state(trade_state)


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
    logging.info(
        "启动 OKX REST API 轮询交易程序 (V4.9 防御性取消修复版 - 无Hurst版)..."
    )
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

            # --- V4.9 核心应用点 ---
            if open_algos := trader.fetch_open_algo_orders(symbol):
                if not trader.cancel_algo_orders(
                    symbol, [order["algoId"] for order in open_algos]
                ):
                    trade_flow_logger.critical(
                        f"[{symbol}] [接管失败] 无法在启动时清除旧的算法订单！为保证安全，程序将退出。"
                    )
                    return  # 关键安全检查：如果无法清理环境，则不继续

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
                        if not trader.cancel_algo_orders(
                            symbol, [order["algoId"] for order in open_algos]
                        ):
                            trade_flow_logger.critical(
                                f"[{symbol}] [紧急接管失败] 无法清除旧的算法订单！为保证安全，程序将暂停对该交易对的操作。"
                            )
                            is_globally_paused = True
                            continue

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
                                current_price,
                                current_position_data,
                            )
                        else:
                            logging.warning(f"[{symbol}] 无法获取价格，跳过退出检查。")
                elif strategy.position is not None:
                    trade_flow_logger.info(f"[{symbol}] 检测到仓位已平仓。")
                    strategy.cool_down_until = datetime.utcnow() + timedelta(
                        seconds=COOL_DOWN_PERIOD_SECONDS
                    )
                    trade_flow_logger.warning(
                        f"[{symbol}] 交易结束，启动 {COOL_DOWN_PERIOD_SECONDS} 秒冷静期。"
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

                    # --- V4.8 辅助修复：交易后全面清扫 ---
                    trade_flow_logger.info(
                        f"[{symbol}] [清扫] 交易已结束，开始全面清扫所有未成交的条件单..."
                    )
                    if open_algos := trader.fetch_open_algo_orders(symbol):
                        if not trader.cancel_algo_orders(
                            symbol, [o["algoId"] for o in open_algos]
                        ):
                            trade_flow_logger.critical(
                                f"[{symbol}] [清扫失败] 交易结束后未能清除所有残留订单！请手动检查！"
                            )
                    else:
                        trade_flow_logger.info(
                            f"[{symbol}] [清扫] 无残留条件单，环境干净。"
                        )
                    # --- 清扫结束 ---

                    strategy.set_position(None)
                    clear_trade_state()
                    trade_flow_logger.info(f"[{symbol}] 策略状态已重置。")

                current_price = trader.fetch_ticker_price(symbol)
                price_str = f"{current_price:.4f}" if current_price else "N/A"
                equity_str = f"{strategy.equity:.2f}" if strategy.equity else "N/A"
                pos_str = strategy.position if strategy.position is not None else "无"

                # <<< [核心修改] START >>>
                # 从策略数据中安全地获取最新的市场状态
                regime_str = "计算中..."
                if not strategy.data.empty and "trend_regime" in strategy.data.columns:
                    regime_str = strategy.data.iloc[-1].get("trend_regime", "计算中...")

                logging.info(
                    f"[{symbol}] 等待新K線... 仓位: {pos_str} | 市场: {regime_str} | 权益: {equity_str} | 价格: {price_str} | K線: {strategy.data.index[-1]}"
                )
                # <<< [核心修改] END >>>

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
