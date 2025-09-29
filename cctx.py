import requests
import pandas as pd
import numpy as np
from scipy import stats
import time
import json
import logging
import sys
import ta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# --- 配置类 ---
@dataclass
class TradingConfig:
    """交易策略配置"""
    # 基础配置
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    data_limit: int = 200  # 增加数据量以提高指标准确性
    loop_sleep_seconds: int = 30
    retry_attempts: int = 3
    retry_sleep_seconds: int = 5
    
    # 技术指标参数
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    volume_sma_period: int = 20
    
    # 风险管理参数
    stop_loss_pct: float = 0.02  # 2% 止损
    take_profit_pct: float = 0.04  # 4% 止盈
    max_position_size: float = 1.0  # 最大仓位
    risk_per_trade: float = 0.01  # 每笔交易风险1%
    
    # 信号过滤参数
    min_signal_strength: float = 0.6  # 最小信号强度
    trend_lookback: int = 50  # 趋势判断回看期
    volatility_threshold: float = 0.02  # 波动率阈值

class MarketState(Enum):
    """市场状态枚举"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"

class SignalType(Enum):
    """信号类型枚举"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

# --- 全局配置 ---
config = TradingConfig()

# --- 日志配置 ---
def setup_logger():
    """配置日志系统"""
    logger = logging.getLogger("AdvancedTradingStrategy")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # 文件处理器
    file_handler = logging.FileHandler("advanced_trading_log.log", mode="a", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter("%(asctime)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()

def get_binance_klines(
    symbol: str = config.symbol, 
    interval: str = config.interval, 
    limit: int = config.data_limit, 
    retries: int = config.retry_attempts
) -> pd.DataFrame:
    """获取币安K线数据"""
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for attempt in range(retries):
        try:
            response = requests.get(
                base_url + endpoint,
                params=params,
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ],
            )

            # 数据类型转换
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            return df[["open", "high", "low", "close", "volume"]]

        except requests.exceptions.RequestException as e:
            logger.warning(f"网络错误 (尝试 {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(config.retry_sleep_seconds * (attempt + 1))
            else:
                logger.error("多次尝试后网络请求失败")
                raise
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"数据解析错误: {e}")
            raise

def calculate_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算高级技术指标"""
    df_copy = df.copy()
    
    # 1. 趋势指标
    # MACD
    macd = ta.trend.MACD(
        close=df_copy["close"], 
        window_slow=config.macd_slow, 
        window_fast=config.macd_fast, 
        window_sign=config.macd_signal
    )
    df_copy["macd"] = macd.macd()
    df_copy["macd_signal"] = macd.macd_signal()
    df_copy["macd_histogram"] = macd.macd_diff()
    
    # EMA
    df_copy["ema_12"] = ta.trend.EMAIndicator(close=df_copy["close"], window=12).ema_indicator()
    df_copy["ema_26"] = ta.trend.EMAIndicator(close=df_copy["close"], window=26).ema_indicator()
    df_copy["ema_50"] = ta.trend.EMAIndicator(close=df_copy["close"], window=50).ema_indicator()
    
    # 2. 动量指标
    # RSI
    df_copy["rsi"] = ta.momentum.RSIIndicator(close=df_copy["close"], window=config.rsi_period).rsi()
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high=df_copy["high"], low=df_copy["low"], close=df_copy["close"]
    )
    df_copy["stoch_k"] = stoch.stoch()
    df_copy["stoch_d"] = stoch.stoch_signal()
    
    # Williams %R
    df_copy["williams_r"] = ta.momentum.WilliamsRIndicator(
        high=df_copy["high"], low=df_copy["low"], close=df_copy["close"]
    ).williams_r()
    
    # 3. 波动率指标
    # 布林带
    bb = ta.volatility.BollingerBands(close=df_copy["close"], window=config.bb_period, window_dev=config.bb_std)
    df_copy["bb_upper"] = bb.bollinger_hband()
    df_copy["bb_middle"] = bb.bollinger_mavg()
    df_copy["bb_lower"] = bb.bollinger_lband()
    df_copy["bb_width"] = (df_copy["bb_upper"] - df_copy["bb_lower"]) / df_copy["bb_middle"]
    df_copy["bb_position"] = (df_copy["close"] - df_copy["bb_lower"]) / (df_copy["bb_upper"] - df_copy["bb_lower"])
    
    # ATR (平均真实波幅)
    df_copy["atr"] = ta.volatility.AverageTrueRange(
        high=df_copy["high"], low=df_copy["low"], close=df_copy["close"]
    ).average_true_range()
    
    # 4. 成交量指标
    # 成交量移动平均 (使用简单移动平均)
    df_copy["volume_sma"] = df_copy["volume"].rolling(window=config.volume_sma_period).mean()
    
    # OBV (能量潮)
    df_copy["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df_copy["close"], volume=df_copy["volume"]
    ).on_balance_volume()
    
    # 5. 自定义指标
    # 价格动量
    df_copy["price_momentum"] = df_copy["close"].pct_change(periods=5)
    
    # 趋势强度
    df_copy["trend_strength"] = (df_copy["ema_12"] - df_copy["ema_26"]) / df_copy["close"]
    
    # 波动率
    df_copy["volatility"] = df_copy["close"].rolling(window=20).std() / df_copy["close"].rolling(window=20).mean()
    
    return df_copy

def identify_market_state(df: pd.DataFrame) -> MarketState:
    """识别市场状态"""
    latest_data = df.iloc[-config.trend_lookback:]
    
    # 计算趋势指标
    price_change = (latest_data["close"].iloc[-1] - latest_data["close"].iloc[0]) / latest_data["close"].iloc[0]
    volatility = latest_data["volatility"].iloc[-1]
    
    # 判断市场状态
    if volatility > config.volatility_threshold:
        return MarketState.HIGH_VOLATILITY
    elif price_change > 0.05:  # 5%以上涨幅
        return MarketState.TRENDING_UP
    elif price_change < -0.05:  # 5%以上跌幅
        return MarketState.TRENDING_DOWN
    else:
        return MarketState.SIDEWAYS

def calculate_signal_strength(df: pd.DataFrame) -> float:
    """计算信号强度 (0-1)"""
    latest = df.iloc[-1]
    
    # 各种信号权重
    weights = {
        'trend': 0.3,
        'momentum': 0.25,
        'volume': 0.2,
        'volatility': 0.15,
        'pattern': 0.1
    }
    
    scores = {}
    
    # 1. 趋势信号
    if latest["ema_12"] > latest["ema_26"] > latest["ema_50"]:
        scores['trend'] = 1.0
    elif latest["ema_12"] < latest["ema_26"] < latest["ema_50"]:
        scores['trend'] = -1.0
    else:
        scores['trend'] = 0.0
    
    # 2. 动量信号
    momentum_score = 0
    if latest["macd"] > latest["macd_signal"]:
        momentum_score += 0.4
    if latest["rsi"] > 30 and latest["rsi"] < 70:
        momentum_score += 0.3
    if latest["stoch_k"] > latest["stoch_d"]:
        momentum_score += 0.3
    scores['momentum'] = momentum_score
    
    # 3. 成交量信号
    volume_ratio = latest["volume"] / latest["volume_sma"]
    scores['volume'] = min(volume_ratio / 2, 1.0)  # 标准化到0-1
    
    # 4. 波动率信号
    scores['volatility'] = 1.0 - min(latest["volatility"] / config.volatility_threshold, 1.0)
    
    # 5. 形态信号
    bb_pos = latest["bb_position"]
    if 0.2 <= bb_pos <= 0.8:  # 在布林带中间区域
        scores['pattern'] = 1.0
    else:
        scores['pattern'] = 0.5
    
    # 计算加权总分
    total_score = sum(scores[key] * weights[key] for key in weights.keys())
    return max(0, min(1, total_score))  # 确保在0-1范围内

def generate_advanced_signals(df: pd.DataFrame) -> pd.DataFrame:
    """生成高级交易信号"""
    df_copy = df.copy()
    df_copy["signal"] = 0
    df_copy["signal_strength"] = 0.0
    df_copy["market_state"] = ""
    
    for i in range(len(df_copy)):
        if i < 50:  # 需要足够的历史数据
            continue
            
        current_df = df_copy.iloc[:i+1]
        latest = current_df.iloc[-1]
        
        # 识别市场状态
        market_state = identify_market_state(current_df)
        df_copy.iloc[i, df_copy.columns.get_loc("market_state")] = market_state.value
        
        # 计算信号强度
        signal_strength = calculate_signal_strength(current_df)
        df_copy.iloc[i, df_copy.columns.get_loc("signal_strength")] = signal_strength
        
        # 生成买入信号条件
        buy_conditions = [
            latest["macd"] > latest["macd_signal"],  # MACD金叉
            latest["rsi"] > 30 and latest["rsi"] < 70,  # RSI在合理区间
            latest["close"] > latest["ema_12"],  # 价格在短期均线上方
            latest["bb_position"] < 0.8,  # 不在布林带上轨附近
            latest["volume"] > latest["volume_sma"] * 1.2,  # 成交量放大
            signal_strength > config.min_signal_strength  # 信号强度足够
        ]
        
        # 生成卖出信号条件
        sell_conditions = [
            latest["macd"] < latest["macd_signal"],  # MACD死叉
            latest["rsi"] > 70 or latest["rsi"] < 30,  # RSI超买或超卖
            latest["close"] < latest["ema_12"],  # 价格在短期均线下方
            latest["bb_position"] > 0.2,  # 不在布林带下轨附近
            latest["volume"] > latest["volume_sma"] * 1.2,  # 成交量放大
            signal_strength > config.min_signal_strength  # 信号强度足够
        ]
        
        # 根据市场状态调整信号
        if market_state == MarketState.TRENDING_UP:
            if sum(buy_conditions) >= 4:
                df_copy.iloc[i, df_copy.columns.get_loc("signal")] = SignalType.STRONG_BUY.value if signal_strength > 0.8 else SignalType.BUY.value
        elif market_state == MarketState.TRENDING_DOWN:
            if sum(sell_conditions) >= 4:
                df_copy.iloc[i, df_copy.columns.get_loc("signal")] = SignalType.STRONG_SELL.value if signal_strength > 0.8 else SignalType.SELL.value
        elif market_state == MarketState.SIDEWAYS:
            # 震荡市场中更保守
            if sum(buy_conditions) >= 5:
                df_copy.iloc[i, df_copy.columns.get_loc("signal")] = SignalType.BUY.value
            elif sum(sell_conditions) >= 5:
                df_copy.iloc[i, df_copy.columns.get_loc("signal")] = SignalType.SELL.value
    
    return df_copy

def calculate_position_size(signal_strength: float, current_price: float, atr: float) -> float:
    """计算仓位大小"""
    # 基于ATR的风险调整仓位
    risk_amount = config.risk_per_trade
    stop_distance = atr * 2  # 使用2倍ATR作为止损距离
    
    if stop_distance > 0:
        position_size = (risk_amount * current_price) / stop_distance
        position_size *= signal_strength  # 根据信号强度调整
        return min(position_size, config.max_position_size)
    
    return 0.1  # 默认小仓位

def calculate_stop_loss_take_profit(entry_price: float, signal_type: SignalType, atr: float) -> Tuple[float, float]:
    """计算止损止盈价格"""
    if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
        # 买入信号：止损在下方，止盈在上方
        stop_loss = entry_price * (1 - config.stop_loss_pct)
        take_profit = entry_price * (1 + config.take_profit_pct)
        
        # 使用ATR动态调整
        atr_stop = entry_price - (atr * 2)
        atr_profit = entry_price + (atr * 3)
        
        # 选择更保守的止损和更激进的止盈
        stop_loss = max(stop_loss, atr_stop)  # 止损不要太远
        take_profit = max(take_profit, atr_profit)  # 止盈可以更远
        
    else:
        # 卖出信号：止损在上方，止盈在下方
        stop_loss = entry_price * (1 + config.stop_loss_pct)
        take_profit = entry_price * (1 - config.take_profit_pct)
        
        # 使用ATR动态调整
        atr_stop = entry_price + (atr * 2)
        atr_profit = entry_price - (atr * 3)
        
        # 选择更保守的止损和更激进的止盈
        stop_loss = min(stop_loss, atr_stop)  # 止损不要太远
        take_profit = min(take_profit, atr_profit)  # 止盈可以更远
    
    return stop_loss, take_profit

def run_advanced_strategy():
    """运行高级交易策略"""
    df = get_binance_klines()
    df = calculate_advanced_indicators(df)
    df = generate_advanced_signals(df)
    
    # 获取最新数据
    latest_data = df.iloc[-1]
    signal = int(latest_data["signal"])
    signal_strength = latest_data["signal_strength"]
    market_state = latest_data["market_state"]
    
    # 信号类型转换
    signal_type = SignalType(signal)
    signal_text = {
        SignalType.STRONG_BUY: "强烈买入",
        SignalType.BUY: "买入",
        SignalType.HOLD: "持有",
        SignalType.SELL: "卖出",
        SignalType.STRONG_SELL: "强烈卖出"
    }[signal_type]
    
    # 计算仓位和风险管理参数
    position_size = 0
    stop_loss = 0
    take_profit = 0
    
    if signal != 0:
        position_size = calculate_position_size(signal_strength, latest_data["close"], latest_data["atr"])
        stop_loss, take_profit = calculate_stop_loss_take_profit(
            latest_data["close"], signal_type, latest_data["atr"]
        )
    
    # 只在有买入或卖出信号时记录日志
    if signal != 0:  # 只记录买入或卖出信号
        logger.info(
            f"=== 交易信号 ===\n"
            f"时间: {latest_data.name}\n"
            f"价格: {latest_data['close']:.2f}\n"
            f"信号: {signal_text} (强度: {signal_strength:.2f})\n"
            f"建议仓位: {position_size:.2f}\n"
            f"止损价格: {stop_loss:.2f}\n"
            f"止盈价格: {take_profit:.2f}\n"
            f"风险回报比: {abs(take_profit-latest_data['close'])/abs(latest_data['close']-stop_loss):.2f}"
        )
    
    # 显示最近5个数据点的概览
    print("\n最近5个数据点概览:")
    display_columns = ["close", "signal", "signal_strength", "rsi", "macd", "bb_position", "market_state"]
    print(df[display_columns].tail().round(3))
    print("\n" + "=" * 80 + "\n")

def main():
    """主程序循环"""
    logger.info(
        f"启动高级交易策略监控 - {config.symbol} ({config.interval}间隔)"
    )
    
    while True:
        try:
            run_advanced_strategy()
            time.sleep(config.loop_sleep_seconds)
        except KeyboardInterrupt:
            logger.info("程序被手动中断，正在退出...")
            break
        except Exception as e:
            logger.error(f"主循环发生意外错误: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
