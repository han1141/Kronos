# -*- coding: utf-8 -*-
# 简化版：仅保留 15m 模型进行回测
# 变更概要：
# - 仅使用 15m V3 模型信号入场/离场
# - 移除 4 小时模型相关逻辑
# - 移除基础指标过滤（如 EMA/ADX/日线MTF过滤等）在策略决策中的参与

# --- 1. 导入库与配置 ---
# (此部分代码未变，保持原样)
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.font_manager
import joblib
import os
import glob
import warnings
from scipy.stats import linregress
from scipy.signal import find_peaks
import pandas_ta as pta

try:
    import numba

    jit = numba.jit(nopython=True, cache=True)
    NUMBA_INSTALLED = True
except ImportError:

    def jit(func):
        return func

    NUMBA_INSTALLED = False

try:
    import lightgbm as lgb

    ML_LIBS_INSTALLED = True
except ImportError:
    ML_LIBS_INSTALLED = False

try:
    import tensorflow as tf

    ADVANCED_ML_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_ML_LIBS_INSTALLED = False

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_filename = f"trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def set_chinese_font():
    try:
        font_names = [
            "PingFang SC",
            "Microsoft YaHei",
            "SimHei",
            "Heiti TC",
            "sans-serif",
        ]
        for font in font_names:
            if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
                plt.rcParams["font.sans-serif"] = [font]
                plt.rcParams["axes.unicode_minus"] = False
                logger.info(f"成功设置中文字体: {font}")
                return
        logger.warning("未找到指定的中文字体")
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")


set_chinese_font()

# --- 核心配置 ---
# (此部分代码未变，保持原样)
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],
    "interval": "15m",
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-11-07",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0003,
    "show_plots": False,
    "training_window_days": 365 * 1.5,
    "enable_ml_component": True,
    # 网格搜索阈值配置
    "threshold_search": {
        "enabled": False,            # 启用/禁用阈值网格搜索
        "metric": "sharpe",        # 评估指标: sharpe | return
        "grid": None,               # 自定义阈值列表(如 [0.30,0.35,0.40])；None 则使用默认网格
        "rerun_with_best": True,    # 是否用最佳阈值再跑一次并展示完整结果
    },
}

# --- 模型路径配置 ---
# (此部分代码未变，保持原样)
V3_ML_MODEL_15M_PATH = "models/eth_model_high_precision_v3_15m.joblib"
V3_ML_SCALER_15M_PATH = "models/eth_scaler_high_precision_v3_15m.joblib"
V3_ML_FEATURE_COLUMNS_15M_PATH = "models/feature_columns_high_precision_v3_15m.joblib"
V3_ML_FLATTENED_COLUMNS_15M_PATH = (
    "models/flattened_columns_high_precision_v3_15m.joblib"
)
V3_ML_THRESHOLD_15M = 0.3204
V3_ML_SEQUENCE_LENGTH_15M = 60


# --- 策略参数 ---
# 仅保留在当前策略中实际使用到的字段
STRATEGY_PARAMS = {
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
}
ASSET_SPECIFIC_OVERRIDES = {}


# --- 函数定义 ---
# (所有数据获取和特征工程函数保持不变)
def fetch_binance_klines(s, i, st, en=None, l=1000):
    url, cols = "https://api.binance.com/api/v3/klines", [
        "timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    sts, ets = int(pd.to_datetime(st).timestamp() * 1000), (
        int(pd.to_datetime(en).timestamp() * 1000) if en else int(time.time() * 1000)
    )
    all_d, retries, last_e = [], 5, None
    while sts < ets:
        p = {
            "symbol": s.upper(),
            "interval": i,
            "startTime": sts,
            "endTime": ets,
            "limit": l,
        }
        for attempt in range(retries):
            try:
                r = requests.get(url, params=p, timeout=15)
                r.raise_for_status()
                d = r.json()
                if not d:
                    sts = ets
                    break
                all_d.extend(d)
                sts = d[-1][0] + 1
                break
            except requests.exceptions.RequestException as e:
                last_e = e
                time.sleep(2**attempt)
        else:
            logger.error(f"获取 {s} 失败: {last_e}")
            return pd.DataFrame()
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d, columns=cols)[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


def get_market_structure_features(df, order=5):
    df = df.copy()
    high_peaks_idx, _ = find_peaks(
        df["High"], distance=order, prominence=df["High"].std() * 0.5
    )
    low_peaks_idx, _ = find_peaks(
        -df["Low"], distance=order, prominence=df["Low"].std() * 0.5
    )
    df["swing_high_price"], df["swing_low_price"] = np.nan, np.nan
    df.iloc[high_peaks_idx, df.columns.get_loc("swing_high_price")] = df.iloc[
        high_peaks_idx
    ]["High"]
    df.iloc[low_peaks_idx, df.columns.get_loc("swing_low_price")] = df.iloc[
        low_peaks_idx
    ]["Low"]
    df["swing_high_price"], df["swing_low_price"] = (
        df["swing_high_price"].ffill(),
        df["swing_low_price"].ffill(),
    )
    df["is_uptrend"] = (
        (df["swing_high_price"] > df["swing_high_price"].shift(1))
        & (df["swing_low_price"] > df["swing_low_price"].shift(1))
    ).astype(int)
    df["is_downtrend"] = (
        (df["swing_high_price"] < df["swing_high_price"].shift(1))
        & (df["swing_low_price"] < df["swing_low_price"].shift(1))
    ).astype(int)
    df["market_structure"] = df["is_uptrend"] - df["is_downtrend"]
    return df[["market_structure"]]


def feature_engineering_v3(df_in):
    df = df_in.copy()
    df["RSI_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    macd_indicator = ta.trend.MACD(
        close=df["Close"], window_fast=12, window_slow=26, window_sign=9
    )
    df["MACD_12_26_9"], df["MACDs_12_26_9"], df["MACDh_12_26_9"] = (
        macd_indicator.macd(),
        macd_indicator.macd_signal(),
        macd_indicator.macd_diff(),
    )
    bb_indicator = ta.volatility.BollingerBands(
        close=df["Close"], window=20, window_dev=2
    )
    (
        df["BBL_20_2.0"],
        df["BBM_20_2.0"],
        df["BBU_20_2.0"],
        df["BBB_20_2.0"],
        df["BBP_20_2.0"],
    ) = (
        bb_indicator.bollinger_lband(),
        bb_indicator.bollinger_mavg(),
        bb_indicator.bollinger_hband(),
        bb_indicator.bollinger_wband(),
        bb_indicator.bollinger_pband(),
    )
    adx_indicator = ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    )
    df["ADX_14"], df["DMP_14"], df["DMN_14"] = (
        adx_indicator.adx(),
        adx_indicator.adx_pos(),
        adx_indicator.adx_neg(),
    )
    df["volatility"] = (
        (np.log(df["Close"] / df["Close"].shift(1))).rolling(window=20).std()
    )
    market_structure_df = get_market_structure_features(df)
    macd_long_indicator = ta.trend.MACD(
        close=df["Close"], window_fast=24, window_slow=52, window_sign=18
    )
    df["MACD_long"], df["MACDh_long"], df["MACDs_long"] = (
        macd_long_indicator.macd(),
        macd_long_indicator.macd_diff(),
        macd_long_indicator.macd_signal(),
    )
    all_features_df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])
    all_features_df = pd.concat([all_features_df, market_structure_df], axis=1)
    all_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return all_features_df


def generate_v3_ml_predictions(
    df_with_ohlcv: pd.DataFrame,
    model_path: str,
    scaler_path: str,
    orig_cols_path: str,
    flat_cols_path: str,
    seq_len: int,
    log_prefix: str = "[V3 MODEL]",
) -> pd.Series:
    logger.info(f"--- {log_prefix} 开始生成ML预测 ---")
    if not all(
        os.path.exists(p)
        for p in [model_path, scaler_path, orig_cols_path, flat_cols_path]
    ):
        logger.warning(f"{log_prefix} 缺少模型文件，ML预测将为0。")
        return pd.Series(0, index=df_with_ohlcv.index)
    try:
        model, scaler, original_columns, flattened_columns = (
            joblib.load(model_path),
            joblib.load(scaler_path),
            joblib.load(orig_cols_path),
            joblib.load(flat_cols_path),
        )
        features_df = feature_engineering_v3(df_with_ohlcv).dropna()
        features_aligned = features_df.reindex(columns=original_columns, fill_value=0)
        scaled_features = scaler.transform(features_aligned)
        predictions = []
        for i in range(seq_len, len(scaled_features)):
            input_sequence = (
                scaled_features[i - seq_len : i, :].flatten().reshape(1, -1)
            )
            input_df = pd.DataFrame(input_sequence, columns=flattened_columns)
            pred_prob = model.predict_proba(input_df)[0][1]
            predictions.append(pred_prob)
        prediction_index = features_aligned.index[seq_len:]
        final_probs = pd.Series(predictions, index=prediction_index)
        logger.info(f"--- {log_prefix} ML预测生成完毕 ---")
        return final_probs.reindex(df_with_ohlcv.index, fill_value=0)
    except Exception as e:
        logger.error(f"{log_prefix} 生成ML预测时出错: {e}", exc_info=True)
        return pd.Series(0, index=df_with_ohlcv.index)




def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    logger.info(
        f"[{symbol}] 开始数据预处理 (数据范围: {df.index.min()} to {df.index.max()})..."
    )

    # 仅生成 15m 模型信号
    df["v3_ml_prob_15m"] = generate_v3_ml_predictions(
        df,
        V3_ML_MODEL_15M_PATH,
        V3_ML_SCALER_15M_PATH,
        V3_ML_FEATURE_COLUMNS_15M_PATH,
        V3_ML_FLATTENED_COLUMNS_15M_PATH,
        V3_ML_SEQUENCE_LENGTH_15M,
        log_prefix="[V3 MODEL 15M]",
    )
    df["v3_ml_signal_15m"] = (df["v3_ml_prob_15m"] > V3_ML_THRESHOLD_15M).astype(int)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。数据行数: {len(df)}")
    return df


TREND_CONFIG = {"look_forward_steps": 5, "ema_length": 8}


def analyze_v3_standalone_performance(df: pd.DataFrame, signal_col="v3_ml_signal_15m"):
    print(f"\n{'-'*40}\n       V3 高精度模型独立表现分析 ({signal_col}) \n{'-'*40}")
    required_cols = [signal_col, "Close"]
    if not all(col in df.columns for col in required_cols):
        print(f"缺少必要列 {signal_col}，无法分析。")
        return
    look_forward_steps, ema_length = (
        TREND_CONFIG["look_forward_steps"],
        TREND_CONFIG["ema_length"],
    )
    n = len(df)
    df_reset = df.reset_index(drop=True)
    df_reset[f"EMA_{ema_length}"] = pta.ema(close=df_reset["Close"], length=ema_length)
    macd_result = pta.macd(close=df_reset["Close"], fast=24, slow=52, signal=18)
    df_reset["MACD_long"], df_reset["MACDs_long"] = (
        macd_result["MACD_24_52_18"],
        macd_result["MACDs_24_52_18"],
    )
    valid_mask = (
        (df_reset.index <= n - look_forward_steps - 1)
        & (df_reset[signal_col] == 1)
        & (df_reset["MACD_long"] > df_reset["MACDs_long"])
        & (df_reset["MACD_long"] > 0)
    )
    trade_signals = df_reset[valid_mask].copy()
    if trade_signals.empty:
        print("无有效信号（可能因MACD过滤或边界限制）。")
        return
    total_trades = len(trade_signals)
    current_ema = trade_signals[f"EMA_{ema_length}"]
    future_ema = df_reset.loc[
        trade_signals.index + look_forward_steps, f"EMA_{ema_length}"
    ].values
    wins = (future_ema > current_ema).sum()
    win_rate = (wins / total_trades) * 100
    entry_price = trade_signals["Close"]
    exit_price = df_reset.loc[trade_signals.index + look_forward_steps, "Close"].values
    price_returns = (exit_price - entry_price) / entry_price
    avg_price_return, cum_price_return = (
        price_returns.mean() * 100,
        price_returns.sum() * 100,
    )
    print(f"有效信号总数（含MACD过滤，可观测{look_forward_steps}步）: {total_trades}")
    print(f"✅ 胜率（EMA趋势上涨，与训练目标一致）: {win_rate:.2f}%")
    print(f"📊 平均价格回报率（实际盈亏参考）: {avg_price_return:.4f}%")
    print(f"📈 累计价格回报率: {cum_price_return:.2f}%")
    print(f"{'-'*40}")


# --- 策略定义 ---
class UltimateStrategy(Strategy):
    symbol = None

    def init(self):
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)
        c, h, l = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )

        self.v3_ml_signal_15m = self.I(lambda: self.data.v3_ml_signal_15m)
        self.v3_ml_prob_15m = self.I(lambda: self.data.v3_ml_prob_15m)
        # 与训练/评估一致的长周期 MACD 过滤器 (24, 52, 18)
        self.macd_long = self.I(
            lambda: ta.trend.MACD(
                close=c, window_fast=24, window_slow=52, window_sign=18
            ).macd()
        )
        self.macds_long = self.I(
            lambda: ta.trend.MACD(
                close=c, window_fast=24, window_slow=52, window_sign=18
            ).macd_signal()
        )

    def next(self):
        price = self.data.Close[-1]
        current_bar = len(self.data) - 1

        # 入场条件与模型评估一致：阈值信号 + 长周期MACD过滤
        entry_signal = (
            (self.v3_ml_signal_15m[-1] > 0)
            and (self.macd_long[-1] > self.macds_long[-1])
            and (self.macd_long[-1] > 0)
        )
        exit_signal = self.v3_ml_signal_15m[-1] <= 0

        if not self.position:
            if entry_signal:
                self.open_dynamic_position(price, current_bar)
        elif self.position.is_long:
            if exit_signal:
                self.close_all_positions("15M信号反转")

    def get_confidence_factor(self, probability: float) -> float:
        if probability > 0.65:
            return 2.0
        elif probability > 0.55:
            return 1.5
        elif probability > 0.45:
            return 1.0
        else:
            return 0.5

    def open_dynamic_position(self, price: float, current_bar: int):
        probability = self.v3_ml_prob_15m[-1]
        confidence_factor = self.get_confidence_factor(probability)
        invest_pct = min(self.default_risk_pct * confidence_factor, self.max_risk_pct)
        size = int((self.equity * invest_pct) / price)
        if size > 0:
            self.buy(size=size)

    def close_all_positions(self, reason: str):
        """关闭所有仓位"""
        if self.position:
            self.position.close()

    def _calculate_position_size(self, price, risk_per_share, risk_pct):
        if risk_per_share <= 0 or price <= 0 or risk_pct <= 0:
            return 0
        return int((self.equity * risk_pct) / risk_per_share)


# --- 阈值网格搜索 ---
def _metric_from_stats(stats, metric: str):
    try:
        if metric == "sharpe":
            val = stats.get("Sharpe Ratio", np.nan)
            if val is None:
                val = np.nan
            return float(val)
        elif metric == "return":
            return float(stats.get("Return [%]", np.nan))
    except Exception:
        return np.nan
    return np.nan


def run_backtest_with_threshold(data: pd.DataFrame, symbol: str, threshold: float):
    df = data.copy()
    # 使用给定阈值生成信号
    df["v3_ml_signal_15m"] = (df["v3_ml_prob_15m"] > threshold).astype(int)
    bt = Backtest(
        df,
        UltimateStrategy,
        cash=CONFIG["initial_cash"],
        commission=CONFIG["commission"],
        margin=CONFIG["spread"] / 2,
        finalize_trades=True,
    )
    stats = bt.run(symbol=symbol)
    return stats


def grid_search_threshold(processed_backtest_data: dict, thresholds: list, metric: str):
    results = {}
    best_thr, best_score = None, -np.inf
    for thr in thresholds:
        per_symbol_scores = {}
        per_symbol_stats = {}
        for symbol, data in processed_backtest_data.items():
            if data.empty:
                continue
            stats = run_backtest_with_threshold(data, symbol, thr)
            score = _metric_from_stats(stats, metric)
            # 若主指标无效，用收益率兜底
            if not np.isfinite(score):
                score = _metric_from_stats(stats, "return")
            per_symbol_scores[symbol] = score if np.isfinite(score) else -np.inf
            per_symbol_stats[symbol] = stats
        # 汇总分数（平均）
        valid_scores = [s for s in per_symbol_scores.values() if np.isfinite(s)]
        agg = np.mean(valid_scores) if valid_scores else -np.inf
        results[thr] = {"aggregate": agg, "scores": per_symbol_scores, "stats": per_symbol_stats}
        if agg > best_score:
            best_thr, best_score = thr, agg
    return best_thr, best_score, results

# --- 主程序入口 ---
if __name__ == "__main__":
    logger.info(f"🚀 简化版策略 - 仅使用15M模型 开始运行...")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_fetch_start_date_str = (
        backtest_start_dt - pd.Timedelta(days=365 * 2)
    ).strftime("%Y-%m-%d")
    if CONFIG["enable_ml_component"]:
        training_window = timedelta(days=CONFIG["training_window_days"])
        data_fetch_start_date_str = (backtest_start_dt - training_window).strftime(
            "%Y-%m-%d"
        )

    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期 (包含训练窗口): {data_fetch_start_date_str}")
    raw_data = {
        s: fetch_binance_klines(
            s,
            CONFIG["interval"],
            data_fetch_start_date_str,
            CONFIG["backtest_end_date"],
        )
        for s in CONFIG["symbols_to_test"]
    }
    if not any(not df.empty for df in raw_data.values()):
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    logger.info("### 模式: 跳过动态训练，使用静态模型进行回测 ###")
    logger.info(f"### 准备完整回测数据 ###")
    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        if data.empty:
            continue
        logger.info(f"为 {symbol} 预处理完整时段数据...")
        full_processed_data = preprocess_data_for_strategy(data, symbol)
        backtest_slice = full_processed_data.loc[CONFIG["backtest_start_date"] :].copy()
        if not backtest_slice.empty:
            processed_backtest_data[symbol] = backtest_slice

    if not processed_backtest_data:
        logger.error("无回测数据，程序终止。")
        exit()

    # --- 阈值网格搜索 ---
    ts_cfg = CONFIG.get("threshold_search", {})
    if ts_cfg.get("enabled", False):
        logger.info("### 启用阈值网格搜索 ###")
        if ts_cfg.get("grid"):
            thresholds = [float(x) for x in ts_cfg["grid"]]
        else:
            thresholds = [round(x, 3) for x in np.arange(0.25, 0.651, 0.025)]
        metric = ts_cfg.get("metric", "sharpe").lower()
        best_thr, best_score, grid_results = grid_search_threshold(
            processed_backtest_data, thresholds, metric
        )
        print(f"\n{'#'*80}\n阈值网格搜索结果 (metric={metric})\n{'#'*80}")
        for thr in thresholds:
            agg = grid_results[thr]["aggregate"]
            print(f"  阈值={thr:.3f} -> 综合得分={agg:.4f}")
        print(f"\n>>> 最佳阈值: {best_thr:.3f}, 综合得分={best_score:.4f}")

        if ts_cfg.get("rerun_with_best", True):
            logger.info(f"### 使用最佳阈值 {best_thr:.3f} 进行最终回测 ###")
            all_stats = {}
            for symbol, data in processed_backtest_data.items():
                print(f"\n{'='*80}\n正在回测品种: {symbol} (best_thr={best_thr:.3f})\n{'='*80}")
                final_stats = run_backtest_with_threshold(data, symbol, best_thr)
                all_stats[symbol] = final_stats
                print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
                print(final_stats)
                if CONFIG["show_plots"]:
                    # 生成包含最佳阈值信号的数据用于绘图
                    plot_df = data.copy()
                    plot_df["v3_ml_signal_15m"] = (
                        plot_df["v3_ml_prob_15m"] > best_thr
                    ).astype(int)
                    Backtest(
                        plot_df,
                        UltimateStrategy,
                        cash=CONFIG["initial_cash"],
                        commission=CONFIG["commission"],
                        margin=CONFIG["spread"] / 2,
                        finalize_trades=True,
                    ).plot()
            if all_stats:
                initial_total = CONFIG["initial_cash"] * len(all_stats)
                total_equity = sum(s["Equity Final [$]"] for s in all_stats.values())
                ret = ((total_equity - initial_total) / initial_total) * 100
                print(f"\n{'#'*80}\n                 组合策略表现总览 (best_thr={best_thr:.3f})\n{'#'*80}")
                for symbol, stats in all_stats.items():
                    print(
                        f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A')}"
                    )
                print(
                    f"\n--- 投资组合整体表现 ---\n总初始资金: ${initial_total:,.2f}\n总最终权益: ${total_equity:,.2f}\n组合总回报率: {ret:.2f}%"
                )
        else:
            logger.info("### 网格搜索已完成，未启用最终回测 ###")
    else:
        logger.info(f"### 进入回测模式 ###")
        all_stats = {}
        for symbol, data in processed_backtest_data.items():
            print(f"\n{'='*80}\n正在回测品种: {symbol}\n{'='*80}")
            bt = Backtest(
                data,
                UltimateStrategy,
                cash=CONFIG["initial_cash"],
                commission=CONFIG["commission"],
                margin=CONFIG["spread"] / 2,
                finalize_trades=True,
            )
            stats = bt.run(symbol=symbol)
            all_stats[symbol] = stats
            print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
            print(stats)
            
            # --- 模型独立表现分析 ---
            analyze_v3_standalone_performance(data, signal_col="v3_ml_signal_15m")
            
            if CONFIG["show_plots"]:
                bt.plot()

    if all_stats:
        initial_total = CONFIG["initial_cash"] * len(all_stats)
        total_equity = sum(stats["Equity Final [$]"] for stats in all_stats.values())
        ret = ((total_equity - initial_total) / initial_total) * 100
        print(f"\n{'#'*80}\n                 组合策略表现总览\n{'#'*80}")
        for symbol, stats in all_stats.items():
            print(
                f"  - {symbol}:\n    - 最终权益: ${stats['Equity Final [$]']:,.2f} (回报率: {stats['Return [%]']:.2f}%)\n    - 最大回撤: {stats['Max. Drawdown [%]']:.2f}%\n    - 夏普比率: {stats.get('Sharpe Ratio', 'N/A')}"
            )
        print(
            f"\n--- 投资组合整体表现 ---\n总初始资金: ${initial_total:,.2f}\n总最终权益: ${total_equity:,.2f}\n组合总回报率: {ret:.2f}%"
        )
