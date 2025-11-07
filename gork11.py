# -*- coding: utf-8 -*-
# V49.2-Fetch-Fix

# --- 1. 导入库与配置 ---
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
import warnings

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import ta

# --- 日志配置 (无变化) ---
logger = logging.getLogger(__name__)
# ... (日志配置保持不变)
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


def set_chinese_font():  # (无变化)
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


# --- 核心配置 (无变化) ---
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],
    "interval": "15m",
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-11-06",
    "initial_cash": 500_000,
    "commission": 0.00075,
    "spread": 0.0002,
    "show_plots": False,
    "data_lookback_days": 250,
}

# --- 模型文件路径配置 (无变化) ---
LGBM_4H_MODEL_PATH = "models/eth_trend_model_lgb_4h.joblib"
LGBM_4H_SCALER_PATH = "models/eth_trend_scaler_lgb_4h.joblib"
LGBM_4H_FEATURE_COLUMNS_PATH = "models/feature_columns_lgb_4h.joblib"
LGBM_4H_THRESHOLD = 0.3159
LGBM_SEQUENCE_LENGTH = 60

# --- 策略参数 (无变化) ---
STRATEGY_PARAMS = {
    "tactical_ema_period": 50,
    "tactical_adx_period": 14,
    "long_entry_threshold": 0.3,
    "short_entry_threshold": -0.3,
    "score_weights": {
        "ema_direction": 0.5,
        "ml_signal": 0.4,
        "adx_score": 0.1,
    },
    "tsl_enabled": True,
    "tsl_activation_atr_mult": 1.5,
    "tsl_trailing_atr_mult": 2.0,
    "kelly_trade_history": 20,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.5,
}


# --- 函数定义 ---
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

                # ### <<< 核心修正：从列表中获取时间戳 >>> ###
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


def add_features_for_lgbm_model(df: pd.DataFrame) -> pd.DataFrame:  # (无变化)
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    df["volatility"] = (
        np.log(df["Close"] / df["Close"].shift(1)).rolling(window=20).std()
    )
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
    (
        df["BBU_20_2.0"],
        df["BBM_20_2.0"],
        df["BBL_20_2.0"],
        df["BBB_20_2.0"],
        df["BBP_20_2.0"],
    ) = (
        bb_indicator.bollinger_hband(),
        bb_indicator.bollinger_mavg(),
        bb_indicator.bollinger_lband(),
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


def create_flattened_sequences(data, look_back=60):  # (无变化)
    X = []
    for i in range(len(data) - look_back + 1):
        X.append(data[i : (i + look_back), :].flatten())
    return np.array(X, dtype=np.float32) if X else np.array([])


def generate_lgbm_signals(symbol: str, interval: str) -> pd.Series:  # (无变化)
    logger.info(f"--- 正在为 [{symbol}] 生成 [{interval}] 级别的LGBM信号 ---")
    if lgb is None:
        logger.warning("lightgbm库未安装，无法生成LGBM信号。")
        return pd.Series(dtype="float64")
    if not all(
        os.path.exists(p)
        for p in [LGBM_4H_MODEL_PATH, LGBM_4H_SCALER_PATH, LGBM_4H_FEATURE_COLUMNS_PATH]
    ):
        logger.warning(f"缺少 {interval} 模型的必要文件，将返回空信号。")
        return pd.Series(dtype="float64")
    try:
        model, scaler, feature_columns = (
            joblib.load(LGBM_4H_MODEL_PATH),
            joblib.load(LGBM_4H_SCALER_PATH),
            joblib.load(LGBM_4H_FEATURE_COLUMNS_PATH),
        )
        start_date = (
            datetime.now() - timedelta(days=CONFIG["data_lookback_days"] + 200)
        ).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        df_lgbm = fetch_binance_klines(symbol, interval, start_date, end_date)
        if df_lgbm.empty:
            return pd.Series(dtype="float64")

        df_featured = add_features_for_lgbm_model(df_lgbm.copy())
        for col in feature_columns:
            if col not in df_featured.columns:
                df_featured[col] = 0
        df_aligned = df_featured[feature_columns].dropna()
        if df_aligned.empty:
            return pd.Series(dtype="float64")

        scaled_features = scaler.transform(df_aligned)
        X_sequences = create_flattened_sequences(
            scaled_features, look_back=LGBM_SEQUENCE_LENGTH
        )
        if X_sequences.shape[0] == 0:
            return pd.Series(dtype="float64")

        probs = model.predict_proba(X_sequences)[:, 1]
        signals = np.where(probs > LGBM_4H_THRESHOLD, 1, -1)
        signal_index = df_aligned.index[LGBM_SEQUENCE_LENGTH - 1 :]
        signal_series = pd.Series(signals, index=signal_index)

        logger.info(f"✅ 成功生成 {len(signal_series)} 条 [{interval}] LGBM信号。")
        return signal_series
    except Exception as e:
        logger.error(f"生成 {interval} LGBM信号时出错: {e}", exc_info=True)
        return pd.Series(dtype="float64")


def preprocess_data_for_strategy(
    data_in: pd.DataFrame, symbol: str
) -> pd.DataFrame:  # (无变化)
    df_15m = data_in.copy()
    logger.info(f"[{symbol}] 开始 4h 周期策略的数据预处理...")
    p = STRATEGY_PARAMS
    start_date = (
        df_15m.index.min() - timedelta(days=CONFIG["data_lookback_days"] + 50)
    ).strftime("%Y-%m-%d")
    end_date = (df_15m.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
    df_4h = fetch_binance_klines(symbol, "4h", start_date, end_date)
    if df_4h.empty:
        logger.error(f"无法获取 {symbol} 的 4h 数据，策略无法运行。")
        for col in ["trend_direction", "adx_score", "entry_signal"]:
            df_15m[col] = 0
        return df_15m
    logger.info(f"[{symbol}] 在 4h 数据上计算 EMA, ADX, 和 LGBM 信号...")
    ema_4h = ta.trend.EMAIndicator(
        df_4h["Close"], window=p["tactical_ema_period"]
    ).ema_indicator()
    df_4h["trend_direction"] = np.where(df_4h["Close"] > ema_4h, 1, -1)
    adx_indicator = ta.trend.ADXIndicator(
        df_4h["High"], df_4h["Low"], df_4h["Close"], window=p["tactical_adx_period"]
    )
    adx_values = adx_indicator.adx()
    df_4h["adx_score"] = (adx_values / 60).clip(0, 1)
    lgbm_signal_4h = generate_lgbm_signals(symbol, "4h")
    df_4h["entry_signal"] = lgbm_signal_4h.reindex(df_4h.index).fillna(0)
    logger.info(f"[{symbol}] 将 4h 信号广播到 15m 数据...")
    for signal_col in ["trend_direction", "adx_score", "entry_signal"]:
        df_15m[signal_col] = (
            df_4h[signal_col].reindex(df_15m.index, method="ffill").fillna(0)
        )
    df_15m["atr_15m"] = ta.volatility.AverageTrueRange(
        df_15m["High"], df_15m["Low"], df_15m["Close"], p["tf_atr_period"]
    ).average_true_range()
    df_15m.dropna(inplace=True)
    logger.info(f"[{symbol}] 数据预处理完成。")
    return df_15m


# --- 策略类定义 ---
class UltimateStrategy(Strategy):  # (无变化)
    symbol, vol_weight = (None, 1.0)

    def init(self):
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()
        self.trend_direction = self.I(lambda: self.data.trend_direction)
        self.adx_score = self.I(lambda: self.data.adx_score)
        self.entry_signal = self.I(lambda: self.data.entry_signal)
        self.final_score = self.I(self._calculate_score)
        self.atr = self.I(lambda: self.data.atr_15m)

    def _calculate_score(self):
        w = self.score_weights
        direction_scores = w["ema_direction"] * self.trend_direction
        ml_scores = w["ml_signal"] * self.entry_signal
        adx_scores = w["adx_score"] * self.adx_score
        return direction_scores + ml_scores + adx_scores

    def next(self):
        if self.position:
            self.manage_open_position(self.data.Close[-1])
            return

        long_signal = crossover(self.final_score, self.long_entry_threshold)
        short_signal = crossover(self.short_entry_threshold, self.final_score)

        if long_signal:
            self.open_position(self.data.Close[-1], is_long=True)
        elif short_signal:
            self.open_position(self.data.Close[-1], is_long=False)

    def reset_trade_state(self):
        self.stop_loss_price = 0.0
        self.trailing_stop_active = False

    def manage_open_position(self, p):
        if (self.position.is_long and p < self.stop_loss_price) or (
            self.position.is_short and p > self.stop_loss_price
        ):
            self.position.close()
        elif self.tsl_enabled:
            self._manage_trailing_stop_loss()

    def _manage_trailing_stop_loss(self):
        if not self.position:
            return
        is_active, entry_price, current_price = (
            self.trailing_stop_active,
            self.trades[-1].entry_price,
            self.data.Close[-1],
        )
        if not is_active:
            activation_dist = self.atr[-1] * self.tsl_activation_atr_mult
            if (
                self.position.is_long and current_price >= entry_price + activation_dist
            ) or (
                self.position.is_short
                and current_price <= entry_price - activation_dist
            ):
                self.trailing_stop_active = True
        if self.trailing_stop_active:
            trail_dist = self.atr[-1] * self.tsl_trailing_atr_mult
            if self.position.is_long:
                self.stop_loss_price = max(
                    self.stop_loss_price, current_price - trail_dist
                )
            else:
                self.stop_loss_price = min(
                    self.stop_loss_price, current_price + trail_dist
                )

    def open_position(self, p, is_long):
        risk_ps = self.atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(p, risk_ps, self._calculate_dynamic_risk())
        if size <= 0:
            return
        self.reset_trade_state()
        if is_long:
            self.buy(size=size)
            self.stop_loss_price = p - risk_ps
        else:
            self.sell(size=size)
            self.stop_loss_price = p + risk_ps

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0:
            return 0
        return int(min((self.equity * risk_pct) / rps, (self.equity * 0.95) / p))

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins, losses = [r for r in self.recent_trade_returns if r > 0], [
            r for r in self.recent_trade_returns if r < 0
        ]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate, avg_win, avg_loss = (
            len(wins) / len(self.recent_trade_returns),
            sum(wins) / len(wins),
            abs(sum(losses) / len(losses)),
        )
        if avg_loss == 0 or (reward_ratio := avg_win / avg_loss) == 0:
            return self.default_risk_pct * self.vol_weight
        return min(
            max(0.005, (win_rate - (1 - win_rate) / reward_ratio) * 0.5)
            * self.vol_weight,
            self.max_risk_pct,
        )


# --- 主程序入口 ---
if __name__ == "__main__":
    logger.info(f"🚀 (V49.2-Fetch-Fix & Crossover) 开始运行...")

    import sys

    if len(sys.argv) == 3:
        CONFIG["backtest_start_date"] = sys.argv[1]
        CONFIG["backtest_end_date"] = sys.argv[2]

    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_fetch_start_date = (
        backtest_start_dt - timedelta(days=CONFIG["data_lookback_days"])
    ).strftime("%Y-%m-%d")

    logger.info(
        f"回测时间段: {CONFIG['backtest_start_date']} to {CONFIG['backtest_end_date']}"
    )
    logger.info(f"数据获取起始日期: {data_fetch_start_date}")

    raw_data = {
        s: fetch_binance_klines(
            s, CONFIG["interval"], data_fetch_start_date, CONFIG["backtest_end_date"]
        )
        for s in CONFIG["symbols_to_test"]
    }
    raw_data = {s: d for s, d in raw_data.items() if not d.empty}
    if not raw_data:
        logger.error("所有品种数据获取失败，程序终止。")
        exit()

    processed_backtest_data = {}
    for symbol, data in raw_data.items():
        logger.info(f"为 {symbol} 预处理完整时段数据...")
        full_processed_data = preprocess_data_for_strategy(data, symbol)
        backtest_period_slice = full_processed_data.loc[
            CONFIG["backtest_start_date"] : CONFIG["backtest_end_date"]
        ].copy()
        if not backtest_period_slice.empty:
            processed_backtest_data[symbol] = backtest_period_slice

    if not processed_backtest_data:
        logger.error("无回测数据，程序终止。")
        exit()

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
