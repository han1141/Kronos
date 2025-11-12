# --- 核心库导入 ---
import logging
import time
import warnings

# --- 数据分析和机器学习库 ---
import numpy as np
import pandas as pd
import requests
import ta
from backtesting import Backtest, Strategy

# 忽略一些常见的未来警告
warnings.simplefilter(action="ignore", category=FutureWarning)

# --- 1. 全局配置和策略参数 ---

# 回测配置
SYMBOL = "ETHUSDT"
INTERVAL = "15m"
START_DATE = "2025-01-01"
END_DATE = "2025-11-11"
INITIAL_CASH = 100_000
COMMISSION = 0.0006  # 币安费率: 0.06%

# 策略核心参数
STRATEGY_PARAMS = {
    # --- 市场状态检测参数 ---
    "regime_adx_period": 14,
    "regime_atr_period": 14,
    "regime_atr_slope_period": 6,
    "regime_rsi_period": 14,
    "regime_rsi_vol_period": 14,
    "regime_norm_period": 252,
    "regime_hurst_period": 80,
    "regime_score_weight_adx": 0.55,
    "regime_score_weight_atr": 0.3,
    "regime_score_weight_rsi": 0.1,
    "regime_score_weight_hurst": 0.05,
    "regime_score_threshold": 0.4,
    # --- 趋势跟随(TF)模块参数 ---
    "tf_donchian_period": 24,
    "tf_ema_fast_period": 21,
    "tf_ema_slow_period": 60,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.6,
    # --- 均值回归(MR)模块参数 ---
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_stop_loss_atr_multiplier": 1.5,
    # --- 多周期过滤和信号权重 ---
    "mtf_period": 40,
    "score_entry_threshold": 0.5,
    # 纯指标打分（临时禁用 ML 权重，重归一化其余权重）
    "score_weights_tf": {
        "breakout": 0.42,
        "momentum": 0.35,
        "mtf": 0.23,
        # "ml": 0.0,  # 暂不参与
        # "advanced_ml": 0.0,
    },
}

# --- 2. 日志系统设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# --- 3. 特征工程 & 原始策略逻辑 ---
# 这部分代码是从 `ultimate_strategy.py` 移植过来的，用于预先计算所有指标。


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
        hurst = np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0]
        return max(0.0, min(1.0, hurst))
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
        df["regime_score"] > STRATEGY_PARAMS["regime_score_threshold"], "趋势", "震荡"
    )
    df["market_regime"] = np.where(df["trend_regime"] == "趋势", 1, -1)
    return df


class UltimateStrategyCalculator:
    """
    这个类仅用于在回测前，一次性地计算所有需要的技术指标。
    它不是一个可交易的策略，而是一个特征计算器。
    """

    def __init__(self, df: pd.DataFrame, symbol: str):
        self.data = df.copy()
        self.symbol = symbol
        # 将策略参数加载到实例中
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)

    def compute_all_features(self, trader, kline_interval: str):
        logger.debug(f"[{self.symbol}] 开始计算特征...")
        # 确保数据完整性
        self.data.dropna(
            subset=["Open", "High", "Low", "Close", "Volume"], inplace=True
        )
        if len(self.data) < self.tf_donchian_period:
            logger.warning(f"数据量过少({len(self.data)}条)，无法计算特征。")
            return

        # 计算所有需要的指标
        self.data = add_ml_features_ported(self.data)
        self.data = add_market_regime_features(self.data)

        # 多时间框架(MTF)信号
        # --- !!! 这里是修改点 !!! ---
        if (
            data_1d := trader.fetch_history_klines(
                self.symbol, bar="1d", limit=self.mtf_period + 50
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
            self.data["mtf_signal"] = 0  # 如果获取失败，则为中性信号

        # 宏观趋势过滤
        df_4h = self.data["Close"].resample("4H").last().to_frame()
        df_4h["macro_ema"] = ta.trend.EMAIndicator(
            df_4h["Close"], window=50
        ).ema_indicator()
        df_4h["macro_trend"] = np.where(df_4h["Close"] > df_4h["macro_ema"], 1, -1)
        self.data["macro_trend_filter"] = (
            df_4h["macro_trend"].reindex(self.data.index, method="ffill").fillna(0)
        )

        # 趋势跟随(TF)指标
        self.data["tf_atr"] = ta.volatility.AverageTrueRange(
            self.data.High, self.data.Low, self.data.Close, self.tf_atr_period
        ).average_true_range()
        self.data["tf_donchian_h"] = (
            self.data.High.rolling(self.tf_donchian_period).max().shift(1)
        )
        self.data["tf_donchian_l"] = (
            self.data.Low.rolling(self.tf_donchian_period).min().shift(1)
        )
        self.data["tf_ema_fast"] = ta.trend.EMAIndicator(
            self.data.Close, self.tf_ema_fast_period
        ).ema_indicator()
        self.data["tf_ema_slow"] = ta.trend.EMAIndicator(
            self.data.Close, self.tf_ema_slow_period
        ).ema_indicator()

        # 均值回归(MR)指标
        bb = ta.volatility.BollingerBands(
            self.data.Close, self.mr_bb_period, self.mr_bb_std
        )
        self.data["mr_bb_upper"] = bb.bollinger_hband()
        self.data["mr_bb_lower"] = bb.bollinger_lband()
        self.data["mr_bb_mid"] = bb.bollinger_mavg()
        stoch_rsi = ta.momentum.StochRSIIndicator(
            self.data.Close, window=14, smooth1=3, smooth2=3
        )
        self.data["mr_stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        self.data["mr_stoch_rsi_d"] = stoch_rsi.stochrsi_d()

        # 清理数据
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.fillna(method="ffill", inplace=True)
        self.data.dropna(inplace=True)  # 删除计算后仍然存在的NaN行
        logger.debug(f"[{self.symbol}] 特征计算完成。")


# --- 4. 数据获取函数 ---


def fetch_binance_klines(symbol, interval, start_str, end_str=None, limit=1000):
    if end_str is None:
        end_str = pd.to_datetime("now", utc=True).strftime("%Y-%m-%d %H:%M:%S")

    url = "https://api.binance.com/api/v3/klines"
    cols = ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)
    all_data = []

    logger.info(f"正在从币安获取 {symbol} 从 {start_str} 到 {end_str} 的K线数据...")

    while start_ts < end_ts:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ts,
            "limit": limit,
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            all_data.extend(data)
            start_ts = data[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.error(f"获取数据失败: {e}")
            time.sleep(5)

    if not all_data:
        logger.warning(f"未能获取到 {symbol} 的任何数据。")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[*cols, "c1", "c2", "c3", "c4", "c5", "c6"])[
        cols
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"✅ 成功获取 {len(df)} 条 {symbol} 的K线数据")
    return df.set_index("timestamp").sort_index()


# --- 5. Backtesting.py 策略集成 ---


class UltimateBacktestStrategy(Strategy):
    """
    这是与 `backtesting.py` 库兼容的策略封装类。
    """

    def init(self):
        # 将预先计算好的指标作为 `self.I` 注册，以便在图表中查看
        self.market_regime = self.I(
            lambda x: self.data.market_regime, self.data.Close, name="Market_Regime"
        )
        self.macro_trend_filter = self.I(
            lambda x: self.data.macro_trend_filter, self.data.Close, name="Macro_Trend"
        )
        self.tf_donchian_h = self.I(
            lambda x: self.data.tf_donchian_h, self.data.Close, name="Donchian_H"
        )
        self.tf_donchian_l = self.I(
            lambda x: self.data.tf_donchian_l, self.data.Close, name="Donchian_L"
        )
        self.mr_bb_upper = self.I(
            lambda x: self.data.mr_bb_upper, self.data.Close, name="BB_Upper"
        )
        self.mr_bb_lower = self.I(
            lambda x: self.data.mr_bb_lower, self.data.Close, name="BB_Lower"
        )

        # 存储策略参数
        self.p = STRATEGY_PARAMS
        self.score_threshold = self.p["score_entry_threshold"]
        self.sl_atr_mult_tf = self.p["tf_stop_loss_atr_multiplier"]
        self.sl_atr_mult_mr = self.p["mr_stop_loss_atr_multiplier"]

    def next(self):
        # 获取当前K线的价格和指标
        price = self.data.Close[-1]
        atr = self.data.tf_atr[-1]

        # --- 入场逻辑 ---
        if not self.position:
            action = None

            # 趋势跟随 (TF) 逻辑
            if self.market_regime[-1] == 1:  # 趋势市场
                score = self._calculate_entry_score()
                if self.macro_trend_filter[-1] == 1 and score > self.score_threshold:
                    action = "BUY_TF"
                elif (
                    self.macro_trend_filter[-1] == -1 and score < -self.score_threshold
                ):
                    action = "SELL_TF"

            # 均值回归 (MR) 逻辑
            elif self.market_regime[-1] == -1:  # 震荡市场
                mr_signal = self._define_mr_entry_signal()
                if self.macro_trend_filter[-1] == 1 and mr_signal == 1:
                    action = "BUY_MR"
                elif self.macro_trend_filter[-1] == -1 and mr_signal == -1:
                    action = "SELL_MR"

            # 如果有交易信号，则执行
            if action:
                sl_atr_mult = (
                    self.sl_atr_mult_tf if "TF" in action else self.sl_atr_mult_mr
                )
                stop_loss_dist = atr * sl_atr_mult

                if "BUY" in action:
                    self.buy(sl=price - stop_loss_dist, size=0.95)  # 使用95%的资金开仓
                elif "SELL" in action:
                    self.sell(sl=price + stop_loss_dist, size=0.95)

        # --- 出场逻辑 ---
        # 初始止损已在 self.buy/sell 中设置。
        # 这里可以添加更复杂的出场逻辑，例如均值回归的目标止盈。
        else:
            if self.position.is_long and self.data.Close[-1] >= self.data.mr_bb_mid[-1]:
                self.position.close()
            elif (
                self.position.is_short
                and self.data.Close[-1] <= self.data.mr_bb_mid[-1]
            ):
                self.position.close()

    # 策略内部的辅助计算函数
    def _calculate_entry_score(self):
        w = self.p["score_weights_tf"]
        last = self.data

        b_s = (
            1
            if last.High[-1] > self.tf_donchian_h[-1]
            else -1 if last.Low[-1] < self.tf_donchian_l[-1] else 0
        )
        # 使用已在数据中预计算的 EMA 列，避免访问未注册的属性
        mo_s = 1 if last.tf_ema_fast[-1] > last.tf_ema_slow[-1] else -1

        # 在回测中简化ML信号
        ml_score, adv_ml_score = 0, 0

        score = (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + last.mtf_signal[-1] * w.get("mtf", 0)
            + ml_score * w.get("ml", 0)
            + adv_ml_score * w.get("advanced_ml", 0)
        )
        return score

    def _define_mr_entry_signal(self):
        # 使用属性访问来获取当前和前一个 K 线的数据
        last_close, prev_close = self.data.Close[-1], self.data.Close[-2]
        last_k, last_d = self.data.mr_stoch_rsi_k[-1], self.data.mr_stoch_rsi_d[-1]
        prev_k, prev_d = self.data.mr_stoch_rsi_k[-2], self.data.mr_stoch_rsi_d[-2]
        bb_lower, bb_upper = (
            self.data.mr_bb_lower,
            self.data.mr_bb_upper,
        )  # 获取布林带上下轨

        # 定义买入信号
        if (
            prev_close < bb_lower[-2]
            and last_close > bb_lower[-1]
            and last_k > last_d
            and prev_k <= prev_d
            and last_k < 40
        ):
            return 1  # 买入信号

        # 定义卖出信号
        if (
            prev_close > bb_upper[-2]
            and last_close < bb_upper[-1]
            and last_k < last_d
            and prev_k >= prev_d
            and last_k > 60
        ):
            return -1  # 卖出信号

        return 0  # 无信号


# --- 6. 主执行程序 ---
if __name__ == "__main__":
    # 步骤 1: 获取历史数据
    data_df = fetch_binance_klines(SYMBOL, INTERVAL, START_DATE, END_DATE)

    if not data_df.empty:
        # 步骤 2: 预先计算所有策略指标
        logger.info("正在预计算所有策略指标...")

        # 创建一个模拟的'trader'对象，以满足特征计算函数的需求
        class MockTrader:
            def fetch_history_klines(self, symbol, bar, limit):
                # 为MTF指标获取真实日线数据
                return fetch_binance_klines(symbol, bar, START_DATE, END_DATE, limit)

        # 使用计算器类来生成带有所有指标的DataFrame
        calculator = UltimateStrategyCalculator(df=data_df.copy(), symbol=SYMBOL)
        calculator.compute_all_features(trader=MockTrader(), kline_interval=INTERVAL)
        augmented_df = calculator.data

        # 步骤 3: 设置并运行回测
        logger.info("设置并运行回测...")
        bt = Backtest(
            augmented_df,
            UltimateBacktestStrategy,
            cash=INITIAL_CASH,
            commission=COMMISSION,
            trade_on_close=True,
        )

        stats = bt.run()

        # 步骤 4: 打印结果并生成图表
        logger.info("回测完成，输出结果:")
        print(stats)
