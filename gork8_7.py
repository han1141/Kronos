# -*- coding-utf-8 -*-
# File: gork8_11_full_features.py (Integrates All Features from Training Script)

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
import warnings

# NumPy 2.0 compatibility for older pandas_ta
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "Inf"):
    np.Inf = np.inf

# 静音 pandas_ta 使用 pkg_resources 的弃用提醒
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

try:
    import pandas_ta as ta

    if hasattr(ta, "Imports"):
        ta.Imports["talib"] = False
except ImportError:
    print("错误: 'pandas_ta' 库未安装。请运行 'pip install pandas_ta' 来安装。")
    exit()

from backtesting import Backtest, Strategy

# --- 2. 日志与全局设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def set_chinese_font():
    try:
        font = next(
            f
            for f in [
                "PingFang SC",
                "Microsoft YaHei",
                "SimHei",
                "Heiti TC",
                "sans-serif",
            ]
            if f in [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        )
        plt.rcParams["font.sans-serif"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
        logger.info(f"成功设置中文字体: {font}")
    except Exception:
        logger.warning("未找到指定的中文字体，绘图可能出现乱码。")


set_chinese_font()

# --- 3. 核心配置 ---
CONFIG = {
    "symbol_to_test": "ETHUSDT",
    "interval": "1h",  # 与模型训练脚本保持一致
    "backtest_start_date": "2025-01-01",  # 使用测试集时间范围
    "backtest_end_date": "2025-11-17",
    "initial_cash": 500_000,
    "commission": 0.00075,
}
MODEL_PATH = "models_gbm2/"
LOOK_BACK = 60  # 与模型训练脚本保持一致

# --- 4. 策略参数 ---
STRATEGY_PARAMS = {
    "kelly_trade_history": 20,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "dd_grace_period_bars": 240,
    "dd_initial_scale": 0.35,
    "dd_decay_bars": 4320,
    "max_consecutive_losses": 5,
    "trading_pause_bars": 96,
    "regime_bbw_period": 20,
    "regime_bbw_std": 2.0,
    "regime_percentile_period": 252,
    "regime_squeeze_threshold": 0.1,
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.0,
    # TF 子策略入场评分阈值（越高越少交易）
    "score_entry_threshold": 0.8,
    "score_weights_tf": {
        "ml_signal": 0.35,
        "breakout": 0.25,
        "momentum": 0.25,
        "mtf": 0.15,
    },
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 30,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mr_ml_entry_threshold": 0.4,  # MR 子策略中 ML 信号触发方向的最小绝对值 (|ML_signal|)
    "tf_ml_entry_threshold": 0.4,  # TF 子策略中 ML 信号的硬阈值
    "volatility_filter_long_period": 100,
    "volatility_filter_short_period": 14,
    "volatility_filter_multiplier": 2.5,
    "tf_rsi_filter_period": 14,
    "tf_rsi_long_threshold": 55,
    "tf_rsi_short_threshold": 45,
    "mtf_period": 20,
}


# --- 5. 数据获取与特征工程 ---
def fetch_binance_klines(s, i, st, en=None, l=1000):
    # ... (此函数无变动) ...
    url, cols = "https://api.binance.com/api/v3/klines", [
        "timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
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


# [!!! 关键实现 1 !!!] - 移植自训练脚本
def get_market_structure_features_trailing(df, window=5):
    """在线、仅依赖过去数据的市场结构近似，无前视偏差。"""
    df_copy = df.copy()
    rolling_high = df_copy["High"].rolling(window=window, min_periods=1).max()
    rolling_low = df_copy["Low"].rolling(window=window, min_periods=1).min()
    is_higher_high = rolling_high > rolling_high.shift(1)
    is_higher_low = rolling_low > rolling_low.shift(1)
    is_lower_high = rolling_high < rolling_high.shift(1)
    is_lower_low = rolling_low < rolling_low.shift(1)
    df_copy["is_uptrend"] = (is_higher_high & is_higher_low).astype(int)
    df_copy["is_downtrend"] = (is_lower_high & is_lower_low).astype(int)
    df_copy["market_structure"] = df_copy["is_uptrend"] - df_copy["is_downtrend"]
    return df_copy[["market_structure"]]


# [!!! 关键实现 2 !!!] - 移植自训练脚本
def calculate_base_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算与模型训练时完全一致的【原始、未平铺】特征。"""
    df = df.copy()

    # 使用 try-except 块和长度检查来稳健地计算指标
    try:
        df.ta.rsi(length=14, append=True)
    except Exception:
        pass
    try:
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
    except Exception:
        pass
    try:
        df.ta.bbands(length=20, std=2, append=True)
    except Exception:
        pass
    try:
        df.ta.adx(length=14, append=True)
    except Exception:
        pass
    try:
        df.ta.atr(length=14, append=True)
    except Exception:
        pass
    try:
        df.ta.obv(append=True)
    except Exception:
        pass
    try:
        df.ta.macd(
            fast=24,
            slow=52,
            signal=18,
            append=True,
            col_names=("MACD_long", "MACDh_long", "MACDs_long"),
        )
    except Exception:
        pass

    # 自定义特征
    df["volatility_log_ret"] = (
        (np.log(df["Close"] / df["Close"].shift(1))).rolling(window=20).std()
    )
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_4"] = df["Close"].pct_change(4)
    df["ret_16"] = df["Close"].pct_change(16)

    # 依赖于其他指标的特征
    if "RSI_14" in df.columns:
        df["rsi_delta_1"] = df["RSI_14"].diff(1)
    if "MACDh_12_26_9" in df.columns:
        df["macd_delta_1"] = df["MACDh_12_26_9"].diff(1)

    # 市场结构特征
    market_structure_df = get_market_structure_features_trailing(df, window=5)

    # 组合并清理
    all_features_df = df.drop(columns=["Open", "High", "Low", "Close", "Volume"])
    all_features_df = pd.concat([all_features_df, market_structure_df], axis=1)
    all_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return all_features_df


# [!!! 关键实现 3 !!!] - 移植自训练脚本
def flatten_features(
    df_scaled: pd.DataFrame, flattened_columns: list, look_back: int
) -> pd.DataFrame:
    """将标准化后的基础特征进行平铺（lagging）。"""
    # 这是一个高效的实现，避免了循环
    vals = df_scaled.values

    # 创建一个大的3D数组，然后重塑
    # 这是基于numpy的高级索引技巧，比循环快得多
    n_features = vals.shape[1]
    n_rows = len(df_scaled)

    # 创建索引矩阵
    idx = np.arange(look_back)[:, None] + np.arange(n_rows - look_back + 1)

    # 从vals中提取所有序列
    sequences = vals[idx, :]

    # 重塑为2D，并移动轴以匹配flatten()的顺序
    flattened_data = np.transpose(sequences, (1, 0, 2)).reshape(
        n_rows - look_back + 1, -1
    )

    # 创建 DataFrame，索引需要对齐
    final_X = pd.DataFrame(
        flattened_data,
        index=df_scaled.index[look_back - 1 :],
        columns=flattened_columns,
    )

    # 重新索引以匹配原始DataFrame，确保开头有NaN
    return final_X.reindex(df_scaled.index)


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = data_in.copy()
    # --- 市场状态 (趋势 / 震荡) 与多周期 MTF 信号 ---
    # 1) 使用 ADX 判断当前周期是趋势市还是震荡市（仅依赖过去K线，无前视偏差）
    try:
        adx_df = ta.adx(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            length=STRATEGY_PARAMS["tf_atr_period"],
        )
        adx_series = adx_df[adx_df.columns[0]]
        adx_threshold = 20  # ADX 大于该值视为趋势市，反之视为震荡市
        df["market_regime"] = np.where(adx_series > adx_threshold, 1, -1)
    except Exception as e:
        logger.error(f"[{symbol}] 计算 ADX 市场状态失败，使用默认趋势模式: {e}")
        df["market_regime"] = 1

    # 2) 计算日线级别的 MTF 趋势信号，并整体右移一根K线以避免前视偏差
    try:
        daily_start = df.index.min().normalize() - pd.Timedelta(
            days=STRATEGY_PARAMS["mtf_period"] + 5
        )
        daily_end = df.index.max().normalize()
        data_1d = fetch_binance_klines(
            symbol,
            "1d",
            daily_start.strftime("%Y-%m-%d"),
            daily_end.strftime("%Y-%m-%d"),
        )
        if not data_1d.empty:
            sma_1d = ta.sma(
                data_1d["Close"], length=STRATEGY_PARAMS["mtf_period"]
            )
            mtf_signal_1d = pd.Series(
                np.where(data_1d["Close"] > sma_1d, 1, -1), index=data_1d.index
            )
            # 避免使用当日未收盘的日线信息：整体右移一根
            mtf_signal_1d = mtf_signal_1d.shift(1)
            df["mtf_signal"] = (
                mtf_signal_1d.reindex(df.index, method="ffill").fillna(0)
            )
        else:
            df["mtf_signal"] = 0
    except Exception as e:
        logger.error(f"[{symbol}] 计算多周期 MTF 信号失败，使用默认0: {e}")
        df["mtf_signal"] = 0

    logger.info("正在加载ML模型并按正确顺序生成预测分数...")
    try:
        model = joblib.load(
            f"{MODEL_PATH}{symbol.lower()}_model_high_precision_v6_{CONFIG['interval']}.joblib"
        )
        scaler = joblib.load(
            f"{MODEL_PATH}{symbol.lower()}_scaler_high_precision_v6_{CONFIG['interval']}.joblib"
        )
        base_feature_columns = joblib.load(
            f"{MODEL_PATH}{symbol.lower()}_feature_columns_high_precision_v6_{CONFIG['interval']}.joblib"
        )
        flattened_columns = joblib.load(
            f"{MODEL_PATH}{symbol.lower()}_flattened_columns_high_precision_v6_{CONFIG['interval']}.joblib"
        )

        # 检查是否存在特征选择后的列
        selected_flattened_columns_path = f"{MODEL_PATH}{symbol.lower()}_selected_flattened_columns_high_precision_v6_{CONFIG['interval']}.joblib"
        if joblib.os.path.exists(selected_flattened_columns_path):
            inference_columns = joblib.load(selected_flattened_columns_path)
            logger.info(
                f"检测到特征选择文件，将使用 {len(inference_columns)} 个特征进行推理。"
            )
        else:
            inference_columns = flattened_columns

        # --- 正确的预测流程 ---
        df_features = calculate_base_ml_features(df.copy())

        missing_cols = [
            col for col in base_feature_columns if col not in df_features.columns
        ]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=df_features.index, columns=missing_cols)
            df_features = pd.concat([df_features, missing_df], axis=1)

        X_base = df_features[base_feature_columns].fillna(0)

        X_scaled_base = scaler.transform(X_base)
        df_scaled_base = pd.DataFrame(
            X_scaled_base, index=X_base.index, columns=X_base.columns
        )

        X_flattened = flatten_features(df_scaled_base, flattened_columns, LOOK_BACK)

        # 准备最终用于预测的数据
        X_predict = X_flattened[inference_columns].dropna()

        if not X_predict.empty:
            pred_probs = model.predict_proba(X_predict)[:, 1]
            # 将概率结果对齐回原始DataFrame
            pred_series = pd.Series(pred_probs, index=X_predict.index)
            df["ml_score"] = pred_series.shift(1)  # 使用shift(1)确保无前视
        else:
            df["ml_score"] = np.nan

        logger.info("✅ ML模型预测分数计算完成。")

    except FileNotFoundError as e:
        logger.error(f"加载ML模型文件失败: {e}。将使用 0.5 作为中性 ML 分数。")
        # 使用中性概率 0.5，等价于“无观点”，避免被误解为强烈看空
        df["ml_score"] = 0.5
    except Exception as e:
        logger.error(f"处理ML特征时发生未知错误: {e}", exc_info=True)
        df["ml_score"] = 0.5

    df.dropna(
        subset=["Open", "High", "Low", "Close", "Volume", "ml_score"], inplace=True
    )
    logger.info(f"[{symbol}] 数据预处理完成。有效数据行数: {len(df)}")
    return df


# --- 6. 策略类定义 & 7. 回测执行 ---
class UltimateStrategy(Strategy):
    def init(self):
        for key, value in STRATEGY_PARAMS.items():
            setattr(self, key, value)
        close, high, low = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )
        # 交易与绩效跟踪
        self.initial_equity = float(self.equity)
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        # ML 相关统计
        self.ml_based_profit = 0.0
        self.ml_based_trades = 0
        self.ml_long_profit = 0.0
        self.ml_short_profit = 0.0
        self.ml_long_trades = 0
        self.ml_short_trades = 0
        self.ml_wins = 0
        self.ml_losses = 0
        self.ml_long_wins = 0
        self.ml_long_losses = 0
        self.ml_short_wins = 0
        self.ml_short_losses = 0
        # 不含手续费胜率统计
        self.ml_wins_gross = 0
        self.ml_losses_gross = 0
        self.ml_long_wins_gross = 0
        self.ml_long_losses_gross = 0
        self.ml_short_wins_gross = 0
        self.ml_short_losses_gross = 0
        self.total_trades = 0
        self.current_trade_is_ml_based = False
        self.last_mr_signal_from_ml = False
        self.reset_trade_state()
        self.equity_peak, self.bars_since_equity_peak = self.equity, 0
        self.consecutive_losses, self.paused_until_bar = 0, 0
        self.market_regime = self.I(lambda: self.data.market_regime)
        self.mtf_signal = self.I(lambda: self.data.mtf_signal)
        self.ml_score = self.I(lambda: self.data.ml_score, name="ML_Score")

        # [!!! 关键修复 !!!] 修正 pandas_ta API 调用方式
        self.tf_atr = self.I(ta.atr, high, low, close, length=self.tf_atr_period)
        self.tf_donchian_h = self.I(
            lambda: high.rolling(self.tf_donchian_period).max().shift(1)
        )
        self.tf_donchian_l = self.I(
            lambda: low.rolling(self.tf_donchian_period).min().shift(1)
        )
        self.tf_ema_fast = self.I(ta.ema, close, length=self.tf_ema_fast_period)
        self.tf_ema_slow = self.I(ta.ema, close, length=self.tf_ema_slow_period)

        # bbands 使用 pandas_ta 先一次性计算，再分别注册为三个独立指标
        bbands_df = ta.bbands(close=close, length=self.mr_bb_period, std=self.mr_bb_std)
        self.mr_bb_lower = self.I(
            lambda: bbands_df.iloc[:, 0], name="MR_BB_Lower"
        )
        self.mr_bb_mid = self.I(
            lambda: bbands_df.iloc[:, 1], name="MR_BB_Mid"
        )
        self.mr_bb_upper = self.I(
            lambda: bbands_df.iloc[:, 2], name="MR_BB_Upper"
        )

        self.mr_rsi = self.I(ta.rsi, close, length=self.mr_rsi_period)
        self.tf_rsi_filter = self.I(ta.rsi, close, length=self.tf_rsi_filter_period)

        # 使用 pandas_ta 先计算短周期 ATR，再在其基础上计算长周期平滑
        base_atr_series = ta.atr(
            high=high,
            low=low,
            close=close,
            length=self.volatility_filter_short_period,
        )
        self.vol_filter_short_atr = self.I(
            lambda: base_atr_series, name="Vol_ATR_Short"
        )
        self.vol_filter_long_atr = self.I(
            lambda: base_atr_series.rolling(
                self.volatility_filter_long_period
            ).mean(),
            name="Vol_ATR_Long",
        )

    def next(self):
        if self.equity > self.equity_peak:
            self.equity_peak, self.bars_since_equity_peak = self.equity, 0
        else:
            self.bars_since_equity_peak += 1
        if self.position:
            if self.active_sub_strategy == "TF":
                self.manage_trend_following_exit(self.data.Close[-1])
            elif self.active_sub_strategy == "MR":
                self.manage_mean_reversion_exit(self.data.Close[-1])
        else:
            if len(self.data) < self.paused_until_bar:
                return
            if self.market_regime[-1] == 1:
                self.run_scoring_system_entry(self.data.Close[-1])
            else:
                self.run_mean_reversion_entry(self.data.Close[-1])

    def _get_drawdown_risk_scale(self) -> float:
        if self.bars_since_equity_peak < self.dd_grace_period_bars:
            return 1.0
        drawdown_pct = (self.equity_peak - self.equity) / self.equity_peak
        if drawdown_pct <= 0:
            return 1.0
        decay_progress = min(1.0, self.bars_since_equity_peak / self.dd_decay_bars)
        return self.dd_initial_scale + (1 - self.dd_initial_scale) * decay_progress

    def run_scoring_system_entry(self, price):
        # 先用 ML 做一次硬过滤：ML 信号不够强则不参与本 bar 交易
        ml_prob = self.ml_score[-1] if not np.isnan(self.ml_score[-1]) else 0.5
        ml_signal = (ml_prob - 0.5) * 2
        if abs(ml_signal) < self.tf_ml_entry_threshold:
            return

        score = self._calculate_tf_entry_score()
        if (score > 0 and self.mtf_signal[-1] == -1) or (
            score < 0 and self.mtf_signal[-1] == 1
        ):
            return
        # 检查ATR值是否存在
        if np.isnan(self.vol_filter_short_atr[-1]) or np.isnan(
            self.vol_filter_long_atr[-1]
        ):
            return
        if (
            self.vol_filter_short_atr[-1]
            > self.vol_filter_long_atr[-1] * self.volatility_filter_multiplier
        ):
            return
        rsi_ok = (
            score > 0 and self.tf_rsi_filter[-1] > self.tf_rsi_long_threshold
        ) or (score < 0 and self.tf_rsi_filter[-1] < self.tf_rsi_short_threshold)
        if abs(score) >= self.score_entry_threshold and rsi_ok:
            # 趋势跟随子策略始终依赖 ML 信号
            self.open_tf_position(
                price, is_long=(score > 0), confidence_factor=abs(score)
            )

    def run_mean_reversion_entry(self, price):
        signal = self._define_mr_entry_signal()
        if signal != 0:
            # 记录此次 MR 入场是否由 ML 信号触发
            self.current_trade_is_ml_based = bool(
                getattr(self, "last_mr_signal_from_ml", False)
            )
            self.open_mr_position(price, is_long=(signal == 1))

    def open_tf_position(self, p, is_long, confidence_factor):
        # 趋势跟随子策略的开仓视为 ML 驱动
        self.current_trade_is_ml_based = True
        self.current_entry_price = float(p)
        risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_ps <= 0 or np.isnan(risk_ps):
            return
        risk_pct = self._calculate_dynamic_risk() * confidence_factor
        size = self._calculate_position_size(p, risk_ps, risk_pct)
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.tf_initial_stop_loss = p - risk_ps
            self.highest_high_in_trade = self.data.High[-1]
        else:
            self.sell(size=size)
            self.tf_initial_stop_loss = p + risk_ps
            self.lowest_low_in_trade = self.data.Low[-1]

    def open_mr_position(self, p, is_long):
        risk_ps = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
        if risk_ps <= 0 or np.isnan(risk_ps):
            return
        risk_pct = self._calculate_dynamic_risk() * self.mr_risk_multiplier
        size = self._calculate_position_size(p, risk_ps, risk_pct)
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        if is_long:
            self.buy(size=size)
            self.mr_stop_loss = p - risk_ps
        else:
            self.sell(size=size)
            self.mr_stop_loss = p + risk_ps

    def manage_trend_following_exit(self, p):
        atr = self.tf_atr[-1]
        if np.isnan(atr):
            return  # 如果ATR无效则不操作
        if self.position.is_long:
            if p < self.tf_initial_stop_loss:
                self.close_position(exit_price=p)
                return
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            chandelier_exit = (
                self.highest_high_in_trade - atr * self.tf_chandelier_atr_multiplier
            )
            if p < chandelier_exit:
                self.close_position(exit_price=p)
        elif self.position.is_short:
            if p > self.tf_initial_stop_loss:
                self.close_position(exit_price=p)
                return
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            chandelier_exit = (
                self.lowest_low_in_trade + atr * self.tf_chandelier_atr_multiplier
            )
            if p > chandelier_exit:
                self.close_position(exit_price=p)

    def manage_mean_reversion_exit(self, p):
        if (
            self.position.is_long
            and (p >= self.mr_bb_mid[-1] or p <= self.mr_stop_loss)
        ) or (
            self.position.is_short
            and (p <= self.mr_bb_mid[-1] or p >= self.mr_stop_loss)
        ):
            self.close_position(exit_price=p)

    def _calculate_tf_entry_score(self) -> float:
        w = self.score_weights_tf
        breakout_signal = (
            1
            if self.data.High[-1] > self.tf_donchian_h[-1]
            else -1 if self.data.Low[-1] < self.tf_donchian_l[-1] else 0
        )
        momentum_signal = 1 if self.tf_ema_fast[-1] > self.tf_ema_slow[-1] else -1
        mtf_signal = self.mtf_signal[-1]
        ml_prob = self.ml_score[-1] if not np.isnan(self.ml_score[-1]) else 0.5
        ml_signal = (ml_prob - 0.5) * 2
        return (
            ml_signal * w["ml_signal"]
            + breakout_signal * w["breakout"]
            + momentum_signal * w["momentum"]
            + mtf_signal * w["mtf"]
        )

    def _define_mr_entry_signal(self) -> int:
        # 增加对NaN值的检查
        # 默认视为非 ML 触发
        self.last_mr_signal_from_ml = False
        if (
            len(self.data.Close) < 2
            or np.isnan(self.mr_bb_lower[-2])
            or np.isnan(self.mr_rsi[-2])
        ):
            return 0

        is_oversold = (
            self.data.Close[-2] < self.mr_bb_lower[-2]
            and self.mr_rsi[-2] < self.mr_rsi_oversold
        )
        is_overbought = (
            self.data.Close[-2] > self.mr_bb_upper[-2]
            and self.mr_rsi[-2] > self.mr_rsi_overbought
        )
        # 1) 经典均值回归信号：超卖/超买后出现反转（优先级最高）
        if is_oversold and self.data.Close[-1] > self.data.Close[-2]:
            self.last_mr_signal_from_ml = False
            return 1
        if is_overbought and self.data.Close[-1] < self.data.Close[-2]:
            self.last_mr_signal_from_ml = False
            return -1

        # 2) 补充：在震荡市中，如果 ML 信号足够强，则允许 MR 按 ML 方向开仓，
        #    但要求价格位置「不追高不杀低」，只在相对合理价位参与。
        ml_prob = self.ml_score[-1] if not np.isnan(self.ml_score[-1]) else 0.5
        ml_signal = (ml_prob - 0.5) * 2  # 映射到 [-1, 1]
        if abs(ml_signal) >= self.mr_ml_entry_threshold:
            # 多头：ML 看涨，且当前价格不高于中轨（避免高位追多）
            if ml_signal > 0 and self.data.Close[-1] <= self.mr_bb_mid[-1]:
                self.last_mr_signal_from_ml = True
                return 1
            # 空头：ML 看跌，且当前价格不低于中轨（避免低位追空）
            if ml_signal < 0 and self.data.Close[-1] >= self.mr_bb_mid[-1]:
                self.last_mr_signal_from_ml = True
                return -1

        return 0

    def _calculate_position_size(self, price, risk_per_share, risk_pct):
        if risk_per_share <= 0 or price <= 0:
            return 0
        risk_amount = self.equity * min(risk_pct, self.max_risk_pct)
        size = int(risk_amount / risk_per_share)
        return int(self.equity / price * 0.98) if size * price >= self.equity else size

    def _calculate_dynamic_risk(self):
        kelly_risk = self.default_risk_pct
        if len(self.recent_trade_returns) >= self.kelly_trade_history:
            wins, losses = [r for r in self.recent_trade_returns if r > 0], [
                r for r in self.recent_trade_returns if r < 0
            ]
            if wins and losses:
                win_rate, avg_win, avg_loss = (
                    len(wins) / len(self.recent_trade_returns),
                    sum(wins) / len(wins),
                    abs(sum(losses) / len(losses)),
                )
                reward_ratio = avg_win / avg_loss if avg_loss > 0 else 1e9
                kelly = win_rate - (1 - win_rate) / reward_ratio
                kelly_risk = min(max(0.005, kelly * 0.5), self.max_risk_pct)
        return kelly_risk * self._get_drawdown_risk_scale()

    def close_position(self, exit_price=None):
        # 在关闭前记录方向与该笔交易盈亏
        was_long = self.position.is_long
        was_short = self.position.is_short
        eq_before = float(self.equity) if self.equity is not None else 0.0
        trade_pl = float(self.position.pl) if self.position.pl is not None else 0.0
        # 记录价格用于不含手续费胜率统计
        entry_price = float(getattr(self, "current_entry_price", float("nan")))
        if np.isnan(entry_price):
            entry_price = None
        if exit_price is None and len(self.data.Close):
            exit_price = float(self.data.Close[-1])
        # 按原始逻辑关闭仓位，并使用账户权益变化驱动 Kelly 风控
        self.position.close()
        pnl_pct = self.equity / eq_before - 1 if eq_before != 0 else 0.0
        pnl_abs = trade_pl
        self.recent_trade_returns.append(pnl_pct)
        # 累计统计整体与基于 ML 的交易表现
        self.total_trades += 1
        if self.current_trade_is_ml_based:
            self.ml_based_trades += 1
            self.ml_based_profit += pnl_abs
            # 含手续费胜负判断
            if was_long:
                self.ml_long_trades += 1
                self.ml_long_profit += pnl_abs
                if pnl_abs > 0:
                    self.ml_wins += 1
                    self.ml_long_wins += 1
                elif pnl_abs < 0:
                    self.ml_losses += 1
                    self.ml_long_losses += 1
            elif was_short:
                self.ml_short_trades += 1
                self.ml_short_profit += pnl_abs
                if pnl_abs > 0:
                    self.ml_wins += 1
                    self.ml_short_wins += 1
                elif pnl_abs < 0:
                    self.ml_losses += 1
                    self.ml_short_losses += 1
            # 不含手续费胜负判断（仅看价格方向）
            if entry_price is not None and exit_price is not None:
                if was_long:
                    if exit_price > entry_price:
                        self.ml_wins_gross += 1
                        self.ml_long_wins_gross += 1
                    elif exit_price < entry_price:
                        self.ml_losses_gross += 1
                        self.ml_long_losses_gross += 1
                elif was_short:
                    if exit_price < entry_price:
                        self.ml_wins_gross += 1
                        self.ml_short_wins_gross += 1
                    elif exit_price > entry_price:
                        self.ml_losses_gross += 1
                        self.ml_short_losses_gross += 1
        if pnl_pct < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.paused_until_bar = len(self.data) + self.trading_pause_bars
                self.consecutive_losses = 0
        else:
            self.consecutive_losses = 0
        self.reset_trade_state()
        # 当前交易已结束，重置 ML 标记
        self.current_trade_is_ml_based = False

    def reset_trade_state(self):
        self.active_sub_strategy, self.mr_stop_loss, self.tf_initial_stop_loss = (
            None,
            0.0,
            0.0,
        )
        self.highest_high_in_trade, self.lowest_low_in_trade = 0, float("inf")


if __name__ == "__main__":
    logger.info(f"🚀 (Advanced Framework v19 - Full Features) 开始运行...")
    symbol, start_date, end_date = (
        CONFIG["symbol_to_test"],
        CONFIG["backtest_start_date"],
        CONFIG["backtest_end_date"],
    )
    # 需要足够长的历史数据来计算所有指标和滞后特征
    data_fetch_start_date = (pd.to_datetime(start_date) - timedelta(days=365)).strftime(
        "%Y-%m-%d"
    )

    logger.info(
        f"回测品种: {symbol}\n回测时间段: {start_date} to {end_date}\n数据获取起始日期: {data_fetch_start_date}"
    )
    raw_data = fetch_binance_klines(
        symbol, CONFIG["interval"], data_fetch_start_date, end_date
    )
    if raw_data.empty:
        logger.error("数据获取失败，程序终止。")
        exit()

    logger.info("### 准备完整回测数据 ###")
    processed_data = preprocess_data_for_strategy(raw_data, symbol)
    backtest_data = processed_data.loc[start_date:end_date].copy()
    if backtest_data.empty:
        logger.error("在指定回测周期内无有效数据，程序终止。")
        exit()

    logger.info("### 进入回测模式 ###")
    print("\n" + "=" * 80 + f"\n正在回测品种: {symbol}\n" + "=" * 80)
    bt = Backtest(
        backtest_data,
        UltimateStrategy,
        cash=CONFIG["initial_cash"],
        commission=CONFIG["commission"],
        finalize_trades=True,
    )
    stats = bt.run()
    print("\n" + "-" * 40 + f"\n          {symbol} 回测结果摘要\n" + "-" * 40)
    print(stats)
    # --- 基于 ML 信号的胜率统计 ---
    try:
        strat = stats.get("_strategy", None)
    except AttributeError:
        strat = stats["_strategy"] if "_strategy" in stats else None
    if strat is not None and hasattr(strat, "ml_based_trades"):
        total_trades = getattr(strat, "total_trades", 0)
        ml_trades = getattr(strat, "ml_based_trades", 0)
        ml_wins = getattr(strat, "ml_wins", 0)
        ml_losses = getattr(strat, "ml_losses", 0)
        ml_long_trades = getattr(strat, "ml_long_trades", 0)
        ml_short_trades = getattr(strat, "ml_short_trades", 0)
        ml_long_wins = getattr(strat, "ml_long_wins", 0)
        ml_short_wins = getattr(strat, "ml_short_wins", 0)

        ml_win_rate = (ml_wins / ml_trades * 100) if ml_trades > 0 else float("nan")
        ml_long_win_rate = (
            ml_long_wins / ml_long_trades * 100 if ml_long_trades > 0 else float("nan")
        )
        ml_short_win_rate = (
            ml_short_wins / ml_short_trades * 100
            if ml_short_trades > 0
            else float("nan")
        )

        print("\n" + "-" * 40 + "\n   基于ML信号的胜率（含手续费）\n" + "-" * 40)
        print(f"总成交笔数: {total_trades}")
        print(f"ML 信号相关成交笔数: {ml_trades}")
        if ml_win_rate == ml_win_rate:
            print(f"ML 信号总体胜率 [%]: {ml_win_rate:.2f}")
        else:
            print("ML 信号总体胜率 [%]: NaN")

        print("\n   ML 多头 / 空头胜率分解（含手续费）")
        print(f"ML 多头成交笔数: {ml_long_trades}")
        if ml_long_win_rate == ml_long_win_rate:
            print(f"ML 多头胜率 [%]: {ml_long_win_rate:.2f}")
        else:
            print("ML 多头胜率 [%]: NaN")
        print(f"ML 空头成交笔数: {ml_short_trades}")
        if ml_short_win_rate == ml_short_win_rate:
            print(f"ML 空头胜率 [%]: {ml_short_win_rate:.2f}")
        else:
            print("ML 空头胜率 [%]: NaN")

        # 不含手续费的胜率（仅看价格方向）
        ml_wins_gross = getattr(strat, "ml_wins_gross", 0)
        ml_losses_gross = getattr(strat, "ml_losses_gross", 0)
        ml_long_wins_gross = getattr(strat, "ml_long_wins_gross", 0)
        ml_short_wins_gross = getattr(strat, "ml_short_wins_gross", 0)

        ml_win_rate_gross = (
            ml_wins_gross / ml_trades * 100 if ml_trades > 0 else float("nan")
        )
        ml_long_win_rate_gross = (
            ml_long_wins_gross / ml_long_trades * 100
            if ml_long_trades > 0
            else float("nan")
        )
        ml_short_win_rate_gross = (
            ml_short_wins_gross / ml_short_trades * 100
            if ml_short_trades > 0
            else float("nan")
        )

        print("\n" + "-" * 40 + "\n   基于ML信号的胜率（不含手续费，仅看方向）\n" + "-" * 40)
        print(f"ML 信号相关成交笔数: {ml_trades}")
        if ml_win_rate_gross == ml_win_rate_gross:
            print(f"ML 信号总体胜率（不含手续费）[%]: {ml_win_rate_gross:.2f}")
        else:
            print("ML 信号总体胜率（不含手续费）[%]: NaN")

        print("\n   ML 多头 / 空头胜率分解（不含手续费）")
        print(f"ML 多头成交笔数: {ml_long_trades}")
        if ml_long_win_rate_gross == ml_long_win_rate_gross:
            print(f"ML 多头胜率（不含手续费）[%]: {ml_long_win_rate_gross:.2f}")
        else:
            print("ML 多头胜率（不含手续费）[%]: NaN")
        print(f"ML 空头成交笔数: {ml_short_trades}")
        if ml_short_win_rate_gross == ml_short_win_rate_gross:
            print(f"ML 空头胜率（不含手续费）[%]: {ml_short_win_rate_gross:.2f}")
        else:
            print("ML 空头胜率（不含手续费）[%]: NaN")
    # --- 按月收益率统计 ---
    try:
        equity_curve = stats["_equity_curve"]
        if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
            # 使用 'ME' (MonthEnd) 避免未来 pandas 版本中 'M' 的弃用告警
            monthly_equity = equity_curve["Equity"].resample("ME").last()
            monthly_returns = monthly_equity.pct_change().dropna() * 100
            print("\n" + "-" * 40 + f"\n          {symbol} 每月收益率 [%]\n" + "-" * 40)
            print(monthly_returns.to_frame(name="Monthly Return [%]").round(2))
    except Exception as e:
        logger.error(f"计算每月收益率时出错: {e}")
