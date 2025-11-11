# -*- coding: utf-8 -*-
# File: gork8_final.py (Single-File Version)

# --- 1. 导入所有需要的库 ---
import pandas as pd
import numpy as np
import ta
import joblib
import lightgbm as lgb
import time
import requests
from datetime import datetime, timedelta
import logging
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager
from collections import deque
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
        plt.rcParams["font.sans-serif"], plt.rcParams["axes.unicode_minus"] = [
            font
        ], False
        logger.info(f"成功设置中文字体: {font}")
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")


set_chinese_font()

# --- 3. 共享的函数和配置 (训练和回测都会用到) ---

# 全局配置
CONFIG = {
    "symbol": "ETHUSDT",
    "interval": "15m",
    # 回测数据周期
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-11-10",
    "data_lookback_days": 100,
    # 模型相关
    "feature_lookback": 60,
    # 仅推理用的序列长度（训练功能已移除）
    "label_look_forward": 24,  # 保留占位，避免历史模型依赖出错
    "output_model_path": "models/eth_trend_artifacts_15m.joblib",
    # 回测相关
    "initial_cash": 500_000,
    "commission": 0.00085,
    "spread": 0.0002,
    "show_plots": False,
}


# 共享的数据获取函数
def fetch_binance_klines(s, i, st, en=None, l=1000):
    url, cols = "https://api.binance.com/api/v3/klines", [
        "timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    sts, ets = int(pd.to_datetime(st).timestamp() * 1000), int(
        pd.to_datetime(en).timestamp() * 1000
    )
    all_d = []
    while sts < ets:
        p = {"symbol": s.upper(), "interval": i, "startTime": sts, "limit": l}
        try:
            r = requests.get(url, params=p, timeout=15)
            r.raise_for_status()
            d = r.json()
            if not d:
                break
            all_d.extend(d)
            sts = d[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.error(f"获取 {s} 失败: {e}")
            time.sleep(5)
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d, columns=[*cols, "c1", "c2", "c3", "c4", "c5", "c6"])[
        cols
    ].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()


# 共享的特征工程函数
def feature_engineering_for_regime_model(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["RSI_14"] = ta.momentum.RSIIndicator(x["Close"], window=14).rsi()
    macd = ta.trend.MACD(x["Close"], window_fast=12, window_slow=26, window_sign=9)
    x["MACD_12_26_9"] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(x["High"], x["Low"], x["Close"], window=14)
    x["ADX_14"] = adx.adx()
    bb = ta.volatility.BollingerBands(x["Close"], window=20, window_dev=2)
    x["BB_WIDTH_20"] = bb.bollinger_wband()
    x["ATR_14"] = ta.volatility.AverageTrueRange(
        x["High"], x["Low"], x["Close"], window=14
    ).average_true_range()
    close_4h = df["Close"].resample("4h").last()
    ema_4h_50 = ta.trend.EMAIndicator(close_4h, window=50).ema_indicator()
    x["ema_4h_50"] = ema_4h_50.reindex(x.index, method="ffill")
    x["price_dist_ema_4h"] = (x["Close"] - x["ema_4h_50"]) / x["Close"]

    def get_hurst(ts):
        if len(ts) < 100:
            return 0.5
        lags = range(2, 100)
        tau = [pd.Series(ts).diff(lag).std() for lag in lags]
        tau = [t for t in tau if t > 0]
        if len(tau) < 2:
            return 0.5
        lags = range(2, len(tau) + 2)
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    x["hurst_100"] = x["Close"].rolling(window=100).apply(get_hurst, raw=False)
    return x.replace([np.inf, -np.inf], np.nan)


# --- 4. 训练模式相关函数 ---


    # 训练功能已移除：仅支持加载既有模型工件进行推理


# --- 5. 回测模式相关函数和类 ---

_TREND_ARTIFACTS_CACHE = None


def _load_trend_artifacts():
    global _TREND_ARTIFACTS_CACHE
    if _TREND_ARTIFACTS_CACHE:
        return _TREND_ARTIFACTS_CACHE
    if not os.path.exists(CONFIG["output_model_path"]):
        logger.warning(
            f"未找到模型文件: {CONFIG['output_model_path']}。请先运行训练模式。"
        )
        return None
    try:
        _TREND_ARTIFACTS_CACHE = joblib.load(CONFIG["output_model_path"])
        logger.info("✅ 已加载市场状态分类模型。")
        return _TREND_ARTIFACTS_CACHE
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None


def _predict_trend_ranging_series(df_ohlcv: pd.DataFrame, arts: dict) -> pd.Series:
    if arts is None or df_ohlcv.empty:
        return pd.Series(dtype=float)
    model, scaler, feat_cols = arts["model"], arts["scaler"], arts["feature_columns"]
    flat_cols = arts.get("flattened_columns")

    Xdf = feature_engineering_for_regime_model(df_ohlcv)
    missing = [c for c in feat_cols if c not in Xdf.columns]
    if missing:
        for c in missing:
            Xdf[c] = 0
    Xdf = Xdf.reindex(columns=feat_cols, fill_value=0).fillna(0)

    X_scaled = scaler.transform(Xdf.values)
    lookback = CONFIG["feature_lookback"]
    if len(X_scaled) <= lookback:
        return pd.Series(dtype=float)

    preds = []
    for i in range(lookback, len(X_scaled)):
        seq = X_scaled[i - lookback : i].flatten().reshape(1, -1)
        in_df = pd.DataFrame(seq, columns=flat_cols)
        preds.append(int(model.predict(in_df)[0]))

    return pd.Series(preds, index=df_ohlcv.index[lookback:], name="trend_range_label")


def preprocess_data_for_strategy(data_in: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """仅使用训练好的模型识别市场状态，去除 ADX 判别与对比模式。

    输出列:
    - market_regime: Trending=1, Ranging=-1, Unknown=0
    - macro_trend_filter: 4h EMA(50) 上下的方向过滤
    """
    df = data_in.copy()
    logger.info(f"[{symbol}] 开始数据预处理...")

    df["market_regime"] = 0

    artifacts = _load_trend_artifacts()
    if artifacts:
        try:
            pred_sr = _predict_trend_ranging_series(df, artifacts)
            if not pred_sr.empty:
                # 训练时约定 0=Trending, 1=Ranging；映射为: Trending=1, Ranging=-1
                df.loc[pred_sr.index, "market_regime"] = np.where(
                    pred_sr == 1, -1, 1
                )
        except Exception as e:
            logger.error(f"[{symbol}] 应用模型失败: {e}")
    else:
        logger.warning(f"[{symbol}] 未找到模型工件，regime 将为 Unknown(0)")

    df_4h = df["Close"].resample("4h").last().to_frame()
    df_4h["macro_ema"] = ta.trend.EMAIndicator(
        df_4h["Close"], window=50
    ).ema_indicator()
    df_4h["macro_trend"] = np.where(df_4h["Close"] > df_4h["macro_ema"], 1, -1)
    df["macro_trend_filter"] = (
        df_4h["macro_trend"].reindex(df.index, method="ffill").fillna(0)
    )
    df.dropna(subset=["macro_trend_filter"], inplace=True)
    cnt = (
        df["market_regime"]
        .map({1: "Trending", -1: "Ranging", 0: "Unknown"})
        .value_counts(normalize=True)
        * 100
    )
    logger.info(
        f"[{symbol}] 模型输出市场识别: Trending={cnt.get('Trending',0):.2f}%, Ranging={cnt.get('Ranging',0):.2f}%, Unknown={cnt.get('Unknown',0):.2f}%"
    )
    logger.info(f"[{symbol}] 数据预处理完成，行数: {len(df)}")
    return df


# 策略参数
STRATEGY_PARAMS = {
    "tsl_enabled": True,
    "tsl_activation_profit_pct": 0.005,
    "tsl_activation_atr_mult": 1.5,
    "tsl_trailing_atr_mult": 2.0,
    "default_risk_pct": 0.015,
    "max_risk_pct": 0.04,
    "kelly_trade_history": 20,
    "tf_atr_period": 14,
    "tf_ema_fast_window": 20,
    "tf_ema_slow_window": 50,
    "tf_stop_loss_atr_multiplier": 2.0,
    "tf_risk_multiplier": 1.0,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mr_rsi_period": 14,
    "daily_max_entries": 2,
}
ASSET_SPECIFIC_OVERRIDES = {"ETHUSDT": {"strategy_class": "ETHStrategy"}}


# 策略类
class BaseAssetStrategy:
    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy


class ETHStrategy(BaseAssetStrategy):
    pass


STRATEGY_MAPPING = {"BaseAssetStrategy": BaseAssetStrategy, "ETHStrategy": ETHStrategy}


class UltimateStrategy(Strategy):
    symbol = None
    vol_weight = 1.0

    def init(self):
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)
        strategy_class_name = ASSET_SPECIFIC_OVERRIDES.get(self.symbol, {}).get(
            "strategy_class", "BaseAssetStrategy"
        )
        self.asset_strategy = STRATEGY_MAPPING.get(
            strategy_class_name, BaseAssetStrategy
        )(self)
        self.market_regime = self.I(lambda: self.data.market_regime)
        self.macro_trend = self.I(lambda: self.data.macro_trend_filter)
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                pd.Series(self.data.High),
                pd.Series(self.data.Low),
                pd.Series(self.data.Close),
                self.tf_atr_period,
            ).average_true_range()
        )
        # 趋势交易用 EMA
        self.tf_ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(
                pd.Series(self.data.Close), window=self.tf_ema_fast_window
            ).ema_indicator()
        )
        self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(
                pd.Series(self.data.Close), window=self.tf_ema_slow_window
            ).ema_indicator()
        )
        bb = ta.volatility.BollingerBands(
            pd.Series(self.data.Close),
            window=self.mr_bb_period,
            window_dev=self.mr_bb_std,
        )
        self.mr_bb_upper = self.I(lambda: bb.bollinger_hband())
        self.mr_bb_middle = self.I(lambda: bb.bollinger_mavg())
        self.mr_bb_lower = self.I(lambda: bb.bollinger_lband())
        stoch_rsi = ta.momentum.StochRSIIndicator(
            pd.Series(self.data.Close), window=14, smooth1=3, smooth2=3
        )
        self.mr_stoch_rsi_k = self.I(lambda: stoch_rsi.stochrsi_k())
        self.mr_stoch_rsi_d = self.I(lambda: stoch_rsi.stochrsi_d())
        self.recent_trade_returns = deque(maxlen=self.kelly_trade_history)
        self.reset_trade_state()

    def _calculate_position_size(self, p, rps, risk_pct):
        if rps <= 0 or p <= 0:
            return 0
        risk_amount = self.equity * min(max(float(risk_pct), 0.0), self.max_risk_pct)
        units = risk_amount / rps
        if units * p > self.equity:
            units = (self.equity * 0.95) / p
        return max(1, int(units))

    def _calculate_dynamic_risk(self):
        if len(self.recent_trade_returns) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        wins = [r for r in self.recent_trade_returns if r > 0]
        losses = [r for r in self.recent_trade_returns if r < 0]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate = len(wins) / len(self.recent_trade_returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0
        if avg_loss == 0:
            return self.default_risk_pct * self.vol_weight
        rr = avg_win / avg_loss if avg_loss else 0
        kelly = win_rate - (1 - win_rate) / rr if rr != 0 else 0
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)

    def _update_day_counter(self):
        try:
            cur_ts = self.data.df.index[-1]
        except Exception:
            return
        cur_day = cur_ts.date()
        if getattr(self, "_current_day", None) != cur_day:
            self._current_day = cur_day
            self._entries_today = 0

    def _can_enter_today(self) -> bool:
        self._update_day_counter()
        return getattr(self, "_entries_today", 0) < self.daily_max_entries

    def _mark_entered(self):
        self._update_day_counter()
        self._entries_today = getattr(self, "_entries_today", 0) + 1

    def manage_mean_reversion_exit(self, p):
        mid = float(self.mr_bb_middle[-1]) if np.isfinite(self.mr_bb_middle[-1]) else p
        if (self.position.is_long and (p >= mid or p <= self.stop_loss_price)) or (
            self.position.is_short and (p <= mid or p >= self.stop_loss_price)
        ):
            self.close_position("MR_Exit")

    def manage_trend_exit(self, p):
        # 趋势跟随退出与移动止损
        if not self.position:
            return
        atr = float(self.tf_atr[-1]) if np.isfinite(self.tf_atr[-1]) else 0.0
        fast_now = float(self.tf_ema_fast[-1]) if np.isfinite(self.tf_ema_fast[-1]) else p
        slow_now = float(self.tf_ema_slow[-1]) if np.isfinite(self.tf_ema_slow[-1]) else p
        fast_prev = (
            float(self.tf_ema_fast[-2])
            if len(self.tf_ema_fast) >= 2 and np.isfinite(self.tf_ema_fast[-2])
            else fast_now
        )
        slow_prev = (
            float(self.tf_ema_slow[-2])
            if len(self.tf_ema_slow) >= 2 and np.isfinite(self.tf_ema_slow[-2])
            else slow_now
        )

        # 移动止损激活与更新
        if self.tsl_enabled and atr > 0:
            current_price = float(self.data.Close[-1])
            entry_price = (
                float(self.trades[-1].entry_price)
                if len(self.trades) > 0 and hasattr(self.trades[-1], "entry_price")
                else current_price
            )
            if self.position.is_long:
                profit_pct = current_price / entry_price - 1.0
                if (not getattr(self, "tsl_active", False)) and profit_pct >= float(self.tsl_activation_profit_pct):
                    self.tsl_active = True
                if getattr(self, "tsl_active", False):
                    new_sl = current_price - atr * float(self.tsl_trailing_atr_mult)
                    self.stop_loss_price = max(self.stop_loss_price, new_sl)
            elif self.position.is_short:
                profit_pct = entry_price / current_price - 1.0
                if (not getattr(self, "tsl_active", False)) and profit_pct >= float(self.tsl_activation_profit_pct):
                    self.tsl_active = True
                if getattr(self, "tsl_active", False):
                    new_sl = current_price + atr * float(self.tsl_trailing_atr_mult)
                    self.stop_loss_price = min(self.stop_loss_price, new_sl)

        # 趋势反转或触发止损则退出
        macro = self.macro_trend[-1]
        if self.position.is_long:
            trend_reversal = (fast_prev >= slow_prev and fast_now < slow_now)
            if p <= self.stop_loss_price or macro == -1 or trend_reversal:
                self.close_position("TF_Exit")
        elif self.position.is_short:
            trend_reversal = (fast_prev <= slow_prev and fast_now > slow_now)
            if p >= self.stop_loss_price or macro == 1 or trend_reversal:
                self.close_position("TF_Exit")

    def close_position(self, reason: str):
        eq_before = self.equity
        self.position.close()
        try:
            self.recent_trade_returns.append(self.equity / eq_before - 1)
        except Exception:
            pass
        self.reset_trade_state()

    def reset_trade_state(self):
        self.active_sub_strategy = None
        self.stop_loss_price = 0.0
        self.tsl_active = False

    def next(self):
        price = self.data.Close[-1]
        # 已持仓：按趋势策略管理；旧 MR 持仓也能兼容退出
        if self.position:
            if self.active_sub_strategy == "TF":
                self.manage_trend_exit(price)
            else:
                self.manage_mean_reversion_exit(price)
            return

        # 仅在“趋势”市场交易；震荡不交易
        if self.market_regime[-1] != 1:
            return

        macro = self.macro_trend[-1]
        if len(self.tf_ema_fast) < 2 or len(self.tf_ema_slow) < 2:
            return

        fast_now = self.tf_ema_fast[-1]
        slow_now = self.tf_ema_slow[-1]
        fast_prev = self.tf_ema_fast[-2]
        slow_prev = self.tf_ema_slow[-2]

        # 趋势方向由宏观过滤给定，入场信号用均线金叉/死叉
        long_signal = macro == 1 and (fast_now > slow_now and fast_prev <= slow_prev)
        short_signal = macro == -1 and (fast_now < slow_now and fast_prev >= slow_prev)

        if long_signal and self._can_enter_today():
            atr = self.tf_atr[-1]
            risk_ps = atr * self.tf_stop_loss_atr_multiplier
            size = self._calculate_position_size(
                price, risk_ps, self._calculate_dynamic_risk() * self.tf_risk_multiplier
            )
            if size > 0:
                self.reset_trade_state()
                self.active_sub_strategy = "TF"
                self.buy(size=size)
                self.stop_loss_price = price - risk_ps
                self._mark_entered()
            return

        if short_signal and self._can_enter_today():
            atr = self.tf_atr[-1]
            risk_ps = atr * self.tf_stop_loss_atr_multiplier
            size = self._calculate_position_size(
                price, risk_ps, self._calculate_dynamic_risk() * self.tf_risk_multiplier
            )
            if size > 0:
                self.reset_trade_state()
                self.active_sub_strategy = "TF"
                self.sell(size=size)
                self.stop_loss_price = price + risk_ps
                self._mark_entered()


# 回测主流程
def run_backtest():
    logger.info("====== 进入回测模式 ======")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"])
    data_lookback = timedelta(days=CONFIG["data_lookback_days"])
    data_fetch_start_date = (backtest_start_dt - data_lookback).strftime("%Y-%m-%d")
    raw_data = {
        s: fetch_binance_klines(
            s, CONFIG["interval"], data_fetch_start_date, CONFIG["backtest_end_date"]
        )
        for s in [CONFIG["symbol"]]
    }
    raw_data = {s: d for s, d in raw_data.items() if not d.empty}
    if not raw_data:
        logger.error("数据获取失败，终止。")
        return

    for symbol, data in raw_data.items():
        logger.info(f"为 {symbol} 预处理数据...")
        processed_full = preprocess_data_for_strategy(data, symbol)
        data_slice = processed_full.loc[CONFIG["backtest_start_date"] :].copy()
        if data_slice.empty:
            logger.warning(f"{symbol} 在模型模式下无数据，跳过。")
            continue

        print(f"\n{'='*80}\n正在回测品种: {symbol}\n{'='*80}")
        bt = Backtest(
            data_slice,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
            finalize_trades=True,
        )
        stats = bt.run(symbol=symbol)
        print(f"\n{'-'*40}\n          {symbol} 回测结果摘要\n{'-'*40}")
        print(stats)

        # 附加汇总：分月收益、长/短PnL、手续费占比、单笔收益分布
        try:
            trades = stats.get("_trades", None)
            equity_curve = stats.get("_equity_curve", None)
            commissions = float(stats.get("Commissions [$]", 0.0))
            eq_final = float(stats.get("Equity Final [$]", np.nan))
            eq_start = float(CONFIG.get("initial_cash", np.nan))
            net_profit = eq_final - eq_start if np.isfinite(eq_final) and np.isfinite(eq_start) else np.nan
            gross_profit = (net_profit + commissions) if np.isfinite(net_profit) else np.nan

            print(f"\n{'-'*40}\n          {symbol} 附加汇总\n{'-'*40}")

            # 1) 分月收益
            if equity_curve is not None and not getattr(equity_curve, "empty", True):
                try:
                    ec = equity_curve.copy()
                    # 兼容不同列名，优先 'Equity'
                    eq_col = "Equity" if "Equity" in ec.columns else ec.columns[0]
                    ec = ec[[eq_col]].rename(columns={eq_col: "Equity"})
                    # Pandas 2.2+ deprecates alias "M"; use "ME" (month-end)
                    monthly = ec["Equity"].resample("ME").agg(["first", "last"])  # type: ignore
                    monthly["Return_%"] = (monthly["last"] / monthly["first"] - 1.0) * 100
                    print("分月收益 [%]:")
                    for idx, row in monthly.iterrows():
                        print(f"  {idx.strftime('%Y-%m')}: {row['Return_%']:.2f}%")
                except Exception as e:
                    print(f"分月收益统计失败: {e}")
            else:
                print("分月收益: 无 equity 曲线数据")

            # 2) 长/短侧 PnL 汇总
            if trades is not None and not trades.empty:
                try:
                    # 兼容列名
                    pnl_col = "PnL" if "PnL" in trades.columns else None
                    size_col = "Size" if "Size" in trades.columns else None
                    ret_col = "Return [%]" if "Return [%]" in trades.columns else None
                    if pnl_col and size_col:
                        long_pnl = trades.loc[trades[size_col] > 0, pnl_col].sum()
                        short_pnl = trades.loc[trades[size_col] < 0, pnl_col].sum()
                        print(f"长侧PnL: {long_pnl:.2f} | 短侧PnL: {short_pnl:.2f}")
                        print(f"长侧笔数: {(trades[size_col] > 0).sum()} | 短侧笔数: {(trades[size_col] < 0).sum()}")
                    else:
                        print("长/短PnL: 缺少列，跳过")

                    # 3) 手续费占比
                    if np.isfinite(gross_profit) and gross_profit != 0:
                        print(
                            f"手续费: ${commissions:,.2f} | 占毛收益比: {commissions / abs(gross_profit) * 100:.2f}% | 单笔均值: ${commissions / max(1, len(trades)):.2f}"
                        )
                    else:
                        print(f"手续费: ${commissions:,.2f}")

                    # 4) 单笔收益分布
                    if ret_col and ret_col in trades.columns:
                        r = trades[ret_col].dropna().astype(float)
                        if not r.empty:
                            qs = r.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).values
                            print(
                                "单笔收益[%] 分位: min={:.2f}, p10={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, p90={:.2f}, max={:.2f}".format(
                                    *qs
                                )
                            )
                            wins = (r > 0).sum()
                            losses = (r <= 0).sum()
                            print(
                                f"交易分布: 胜 {wins} / 负 {losses} | 胜均值 {r[r>0].mean():.2f}% | 负均值 {r[r<=0].mean():.2f}%"
                            )
                        else:
                            print("单笔收益分布: 无数据")
                    elif pnl_col and pnl_col in trades.columns:
                        r = trades[pnl_col].dropna().astype(float)
                        if not r.empty:
                            qs = r.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).values
                            print(
                                "单笔PnL[$] 分位: min={:.2f}, p10={:.2f}, p25={:.2f}, p50={:.2f}, p75={:.2f}, p90={:.2f}, max={:.2f}".format(
                                    *qs
                                )
                            )
                            wins = (r > 0).sum()
                            losses = (r <= 0).sum()
                            print(
                                f"交易分布: 胜 {wins} / 负 {losses} | 胜均值 ${r[r>0].mean():.2f} | 负均值 ${r[r<=0].mean():.2f}"
                            )
                        else:
                            print("单笔收益分布: 无数据")
                except Exception as e:
                    print(f"长短PnL/分布统计失败: {e}")
            else:
                print("交易明细: 无 _trades 数据")
        except Exception as e:
            print(f"附加汇总生成失败: {e}")
        if CONFIG["show_plots"]:
            bt.plot()


# --- 6. 主程序入口 ---
if __name__ == "__main__":
    # 仅运行回测：模型训练功能已移除
    run_backtest()
