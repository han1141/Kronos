# -*- coding: utf-8 -*-
import os
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        Conv1D,
        LSTM,
        Dense,
        Dropout,
        Input,
        BatchNormalization,
    )
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.regularizers import l2

    ML_LIBS = True
except ImportError:
    ML_LIBS = False

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    import ta
except ImportError:
    pass

import warnings
from pandas.errors import SettingWithCopyWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
if not logger.handlers:
    logger.addHandler(ch)


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


# --- CONFIGURATION ---
CONFIG = {
    "symbols_to_test": ["ETHUSDT", "BTCUSDT"],  # 扩大测试范围，验证普适性
    "btc_symbol": "BTCUSDT",
    "interval": "30m",
    "backtest_start_date": "2025-01-01",
    "backtest_end_date": "2025-10-16",
    "initial_cash": 500_000,
    "commission": 0.0002,
    "spread": 0.0005,
    "show_plots": True,
    "training_window_days": 365 * 2.2,
    # [V49.0 新增] 训练数据隔离期，防止前视偏差
    "ml_training_gap_days": 30,
    # [V49.0 新增] 策略模块开关，用于消融研究(Ablation Study)
    "enabled_modules": {
        "trend_following": True,
        "mean_reversion": True,
        "ml_filter": True,
        "lpr_mode": True,
    },
}

# --- STRATEGY PARAMETERS ---
STRATEGY_PARAMS = {
    "kelly_trade_history": 20,
    "default_risk_pct": 0.015,
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
    "regime_score_threshold": 0.5,  # 从0.45调整为0.5，避免过度拟合
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_adx_confirm_period": 14,
    "tf_adx_confirm_threshold": 18,
    "tf_chandelier_period": 22,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.5,
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 30,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,
    "mtf_period": 50,
    "score_entry_threshold": 0.5,
    "score_weights_tf": {"breakout": 0.5, "momentum": 0.3, "mtf": 0.2},
    "lpr_enabled": True,
    "lpr_trigger_pct": 0.08,
    "lpr_trail_atr_multiplier": 2.0,
    "ml_filter_enabled": True,
    "lstm_sequence_length": 48,
    "lstm_epochs": 60,
    "lstm_batch_size": 128,
    "lstm_l2_reg": 0.001,  # [V49.0 新增] LSTM L2正则化
    "xgb_nrounds": 250,
    "xgb_gamma": 0.3,  # [V49.0 新增] XGBoost Gamma正则化
    "xgb_lambda": 1.5,  # [V49.0 新增] XGBoost Lambda(L2)正则化
    "ensemble_w_lstm": 0.3,
    "ensemble_w_xgb": 0.7,
    "ml_confidence_threshold": 0.6,  # 从0.58调整为0.6，避免过度拟合
    "ml_score_weight": 0.3,
}

def fetch_binance_klines(s, i, st, en=None, l=1000, cache_dir="data_cache"):
    # ... (函数内容保持不变)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{s}_{i}.csv")
    start_dt, end_dt = pd.to_datetime(st), (
        pd.to_datetime(en) if en else datetime.utcnow()
    )
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
            if not df.empty and df.index[0] <= start_dt and df.index[-1] >= end_dt:
                logger.info(f"✅ 从缓存加载数据: {cache_file}")
                return df
        except Exception:
            pass
    logger.info(f"开始从币安API下载 {s} 数据...")
    url, all_data, current_start = (
        "https://api.binance.com/api/v3/klines",
        [],
        int(start_dt.timestamp() * 1000),
    )
    end_ts = int(end_dt.timestamp() * 1000)
    while current_start < end_ts:
        p = {
            "symbol": s.upper(),
            "interval": i,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": l,
        }
        try:
            r = requests.get(url, params=p, timeout=30)
            r.raise_for_status()
            d = r.json()
            if not d:
                break
            all_data.extend(d)
            current_start = d[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.warning(f"获取 {s} 时遇到连接问题: {e}。5秒后重试...")
            time.sleep(5)
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "ct",
            "qav",
            "nt",
            "tbb",
            "tbq",
            "ig",
        ],
    )
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df = df.set_index("timestamp").sort_index()
    df.to_csv(cache_file)
    logger.info(f"✅ 数据已缓存至: {cache_file}")
    return df


def compute_hurst(ts, max_lag=100):
    # ... (函数内容保持不变)
    if len(ts) < 10:
        return 0.5
    lags = range(2, min(max_lag, len(ts) // 2 + 1))
    tau = []
    for lag in lags:
        diff = np.subtract(ts[lag:], ts[:-lag])
        std = np.std(diff)
        if std > 0:
            tau.append(std)
    if len(tau) < 2:
        return 0.5
    try:
        return max(
            0.0, min(1.0, np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0])
        )
    except:
        return 0.5


def add_all_features(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    # ... (函数内容基本不变, 但建议在这里考虑特征选择)
    # 提示: 在此阶段或ML训练前，可以考虑进行特征重要性分析和筛选，以减少维度灾难风险。
    df = df.copy()
    p = STRATEGY_PARAMS
    # --- TA Features for Strategy ---
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
    df["regime_adx_norm"] = norm(adx)
    df["regime_atr_slope_norm"] = norm(
        (atr - atr.shift(p["regime_atr_slope_period"]))
        / atr.shift(p["regime_atr_slope_period"])
    )
    df["regime_rsi_vol_norm"] = 1 - norm(rsi.rolling(p["regime_rsi_vol_period"]).std())
    df["regime_hurst"] = (
        df.Close.rolling(p["regime_hurst_period"])
        .apply(lambda x: compute_hurst(np.log(x + 1e-9)), raw=False)
        .fillna(0.5)
    )
    df["regime_score"] = (
        df["regime_adx_norm"] * p["regime_score_weight_adx"]
        + df["regime_atr_slope_norm"] * p["regime_score_weight_atr"]
        + df["regime_rsi_vol_norm"] * p["regime_score_weight_rsi"]
        + np.clip((df["regime_hurst"] - 0.3) / 0.7, 0, 1)
        * p["regime_score_weight_hurst"]
    )
    df["market_regime"] = np.where(
        df["regime_score"] > p["regime_score_threshold"], 1, -1
    )
    df_1d = (
        df.resample("1D")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    if not df_1d.empty:
        sma = ta.trend.SMAIndicator(
            df_1d["Close"], window=p["mtf_period"]
        ).sma_indicator()
        df["mtf_signal"] = (
            pd.Series(np.where(df_1d["Close"] > sma, 1, -1), index=df_1d.index)
            .reindex(df.index, method="ffill")
            .fillna(0)
        )
    else:
        df["mtf_signal"] = 0
    # --- ML Features ---
    df["feature_log_return_1"] = np.log(df.Close / df.Close.shift(1))
    df["feature_rsi_14"] = ta.momentum.RSIIndicator(df.Close, 14).rsi()
    df["feature_atr_14_pct"] = (
        ta.volatility.AverageTrueRange(
            df.High, df.Low, df.Close, 14
        ).average_true_range()
        / df.Close
    )
    df["feature_bb_width"] = (
        ta.volatility.BollingerBands(df.Close, 20).bollinger_hband()
        - ta.volatility.BollingerBands(df.Close, 20).bollinger_lband()
    ) / df.Close
    df["feature_close_ma_200_ratio"] = (
        df.Close / ta.trend.EMAIndicator(df.Close, 200).ema_indicator()
    )
    df["feature_atr_std_ratio"] = (
        df["feature_atr_14_pct"].rolling(20).std() / df["feature_atr_14_pct"]
    )
    if btc_df is not None and not btc_df.empty:
        btc_log_ret = np.log(btc_df.Close / btc_df.Close.shift(1))
        df["feature_btc_log_return"] = btc_log_ret.reindex(df.index, method="ffill")
        df["feature_btc_corr_120"] = (
            df["feature_log_return_1"].rolling(120).corr(df["feature_btc_log_return"])
        )
    df["feature_hour"] = df.index.hour
    df["feature_dayofweek"] = df.index.dayofweek
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    return df


def make_asymmetric_labels(
    df, look_forward=30, risk_reward_ratio=2.0, sl_atr_multiplier=1.5
):
    # ... (函数内容保持不变)
    atr = ta.volatility.AverageTrueRange(
        df.High, df.Low, df.Close, 14
    ).average_true_range()
    upper_barrier = df.Close + (atr * sl_atr_multiplier * risk_reward_ratio)
    lower_barrier = df.Close - (atr * sl_atr_multiplier)
    labels = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - look_forward):
        entry_idx = df.index[i]
        path = df.iloc[i + 1 : i + 1 + look_forward]
        upper_touch_time = path[path.High >= upper_barrier.loc[entry_idx]].index.min()
        lower_touch_time = path[path.Low <= lower_barrier.loc[entry_idx]].index.min()
        if pd.notna(upper_touch_time) and pd.notna(lower_touch_time):
            labels.loc[entry_idx] = 1 if upper_touch_time <= lower_touch_time else 0
        elif pd.notna(upper_touch_time):
            labels.loc[entry_idx] = 1
        elif pd.notna(lower_touch_time):
            labels.loc[entry_idx] = 0
    return labels


def create_sequences(X, y, seq_len):
    # ... (函数内容保持不变)
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


# --- MACHINE LEARNING MODELS ---


def train_and_save_lstm(training_df, symbol, seq_len, epochs, batch):
    logger.info(f"为 {symbol} 训练 LSTM...")
    labels = make_asymmetric_labels(training_df)
    df = training_df.join(labels.rename("target")).dropna(subset=["target"])
    df.loc[:, "target"] = df["target"].astype(int)
    features = [c for c in df.columns if c.startswith("feature_")]
    # 提示: 在这里可以加入基于训练集的特征选择逻辑，例如基于方差或与目标的互信息。
    X, y = df[features].values, df["target"].values
    if len(np.unique(y)) < 2:
        logger.warning("类别不足，跳过 LSTM。")
        return None

    # [V49.0 确认] 静态分割，但严格遵守时间顺序，这是正确的做法。
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    scaler = RobustScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    joblib.dump(scaler, f"lstm_scaler_{symbol}.pkl")
    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test, seq_len)
    if len(X_train_seq) == 0:
        logger.error("序列样本不足，跳过 LSTM。")
        return None
    if np.unique(y_train_seq).size < 2 or np.unique(y_test_seq).size < 2:
        logger.warning("分割后类别不足，跳过 LSTM。")
        return None

    l2_reg = STRATEGY_PARAMS.get("lstm_l2_reg", 0.001)
    model = Sequential(
        [
            Input(shape=(seq_len, X_train_seq.shape[2])),
            Conv1D(64, kernel_size=5, activation="relu", kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(0.2),  # 增加Dropout率
            LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg)),
            Dropout(0.3),  # 增加Dropout率
            LSTM(64, kernel_regularizer=l2(l2_reg)),
            Dropout(0.3),  # 增加Dropout率
            Dense(64, activation="relu", kernel_regularizer=l2(l2_reg)),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=epochs,
        batch_size=batch,
        callbacks=[es],
        verbose=0,
    )
    loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    logger.info(f"LSTM 在时间外样本上的真实准确率: {acc:.4f}")
    model.save(f"lstm_model_{symbol}.keras")
    return acc


def train_and_save_xgb(training_df, symbol):
    logger.info(f"为 {symbol} 训练 XGBoost...")
    labels = make_asymmetric_labels(training_df)
    df = training_df.join(labels.rename("target")).dropna(subset=["target"])
    df.loc[:, "target"] = df["target"].astype(int)
    features = [c for c in df.columns if c.startswith("feature_")]
    X, y = df[features].values, df["target"].values
    if len(np.unique(y)) < 2:
        logger.warning("类别不足，跳过 XGBoost。")
        return None

    # [V49.0 确认] 静态分割，但严格遵守时间顺序。
    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    # 提示: 可以在这里基于X_train训练一个初步模型，进行特征选择，然后再用筛选后的特征训练最终模型。

    scaler = RobustScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    joblib.dump(scaler, f"xgb_scaler_{symbol}.pkl")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=STRATEGY_PARAMS["xgb_nrounds"],
        learning_rate=0.02,
        max_depth=3,  # 降低最大深度
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=STRATEGY_PARAMS.get("xgb_gamma", 0.3),  # [V49.0] 使用正则化参数
        reg_lambda=STRATEGY_PARAMS.get("xgb_lambda", 1.5),  # [V49.0] 使用正则化参数
        early_stopping_rounds=25,  # 增加早停轮次
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    logger.info(f"XGBoost 在时间外样本上的真实准确率: {acc:.4f}")
    joblib.dump(model, f"xgb_model_{symbol}.joblib")
    return acc


# --- PLOTTING & ANALYSIS ---


def plot_walk_forward_equity(
    equity_curves, combined_trades, initial_cash, title="Walk-Forward Equity Curve"
):
    # ... (函数内容保持不变)
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    full_equity_curve = pd.Series([initial_cash])
    for i, curve in enumerate(equity_curves):
        period_equity = curve["Equity"]
        if i > 0:
            start_val = full_equity_curve.iloc[-1]
            period_equity = period_equity / period_equity.iloc[0] * start_val
        full_equity_curve = pd.concat([full_equity_curve.iloc[:-1], period_equity])
    full_equity_curve.plot(
        ax=ax, label="总权益曲线 (Total Equity)", color="blue", linewidth=2.5
    )
    for curve in equity_curves:
        start_date = curve.index[0]
        ax.axvline(
            x=start_date,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            label="滚动周期分割线" if start_date == equity_curves[0].index[0] else "",
        )
    formatter = mticker.FuncFormatter(lambda x, p: f"${x:,.0f}")
    ax.yaxis.set_major_formatter(formatter)
    plt.title(title, fontsize=16)
    plt.xlabel("日期")
    plt.ylabel("账户权益 ($)")
    plt.legend()
    plt.grid(True)
    filename = "equity_curve.png"
    plt.savefig(filename)
    logger.info(f"✅ 权益曲线图已保存至: {filename}")
    plt.close(fig)


def analyze_trade_distribution(combined_trades, title="交易盈亏 (PnL) 分布"):
    # ... (函数内容保持不变)
    if combined_trades.empty:
        logger.warning("没有交易记录，无法生成收益分布图。")
        return
    pnl = combined_trades["PnL"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.hist(
        losses, bins=50, color="salmon", alpha=0.7, label=f"亏损交易 (n={len(losses)})"
    )
    ax.hist(
        wins, bins=50, color="skyblue", alpha=0.7, label=f"盈利交易 (n={len(wins)})"
    )
    stats_text = (
        f"盈利交易:\n  - 数量: {len(wins)}\n  - 平均盈利: ${wins.mean():,.2f}\n  - 盈利标准差: ${wins.std():,.2f}\n\n"
        f"亏损交易:\n  - 数量: {len(losses)}\n  - 平均亏损: ${losses.mean():,.2f}\n  - 亏损标准差: ${losses.std():,.2f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5),
    )
    plt.title(title, fontsize=16)
    plt.xlabel("单笔交易盈亏 ($)")
    plt.ylabel("交易次数")
    plt.legend()
    plt.grid(True)
    filename = "trade_distribution.png"
    plt.savefig(filename)
    logger.info(f"✅ 交易分布图已保存至: {filename}")
    plt.close(fig)


def generate_optimization_suggestions(stats, model_accuracies, config):
    # ... (函数内容保持不变)
    logger.info("\n" + "#" * 80 + "\n                 🧠 AI 优化建议器\n" + "#" * 80)
    suggestions = []
    if stats["Profit Factor"] < 1.1:
        suggestions.append(
            "🚨 核心问题: 盈亏因子低于1.1，策略盈利能力非常脆弱。\n   >> 优化方向: 优先提高信号质量，而不是增加交易频率。考虑提高 `score_entry_threshold` 以过滤掉更多弱信号。"
        )
    elif stats["Profit Factor"] < 1.3:
        suggestions.append(
            "⚠️ 注意: 盈亏因子 (1.1-1.3) 尚可，但有提升空间。\n   >> 优化方向: 审视亏损交易，检查是否能通过调整止损 (`tf_stop_loss_atr_multiplier`) 或趋势识别参数来规避部分亏损。"
        )
    else:
        suggestions.append(
            "✅ 优势: 盈亏因子 (>=1.3) 表现良好，策略具备不错的盈利能力。"
        )
    if stats["Win Rate [%]"] < 40:
        suggestions.append(
            f"⚠️ 注意: 胜率 ({stats['Win Rate [%]']:.2f}%) 偏低，可能导致回撤较大。\n   >> 优化方向: 1. 提高 `score_entry_threshold`。 2. 提高机器学习的置信度门槛 `ml_confidence_threshold`。"
        )
    else:
        suggestions.append(
            f"✅ 优势: 胜率 ({stats['Win Rate [%]']:.2f}%) 处于可接受范围。"
        )
    avg_lstm_acc = np.mean([acc.get("lstm", 0.5) for acc in model_accuracies])
    avg_xgb_acc = np.mean([acc.get("xgb", 0.5) for acc in model_accuracies])
    if avg_lstm_acc < 0.52:
        suggestions.append(
            f"📉 模型问题: LSTM 平均准确率 ({avg_lstm_acc:.2%}) 过低，几乎没有预测能力。\n   >> 优化方向: 1. 大幅降低其在集成模型中的权重 (`ensemble_w_lstm`)。 2. 考虑从集成中移除LSTM (通过 enabled_modules 设置)。"
        )
    if avg_xgb_acc < 0.53:
        suggestions.append(
            f"📉 模型问题: XGBoost 平均准确率 ({avg_xgb_acc:.2%}) 偏低。\n   >> 优化方向: 1. 进行特征选择，移除重要性低的特征。 2. 调整`make_asymmetric_labels`中的参数。"
        )
    if avg_xgb_acc > avg_lstm_acc + 0.02:
        suggestions.append(
            f"💡 模型洞察: XGBoost (acc: {avg_xgb_acc:.2%}) 表现显著优于 LSTM (acc: {avg_lstm_acc:.2%})。\n   >> 优化方向: 调整集成权重，给予XGBoost更大的话语权，例如 `ensemble_w_xgb` 调整至 0.7 或更高。"
        )
    for i, suggestion in enumerate(suggestions):
        logger.info(f"\n--- 建议 {i+1} ---\n{suggestion}")
    logger.info("\n" + "#" * 80)


# --- STRATEGY IMPLEMENTATION ---


class BaseAssetStrategy:
    # ... (类内容保持不变)
    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy

    def _calculate_entry_score(self) -> float:
        m, w = self.main, self.main.score_weights_tf
        b_s = (
            1
            if m.data.High[-1] > m.tf_donchian_h[-2]
            else -1 if m.data.Low[-1] < m.tf_donchian_l[-2] else 0
        )
        mo_s = 1 if m.tf_ema_fast[-1] > m.tf_ema_slow[-1] else -1
        return (
            b_s * w.get("breakout", 0)
            + mo_s * w.get("momentum", 0)
            + m.mtf_signal[-1] * w.get("mtf", 0)
        )

    def _define_mr_entry_signal(self) -> int:
        m = self.main
        return (
            1
            if crossover(m.data.Close, m.mr_bb_lower)
            and m.mr_rsi[-1] < m.mr_rsi_oversold
            else (
                -1
                if crossover(m.mr_bb_upper, m.data.Close)
                and m.mr_rsi[-1] > m.mr_rsi_overbought
                else 0
            )
        )


class UltimateStrategy(Strategy):
    symbol = None
    vol_weight = 1.0

    # [V49.0 新增] 用于从外部传入配置
    enabled_modules = None

    def init(self):
        # 加载基础参数
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)

        # [V49.0 修改] 从 CONFIG 加载模块开关，如果外部没传，则使用默认值
        if self.enabled_modules is None:
            self.enabled_modules = CONFIG.get("enabled_modules", {})

        # [V49.0 移除] 不再使用资产特异性参数
        self.asset_strategy = BaseAssetStrategy(self)

        c, h, l = (
            pd.Series(self.data.Close),
            pd.Series(self.data.High),
            pd.Series(self.data.Low),
        )
        self.reset_trade_state()
        self.market_regime, self.mtf_signal = self.I(
            lambda: self.data.market_regime
        ), self.I(lambda: self.data.mtf_signal)
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                h, l, c, self.tf_atr_period
            ).average_true_range()
        )
        self.tf_donchian_h, self.tf_donchian_l = self.I(
            lambda: h.rolling(self.tf_donchian_period).max().shift(1)
        ), self.I(lambda: l.rolling(self.tf_donchian_period).min().shift(1))
        self.tf_ema_fast, self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(c, self.tf_ema_fast_period).ema_indicator()
        ), self.I(
            lambda: ta.trend.EMAIndicator(c, self.tf_ema_slow_period).ema_indicator()
        )
        self.tf_adx = self.I(
            lambda: ta.trend.ADXIndicator(h, l, c, self.tf_adx_confirm_period).adx()
        )
        bb = ta.volatility.BollingerBands(c, self.mr_bb_period, self.mr_bb_std)
        self.mr_bb_upper, self.mr_bb_lower, self.mr_bb_mid = (
            self.I(lambda: bb.bollinger_hband()),
            self.I(lambda: bb.bollinger_lband()),
            self.I(lambda: bb.bollinger_mavg()),
        )
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(c, self.mr_rsi_period).rsi()
        )

        self.lstm, self.lstm_scaler, self.xgb, self.xgb_scaler = None, None, None, None
        if self.ml_filter_enabled and self.enabled_modules.get("ml_filter", False):
            try:
                self.lstm = load_model(f"lstm_model_{self.symbol}.keras")
                self.lstm_scaler = joblib.load(f"lstm_scaler_{self.symbol}.pkl")
                logger.info(f"✅ [{self.symbol}] LSTM 模型加载成功。")
            except Exception:
                self.lstm = None
            try:
                self.xgb = joblib.load(f"xgb_model_{self.symbol}.joblib")
                self.xgb_scaler = joblib.load(f"xgb_scaler_{self.symbol}.pkl")
                logger.info(f"✅ [{self.symbol}] XGBoost 模型加载成功。")
            except Exception:
                self.xgb = None

    def run_scoring_system_entry(self, price):
        ta_score = self.asset_strategy._calculate_entry_score()
        ml_boost = 0.0

        if self.ml_filter_enabled and self.enabled_modules.get("ml_filter", False):
            prob_lstm, prob_xgb = 0.5, 0.5
            features = [c for c in self.data.df.columns if c.startswith("feature_")]
            if (
                self.lstm
                and self.lstm_scaler
                and len(self.data.df) >= self.lstm_sequence_length
            ):
                seq = self.data.df[features].iloc[-self.lstm_sequence_length :].values
                if not np.isnan(seq).any():
                    scaled = self.lstm_scaler.transform(seq)
                    prob_lstm = float(
                        self.lstm.predict(np.expand_dims(scaled, 0), verbose=0)[0][0]
                    )
            if self.xgb and self.xgb_scaler:
                cur_feat = self.data.df[features].iloc[-1:].values
                if not np.isnan(cur_feat).any():
                    scaled = self.xgb_scaler.transform(cur_feat)
                    prob_xgb = float(self.xgb.predict_proba(scaled)[0, 1])
            ensemble_prob = (
                self.ensemble_w_lstm * prob_lstm + self.ensemble_w_xgb * prob_xgb
            )

            is_long_ta = ta_score > 0
            if is_long_ta and ensemble_prob > self.ml_confidence_threshold:
                ml_boost = (ensemble_prob - 0.5) * 2 * self.ml_score_weight
            elif not is_long_ta and (1 - ensemble_prob) > self.ml_confidence_threshold:
                ml_boost = ((1 - ensemble_prob) - 0.5) * 2 * self.ml_score_weight * -1

        final_score = ta_score + ml_boost
        if abs(final_score) <= self.score_entry_threshold:
            return
        self.open_tf_position(
            price, is_long=(final_score > 0), confidence_factor=abs(final_score)
        )

    def next(self):
        if len(self.data) < max(
            self.tf_donchian_period, self.tf_ema_slow_period, self.regime_norm_period
        ):
            return
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        else:
            if self.data.market_regime[-1] == 1:
                # [V49.0 修改] 检查趋势跟踪模块是否启用
                if self.enabled_modules.get("trend_following", False):
                    self.run_scoring_system_entry(self.data.Close[-1])
            else:
                # [V49.0 修改] 检查均值回归模块是否启用
                if self.enabled_modules.get("mean_reversion", False):
                    self.run_mean_reversion_entry(self.data.Close[-1])

    def run_mean_reversion_entry(self, price):
        signal = self.asset_strategy._define_mr_entry_signal()
        if signal != 0:
            self.open_mr_position(price, is_long=(signal > 0))

    def reset_trade_state(self):
        # ... (函数内容保持不变)
        self.active_sub_strategy, self.chandelier_exit_level = None, 0.0
        self.highest_high_in_trade, self.lowest_low_in_trade = 0, float("inf")
        self.mr_stop_loss, self.tf_initial_stop_loss = 0.0, 0.0
        self.lpr_is_active, self.lpr_trailing_stop = False, 0.0

    def manage_open_position(self, p):
        if self.lpr_is_active:
            self.manage_lpr_exit(p)
            return

        # [V49.0 修改] 检查LPR模块是否启用
        if (
            self.lpr_enabled
            and self.enabled_modules.get("lpr_mode", False)
            and self.position
        ):
            trade = self.trades[-1]
            entry_price = trade.entry_price
            profit_pct = (
                (p / entry_price - 1)
                if self.position.is_long
                else (entry_price / p - 1)
            )
            if profit_pct >= self.lpr_trigger_pct:
                self.lpr_is_active = True
                logger.info(
                    f"🚀 利润奔跑模式激活! 价格: {p:.2f}, 盈利: {profit_pct:.2%}"
                )
                self.manage_lpr_exit(p)
                return

        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(p)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(p)

    def open_tf_position(self, p, is_long, confidence_factor):
        # ... (函数内容保持不变)
        risk_ps = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * confidence_factor
        )
        if size <= 0:
            return
        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.tf_initial_stop_loss = p - risk_ps
            self.highest_high_in_trade = self.data.High[-1]
            self.chandelier_exit_level = (
                self.highest_high_in_trade
                - self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )
        else:
            self.sell(size=size)
            self.tf_initial_stop_loss = p + risk_ps
            self.lowest_low_in_trade = self.data.Low[-1]
            self.chandelier_exit_level = (
                self.lowest_low_in_trade
                + self.tf_atr[-1] * self.tf_chandelier_atr_multiplier
            )

    def manage_trend_following_exit(self, p):
        # ... (函数内容保持不变)
        atr = self.tf_atr[-1]
        if self.position.is_long:
            if p < self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade - atr * self.tf_chandelier_atr_multiplier
            )
            if p < self.chandelier_exit_level:
                self.close_position("TF_Chandelier")
        elif self.position.is_short:
            if p > self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade + atr * self.tf_chandelier_atr_multiplier
            )
            if p > self.chandelier_exit_level:
                self.close_position("TF_Chandelier")

    def open_mr_position(self, p, is_long):
        # ... (函数内容保持不变)
        risk_ps = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
        if risk_ps <= 0:
            return
        size = self._calculate_position_size(
            p, risk_ps, self._calculate_dynamic_risk() * self.mr_risk_multiplier
        )
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

    def manage_mean_reversion_exit(self, p):
        # ... (函数内容保持不变)
        if (
            self.position.is_long
            and (p >= self.mr_bb_mid[-1] or p <= self.mr_stop_loss)
        ) or (
            self.position.is_short
            and (p <= self.mr_bb_mid[-1] or p >= self.mr_stop_loss)
        ):
            self.close_position("MR")

    def manage_lpr_exit(self, p):
        # ... (函数内容保持不变)
        atr_dist = self.tf_atr[-1] * self.lpr_trail_atr_multiplier
        if self.position.is_long:
            new_stop = self.data.High[-1] - atr_dist
            self.lpr_trailing_stop = max(self.lpr_trailing_stop, new_stop)
            if p <= self.lpr_trailing_stop:
                self.close_position("LPR_Trail_Stop")
        elif self.position.is_short:
            new_stop = self.data.Low[-1] + atr_dist
            self.lpr_trailing_stop = (
                min(self.lpr_trailing_stop, new_stop)
                if self.lpr_trailing_stop > 0
                else new_stop
            )
            if p >= self.lpr_trailing_stop:
                self.close_position("LPR_Trail_Stop")

    def close_position(self, reason: str):
        # ... (函数内容保持不变)
        self.position.close()
        self.reset_trade_state()

    def _calculate_position_size(self, p, rps, risk_pct):
        # ... (函数内容保持不变)
        if rps <= 0 or p <= 0:
            return 0
        cash_at_risk = risk_pct * self.equity
        risk_pct_per_unit = rps / p
        if risk_pct_per_unit == 0:
            return 0
        position_size_quote = cash_at_risk / risk_pct_per_unit
        size = position_size_quote / self.equity
        return min(size, 0.99)

    def _calculate_dynamic_risk(self):
        # ... (函数内容保持不变)
        trades = self.closed_trades
        if len(trades) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight
        recent_trades = trades[-self.kelly_trade_history :]
        returns = [t.pl_pct for t in recent_trades]
        wins, losses = [r for r in returns if r > 0], [r for r in returns if r < 0]
        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight
        win_rate, avg_win, avg_loss = (
            len(wins) / len(recent_trades),
            sum(wins) / len(wins),
            abs(sum(losses) / len(losses)),
        )
        if avg_loss == 0:
            return self.max_risk_pct
        reward_ratio = avg_win / avg_loss
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight
        kelly = win_rate - (1 - win_rate) / reward_ratio
        return min(max(0.005, kelly * 0.5) * self.vol_weight, self.max_risk_pct)


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    set_chinese_font()
    CACHE_DIR = "data_cache"
    logger.info(f"🚀 (V49.0) 开始运行抗过拟合重构版...")
    backtest_start_dt, backtest_end_dt = pd.to_datetime(
        CONFIG["backtest_start_date"]
    ), pd.to_datetime(CONFIG["backtest_end_date"])
    training_window = timedelta(days=CONFIG["training_window_days"])
    overall_start_date = backtest_start_dt - training_window
    logger.info(
        f"数据获取总时间段: {overall_start_date.date()} to {backtest_end_dt.date()}"
    )

    all_featured_data = {}
    btc_data = fetch_binance_klines(
        CONFIG["btc_symbol"],
        CONFIG["interval"],
        overall_start_date,
        backtest_end_dt,
        cache_dir=CACHE_DIR,
    )

    for s in CONFIG["symbols_to_test"]:
        raw_data = fetch_binance_klines(
            s,
            CONFIG["interval"],
            overall_start_date,
            backtest_end_dt,
            cache_dir=CACHE_DIR,
        )
        if raw_data.empty:
            logger.error(f"{s} 数据获取失败，跳过。")
            continue
        logger.info(f"为 {s} 添加所有特征...")
        all_featured_data[s] = add_all_features(raw_data, btc_df=btc_data)

    if not all_featured_data:
        logger.error("所有品种数据处理失败，程序终止。")
        exit()

    walk_forward_periods = pd.date_range(
        start=backtest_start_dt, end=backtest_end_dt, freq="3MS"
    )
    all_equity_curves, all_trades, all_model_accuracies = [], pd.DataFrame(), []
    final_equity = CONFIG["initial_cash"]

    for i, period_start in enumerate(walk_forward_periods):
        period_end = (
            walk_forward_periods[i + 1] - timedelta(seconds=1)
            if i + 1 < len(walk_forward_periods)
            else backtest_end_dt
        )
        training_end_dt = period_start - timedelta(seconds=1)

        # [V49.0 核心修改] 创建带有隔离期的训练数据结束点
        gap_days = CONFIG.get("ml_training_gap_days", 30)
        training_end_with_gap = training_end_dt - timedelta(days=gap_days)

        print(
            "\n"
            + "#" * 80
            + f"\n滚动周期 {i+1}: 回测 {period_start.date()} to {period_end.date()}\n"
            + "#" * 80
        )

        for symbol, data in all_featured_data.items():
            period_accuracies = {}
            if (
                STRATEGY_PARAMS["ml_filter_enabled"]
                and CONFIG["enabled_modules"].get("ml_filter", False)
                and ML_LIBS
            ):
                logger.info(
                    f"准备训练数据，截止于: {training_end_with_gap.date()} (已应用 {gap_days}-day 隔离期)"
                )
                training_slice = data.loc[
                    :training_end_with_gap
                ]  # 使用带有隔离期的数据

                if len(training_slice) > 1000:
                    acc_lstm = train_and_save_lstm(
                        training_slice,
                        symbol,
                        STRATEGY_PARAMS["lstm_sequence_length"],
                        STRATEGY_PARAMS["lstm_epochs"],
                        STRATEGY_PARAMS["lstm_batch_size"],
                    )
                    acc_xgb = train_and_save_xgb(training_slice, symbol)
                    if acc_lstm is not None:
                        period_accuracies["lstm"] = acc_lstm
                    if acc_xgb is not None:
                        period_accuracies["xgb"] = acc_xgb
                else:
                    logger.warning(f"[{symbol}] 训练数据不足 (<1000)，跳过本周期训练。")
            if period_accuracies:
                all_model_accuracies.append(period_accuracies)

            backtest_slice = data.loc[period_start:period_end].copy().dropna()
            if backtest_slice.empty:
                continue

            logger.info(f"开始回测 {symbol}...")
            bt = Backtest(
                backtest_slice,
                UltimateStrategy,
                cash=final_equity,
                commission=CONFIG["commission"],
                finalize_trades=True,
            )
            # [V49.0 修改] 将模块开关配置传入策略
            stats = bt.run(symbol=symbol, enabled_modules=CONFIG["enabled_modules"])

            final_equity = stats["Equity Final [$]"]
            all_equity_curves.append(stats["_equity_curve"])
            if "_trades" in stats and not stats["_trades"].empty:
                trades_df = stats["_trades"]
                trades_df["Rolling Period"] = i + 1
                all_trades = (
                    pd.concat([all_trades, trades_df], ignore_index=True)
                    if not all_trades.empty
                    else trades_df
                )
            print(stats.get("_trades", "本周期无交易。"))

    if not all_trades.empty:
        final_stats = {
            "Profit Factor": (
                all_trades[all_trades["PnL"] > 0]["PnL"].sum()
                / abs(all_trades[all_trades["PnL"] < 0]["PnL"].sum())
            ),
            "Win Rate [%]": len(all_trades[all_trades["ReturnPct"] > 0])
            / len(all_trades)
            * 100,
            "# Trades": len(all_trades),
        }
        logger.info(
            "\n" + "#" * 80 + "\n                 滚动回测表现总览\n" + "#" * 80
        )
        logger.info(f"总初始资金: ${CONFIG['initial_cash']:,.2f}")
        logger.info(f"总最终权益: ${final_equity:,.2f}")
        logger.info(
            f"总回报率: {(final_equity / CONFIG['initial_cash'] - 1) * 100:.2f}%"
        )
        logger.info(f"总交易次数: {final_stats['# Trades']}")
        logger.info(f"整体胜率: {final_stats['Win Rate [%]']:.2f}%")
        logger.info(f"整体盈亏因子 (Profit Factor): {final_stats['Profit Factor']:.2f}")
        plot_walk_forward_equity(all_equity_curves, all_trades, CONFIG["initial_cash"])
        analyze_trade_distribution(all_trades)
        generate_optimization_suggestions(final_stats, all_model_accuracies, CONFIG)
    else:
        logger.info("整个回测期间没有产生任何交易。")
