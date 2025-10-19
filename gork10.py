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

# 尝试导入机器学习相关的库
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
    print(
        "警告: 未找到一个或多个机器学习库 (xgboost, scikit-learn, tensorflow)。ML功能将被禁用。"
    )


# 尝试导入回测相关的库
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    import ta
except ImportError:
    print(
        "错误: 未找到回测库 (backtesting, ta)。请运行 'pip install backtesting ta' 进行安装。"
    )
    exit()


import warnings
from pandas.errors import SettingWithCopyWarning

# --- 全局设置 ---
# 忽略特定的警告信息，保持输出整洁
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 配置日志记录器
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
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def set_chinese_font():
    """
    自动查找并设置可用的中文字体，用于matplotlib绘图。
    """
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
        logger.warning("未找到指定的中文字体，绘图可能出现乱码。")
    except Exception as e:
        logger.error(f"设置中文字体时出错: {e}")


# --- CONFIGURATION ---
# 核心配置文件，用于调整策略行为和回测设置
CONFIG = {
    "symbols_to_test": ["ETHUSDT"],  # 测试的交易对列表
    "btc_symbol": "BTCUSDT",  # 用于计算相关性的基准资产
    "interval": "1h",  # K线时间周期
    "backtest_start_date": "2025-05-01",  # 回测开始日期
    "backtest_end_date": "2025-10-18",  # 回测结束日期
    "initial_cash": 500_000,  # 初始资金
    "commission": 0.001,  # 交易手续费
    "spread": 0.0005,  # 模拟买卖价差
    "show_plots": False,  # 是否显示并保存结果图表
    "training_window_days": 365 * 3,  # 机器学习模型训练数据的时间窗口（天）
    "ml_training_gap_days": 30,  # 训练数据和回测数据之间的时间隔离期，防止前视偏差
    # [V49.1 新增] 开关：是否使用已存在的ML模型进行回测。
    # 若为True，则跳过训练过程，直接加载本地模型文件。这在调试非ML参数时非常有用。
    "use_pretrained_models": True,
    # 策略模块开关，用于进行消融研究(Ablation Study)或关闭特定逻辑
    "enabled_modules": {
        "trend_following": True,  # 趋势跟踪模块
        "mean_reversion": True,  # 均值回归模块
        "ml_filter": True,  # 机器学习过滤器模块
        "lpr_mode": True,  # 利润奔跑模式 (LPR)
    },
}

# --- STRATEGY PARAMETERS ---
# 策略内部使用的详细参数，用于技术指标和交易逻辑
STRATEGY_PARAMS = {
    # 风险管理
    "kelly_trade_history": 20,  # 计算凯利公式所用的历史交易次数
    "default_risk_pct": 0.015,  # 默认单笔风险百分比
    "max_risk_pct": 0.04,  # 最大单笔风险百分比
    # 市场状态判断 (Regime Filter)
    "regime_adx_period": 14,
    "regime_atr_period": 14,
    "regime_atr_slope_period": 5,
    "regime_rsi_period": 14,
    "regime_rsi_vol_period": 14,
    "regime_norm_period": 252,  # 归一化周期
    "regime_hurst_period": 100,  # Hurst指数计算周期
    "regime_score_weight_adx": 0.6,
    "regime_score_weight_atr": 0.3,
    "regime_score_weight_rsi": 0.05,
    "regime_score_weight_hurst": 0.05,
    "regime_score_threshold": 0.55,  # 区分趋势/震荡市场的阈值
    # 趋势跟踪模块 (Trend Following)
    "tf_donchian_period": 30,
    "tf_ema_fast_period": 20,
    "tf_ema_slow_period": 75,
    "tf_adx_confirm_period": 14,
    "tf_adx_confirm_threshold": 18,
    "tf_chandelier_period": 22,
    "tf_chandelier_atr_multiplier": 3.0,
    "tf_atr_period": 14,
    "tf_stop_loss_atr_multiplier": 2.5,
    # 均值回归模块 (Mean Reversion)
    "mr_bb_period": 20,
    "mr_bb_std": 2.0,
    "mr_rsi_period": 14,
    "mr_rsi_oversold": 30,
    "mr_rsi_overbought": 70,
    "mr_stop_loss_atr_multiplier": 1.5,
    "mr_risk_multiplier": 0.5,  # 均值回归相对趋势跟踪的风险系数
    # 多时间框架信号
    "mtf_period": 50,  # 日线级别SMA周期
    # 进场评分系统
    "score_entry_threshold": 0.65,
    "score_weights_tf": {"breakout": 0.5, "momentum": 0.3, "mtf": 0.2},
    # 利润奔跑模式 (Let Profits Run)
    "lpr_enabled": True,
    "lpr_trigger_pct": 0.05,  # 激活LPR模式的盈利百分比
    "lpr_trail_atr_multiplier": 2.0,  # LPR模式下的移动止损ATR倍数
    # 机器学习过滤器
    "ml_filter_enabled": True,
    "lstm_sequence_length": 48,  # LSTM模型输入序列长度
    "lstm_epochs": 60,
    "lstm_batch_size": 128,
    "lstm_l2_reg": 0.001,  # LSTM L2正则化系数
    "xgb_nrounds": 250,
    "xgb_gamma": 0.3,  # XGBoost Gamma正则化
    "xgb_lambda": 1.5,  # XGBoost Lambda(L2)正则化
    "ensemble_w_lstm": 0.4,  # 集成模型中LSTM的权重
    "ensemble_w_xgb": 0.6,  # 集成模型中XGBoost的权重
    "ml_confidence_threshold": 0.65,  # ML模型预测概率的置信度阈值
    "ml_score_weight": 0.2,  # ML预测结果对最终得分的贡献权重
}


def fetch_binance_klines(
    symbol, interval, start_str, end_str=None, limit=1000, cache_dir="data_cache"
):
    """
    从币安API获取K线数据，并实现本地缓存机制。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.csv")

    start_dt = pd.to_datetime(start_str, utc=True)
    end_dt = pd.to_datetime(end_str, utc=True) if end_str else datetime.utcnow()

    # 尝试从缓存加载数据
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
            if not df.empty and df.index[0] <= start_dt and df.index[-1] >= end_dt:
                logger.info(f"✅ 从缓存加载 {symbol} 数据: {cache_file}")
                return df.loc[start_dt:end_dt]
        except Exception as e:
            logger.warning(f"读取缓存文件 {cache_file} 失败: {e}")

    logger.info(f"开始从币安API下载 {symbol} 数据...")
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    current_start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    while current_start_ts < end_ts:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": current_start_ts,
            "endTime": end_ts,
            "limit": limit,
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            current_start_ts = data[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.warning(f"获取 {symbol} 时遇到连接问题: {e}。5秒后重试...")
            time.sleep(5)

    if not all_data:
        logger.error(f"未能获取到 {symbol} 的任何数据。")
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
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df = df.set_index("timestamp").sort_index()

    # 保存到缓存
    df.to_csv(cache_file)
    logger.info(f"✅ {symbol} 数据已缓存至: {cache_file}")
    return df.loc[start_dt:end_dt]


def compute_hurst(ts, max_lag=100):
    """计算时间序列的Hurst指数。"""
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
        # 使用对数-对数回归估计Hurst指数
        return max(
            0.0, min(1.0, np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0])
        )
    except np.linalg.LinAlgError:
        return 0.5


def add_all_features(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
    """为给定的DataFrame计算并添加所有技术指标和特征。"""
    df = df.copy()
    p = STRATEGY_PARAMS

    # --- 市场状态判断特征 ---
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
    )  # 1 for Trend, -1 for Mean Reversion

    # --- 多时间框架 (MTF) 特征 ---
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
        sma_daily = ta.trend.SMAIndicator(
            df_1d["Close"], window=p["mtf_period"]
        ).sma_indicator()
        df["mtf_signal"] = (
            pd.Series(np.where(df_1d["Close"] > sma_daily, 1, -1), index=df_1d.index)
            .reindex(df.index, method="ffill")
            .fillna(0)
        )
    else:
        df["mtf_signal"] = 0

    # --- 机器学习 (ML) 特征 ---
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

    # 清理数据
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    return df


def make_asymmetric_labels(
    df, look_forward=30, risk_reward_ratio=2.0, sl_atr_multiplier=1.5
):
    """
    创建非对称标签（三分类：盈利、亏损、未触及），用于机器学习模型训练。
    1: 价格先触及盈利目标 (TP)
    0: 价格先触及止损目标 (SL)
    """
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
    """将数据转换为适用于LSTM的序列格式。"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


# --- MACHINE LEARNING MODELS ---


def train_and_save_lstm(training_df, symbol, seq_len, epochs, batch_size):
    """训练并保存LSTM模型。"""
    logger.info(f"为 {symbol} 训练 LSTM 模型...")
    labels = make_asymmetric_labels(training_df)
    df = training_df.join(labels.rename("target")).dropna(subset=["target"])
    df.loc[:, "target"] = df["target"].astype(int)
    features = [c for c in df.columns if c.startswith("feature_")]
    X, y = df[features].values, df["target"].values

    if len(np.unique(y)) < 2:
        logger.warning(f"[{symbol}] 训练数据中只存在一个类别，跳过 LSTM 训练。")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, f"lstm_scaler_{symbol}.pkl")

    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test, seq_len)

    if len(X_train_seq) == 0:
        logger.error(f"[{symbol}] 转换后的序列样本数量为0，无法训练 LSTM。")
        return None
    if np.unique(y_train_seq).size < 2 or np.unique(y_test_seq).size < 2:
        logger.warning(
            f"[{symbol}] 训练集或测试集序列中只存在一个类别，跳过 LSTM 训练。"
        )
        return None

    l2_reg = STRATEGY_PARAMS.get("lstm_l2_reg", 0.001)
    model = Sequential(
        [
            Input(shape=(seq_len, X_train_seq.shape[2])),
            Conv1D(64, kernel_size=5, activation="relu", kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(128, return_sequences=True, kernel_regularizer=l2(l2_reg)),
            Dropout(0.3),
            LSTM(64, kernel_regularizer=l2(l2_reg)),
            Dropout(0.3),
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
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    loss, acc = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    logger.info(f"LSTM for {symbol} - 时间外样本准确率: {acc:.4f}")
    model.save(f"lstm_model_{symbol}.keras")
    return acc


def train_and_save_xgb(training_df, symbol):
    """训练并保存XGBoost模型。"""
    logger.info(f"为 {symbol} 训练 XGBoost 模型...")
    labels = make_asymmetric_labels(training_df)
    df = training_df.join(labels.rename("target")).dropna(subset=["target"])
    df.loc[:, "target"] = df["target"].astype(int)
    features = [c for c in df.columns if c.startswith("feature_")]
    X, y = df[features].values, df["target"].values

    if len(np.unique(y)) < 2:
        logger.warning(f"[{symbol}] 训练数据中只存在一个类别，跳过 XGBoost 训练。")
        return None

    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, f"xgb_scaler_{symbol}.pkl")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=STRATEGY_PARAMS["xgb_nrounds"],
        learning_rate=0.02,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=STRATEGY_PARAMS.get("xgb_gamma", 0.3),
        reg_lambda=STRATEGY_PARAMS.get("xgb_lambda", 1.5),
        early_stopping_rounds=25,
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    logger.info(f"XGBoost for {symbol} - 时间外样本准确率: {acc:.4f}")
    joblib.dump(model, f"xgb_model_{symbol}.joblib")
    return acc


# --- PLOTTING & ANALYSIS ---


def plot_walk_forward_equity(
    equity_curves, combined_trades, initial_cash, title="滚动回测权益曲线"
):
    """绘制并将滚动回测的权益曲线拼接成一张总图。"""
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    # 拼接所有周期的权益曲线
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

    # 绘制周期分割线
    for curve in equity_curves:
        start_date = curve.index[0]
        ax.axvline(
            x=start_date,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            label="周期分割线" if start_date == equity_curves[0].index[0] else "",
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
    """分析并绘制所有交易的盈亏分布直方图。"""
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
    """根据回测结果，生成智能优化建议。"""
    logger.info("\n" + "#" * 80 + "\n                 🧠 AI 优化建议器\n" + "#" * 80)
    suggestions = []

    # 关于盈亏因子的建议
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

    # 关于胜率的建议
    if stats["Win Rate [%]"] < 40:
        suggestions.append(
            f"⚠️ 注意: 胜率 ({stats['Win Rate [%]']:.2f}%) 偏低，可能导致回撤较大。\n   >> 优化方向: 1. 提高 `score_entry_threshold`。 2. 提高机器学习的置信度门槛 `ml_confidence_threshold`。"
        )
    else:
        suggestions.append(
            f"✅ 优势: 胜率 ({stats['Win Rate [%]']:.2f}%) 处于可接受范围。"
        )

    # 关于机器学习模型表现的建议
    if model_accuracies:
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
    """
    一个辅助类，用于封装特定资产或通用的交易逻辑计算，使主策略类更清晰。
    """

    def __init__(self, main_strategy: Strategy):
        self.main = main_strategy

    def _calculate_entry_score(self) -> float:
        """计算趋势跟踪部分的TA基础得分。"""
        m, w = self.main, self.main.score_weights_tf
        # 突破信号
        breakout_signal = (
            1
            if m.data.High[-1] > m.tf_donchian_h[-2]
            else -1 if m.data.Low[-1] < m.tf_donchian_l[-2] else 0
        )
        # 动量信号
        momentum_signal = 1 if m.tf_ema_fast[-1] > m.tf_ema_slow[-1] else -1
        # 组合得分
        return (
            breakout_signal * w.get("breakout", 0)
            + momentum_signal * w.get("momentum", 0)
            + m.mtf_signal[-1] * w.get("mtf", 0)
        )

    def _define_mr_entry_signal(self) -> int:
        """定义均值回归的进场信号。"""
        m = self.main
        is_long_signal = (
            crossover(m.data.Close, m.mr_bb_lower) and m.mr_rsi[-1] < m.mr_rsi_oversold
        )
        is_short_signal = (
            crossover(m.mr_bb_upper, m.data.Close)
            and m.mr_rsi[-1] > m.mr_rsi_overbought
        )
        return 1 if is_long_signal else -1 if is_short_signal else 0


class UltimateStrategy(Strategy):
    """
    终极混合策略，结合了市场状态判断、趋势跟踪、均值回归以及机器学习过滤器。
    """

    symbol = None  # 将由run方法传入
    vol_weight = 1.0  # 波动率权重，未来可扩展
    enabled_modules = None  # 模块开关，将由run方法传入

    def init(self):
        """策略初始化，加载参数和指标。"""
        # 加载基础参数
        for k, v in STRATEGY_PARAMS.items():
            setattr(self, k, v)

        # 从CONFIG加载模块开关
        if self.enabled_modules is None:
            self.enabled_modules = CONFIG.get("enabled_modules", {})

        self.asset_strategy = BaseAssetStrategy(self)

        # 准备数据序列，用于指标计算
        c = pd.Series(self.data.Close)
        h = pd.Series(self.data.High)
        l = pd.Series(self.data.Low)

        self.reset_trade_state()

        # --- 初始化指标 ---
        self.market_regime = self.I(lambda: self.data.market_regime)
        self.mtf_signal = self.I(lambda: self.data.mtf_signal)

        # 趋势跟踪指标
        self.tf_atr = self.I(
            lambda: ta.volatility.AverageTrueRange(
                h, l, c, self.tf_atr_period
            ).average_true_range(),
            plot=False,
        )
        self.tf_donchian_h = self.I(
            lambda: h.rolling(self.tf_donchian_period).max().shift(1)
        )
        self.tf_donchian_l = self.I(
            lambda: l.rolling(self.tf_donchian_period).min().shift(1)
        )
        self.tf_ema_fast = self.I(
            lambda: ta.trend.EMAIndicator(c, self.tf_ema_fast_period).ema_indicator(),
            plot=False,
        )
        self.tf_ema_slow = self.I(
            lambda: ta.trend.EMAIndicator(c, self.tf_ema_slow_period).ema_indicator(),
            plot=False,
        )

        # 均值回归指标
        bb = ta.volatility.BollingerBands(c, self.mr_bb_period, self.mr_bb_std)
        self.mr_bb_upper = self.I(lambda: bb.bollinger_hband())
        self.mr_bb_lower = self.I(lambda: bb.bollinger_lband())
        self.mr_bb_mid = self.I(lambda: bb.bollinger_mavg())
        self.mr_rsi = self.I(
            lambda: ta.momentum.RSIIndicator(c, self.mr_rsi_period).rsi(),
            plot=False,
        )

        # --- 尝试加载机器学习模型 ---
        self.lstm, self.lstm_scaler, self.xgb, self.xgb_scaler = None, None, None, None
        if self.ml_filter_enabled and self.enabled_modules.get("ml_filter", False):
            try:
                self.lstm = load_model(f"lstm_model_{self.symbol}.keras")
                self.lstm_scaler = joblib.load(f"lstm_scaler_{self.symbol}.pkl")
                logger.info(f"✅ [{self.symbol}] LSTM 模型及scaler加载成功。")
            except Exception:
                logger.warning(f"[{self.symbol}] 未能加载LSTM模型或scaler。")
                self.lstm = None
            try:
                self.xgb = joblib.load(f"xgb_model_{self.symbol}.joblib")
                self.xgb_scaler = joblib.load(f"xgb_scaler_{self.symbol}.pkl")
                logger.info(f"✅ [{self.symbol}] XGBoost 模型及scaler加载成功。")
            except Exception:
                logger.warning(f"[{self.symbol}] 未能加载XGBoost模型或scaler。")
                self.xgb = None

    def next(self):
        """策略的核心逻辑，在每个时间步被调用。"""
        # 数据预热期
        if len(self.data) < max(
            self.tf_donchian_period, self.tf_ema_slow_period, self.regime_norm_period
        ):
            return

        # 如果有持仓，则管理持仓
        if self.position:
            self.manage_open_position(self.data.Close[-1])
        # 如果无持仓，则寻找开仓机会
        else:
            # 趋势市场
            if self.market_regime[-1] == 1:
                if self.enabled_modules.get("trend_following", False):
                    self.run_scoring_system_entry(self.data.Close[-1])
            # 震荡市场
            else:
                if self.enabled_modules.get("mean_reversion", False):
                    self.run_mean_reversion_entry(self.data.Close[-1])

    def run_scoring_system_entry(self, price):
        """执行基于评分系统的趋势跟踪进场逻辑。"""
        ta_score = self.asset_strategy._calculate_entry_score()
        ml_boost = 0.0

        # 如果启用了ML过滤器
        if self.ml_filter_enabled and self.enabled_modules.get("ml_filter", False):
            prob_lstm, prob_xgb = 0.5, 0.5
            features = [c for c in self.data.df.columns if c.startswith("feature_")]

            # LSTM预测
            if (
                self.lstm
                and self.lstm_scaler
                and len(self.data.df) >= self.lstm_sequence_length
            ):
                seq = self.data.df[features].iloc[-self.lstm_sequence_length :].values
                if not np.isnan(seq).any():
                    scaled_seq = self.lstm_scaler.transform(seq)
                    prob_lstm = float(
                        self.lstm.predict(np.expand_dims(scaled_seq, 0), verbose=0)[0][
                            0
                        ]
                    )

            # XGBoost预测
            if self.xgb and self.xgb_scaler:
                current_features = self.data.df[features].iloc[-1:].values
                if not np.isnan(current_features).any():
                    scaled_features = self.xgb_scaler.transform(current_features)
                    prob_xgb = float(self.xgb.predict_proba(scaled_features)[0, 1])

            # 集成预测
            ensemble_prob = (
                self.ensemble_w_lstm * prob_lstm + self.ensemble_w_xgb * prob_xgb
            )

            # 根据预测结果调整得分
            is_long_ta = ta_score > 0
            if is_long_ta and ensemble_prob > self.ml_confidence_threshold:
                ml_boost = (ensemble_prob - 0.5) * 2 * self.ml_score_weight
            elif not is_long_ta and (1 - ensemble_prob) > self.ml_confidence_threshold:
                ml_boost = ((1 - ensemble_prob) - 0.5) * 2 * self.ml_score_weight * -1

        final_score = ta_score + ml_boost

        if abs(final_score) > self.score_entry_threshold:
            self.open_tf_position(
                price, is_long=(final_score > 0), confidence_factor=abs(final_score)
            )

    def run_mean_reversion_entry(self, price):
        """执行均值回归进场逻辑。"""
        signal = self.asset_strategy._define_mr_entry_signal()
        if signal != 0:
            self.open_mr_position(price, is_long=(signal > 0))

    def reset_trade_state(self):
        """重置与单笔交易相关的状态变量。"""
        self.active_sub_strategy = None
        self.chandelier_exit_level = 0.0
        self.highest_high_in_trade = 0
        self.lowest_low_in_trade = float("inf")
        self.mr_stop_loss = 0.0
        self.tf_initial_stop_loss = 0.0
        self.lpr_is_active = False
        self.lpr_trailing_stop = 0.0

    def manage_open_position(self, price):
        """管理当前持仓的退出逻辑。"""
        # 优先处理LPR模式
        if self.lpr_is_active:
            self.manage_lpr_exit(price)
            return

        # 检查是否激活LPR模式
        if (
            self.lpr_enabled
            and self.enabled_modules.get("lpr_mode", False)
            and self.position
        ):
            trade = self.trades[-1]
            profit_pct = (
                (price / trade.entry_price - 1)
                if self.position.is_long
                else (trade.entry_price / price - 1)
            )
            if profit_pct >= self.lpr_trigger_pct:
                self.lpr_is_active = True
                self.lpr_trailing_stop = 0.0  # Reset trailing stop on activation
                logger.info(
                    f"🚀 利润奔跑模式激活! 价格: {price:.2f}, 盈利: {profit_pct:.2%}"
                )
                self.manage_lpr_exit(price)
                return

        # 根据子策略管理退出
        if self.active_sub_strategy == "TF":
            self.manage_trend_following_exit(price)
        elif self.active_sub_strategy == "MR":
            self.manage_mean_reversion_exit(price)

    def open_tf_position(self, price, is_long, confidence_factor):
        """开立趋势跟踪仓位。"""
        risk_per_share = self.tf_atr[-1] * self.tf_stop_loss_atr_multiplier
        if risk_per_share <= 0:
            return
        size = self._calculate_position_size(
            price,
            risk_per_share,
            self._calculate_dynamic_risk() * confidence_factor,
        )
        if size <= 0:
            return

        self.reset_trade_state()
        self.active_sub_strategy = "TF"
        if is_long:
            self.buy(size=size)
            self.tf_initial_stop_loss = price - risk_per_share
            self.highest_high_in_trade = self.data.High[-1]
        else:
            self.sell(size=size)
            self.tf_initial_stop_loss = price + risk_per_share
            self.lowest_low_in_trade = self.data.Low[-1]

    def manage_trend_following_exit(self, price):
        """管理趋势跟踪仓位的退出。"""
        atr = self.tf_atr[-1]
        if self.position.is_long:
            # 硬止损
            if price < self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            # 动态吊灯止损
            self.highest_high_in_trade = max(
                self.highest_high_in_trade, self.data.High[-1]
            )
            self.chandelier_exit_level = (
                self.highest_high_in_trade - atr * self.tf_chandelier_atr_multiplier
            )
            if price < self.chandelier_exit_level:
                self.close_position("TF_Chandelier")
        elif self.position.is_short:
            # 硬止损
            if price > self.tf_initial_stop_loss:
                self.close_position("TF_Initial_SL")
                return
            # 动态吊灯止损
            self.lowest_low_in_trade = min(self.lowest_low_in_trade, self.data.Low[-1])
            self.chandelier_exit_level = (
                self.lowest_low_in_trade + atr * self.tf_chandelier_atr_multiplier
            )
            if price > self.chandelier_exit_level:
                self.close_position("TF_Chandelier")

    def open_mr_position(self, price, is_long):
        """开立均值回归仓位。"""
        risk_per_share = self.tf_atr[-1] * self.mr_stop_loss_atr_multiplier
        if risk_per_share <= 0:
            return
        size = self._calculate_position_size(
            price,
            risk_per_share,
            self._calculate_dynamic_risk() * self.mr_risk_multiplier,
        )
        if size <= 0:
            return

        self.reset_trade_state()
        self.active_sub_strategy = "MR"
        if is_long:
            self.buy(size=size)
            self.mr_stop_loss = price - risk_per_share
        else:
            self.sell(size=size)
            self.mr_stop_loss = price + risk_per_share

    def manage_mean_reversion_exit(self, price):
        """管理均值回归仓位的退出。"""
        is_long_exit = self.position.is_long and (
            price >= self.mr_bb_mid[-1] or price <= self.mr_stop_loss
        )
        is_short_exit = self.position.is_short and (
            price <= self.mr_bb_mid[-1] or price >= self.mr_stop_loss
        )
        if is_long_exit or is_short_exit:
            self.close_position("MR_Exit")

    def manage_lpr_exit(self, price):
        """管理利润奔跑模式下的移动止损。"""
        atr_dist = self.tf_atr[-1] * self.lpr_trail_atr_multiplier
        if self.position.is_long:
            new_stop = self.data.High[-1] - atr_dist
            self.lpr_trailing_stop = max(self.lpr_trailing_stop, new_stop)
            if price <= self.lpr_trailing_stop:
                self.close_position("LPR_Trail_Stop")
        elif self.position.is_short:
            new_stop = self.data.Low[-1] + atr_dist
            # 首次设置时需要特殊处理
            self.lpr_trailing_stop = (
                min(self.lpr_trailing_stop, new_stop)
                if self.lpr_trailing_stop > 0
                else new_stop
            )
            if price >= self.lpr_trailing_stop:
                self.close_position("LPR_Trail_Stop")

    def close_position(self, reason: str):
        """平仓并重置状态。"""
        self.position.close()
        self.reset_trade_state()

    def _calculate_position_size(self, price, risk_per_unit, risk_pct):
        """根据风险计算仓位大小。"""
        if risk_per_unit <= 0 or price <= 0:
            return 0
        cash_at_risk = risk_pct * self.equity
        position_size_in_quote = cash_at_risk / (risk_per_unit / price)
        # 转换为backtesting.py的size格式 (占权益的百分比)
        size = position_size_in_quote / self.equity
        return min(size, 0.99)  # 避免满仓

    def _calculate_dynamic_risk(self):
        """使用简化的凯利公式动态调整风险百分比。"""
        trades = self.closed_trades
        if len(trades) < self.kelly_trade_history:
            return self.default_risk_pct * self.vol_weight

        recent_trades = trades[-self.kelly_trade_history :]
        returns = [t.pl_pct for t in recent_trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        if not wins or not losses:
            return self.default_risk_pct * self.vol_weight

        win_rate = len(wins) / len(recent_trades)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        if avg_loss == 0:
            return self.max_risk_pct
        reward_ratio = avg_win / avg_loss
        if reward_ratio == 0:
            return self.default_risk_pct * self.vol_weight

        # 使用半凯利以降低风险
        kelly_fraction = win_rate - (1 - win_rate) / reward_ratio
        risk = max(0.005, kelly_fraction * 0.5) * self.vol_weight
        return min(risk, self.max_risk_pct)


# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    set_chinese_font()
    CACHE_DIR = "data_cache"
    logger.info(f"🚀 (V49.1) 开始运行抗过拟合重构版...")

    # --- 1. 数据准备 ---
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"], utc=True)
    backtest_end_dt = pd.to_datetime(CONFIG["backtest_end_date"], utc=True)

    if CONFIG.get("use_pretrained_models", False):
        # 使用预训练模型时，仅需获取少量历史数据用于指标预热
        warmup_days = 15
        overall_start_date = backtest_start_dt - timedelta(days=warmup_days)
        logger.info(
            f"💡 'use_pretrained_models' is True. Fetching data from {overall_start_date.date()} for indicator warm-up."
        )
    else:
        # 训练模型时，需要获取完整的训练数据窗口
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

    # --- 2. 回测执行 ---
    all_equity_curves, all_trades, all_model_accuracies = [], pd.DataFrame(), []
    current_equity = CONFIG["initial_cash"]

    # MODIFICATION: 根据 use_pretrained_models 的值选择回测模式
    if CONFIG.get("use_pretrained_models", False):
        # --- 模式一: 单次连续回测 (不训练模型) ---
        logger.info("🚀 'use_pretrained_models' is True. 运行单次连续回测模式。")
        print(
            "#" * 80
            + f"\n单次连续回测: {backtest_start_dt.date()} to {backtest_end_dt.date()}\n"
            + "#" * 80
        )

        for symbol in CONFIG["symbols_to_test"]:
            if symbol not in all_featured_data:
                continue

            data = all_featured_data[symbol]
            backtest_slice = data.loc[backtest_start_dt:backtest_end_dt].copy().dropna()

            if backtest_slice.empty:
                logger.warning(f"[{symbol}] 在指定回测期间内无数据，跳过。")
                continue

            logger.info(f"开始回测 {symbol}...")
            logger.info(f"[{symbol}] 将尝试加载已存在的模型文件。")

            bt = Backtest(
                backtest_slice,
                UltimateStrategy,
                cash=current_equity,
                commission=CONFIG["commission"],
                finalize_trades=True,
            )
            stats = bt.run(symbol=symbol, enabled_modules=CONFIG["enabled_modules"])

            current_equity = stats["Equity Final [$]"]
            all_equity_curves.append(stats["_equity_curve"])
            if "_trades" in stats and not stats["_trades"].empty:
                trades_df = stats["_trades"]
                trades_df["Symbol"] = symbol
                all_trades = pd.concat([all_trades, trades_df], ignore_index=True)

            print(stats)

    else:
        # --- 模式二: 滚动窗口回测 (训练模型) ---
        logger.info(
            "🚀 'use_pretrained_models' is False. 运行滚动窗口回测 (Walk-Forward) 模式。"
        )
        walk_forward_periods = pd.date_range(
            start=backtest_start_dt, end=backtest_end_dt, freq="3MS"
        )

        for i, period_start in enumerate(walk_forward_periods):
            period_end = (
                walk_forward_periods[i + 1] - timedelta(seconds=1)
                if i + 1 < len(walk_forward_periods)
                else backtest_end_dt
            )
            training_end_dt = period_start - timedelta(seconds=1)
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

                # 模型训练
                if (
                    STRATEGY_PARAMS["ml_filter_enabled"]
                    and CONFIG["enabled_modules"].get("ml_filter", False)
                    and ML_LIBS
                ):
                    logger.info(
                        f"[{symbol}] 准备训练数据，截止于: {training_end_with_gap.date()} (已应用 {gap_days}-day 隔离期)"
                    )
                    training_slice = data.loc[:training_end_with_gap]

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
                        logger.warning(
                            f"[{symbol}] 训练数据不足 (<1000)，跳过本周期训练。"
                        )

                if period_accuracies:
                    all_model_accuracies.append(period_accuracies)

                # 执行回测
                backtest_slice = data.loc[period_start:period_end].copy().dropna()
                if backtest_slice.empty:
                    logger.warning(f"[{symbol}] 在周期 {i+1} 内没有数据，跳过回测。")
                    continue

                logger.info(f"开始回测 {symbol} for period {i+1}...")
                bt = Backtest(
                    backtest_slice,
                    UltimateStrategy,
                    cash=current_equity,
                    commission=CONFIG["commission"],
                    finalize_trades=True,
                )
                stats = bt.run(symbol=symbol, enabled_modules=CONFIG["enabled_modules"])

                current_equity = stats["Equity Final [$]"]
                all_equity_curves.append(stats["_equity_curve"])
                if "_trades" in stats and not stats["_trades"].empty:
                    trades_df = stats["_trades"]
                    trades_df["Symbol"] = symbol
                    trades_df["Rolling Period"] = i + 1
                    all_trades = pd.concat([all_trades, trades_df], ignore_index=True)

                print(stats)

    # --- 3. 结果分析与可视化 ---
    if not all_trades.empty:
        final_stats = {
            "Profit Factor": (
                all_trades[all_trades["PnL"] > 0]["PnL"].sum()
                / abs(all_trades[all_trades["PnL"] < 0]["PnL"].sum())
                if abs(all_trades[all_trades["PnL"] < 0]["PnL"].sum()) != 0
                else float("inf")
            ),
            "Win Rate [%]": len(all_trades[all_trades["ReturnPct"] > 0])
            / len(all_trades)
            * 100,
            "# Trades": len(all_trades),
        }
        logger.info("\n" + "#" * 80 + "\n                 回测表现总览\n" + "#" * 80)
        logger.info(f"总初始资金: ${CONFIG['initial_cash']:,.2f}")
        logger.info(f"总最终权益: ${current_equity:,.2f}")
        logger.info(
            f"总回报率: {(current_equity / CONFIG['initial_cash'] - 1) * 100:.2f}%"
        )
        logger.info(f"总交易次数: {final_stats['# Trades']}")
        logger.info(f"整体胜率: {final_stats['Win Rate [%]']:.2f}%")
        logger.info(f"整体盈亏因子 (Profit Factor): {final_stats['Profit Factor']:.2f}")

        if CONFIG["show_plots"]:
            plot_title = (
                "策略权益曲线"
                if CONFIG.get("use_pretrained_models")
                else "滚动回测权益曲线"
            )
            plot_walk_forward_equity(
                all_equity_curves, all_trades, CONFIG["initial_cash"], title=plot_title
            )
            analyze_trade_distribution(all_trades)

        generate_optimization_suggestions(final_stats, all_model_accuracies, CONFIG)
    else:
        logger.info("整个回测期间没有产生任何交易。")
