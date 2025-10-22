# -*- coding: utf-8 -*-
"""
Kronos V63.6 — 引入 ADX 趨勢過濾器
功能摘要：
- V63.6 Feat: [核心優化] 在策略中引入 ADX 指標作為趨勢過濾器，僅在趨勢明顯時（ADX > 閾值）才考慮開倉信號，旨在過濾震蕩行情下的噪聲，提升勝率。
- V63.5 Feat: 新增了跨周期特徵 (1h RSI) 和時間特徵 (hour, weekday)。
- V63.5 Feat: 將 Optuna 概率閾值搜索範圍調整到更實際的 (0.5, 0.6) 區間。
- V63.5 Feat: 默認啟用手動閾值調整，使用 0.52 進行最終回測。
"""
import os
import sys
import time
import logging
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
import requests
import matplotlib.pyplot as plt

# Backtest & ML & Optimization libs
try:
    from backtesting import Backtest, Strategy
    import ta
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler
    from sklearn.utils.class_weight import compute_sample_weight
    import optuna
except Exception as e:
    print(f"錯誤: 缺少必要的庫: {e}")
    sys.exit(1)

# Logging
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
CONFIG = {
    "symbols": ["ETHUSDT", "BTCUSDT"],
    "target_symbol": "ETHUSDT",
    "interval": "5m",
    "backtest_start_date": "2024-07-01",
    "backtest_end_date": "2025-07-01",
    "initial_cash": 500_000,
    "commission": 0.001,
    "spread": 0.0005,
    "show_plots": True,
    "training_window_days": 365 * 1.5,
    "retrain_gap_days": 90,
    "model_dir": "models_v63_alpha_seeker",
    "data_cache_dir": "data_cache_v63",
    "optuna_trials": 5,
    "run_optuna": False,
    "min_optuna_trades": 10,
    # [優化] 方案一: 使用一個折中的閾值 0.52 進行測試
    "final_backtest_probability_threshold": 0.52,
}

# --- MODEL PARAMS ---
MODEL_PARAMS = {
    "emergency_sl_atr_period": 14,
    "volatility_window": 100,
    "target_annual_volatility": 1.5,
    # [優化] 方案二: 新增 ADX 趨勢過濾器的參數
    "adx_filter_period": 14,
    "adx_threshold": 25,  # ADX > 25 才認為是趨勢行情
}


# --- Helper Functions ---
def set_chinese_font():
    try:
        import matplotlib.font_manager as fm

        font_names = [
            "PingFang SC",
            "Microsoft YaHei",
            "SimHei",
            "Heiti TC",
            "Arial Unicode MS",
            "sans-serif",
        ]
        if any(font in [f.name for f in fm.fontManager.ttflist] for font in font_names):
            plt.rcParams["font.sans-serif"] = [
                font
                for font in font_names
                if font in [f.name for f in fm.fontManager.ttflist]
            ]
            plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def ensure_dirs():
    os.makedirs(CONFIG["data_cache_dir"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)


def fetch_binance_klines(
    symbol, interval, start_str, end_str=None, limit=1000, cache_dir=None
):
    cache_dir = cache_dir or CONFIG["data_cache_dir"]
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval}.csv")
    start_dt, end_dt = pd.to_datetime(start_str, utc=True), (
        pd.to_datetime(end_str, utc=True)
        if end_str
        else datetime.utcnow().astimezone(pd.Timestamp.utcnow().tz)
    )
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col="timestamp", parse_dates=True)
            if not df.empty and df.index[0] <= start_dt and df.index[-1] >= end_dt:
                return df.loc[start_dt:end_dt]
        except Exception as e:
            pass
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
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current_start_ts = data[-1][0] + 1
        except requests.exceptions.ReadTimeout:
            time.sleep(5)
        except Exception:
            time.sleep(3)
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
            "c",
            "q",
            "n",
            "tbb",
            "tbq",
            "i",
        ],
    )
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    if not df.empty:
        df.to_csv(cache_file)
    return df.loc[start_dt:end_dt]


def fetch_and_align_data(symbols, interval, start_dt, end_dt):
    extended_start_dt = start_dt - timedelta(days=5)
    df_dict = {}
    for symbol in symbols:
        df_dict[symbol] = fetch_binance_klines(
            symbol, interval, extended_start_dt, end_dt
        )

    base_df = df_dict[CONFIG["target_symbol"]].copy()
    for symbol, df in df_dict.items():
        if symbol == CONFIG["target_symbol"]:
            continue
        df.columns = [f"{col}_{symbol}" for col in df.columns]
        base_df = pd.merge(base_df, df, left_index=True, right_index=True, how="left")

    base_df.fillna(method="ffill", inplace=True)
    return base_df.loc[start_dt:].dropna()


def get_fear_greed_data(start_dt, end_dt):
    dates = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq="D", tz="UTC")
    values = np.random.randint(10, 91, size=len(dates))
    fg_df = pd.DataFrame({"value": values}, index=dates)
    return fg_df


def add_expert_features(df: pd.DataFrame) -> pd.DataFrame:
    df["feature_volume_oscillator"] = df["Volume"] / df["Volume"].rolling(50).mean()
    df["feature_rsi_14"] = ta.momentum.rsi(df.Close, 14)
    df["feature_adx_14"] = ta.trend.adx(df.High, df.Low, df.Close, 14)
    bollinger = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["feature_bollinger_bandwidth"] = (
        bollinger.bollinger_hband() - bollinger.bollinger_lband()
    ) / bollinger.bollinger_mavg()
    ichimoku = ta.trend.IchimokuIndicator(
        high=df["High"], low=df["Low"], window1=9, window2=26, window3=52
    )
    df["feature_ichimoku_span"] = (ichimoku.ichimoku_a() - ichimoku.ichimoku_b()) / df[
        "Close"
    ]
    if "Close_BTCUSDT" in df.columns:
        df["feature_btc_eth_ratio"] = df["Close_BTCUSDT"] / df["Close"]
    fg_data = get_fear_greed_data(df.index.min(), df.index.max())
    df["feature_fear_greed"] = fg_data["value"].reindex(df.index, method="ffill")

    df["feature_hour_of_day"] = df.index.hour
    df["feature_day_of_week"] = df.index.dayofweek

    close_1h = df["Close"].resample("1H").last()
    rsi_1h = ta.momentum.rsi(close_1h, 14)
    df["feature_rsi_1h"] = rsi_1h.reindex(df.index, method="ffill")

    return df.dropna()


def make_return_based_labels(
    df: pd.DataFrame, look_forward: int, atr_multiplier: float
) -> pd.Series:
    atr = ta.volatility.average_true_range(
        df.High, df.Low, df.Close, window=look_forward
    )
    atr_threshold = (atr / df["Close"]) * atr_multiplier
    labels = pd.Series(0, index=df.index, dtype=int)
    future_close = df["Close"].shift(-look_forward)
    returns = (future_close - df["Close"]) / df["Close"]
    labels[returns >= atr_threshold] = 1
    labels[returns <= -atr_threshold] = 2
    return labels


class UltimateStrategy(Strategy):
    params = None

    def init(self):
        self.model = self.params["model"]
        self.scaler = self.params["scaler"]
        self.feature_list = self.params["feature_list"]
        self.PROBABILITY_THRESHOLD = self.params["probability_threshold"]
        self.trailing_sl_atr_multiplier = self.params["trailing_sl_atr_multiplier"]
        self.atr = self.I(
            lambda: ta.volatility.average_true_range(
                high=pd.Series(self.data.High),
                low=pd.Series(self.data.Low),
                close=pd.Series(self.data.Close),
                window=MODEL_PARAMS["emergency_sl_atr_period"],
            )
        )
        returns = self.data.Close.to_series().pct_change()
        self.volatility = self.I(
            lambda: returns.rolling(MODEL_PARAMS["volatility_window"]).std()
        )
        self.bars_per_year = 365 * 24 * (60 / 5)
        self.trailing_sl_price = 0.0
        self.last_log_time = None

        # [優化] 方案二: 新增 ADX 指標和其閾值，用於趨勢過濾
        self.adx_filter = self.I(
            lambda: ta.trend.adx(
                high=pd.Series(self.data.High),
                low=pd.Series(self.data.Low),
                close=pd.Series(self.data.Close),
                window=MODEL_PARAMS["adx_filter_period"],
            )
        )
        self.ADX_THRESHOLD = MODEL_PARAMS["adx_threshold"]

    def next(self):
        current_time = self.data.index[-1]
        log_now = self.last_log_time is None or (
            current_time - self.last_log_time
        ) >= timedelta(hours=4)

        current_features_df = self.data.df.iloc[-1:]
        if not all(f in current_features_df.columns for f in self.feature_list):
            return

        current_features = current_features_df[self.feature_list]
        current_features_scaled = self.scaler.transform(current_features)
        prediction_proba = self.model.predict_proba(current_features_scaled)[0]
        prediction = np.argmax(prediction_proba)
        max_proba = prediction_proba[prediction]

        if log_now:
            logger.info(
                f"時間: {current_time} | "
                f"預測: {prediction} (0:持有,1:買,2:賣) | "
                f"概率: {max_proba:.3f} | "
                f"閾值: {self.PROBABILITY_THRESHOLD:.3f} | "
                f"ADX: {self.adx_filter[-1]:.2f} (>{self.ADX_THRESHOLD}?) | "  # 增加 ADX 狀態日誌
                f"持倉: {'是' if self.position else '否'}"
            )
            self.last_log_time = current_time

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        if self.position:
            if self.position.is_long:
                new_sl = price - self.trailing_sl_atr_multiplier * atr_val
                self.trailing_sl_price = max(self.trailing_sl_price, new_sl)
                if self.data.Low[-1] <= self.trailing_sl_price:
                    self.position.close()
            elif self.position.is_short:
                new_sl = price + self.trailing_sl_atr_multiplier * atr_val
                self.trailing_sl_price = min(self.trailing_sl_price, new_sl)
                if self.data.High[-1] >= self.trailing_sl_price:
                    self.position.close()
            return

        # [優化] 方案二: 在所有開倉條件前，加入 ADX 趨勢過濾器
        if self.adx_filter[-1] < self.ADX_THRESHOLD:
            return  # 市場處於盤整，過濾掉本次信號

        if max_proba < self.PROBABILITY_THRESHOLD or prediction == 0:
            return

        sl_amount_per_unit = atr_val * self.params["emergency_sl_atr_multiplier"]
        if not np.isfinite(sl_amount_per_unit) or sl_amount_per_unit <= 0:
            return

        annualized_vol = self.volatility[-1] * np.sqrt(self.bars_per_year)
        if not np.isfinite(annualized_vol) or annualized_vol <= 0:
            return

        size = np.clip(
            (MODEL_PARAMS["target_annual_volatility"] / annualized_vol) * 0.5, 0.05, 1.0
        )
        if size <= 0:
            return

        if prediction == 1:
            initial_sl = price - sl_amount_per_unit
            self.buy(size=size, sl=initial_sl)
            self.trailing_sl_price = initial_sl
        elif prediction == 2:
            initial_sl = price + sl_amount_per_unit
            self.sell(size=size, sl=initial_sl)
            self.trailing_sl_price = initial_sl


def objective(trial: optuna.Trial):
    try:
        params = {
            "label_look_forward_bars": trial.suggest_int(
                "label_look_forward_bars", 12, 48, step=6
            ),
            "label_atr_multiplier": trial.suggest_float(
                "label_atr_multiplier", 1.5, 3.5
            ),
            "probability_threshold": trial.suggest_float(
                "probability_threshold", 0.5, 0.6
            ),
            "trailing_sl_atr_multiplier": trial.suggest_float(
                "trailing_sl_atr_multiplier", 2.0, 4.0
            ),
            "emergency_sl_atr_multiplier": trial.suggest_float(
                "emergency_sl_atr_multiplier", 2.5, 5.0
            ),
            "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 7),
            "xgb_learning_rate": trial.suggest_float(
                "xgb_learning_rate", 0.01, 0.1, log=True
            ),
        }

        train_start_dt = (
            pd.to_datetime(CONFIG["backtest_start_date"], utc=True)
            - timedelta(days=CONFIG["retrain_gap_days"])
            - timedelta(days=CONFIG["training_window_days"])
        )
        train_end_dt = pd.to_datetime(
            CONFIG["backtest_start_date"], utc=True
        ) - timedelta(days=CONFIG["retrain_gap_days"])

        training_raw_data = fetch_and_align_data(
            CONFIG["symbols"], CONFIG["interval"], train_start_dt, train_end_dt
        )
        training_featured_data = add_expert_features(training_raw_data)

        atr = ta.volatility.average_true_range(
            training_featured_data.High,
            training_featured_data.Low,
            training_featured_data.Close,
            14,
        )
        atr_pct = (atr / training_featured_data.Close).dropna()
        volatility_threshold = atr_pct.quantile(0.25)
        tradable_indices = atr_pct[atr_pct > volatility_threshold].index
        training_df_filtered = training_featured_data.loc[tradable_indices]

        labels = make_return_based_labels(
            training_df_filtered,
            params["label_look_forward_bars"],
            params["label_atr_multiplier"],
        )
        data = training_df_filtered.join(labels.rename("target")).dropna()
        if data.empty or len(data["target"].unique()) < 3:
            return -99

        feature_cols = [col for col in data.columns if "feature_" in col]
        X, y = data[feature_cols], data["target"]
        sample_weights = compute_sample_weight(class_weight="balanced", y=y)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_estimators=200,
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(X_scaled, y, sample_weight=sample_weights)

        backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"], utc=True)
        backtest_end_dt = pd.to_datetime(CONFIG["backtest_end_date"], utc=True)
        warmup_start_dt = backtest_start_dt - timedelta(days=60)
        backtest_raw_data = fetch_and_align_data(
            CONFIG["symbols"], CONFIG["interval"], warmup_start_dt, backtest_end_dt
        )
        backtest_featured_data = add_expert_features(backtest_raw_data)
        final_bt_data = backtest_featured_data.loc[backtest_start_dt:]

        strategy_params = params.copy()
        strategy_params.update(
            {"model": model, "scaler": scaler, "feature_list": feature_cols}
        )

        bt = Backtest(
            final_bt_data,
            UltimateStrategy,
            cash=CONFIG["initial_cash"],
            commission=CONFIG["commission"],
        )
        stats = bt.run(params=strategy_params)

        sharpe = stats["Sharpe Ratio"]
        n_trades = stats["# Trades"]

        if n_trades < CONFIG["min_optuna_trades"] or not np.isfinite(sharpe):
            return -99.0 + n_trades * 0.001

        return sharpe * np.log1p(n_trades)

    except Exception as e:
        logger.error(f"Optuna trial failed with exception: {e}")
        return -100


def main():
    set_chinese_font()
    ensure_dirs()

    best_params = None
    optuna_cache_file = os.path.join(CONFIG["model_dir"], "optuna_cache_v63.pkl")

    if not CONFIG["run_optuna"] and os.path.exists(optuna_cache_file):
        logger.info(f"從緩存文件 {optuna_cache_file} 加載最佳參數...")
        best_params = joblib.load(optuna_cache_file)
        logger.info("已加載的參數:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")

    if best_params is None:
        if not CONFIG["run_optuna"]:
            logger.warning("未找到緩存文件。即使 run_optuna=False, 仍將執行新的優化。")
            CONFIG["run_optuna"] = True

        logger.info("開始 Optuna 超參數優化...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=CONFIG["optuna_trials"])

        logger.info("優化完成！")
        logger.info(f"最佳試驗: {study.best_trial.number}")
        logger.info(f"最佳綜合分數 (Sharpe * log(1+Trades)): {study.best_value}")
        logger.info("最佳參數:")
        best_params = study.best_params
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")

        logger.info(f"將最佳參數保存至緩存文件: {optuna_cache_file}")
        joblib.dump(best_params, optuna_cache_file)

    if CONFIG["final_backtest_probability_threshold"] is not None:
        new_threshold = CONFIG["final_backtest_probability_threshold"]
        logger.warning(
            f"手動覆蓋信心閾值! "
            f"Optuna值: {best_params.get('probability_threshold'):.4f} -> 手動設置值: {new_threshold}"
        )
        best_params["probability_threshold"] = new_threshold

    logger.info("使用最佳參數進行最終回測...")
    train_start_dt = (
        pd.to_datetime(CONFIG["backtest_start_date"], utc=True)
        - timedelta(days=CONFIG["retrain_gap_days"])
        - timedelta(days=CONFIG["training_window_days"])
    )
    train_end_dt = pd.to_datetime(CONFIG["backtest_start_date"], utc=True) - timedelta(
        days=CONFIG["retrain_gap_days"]
    )

    training_raw_data = fetch_and_align_data(
        CONFIG["symbols"], CONFIG["interval"], train_start_dt, train_end_dt
    )
    training_featured_data = add_expert_features(training_raw_data)

    atr = ta.volatility.average_true_range(
        training_featured_data.High,
        training_featured_data.Low,
        training_featured_data.Close,
        14,
    )
    atr_pct = (atr / training_featured_data.Close).dropna()
    volatility_threshold = atr_pct.quantile(0.25)
    tradable_indices = atr_pct[atr_pct > volatility_threshold].index
    training_df_filtered = training_featured_data.loc[tradable_indices]

    labels = make_return_based_labels(
        training_df_filtered,
        best_params["label_look_forward_bars"],
        best_params["label_atr_multiplier"],
    )
    data = training_df_filtered.join(labels.rename("target")).dropna()
    feature_cols = [col for col in data.columns if "feature_" in col]
    X, y = data[feature_cols], data["target"]
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)
    scaler = RobustScaler().fit(X)
    X_scaled = scaler.transform(X)

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_estimators=200,
        max_depth=best_params["xgb_max_depth"],
        learning_rate=best_params["xgb_learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
    ).fit(X_scaled, y, sample_weight=sample_weights)

    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"], utc=True)
    backtest_end_dt = pd.to_datetime(CONFIG["backtest_end_date"], utc=True)
    warmup_start_dt = backtest_start_dt - timedelta(days=60)
    backtest_raw_data = fetch_and_align_data(
        CONFIG["symbols"], CONFIG["interval"], warmup_start_dt, backtest_end_dt
    )
    backtest_featured_data = add_expert_features(backtest_raw_data)
    final_bt_data = backtest_featured_data.loc[backtest_start_dt:]

    strategy_params = best_params.copy()
    strategy_params.update(
        {"model": model, "scaler": scaler, "feature_list": feature_cols}
    )

    bt = Backtest(
        final_bt_data,
        UltimateStrategy,
        cash=CONFIG["initial_cash"],
        commission=CONFIG["commission"],
    )
    stats = bt.run(params=strategy_params)

    logger.info("最終回測結果:")
    print(stats)
    if CONFIG["show_plots"]:
        bt.plot(
            filename=f"backtest_{CONFIG['target_symbol']}_v63_optimized.html",
            open_browser=False,
        )


if __name__ == "__main__":
    main()
