# -*- coding: utf-8 -*-
"""
Kronos V62.2 — 全面優化版
功能摘要：
- V62.2 優化版: 根據文檔建議，整合了多項優化措施。
- 增強特徵工程: 新增布林帶寬度、一目均衡表等指標。
- 模型與標籤優化: 調整了XGBoost超參數並縮短了預測週期。
- 動態風險管理: 採用基於波動率的動態倉位大小和移動止盈機制。
- 強化回測框架: 延長了回測週期，並註解了滾動優化(Walk-Forward)的實現思路。
- V62.1 Hotfix: 修正了因比較“帶時區”和“不帶時區”的時間物件而導致的 TypeError。
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

# Backtest & ML libs
try:
    from backtesting import Backtest, Strategy
    import ta
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler

    # **[修正] 在這裡加入下面這一行**
    from sklearn.utils.class_weight import compute_sample_weight
except Exception as e:
    print(f"錯誤: 缺少必要的庫: {e}")
    sys.exit(1)

# Logging
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- CONFIG (優化後) ---
CONFIG = {
    "symbol": "ETHUSDT",
    "interval": "5m",
    # **優化建議: 延長回測週期以覆蓋多種市場情境**
    "backtest_start_date": "2024-07-01",
    "backtest_end_date": "2025-07-01",
    "initial_cash": 500_000,
    "commission": 0.001,
    "spread": 0.0005,
    "show_plots": True,
    # 訓練窗口保持1.5年，以捕捉足夠的歷史模式
    "training_window_days": 365 * 1.5,
    "retrain_gap_days": 90,
    "model_dir": "models_v62_optimized",
    "data_cache": "data_cache_v62_optimized",
}

# --- MODEL & FEATURE PARAMS (優化後) ---
MODEL_PARAMS = {
    # Labeling
    # **優化建議: 縮短預測週期以適應日內動態**
    "label_look_forward_bars": 24,
    "label_return_threshold": 0.0075,
    # XGBoost
    # **優化建議: 調整超參數以提升性能 (這些值應通過Optuna等工具尋找)**
    "xgb_n_estimators": 200,
    "xgb_max_depth": 5,
    "xgb_learning_rate": 0.03,
    "xgb_subsample": 0.8,
    "xgb_colsample_bytree": 0.8,
    # Strategy
    # **優化建議: 動態風險管理**
    "volatility_window": 100,  # 用於計算波動率以調整倉位大小的窗口
    "target_annual_volatility": 1.5,  # 目標年化波動率，用於倉位大小計算
    "trailing_sl_atr_multiplier": 2.5,  # 移動止盈的ATR乘數
    "emergency_sl_atr_period": 14,
    "emergency_sl_atr_multiplier": 3.0,
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
            logger.info("設置中文字體成功")
    except Exception:
        logger.warning("設置中文字體失敗。")


def ensure_dirs():
    os.makedirs(CONFIG["data_cache"], exist_ok=True)
    os.makedirs(CONFIG["model_dir"], exist_ok=True)


def fetch_binance_klines(
    symbol, interval, start_str, end_str=None, limit=1000, cache_dir=None
):
    cache_dir = cache_dir or CONFIG["data_cache"]
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
                logger.info(f"✅ 從有效缓存載入 {symbol} 資料")
                return df.loc[start_dt:end_dt]
        except Exception as e:
            logger.warning(f"讀取缓存檔案失敗: {e}")
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
        except Exception as e:
            logger.warning(f"獲取資料失敗: {e}. 等待 3s 重試...")
            time.sleep(3)
    if not all_data:
        logger.error("未能獲取任何資料。")
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
    df.to_csv(cache_file)
    logger.info("✅ 資料已獲取並缓存")
    return df.loc[start_dt:end_dt]


def add_expert_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- 原有特徵 ---
    df["feature_volume_oscillator"] = df["Volume"] / df["Volume"].rolling(50).mean()
    # ... (其他現有特徵保持不變)
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

    # **[新增] 宏觀趨勢特徵**
    # 計算短期EMA(50)和長期EMA(200)
    ema_short = ta.trend.ema_indicator(df.Close, 50)
    ema_long = ta.trend.ema_indicator(df.Close, 200)
    # 計算兩者比值。大於1表示短期趨勢強於長期（可能處於多頭），小於1則相反。
    df["feature_ema_trend_context"] = ema_short / ema_long

    return df.dropna()


def make_return_based_labels(df: pd.DataFrame) -> pd.Series:
    p = MODEL_PARAMS
    look_forward = p["label_look_forward_bars"]

    # **[修改] 自適應閾值**
    # 1. 計算未來N根K棒的ATR，作為波動性的衡量
    atr = ta.volatility.average_true_range(
        df.High, df.Low, df.Close, window=look_forward
    )
    # 2. 將回報閾值定義為未來ATR的倍數（例如1.5倍）
    atr_threshold = (atr / df["Close"]) * 1.5  # 這裡的 1.5 是一個可調參數

    labels = pd.Series(0, index=df.index, dtype=int)
    future_close = df["Close"].shift(-look_forward)
    returns = (future_close - df["Close"]) / df["Close"]

    # 使用動態閾值來定義標籤
    labels[returns >= atr_threshold] = 1
    labels[returns <= -atr_threshold] = 2

    return labels


def train_and_save_model(training_df: pd.DataFrame, symbol: str):
    # **[優化 1: 波動率過濾器]**
    # 計算 ATR 作為波動率的代理指標
    atr = ta.volatility.average_true_range(
        training_df.High, training_df.Low, training_df.Close, 14
    )
    # 計算 ATR 佔收盤價的百分比
    atr_pct = (atr / training_df.Close).dropna()
    # 設定波動率門檻，過濾掉波動率最低的 25% 的數據
    volatility_threshold = atr_pct.quantile(0.25)

    # 根據門檻，選出值得學習的數據索引
    tradable_indices = atr_pct[atr_pct > volatility_threshold].index

    original_len = len(training_df)
    training_df_filtered = training_df.loc[tradable_indices]
    filtered_len = len(training_df_filtered)

    logger.info(
        f"波動率過濾器已啟用：從 {original_len} 筆數據中，篩選出 {filtered_len} 筆波動較大的數據進行訓練。"
    )

    # 在過濾後的數據上生成標籤
    logger.info("開始在過濾後的數據上生成標籤...")
    labels = make_return_based_labels(training_df_filtered)
    data = training_df_filtered.join(labels.rename("target")).dropna()

    if data.empty:
        logger.warning("過濾後沒有可供訓練的數據。")
        return

    feature_cols = [col for col in data.columns if "feature_" in col]
    X, y = data[feature_cols], data["target"]

    # **[優化 2: 類別權重]** (繼續保留)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y)

    logger.info(
        f"開始訓練模型... 新樣本數: {len(X)}, 特徵數: {len(X.columns)}, 新類別分佈:\n{y.value_counts(normalize=True)}"
    )

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        n_estimators=MODEL_PARAMS["xgb_n_estimators"],
        max_depth=MODEL_PARAMS["xgb_max_depth"],
        learning_rate=MODEL_PARAMS["xgb_learning_rate"],
        subsample=MODEL_PARAMS["xgb_subsample"],
        colsample_bytree=MODEL_PARAMS["xgb_colsample_bytree"],
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )

    logger.info("模型將使用樣本權重和波動率過濾後的數據進行訓練。")
    model.fit(X_scaled, y, sample_weight=sample_weights)

    model_path, scaler_path, features_path = (
        os.path.join(CONFIG["model_dir"], f"xgb_model_{symbol}.joblib"),
        os.path.join(CONFIG["model_dir"], f"scaler_{symbol}.joblib"),
        os.path.join(CONFIG["model_dir"], "feature_list.txt"),
    )
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        for feature in X.columns:
            f.write(f"{feature}\n")
    logger.info("✅ 模型、Scaler 及特徵列表已保存。")


class UltimateStrategy(Strategy):
    # ... (init 函數不變)
    # ... (init 函數不變)
    def init(self):
        # ... (init 函數代碼不變)
        model_dir, symbol = CONFIG["model_dir"], CONFIG["symbol"]
        self.model = joblib.load(os.path.join(model_dir, f"xgb_model_{symbol}.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, f"scaler_{symbol}.joblib"))
        with open(os.path.join(model_dir, "feature_list.txt"), "r") as f:
            self.feature_list = [line.strip() for line in f.readlines()]

        self.emergency_sl_multiplier = MODEL_PARAMS["emergency_sl_atr_multiplier"]
        self.trailing_sl_atr_multiplier = MODEL_PARAMS["trailing_sl_atr_multiplier"]
        self.atr = self.I(
            lambda: ta.volatility.average_true_range(
                high=pd.Series(self.data.High),
                low=pd.Series(self.data.Low),
                close=pd.Series(self.data.Close),
                window=MODEL_PARAMS["emergency_sl_atr_period"],
            )
        )

        # 用於動態倉位大小的波動率計算
        returns = self.data.Close.to_series().pct_change()
        self.volatility = self.I(
            lambda: returns.rolling(MODEL_PARAMS["volatility_window"]).std()
        )
        self.bars_per_year = 365 * 24 * (60 / 5)  # 5分鐘線每年的bar數

        # **[修正]** 初始化一個變數來手動管理移動止損價格
        self.trailing_sl_price = 0.0

        # **[新增]** 設定概率門檻
        self.PROBABILITY_THRESHOLD = 0.55  # 55% 的概率才交易

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # ... (持倉管理邏輯不變)
        # **（中間的持倉管理邏輯保持不變）**
        # ...

        if self.position:
            if self.position.is_long:
                new_sl = price - self.trailing_sl_atr_multiplier * atr_val
                self.trailing_sl_price = max(self.trailing_sl_price, new_sl)
                if self.data.Low[-1] <= self.trailing_sl_price:
                    self.position.close()
                    return
            elif self.position.is_short:
                new_sl = price + self.trailing_sl_atr_multiplier * atr_val
                self.trailing_sl_price = min(self.trailing_sl_price, new_sl)
                if self.data.High[-1] >= self.trailing_sl_price:
                    self.position.close()
                    return
            return
        # --- 開倉前的數據準備 ---

        current_features = self.data.df.iloc[-1:][self.feature_list]
        current_features_scaled = self.scaler.transform(current_features)

        # **[關鍵修正] 輸出概率而非單一預測**
        prediction_proba = self.model.predict_proba(current_features_scaled)[0]

        # 找出概率最高的類別
        prediction = np.argmax(prediction_proba)
        max_proba = prediction_proba[prediction]

        # **[新增] 概率過濾器**
        # 只有在預測的概率大於我們設定的門檻時，才允許進行交易（即不為0的信號）
        if max_proba < self.PROBABILITY_THRESHOLD or prediction == 0:
            return

        # ... (動態倉位大小邏輯不變)
        # **（後面的倉位大小計算邏輯保持不變）**
        # ...

        sl_amount_per_unit = atr_val * self.emergency_sl_multiplier
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

        # 執行交易 (因為 prediction 已經被過濾，所以只可能是 1 或 2)
        if prediction == 1:
            initial_sl = price - sl_amount_per_unit
            self.buy(size=size, sl=initial_sl)
            self.trailing_sl_price = initial_sl
        elif prediction == 2:
            initial_sl = price + sl_amount_per_unit
            self.sell(size=size, sl=initial_sl)
            self.trailing_sl_price = initial_sl


def main():
    set_chinese_font()
    ensure_dirs()

    # **優化建議: 滾動優化 (Walk-Forward Optimization) 框架**
    # 真正的滾動優化會將訓練和回測包裹在一個循環中。
    # 例如，訓練2022-2023的數據，然後在2024年1-3月進行回測。
    # 接著，將訓練窗口向前滾動三個月（即訓練至2024年3月），在4-6月回測。
    # 如此反覆，直到回測結束。
    # 這能更好地模擬真實交易中模型持續再訓練的過程。
    # 以下代碼為簡化版，僅執行一次訓練和一次較長的回測。

    # Training
    train_start_dt = (
        pd.to_datetime(CONFIG["backtest_start_date"], utc=True)
        - timedelta(days=CONFIG["retrain_gap_days"])
        - timedelta(days=CONFIG["training_window_days"])
    )
    train_end_dt = pd.to_datetime(CONFIG["backtest_start_date"], utc=True) - timedelta(
        days=CONFIG["retrain_gap_days"]
    )
    logger.info(f"數據獲取總時間段: {train_start_dt.date()} -> {train_end_dt.date()}")
    training_raw_data = fetch_binance_klines(
        CONFIG["symbol"],
        CONFIG["interval"],
        train_start_dt,
        train_end_dt,
        cache_dir=CONFIG["data_cache"],
    )
    if training_raw_data.empty:
        return

    logger.info("開始計算專家特徵...")
    training_featured_data = add_expert_features(training_raw_data)
    train_and_save_model(training_featured_data, CONFIG["symbol"])

    # Backtesting
    logger.info("開始回測...")
    backtest_start_dt = pd.to_datetime(CONFIG["backtest_start_date"], utc=True)
    backtest_end_dt = pd.to_datetime(CONFIG["backtest_end_date"], utc=True)
    warmup_start_dt = backtest_start_dt - timedelta(
        days=60
    )  # 預熱期延長至60天以計算指標
    backtest_raw_data = fetch_binance_klines(
        CONFIG["symbol"],
        CONFIG["interval"],
        warmup_start_dt,
        backtest_end_dt,
        cache_dir=CONFIG["data_cache"],
    )
    if backtest_raw_data.empty:
        logger.error("回測數據為空。")
        return

    logger.info("為回測數據計算特徵...")
    backtest_featured_data = add_expert_features(backtest_raw_data)
    final_bt_data = backtest_featured_data.loc[backtest_start_dt:]
    bt = Backtest(
        final_bt_data,
        UltimateStrategy,
        cash=CONFIG["initial_cash"],
        commission=CONFIG["commission"],
    )
    stats = bt.run()
    logger.info("回測完成。")
    print(stats)
    if CONFIG["show_plots"]:
        bt.plot(
            filename=f"backtest_{CONFIG['symbol']}_v62.2_optimized.html",
            open_browser=False,
        )


if __name__ == "__main__":
    main()
