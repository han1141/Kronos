# fusion/predict.py
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from datetime import datetime
from config import *
from dynamic_weights import get_dynamic_weights


def resample_to_timeframe(df_15m, timeframe):
    rule_map = {"15m": "15min", "1h": "1H", "4h": "4H", "8h": "8H"}
    df = df_15m.copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df.set_index("open_time", inplace=True)
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(rule_map[timeframe]).apply(ohlc).dropna()


def calculate_features(df):
    import pandas_ta as ta

    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["macd"] = ta.macd(df["close"])["MACD_12_26_9"]
    bb = ta.bbands(df["close"], length=20)
    df["bb_upper"] = bb["BBU_20_2.0"]
    df["bb_lower"] = bb["BBL_20_2.0"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["price_momentum"] = df["close"].pct_change(5)
    feature_cols = [
        "rsi",
        "macd",
        "bb_upper",
        "bb_lower",
        "atr",
        "volatility",
        "volume_sma",
        "price_momentum",
        "close",
        "volume",
    ]
    return df[feature_cols].fillna(0)


def fusion_predict():
    print(f"\n{'='*60}")
    print(f"  4周期LSTM融合预测系统 ({SYMBOL})")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # 加载模型
    models, scalers, feature_cols_dict = {}, {}, {}
    for tf in TIMEFRAMES:
        model_path = f"{MODEL_DIR}/eth_trend_model_v1_{tf}.keras"
        scaler_path = f"{MODEL_DIR}/eth_trend_scaler_v1_{tf}.joblib"
        cols_path = f"{MODEL_DIR}/feature_columns_{tf}.joblib"
        if not os.path.exists(model_path):
            print(f"模型缺失: {model_path}")
            return
        models[tf] = load_model(model_path)
        scalers[tf] = joblib.load(scaler_path)
        feature_cols_dict[tf] = joblib.load(cols_path)
        print(f"已加载 {tf} 模型")

    # 读取 15m 数据
    data_path = f"{DATA_DIR}/ethusdt_15m_data.csv"
    df_15m = pd.read_csv(data_path, parse_dates=["open_time"])
    latest_15m = df_15m.tail(200).copy()
    print(f"\n使用最新 {len(latest_15m)} 条 15m 数据")

    # 各周期预测
    probs = {}
    for tf in TIMEFRAMES:
        try:
            df_tf = latest_15m if tf == "15m" else resample_to_timeframe(latest_15m, tf)
            if len(df_tf) < 60:
                continue
            df_feat = calculate_features(df_tf)
            cols = feature_cols_dict[tf]
            X = df_feat[cols].tail(1).values
            X_scaled = scalers[tf].transform(X)
            X_3d = X_scaled.reshape(1, 1, -1)
            prob = models[tf].predict(X_3d, verbose=0)[0][0]
            probs[tf] = float(prob)
            print(f"  {tf:>3} → {prob:.4f}")
        except:
            probs[tf] = 0.5

    # 动态权重 + 融合
    weights = get_dynamic_weights()  # 动态更新
    final_prob = sum(weights[tf] * probs.get(tf, 0.5) for tf in TIMEFRAMES)
    print(f"\n融合概率: {final_prob:.4f}")

    # 信号
    if final_prob > FUSION_THRESHOLD:
        strength = "强" if final_prob > 0.55 else "中"
        signal = f"【{strength}看涨】建议做多"
    else:
        signal = f"观望 ({final_prob:.4f} < {FUSION_THRESHOLD})"

    print(f"\n{signal}")
    print(f"\n权重: { {k:f'{v:.3f}' for k,v in weights.items()} }")

    # 推送（可选）
    try:
        from signal_push import push_signal

        push_signal(signal, final_prob)
    except:
        pass


if __name__ == "__main__":
    fusion_predict()
