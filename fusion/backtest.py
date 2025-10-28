# fusion/backtest.py
import pandas as pd
import numpy as np
import time
import requests
import logging
import joblib
import os
import warnings
from datetime import datetime
from config import TIMEFRAMES, SYMBOL, MODEL_DIR, FUSION_THRESHOLD
from dynamic_weights import get_dynamic_weights
from tensorflow.keras.models import load_model

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ==================== 数据获取与处理 (已稳定) ====================
def fetch_binance_klines(s, i, st, en=None, l=1000):
    url, cols = "https://api.binance.com/api/v3/klines", [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    sts, ets = int(pd.to_datetime(st).timestamp() * 1000), (
        int(pd.to_datetime(en).timestamp() * 1000) if en else int(time.time() * 1000)
    )
    all_d, retries = [], 5
    while sts < ets:
        p = {
            "symbol": s.upper(),
            "interval": i,
            "startTime": sts,
            "endTime": ets,
            "limit": l,
        }
        for _ in range(retries):
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
            except Exception:
                time.sleep(1)
        else:
            print(f"获取 {s} {i} 数据失败")
            return pd.DataFrame()
    if not all_d:
        return pd.DataFrame()
    df = pd.DataFrame(all_d)
    df = df.iloc[:, :6]
    df.columns = cols
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("timestamp").sort_index()


def calculate_features(df):
    df = df.copy()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pandas_ta as ta

            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.obv(append=True)
            df["volatility"] = (
                np.log(df["close"] / df["close"].shift(1)).rolling(window=20).std()
            )
            df["volume_change_rate"] = df["volume"].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
    except ImportError:
        print("警告: pandas_ta 未安装，特征计算将受限。")
    except Exception as e:
        print(f"计算特征时出错: {e}")
    return df.fillna(0)


def resample_to_timeframe(df_15m, timeframe):
    if timeframe in ["15m", ""]:
        return df_15m
    # 【【【 最终修正：将 'H' 改为 'h' 以消除警告 】】】
    rule_map = {"_1h": "1h", "_4h": "4h", "_8h": "8h"}
    if timeframe not in rule_map:
        return df_15m
    return (
        df_15m.resample(rule_map[timeframe])
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


# ==================== 回测主函数 (清爽终极版) ====================
def run_backtest(start_date, end_date, step_candles=4):
    LOOK_BACK = 60
    INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE_RATE = 10000.0, 0.001, 0.0005
    print(
        f"\n{'='*70}\n  4周期LSTM融合系统 交易回测\n  时间范围: {start_date} → {end_date}\n{'='*70}\n"
    )

    print("--- 1. 加载模型文件 ---")
    models, scalers, feature_cols_dict, loaded_tfs = {}, {}, {}, []
    for tf_name in TIMEFRAMES:
        try:
            model_path = f"{MODEL_DIR}/eth_trend_model_v1{tf_name}.keras"
            scaler_path = f"{MODEL_DIR}/eth_trend_scaler_v1{tf_name}.joblib"
            cols_path = f"{MODEL_DIR}/feature_columns{tf_name}.joblib"

            if not all(os.path.exists(p) for p in [model_path, scaler_path, cols_path]):
                raise FileNotFoundError(f"文件不存在 for '{tf_name}' in '{MODEL_DIR}'.")

            models[tf_name] = load_model(model_path, compile=False)
            scalers[tf_name] = joblib.load(scaler_path)
            feature_cols_dict[tf_name] = joblib.load(cols_path)
            loaded_tfs.append(tf_name)
            print(f"  ✅ 成功加载 {tf_name}")
        except Exception as e:
            print(f"  ❌ 加载 '{tf_name}' 失败: {e}。将跳过该模型。")

    if not loaded_tfs:
        print("\n错误: 没有任何模型被成功加载，无法继续回测。")
        return
    print(f"\n将使用以下成功加载的模型进行回测: {loaded_tfs}")

    print("\n--- 2. 获取历史数据 ---")
    df_15m = fetch_binance_klines(SYMBOL, "15m", start_date, end_date)
    if df_15m.empty:
        return
    df_15m = df_15m[(df_15m.index >= start_date) & (df_15m.index <= end_date)]
    print(f"获取数据成功: {len(df_15m)} 条 15m K线")

    print("\n--- 3. 开始滚动预测 ---")
    capital, position, entry_price = INITIAL_CAPITAL, 0.0, 0.0
    equity_curve, trade_log = [], []
    min_window = 300

    for i in range(min_window, len(df_15m), step_candles):
        if i % 1000 == 0:
            print(f"  进度: {i}/{len(df_15m)}")
        window_15m = df_15m.iloc[i - min_window : i]
        current_time, current_price = window_15m.index[-1], window_15m["close"].iloc[-1]

        probs = {}
        for tf in loaded_tfs:
            try:
                df_tf = resample_to_timeframe(window_15m, tf)
                df_feat = calculate_features(df_tf)
                cols = feature_cols_dict[tf]
                df_feat_aligned = df_feat.reindex(columns=cols, fill_value=0)
                sequence_data = df_feat_aligned.tail(LOOK_BACK).values

                if len(sequence_data) < LOOK_BACK:
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    sequence_scaled = scalers[tf].transform(sequence_data)

                X_3d = np.expand_dims(sequence_scaled, axis=0)
                probs[tf] = float(models[tf].predict(X_3d, verbose=0)[0][0])
            except Exception:
                probs[tf] = 0.5

        if not probs:
            continue

        weights = get_dynamic_weights()
        final_prob = sum(
            weights.get(tf, 1 / len(loaded_tfs)) * probs.get(tf, 0.5)
            for tf in loaded_tfs
        )
        signal = 1 if final_prob > FUSION_THRESHOLD else 0

        if signal == 1 and position == 0:
            buy_price = current_price * (1 + SLIPPAGE_RATE)
            position = (capital / buy_price) * (1 - COMMISSION_RATE)
            capital = 0.0
            entry_price = buy_price
        elif signal == 0 and position > 0:
            sell_price = current_price * (1 - SLIPPAGE_RATE)
            revenue = position * sell_price * (1 - COMMISSION_RATE)
            trade_log.append({"pnl": revenue - (position * entry_price)})
            capital, position = revenue, 0.0

        equity_curve.append(
            {"time": current_time, "equity": capital + (position * current_price)}
        )

    print("\n--- 4. 生成回测报告 ---")
    if not equity_curve:
        print("\n回测未能生成净值曲线。")
        return
    df_equity = pd.DataFrame(equity_curve).set_index("time")

    if not trade_log:
        print("\n回测期间无任何交易。")
        final_equity = (
            df_equity["equity"].iloc[-1] if not df_equity.empty else INITIAL_CAPITAL
        )
        print(f"期末资产: {final_equity:,.2f} (无变化)")
        return

    df_trades = pd.DataFrame(trade_log)
    total_return = (df_equity["equity"].iloc[-1] / INITIAL_CAPITAL) - 1
    wins, losses = df_trades[df_trades["pnl"] > 0], df_trades[df_trades["pnl"] <= 0]
    win_rate = len(wins) / len(df_trades) if len(df_trades) > 0 else 0
    gross_profit, gross_loss = wins["pnl"].sum(), abs(losses["pnl"].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    equity_values = df_equity["equity"]
    max_drawdown = (equity_values / equity_values.cummax() - 1).min()
    daily_returns = equity_values.resample("D").last().pct_change().dropna()
    sharpe_ratio = (
        np.sqrt(365) * daily_returns.mean() / daily_returns.std()
        if daily_returns.std() != 0
        else 0
    )

    print(f"\n{'='*50}\n回测完成！")
    print(
        f"{'-'*50}\n  期初资产: {INITIAL_CAPITAL:,.2f}\n  期末资产: {df_equity['equity'].iloc[-1]:,.2f}\n  总收益率: {total_return:.2%}"
    )
    print(
        f"{'-'*50}\n  总交易次数: {len(df_trades)}\n  胜率: {win_rate:.1%}\n  盈利因子: {profit_factor:.2f}"
    )
    print(f"  总盈利: {gross_profit:,.2f}\n  总亏损: {gross_loss:,.2f}")
    print(
        f"{'-'*50}\n  最大回撤: {max_drawdown:.2%}\n  年化夏普比率: {sharpe_ratio:.2f}\n{'='*50}"
    )


# ==================== 运行 ====================
if __name__ == "__main__":
    run_backtest(start_date="2025-05-01", end_date="2025-10-23")
