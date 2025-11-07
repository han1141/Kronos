import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import logging
import pandas_ta as ta
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

# from sklearn.utils import class_weight  #不再需要
# import tensorflow as tf #不再需要
# from tensorflow.keras.models import Sequential, load_model #不再需要
# from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input #不再需要
# from tensorflow.keras.callbacks import EarlyStopping #不再需要
# from tensorflow.keras.regularizers import l2 #不再需要
import joblib
import lightgbm as lgb  # 引入 LightGBM

# ... (从这里到 train_and_evaluate 函数的所有代码都保持不变) ...

# --- 0. 设置与数据获取 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SYMBOL = "ETHUSDT"
INTERVAL = "15m"
DATA_START_DATE = "2017-01-01"
TRAIN_START_DATE = "2018-01-01"
TRAIN_END_DATE = "2025-01-01"
TEST_START_DATE = "2025-03-01"
TEST_END_DATE = "2025-11-06"
LOOK_BACK = 60

TREND_CONFIG = {
    "look_forward_steps": 5,
    "ema_length": 8,
}

# 修改模型和Scaler的保存路径以反映模型类型
MODEL_SAVE_PATH = f"models/eth_trend_model_lgb_{INTERVAL}.joblib"
SCALER_SAVE_PATH = f"models/eth_trend_scaler_lgb_{INTERVAL}.joblib"
FEATURE_COLUMNS_PATH = f"models/feature_columns_lgb_{INTERVAL}.joblib"


DATA_CACHE_PATH = f"data/{SYMBOL.lower()}_{INTERVAL}_data.csv"


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
                sts = d[-1][0] + 1
                break
            except requests.exceptions.RequestException as e:
                last_e = e
                logger.warning(f"请求失败，正在重试... ({attempt + 1}/{retries})")
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


# --- 1. 特征工程与标签定义 ---
def feature_engineering(df):
    logger.info("--- 开始计算特征指标 ---")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df["volatility"] = (
        np.log(df["Close"] / df["Close"].shift(1)).rolling(window=20).std()
    )
    df["volume_change_rate"] = df["Volume"].pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def create_trend_labels(df, look_forward_steps=12, ema_length=8):
    logger.info(
        f"--- 创建趋势标签 (向前看{look_forward_steps}根K线, EMA周期{ema_length}) ---"
    )
    df = df.copy()
    ema_col = f"EMA_{ema_length}"
    df.ta.ema(length=ema_length, append=True)
    future_ema = df[ema_col].shift(-look_forward_steps)
    df["label"] = (future_ema > df[ema_col]).astype(int)
    df.dropna(inplace=True)
    logger.info(f"趋势标签创建完成. 剩余数据: {len(df)} 条")
    return df


# --- 2. 数据准备 (修改) ---
def create_flattened_sequences(data, labels, look_back=60):
    """
    为GBDT模型创建展平的序列数据。
    每个样本X是look_back个时间步的所有特征的展平向量。
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        feature_sequence = data[i : (i + look_back), :]
        X.append(feature_sequence.flatten())
        y.append(labels[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# --- 3. 核心训练与评估函数 (使用 LightGBM 重写) ---
def train_and_evaluate(
    train_df, test_df, look_back, model_save_path, scaler_save_path, trend_config
):
    logger.info("--- 开始数据准备和预处理 ---")

    train_featured = feature_engineering(train_df.copy())
    train_labeled = create_trend_labels(train_featured, **trend_config)

    test_featured = feature_engineering(test_df.copy())
    test_labeled = create_trend_labels(test_featured, **trend_config)

    y_train_full = train_labeled["label"].values
    X_train_full_df = train_labeled.drop(columns=["label"])
    y_test_full = test_labeled["label"].values
    X_test_df = test_labeled.drop(columns=["label"])

    X_test_df = X_test_df[X_train_full_df.columns]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_full_df)
    X_test_scaled = scaler.transform(X_test_df)

    joblib.dump(X_train_full_df.columns, FEATURE_COLUMNS_PATH)

    # 使用新的函数创建展平的数据
    X_train, y_train = create_flattened_sequences(
        X_train_scaled, y_train_full, look_back
    )
    X_test, y_test = create_flattened_sequences(X_test_scaled, y_test_full, look_back)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logger.error("数据不足，无法创建训练或测试序列。")
        return None, None, None

    logger.info(
        f"数据准备完成。特征维度: {X_train.shape[1]}, 训练样本: {len(X_train)}, 测试样本: {len(X_test)}"
    )

    train_label_counts = np.bincount(y_train)
    if len(train_label_counts) < 2 or 0 in train_label_counts:
        logger.error("训练数据中只有一个类别，无法继续。")
        return None, None, None
    logger.info(
        f"训练集标签分布: 0 = {train_label_counts[0]}, 1 = {train_label_counts[1]}"
    )

    # 计算 scale_pos_weight 来处理类别不平衡
    scale_pos_weight = train_label_counts[0] / train_label_counts[1]
    logger.info(f"计算出的 scale_pos_weight: {scale_pos_weight:.2f}")

    # 定义 LightGBM 模型
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric="logloss",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
    )

    logger.info("\n开始训练 LightGBM 模型...")
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=True)],
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(lgb_model, model_save_path)  # 使用 joblib 保存模型
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"模型已保存到: {model_save_path}")
    logger.info(f"Scaler已保存到: {scaler_save_path}")

    logger.info("\n--- 开始在测试集上评估模型 ---")
    y_pred_probs = lgb_model.predict_proba(X_test)[:, 1]  # 获取类别1的概率
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)
    f1_scores = np.divide(
        2 * recalls * precisions,
        recalls + precisions,
        out=np.zeros_like(recalls),
        where=(recalls + precisions) != 0,
    )
    best_f1_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
    y_pred_labels_best_f1 = (y_pred_probs > best_f1_threshold).astype(int)

    print("\n--- 测试集评估结果 (使用最佳F1阈值) ---")
    print(f"找到的最佳F1阈值: {best_f1_threshold:.4f}")
    print(f"准确率: {accuracy_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"精确率 (胜率): {precision_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"F1 分数: {f1_score(y_test, y_pred_labels_best_f1):.4f}")

    return lgb_model, scaler, best_f1_threshold


# ==============================================================================
# === 独立的预测演示函数 (已修复) ===
# ==============================================================================
def run_prediction_demo(
    raw_df, model_path, scaler_path, trend_config, look_back, threshold
):
    """
    加载已保存的 LightGBM 模型并对最新数据进行一次预测演示。
    (已修改为使用iloc进行切片，以适应不同时间间隔)
    """
    logger.info("\n" + "=" * 60)
    logger.info("--- 加载已保存的 LightGBM 模型进行预测演示 ---")
    logger.info("=" * 60)

    try:
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        logger.info("模型、Scaler和特征列加载成功。")

        # --- 这是主要的修改点 ---
        # 我们需要 look_back 条数据来构建最终的预测序列，
        # 并且需要额外的数据（例如50条）来确保特征计算（如Boll(20)）不会产生NaN。
        required_rows = look_back + 50
        if len(raw_df) < required_rows:
            logger.error(
                f"数据不足以进行预测。需要 {required_rows} 条，但只有 {len(raw_df)} 条。"
            )
            return

        # 使用 iloc 从末尾获取固定数量的数据，而不是使用时间偏移量
        latest_data = raw_df.iloc[-required_rows:].copy()
        logger.info(f"已获取最新的 {len(latest_data)} 条数据用于预测准备。")
        # --- 修改结束 ---

        latest_featured = feature_engineering(latest_data)
        latest_featured.ta.ema(length=trend_config["ema_length"], append=True)

        # 在对齐特征列之前进行dropna
        latest_featured.dropna(inplace=True)

        if latest_featured.empty:
            logger.error("在特征计算和dropna之后，没有剩下任何数据用于预测。")
            return

        latest_featured_aligned = latest_featured.reindex(
            columns=feature_columns, fill_value=0
        )
        latest_scaled = loaded_scaler.transform(latest_featured_aligned)

        # 检查是否有足够的数据来构建最后的序列
        if len(latest_scaled) < look_back:
            logger.error(f"预处理后数据不足 {look_back} 条，无法构建预测序列。")
            return

        # 准备展平的序列数据用于预测
        last_sequence_unflattened = latest_scaled[-look_back:]
        last_sequence_flattened = last_sequence_unflattened.flatten().reshape(1, -1)

        # 检查特征数量是否匹配
        if last_sequence_flattened.shape[1] != loaded_model.n_features_:
            logger.error(
                f"特征数量不匹配! 模型需要 {loaded_model.n_features_} 个特征, "
                f"但输入数据有 {last_sequence_flattened.shape[1]} 个特征。"
            )
            return

        prediction_prob = loaded_model.predict_proba(last_sequence_flattened)[0][1]

        logger.info(f"对最新的数据序列进行预测...")
        logger.info(f"模型输出的上涨趋势概率: {prediction_prob:.4f}")
        logger.info(f"将使用在测试集上计算出的最佳阈值: {threshold:.4f}")

        if prediction_prob > threshold:
            logger.info(f"预测结果: 看涨趋势 (概率 > {threshold:.4f})")
        else:
            logger.info(f"预测结果: 非看涨趋势 (概率 <= {threshold:.4f})")

    except FileNotFoundError as e:
        logger.error(f"加载文件失败: {e}. 请确保模型等文件都存在于'models/'目录下。")
    except Exception as e:
        logger.error(f"加载模型或进行预测时出错: {e}")


# --- 主流程 (已修改) ---
if __name__ == "__main__":
    if os.path.exists(DATA_CACHE_PATH):
        logger.info(f"从缓存文件 {DATA_CACHE_PATH} 加载数据...")
        raw_df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        logger.info(f"本地没有缓存数据，从币安API获取...")
        raw_df = fetch_binance_klines(
            s=SYMBOL, i=INTERVAL, st=DATA_START_DATE, en=TEST_END_DATE
        )
        if not raw_df.empty:
            os.makedirs(os.path.dirname(DATA_CACHE_PATH), exist_ok=True)
            raw_df.to_csv(DATA_CACHE_PATH)
            logger.info(f"数据已缓存到 {DATA_CACHE_PATH}")

    if raw_df.empty:
        exit()

    train_df = raw_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    test_df = raw_df.loc[TEST_START_DATE:TEST_END_DATE]

    logger.info(f"训练数据周期: {train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"测试数据周期: {test_df.index.min()} to {test_df.index.max()}")

    # 训练并获取返回结果
    trained_model, trained_scaler, best_f1_threshold = train_and_evaluate(
        train_df, test_df, LOOK_BACK, MODEL_SAVE_PATH, SCALER_SAVE_PATH, TREND_CONFIG
    )

    # 检查训练是否成功，然后调用新的预测函数
    if trained_model and trained_scaler and best_f1_threshold is not None:
        run_prediction_demo(
            raw_df=raw_df.loc[:TEST_END_DATE],  # 确保只使用训练和测试期间的数据
            model_path=MODEL_SAVE_PATH,
            scaler_path=SCALER_SAVE_PATH,
            trend_config=TREND_CONFIG,
            look_back=LOOK_BACK,
            threshold=best_f1_threshold,
        )
    else:
        logger.info("模型训练失败，跳过预测演示。")
