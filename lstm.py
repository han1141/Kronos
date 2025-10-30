# ============================================================================
# LSTM加密货币趋势预测模型
# 功能：使用双向LSTM神经网络预测加密货币价格趋势
# 作者：Kronos项目
# ============================================================================

# 导入必要的库
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理和分析库
import matplotlib.pyplot as plt  # 绘图库
import requests  # HTTP请求库，用于API调用
import time  # 时间处理库
import logging  # 日志记录库
import pandas_ta as ta  # 技术分析指标库
import os  # 操作系统接口库

# 机器学习相关库
from sklearn.preprocessing import MinMaxScaler  # 数据标准化
from sklearn.metrics import (  # 模型评估指标
    accuracy_score,  # 准确率
    precision_score,  # 精确率
    recall_score,  # 召回率
    f1_score,  # F1分数
    precision_recall_curve,  # 精确率-召回率曲线
)
from sklearn.utils import class_weight  # 类别权重计算

# TensorFlow和Keras深度学习库
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # 序列模型和模型加载
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Input,
)  # 神经网络层
from tensorflow.keras.callbacks import EarlyStopping  # 早停回调
from tensorflow.keras.regularizers import l2  # L2正则化

import joblib  # 模型序列化库

# ============================================================================
# 0. 日志配置与数据获取模块
# ============================================================================

# 配置日志系统
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_binance_klines(s, i, st, en=None, l=1000):
    """
    从币安API获取K线数据

    参数:
        s (str): 交易对符号，如 'ETHUSDT'
        i (str): K线间隔，如 '15m', '1h', '1d'
        st (str): 开始时间，格式如 '2018-01-01'
        en (str, optional): 结束时间，默认为当前时间
        l (int): 每次请求的数据条数限制，默认1000

    返回:
        pandas.DataFrame: 包含OHLCV数据的DataFrame，以时间戳为索引
    """
    # 币安API端点和返回数据的列名
    url, cols = "https://api.binance.com/api/v3/klines", [
        "timestamp",  # 开盘时间戳
        "Open",  # 开盘价
        "High",  # 最高价
        "Low",  # 最低价
        "Close",  # 收盘价
        "Volume",  # 成交量
        "close_time",  # 收盘时间戳
        "quote_asset_volume",  # 成交额
        "number_of_trades",  # 成交笔数
        "taker_buy_base_volume",  # 主动买入成交量
        "taker_buy_quote_volume",  # 主动买入成交额
        "ignore",  # 忽略字段
    ]

    # 将时间字符串转换为毫秒时间戳
    sts, ets = int(pd.to_datetime(st).timestamp() * 1000), (
        int(pd.to_datetime(en).timestamp() * 1000) if en else int(time.time() * 1000)
    )

    # 初始化变量：所有数据、重试次数、最后的错误
    all_d, retries, last_e = [], 5, None

    # 分批获取数据，直到获取完所有时间范围内的数据
    while sts < ets:
        # 构建API请求参数
        p = {
            "symbol": s.upper(),  # 交易对符号（大写）
            "interval": i,  # K线间隔
            "startTime": sts,  # 开始时间戳
            "endTime": ets,  # 结束时间戳
            "limit": l,  # 数据条数限制
        }

        # 重试机制：最多重试5次
        for attempt in range(retries):
            try:
                # 发送HTTP GET请求
                r = requests.get(url, params=p, timeout=15)
                r.raise_for_status()  # 检查HTTP状态码
                d = r.json()  # 解析JSON响应

                # 如果没有数据，跳出循环
                if not d:
                    sts = ets
                    break

                # 将数据添加到总列表中
                all_d.extend(d)
                # 更新下次请求的开始时间（避免重复数据）
                sts = d[-1][0] + 1
                break  # 成功获取数据，跳出重试循环

            except requests.exceptions.RequestException as e:
                last_e = e
                logger.warning(f"请求失败，正在重试... ({attempt + 1}/{retries})")
                time.sleep(2**attempt)  # 指数退避策略
        else:
            # 所有重试都失败
            logger.error(f"获取 {s} 失败: {last_e}")
            return pd.DataFrame()

    # 如果没有获取到任何数据
    if not all_d:
        return pd.DataFrame()

    # 创建DataFrame并选择需要的列
    df = pd.DataFrame(all_d, columns=cols)[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ].copy()

    # 数据类型转换
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # 时间戳转换
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # 价格和成交量转换为数值

    logger.info(f"✅ 获取 {s} 数据成功: {len(df)} 条")
    return df.set_index("timestamp").sort_index()  # 设置时间戳为索引并排序


# ============================================================================
# 1. 特征工程与标签定义模块
# ============================================================================


def feature_engineering(df):
    """
    对原始OHLCV数据进行特征工程，计算各种技术分析指标

    参数:
        df (pandas.DataFrame): 包含OHLCV数据的DataFrame

    返回:
        pandas.DataFrame: 添加了技术指标特征的DataFrame
    """
    logger.info("--- 开始计算特征指标 ---")

    # 计算RSI指标（相对强弱指数）- 衡量价格变动的速度和幅度
    df.ta.rsi(length=14, append=True)

    # 计算MACD指标（移动平均收敛散度）- 趋势跟踪动量指标
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # 计算布林带指标 - 衡量价格相对于移动平均线的位置
    df.ta.bbands(length=20, std=2, append=True)

    # 计算ADX指标（平均趋向指数）- 衡量趋势强度
    df.ta.adx(length=14, append=True)

    # 计算ATR指标（平均真实波幅）- 衡量市场波动性
    df.ta.atr(length=14, append=True)

    # 计算OBV指标（能量潮）- 结合价格和成交量的动量指标
    df.ta.obv(append=True)

    # 计算价格波动率 - 使用对数收益率的滚动标准差
    df["volatility"] = (
        np.log(df["Close"] / df["Close"].shift(1)).rolling(window=20).std()
    )

    # 计算成交量变化率 - 衡量成交量的相对变化
    df["volume_change_rate"] = df["Volume"].pct_change()

    # 处理无穷大值，将其替换为NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def create_trend_labels(df, look_forward_steps=12, ema_length=8):
    """
    创建趋势预测标签，基于未来EMA的方向

    参数:
        df (pandas.DataFrame): 包含价格数据的DataFrame
        look_forward_steps (int): 向前看的K线数量，默认12
        ema_length (int): EMA的周期长度，默认8

    返回:
        pandas.DataFrame: 添加了趋势标签的DataFrame

    标签定义:
        1: 上涨趋势（未来EMA > 当前EMA）
        0: 下跌趋势（未来EMA <= 当前EMA）
    """
    logger.info(
        f"--- 创建趋势标签 (向前看{look_forward_steps}根K线, EMA周期{ema_length}) ---"
    )

    df = df.copy()
    ema_col = f"EMA_{ema_length}"

    # 计算指定周期的EMA（指数移动平均线）
    df.ta.ema(length=ema_length, append=True)

    # 获取未来时间点的EMA值（向前移动look_forward_steps步）
    future_ema = df[ema_col].shift(-look_forward_steps)

    # 创建二分类标签：未来EMA > 当前EMA 则为1（上涨），否则为0（下跌）
    df["label"] = (future_ema > df[ema_col]).astype(int)

    # 删除包含NaN值的行（由于shift操作产生的）
    df.dropna(inplace=True)

    logger.info(f"趋势标签创建完成. 剩余数据: {len(df)} 条")
    return df


# ============================================================================
# 2. 数据准备模块
# ============================================================================


def create_multivariate_sequences(data, labels, look_back=60):
    """
    创建用于LSTM训练的时间序列数据

    参数:
        data (numpy.ndarray): 标准化后的特征数据
        labels (numpy.ndarray): 对应的标签数据
        look_back (int): 回看窗口大小，默认60

    返回:
        tuple: (X, y) 其中X是3D数组 (样本数, 时间步, 特征数)，y是1D标签数组
    """
    X, y = [], []

    # 滑动窗口创建序列数据
    for i in range(len(data) - look_back):
        # 取look_back长度的历史数据作为输入序列
        X.append(data[i : (i + look_back), :])
        # 对应的标签是序列结束后的下一个标签
        y.append(labels[i + look_back])

    # 转换为numpy数组并指定数据类型
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ============================================================================
# 3. 核心训练与评估函数模块
# ============================================================================


def train_and_evaluate(
    train_df, test_df, look_back, model_save_path, scaler_save_path, trend_config
):
    """
    训练LSTM模型并评估性能

    参数:
        train_df (pandas.DataFrame): 训练数据集
        test_df (pandas.DataFrame): 测试数据集
        look_back (int): LSTM回看窗口大小
        model_save_path (str): 模型保存路径
        scaler_save_path (str): 数据标准化器保存路径
        trend_config (dict): 趋势标签配置参数

    返回:
        tuple: (模型, 标准化器, 最佳F1阈值)
    """
    logger.info("--- 开始数据准备和预处理 ---")

    # 对训练数据进行特征工程和标签创建
    train_featured = feature_engineering(train_df.copy())
    train_labeled = create_trend_labels(train_featured, **trend_config)

    # 对测试数据进行相同的处理
    test_featured = feature_engineering(test_df.copy())
    test_labeled = create_trend_labels(test_featured, **trend_config)

    # 分离特征和标签
    y_train_full = train_labeled["label"].values  # 训练标签
    X_train_full_df = train_labeled.drop(columns=["label"])  # 训练特征
    y_test_full = test_labeled["label"].values  # 测试标签
    X_test_df = test_labeled.drop(columns=["label"])  # 测试特征

    # 确保测试集特征列与训练集一致
    X_test_df = X_test_df[X_train_full_df.columns]

    # 数据标准化：将特征缩放到[0,1]范围
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_full_df)  # 训练集拟合并转换
    X_test_scaled = scaler.transform(X_test_df)  # 测试集仅转换

    # 保存特征列名，用于后续预测时的特征对齐
    joblib.dump(X_train_full_df.columns, "models/feature_columns.joblib")

    # 创建LSTM输入序列
    X_train, y_train = create_multivariate_sequences(
        X_train_scaled, y_train_full, look_back
    )
    X_test, y_test = create_multivariate_sequences(
        X_test_scaled, y_test_full, look_back
    )

    logger.info(f"数据类型检查: X_train -> {X_train.dtype}, y_train -> {y_train.dtype}")

    # 检查数据是否足够
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logger.error("数据不足，无法创建训练或测试序列。")
        return None, None, None

    num_features = X_train.shape[2]  # 特征数量
    logger.info(
        f"数据准备完成。特征: {num_features}, 训练样本: {len(X_train)}, 测试样本: {len(X_test)}"
    )

    # 检查标签分布，确保有两个类别
    train_label_counts = np.bincount(y_train)
    if len(train_label_counts) < 2 or 0 in train_label_counts:
        logger.error("训练数据中只有一个类别，无法继续。")
        return None, None, None
    logger.info(
        f"训练集标签分布: 0 = {train_label_counts[0]}, 1 = {train_label_counts[1]}"
    )

    # 计算类别权重以处理不平衡数据
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    logger.info(f"类别权重: {class_weight_dict}")

    # 构建双向LSTM模型
    L2_REG = 0.001
    model = Sequential(
        [
            Input(shape=(look_back, num_features)),
            # 2. 【关键】简化模型结构 (减少神经元/记忆单元)
            Bidirectional(
                LSTM(32, return_sequences=True, kernel_regularizer=l2(L2_REG))
            ),
            # 3. 【关键】大幅加强Dropout (增加随机“遗忘”)
            Dropout(0.5),
            Bidirectional(LSTM(32, kernel_regularizer=l2(L2_REG))),
            Dropout(0.5),
            Dense(16, activation="relu", kernel_regularizer=l2(L2_REG)),
            Dense(1, activation="sigmoid"),
        ]
    )

    # 编译模型：使用Adam优化器和二元交叉熵损失函数
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()  # 打印模型结构

    # 设置早停回调：监控验证损失，10轮无改善则停止训练
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    logger.info("\n开始训练模型...")
    # 训练模型
    model.fit(
        X_train,  # 训练特征
        y_train,  # 训练标签
        batch_size=32,  # 批次大小
        epochs=100,  # 最大训练轮数
        class_weight=class_weight_dict,  # 类别权重
        validation_data=(X_test, y_test),  # 验证数据
        callbacks=[early_stopping],  # 早停回调
        verbose=1,  # 显示训练过程
    )

    # 保存模型和标准化器
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"模型已保存到: {model_save_path}")
    logger.info(f"Scaler已保存到: {scaler_save_path}")

    # 模型评估
    logger.info("\n--- 开始在测试集上评估模型 ---")

    # 获取预测概率
    y_pred_probs = model.predict(X_test).flatten()

    # 计算精确率-召回率曲线
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_probs)

    # 计算F1分数并找到最佳阈值
    f1_scores = np.divide(
        2 * recalls * precisions,
        recalls + precisions,
        out=np.zeros_like(recalls),
        where=(recalls + precisions) != 0,
    )
    best_f1_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5

    # 使用最佳阈值生成预测标签
    y_pred_labels_best_f1 = (y_pred_probs > best_f1_threshold).astype(int)

    # 打印评估结果
    print("\n--- 测试集评估结果 (使用最佳F1阈值) ---")
    print(f"找到的最佳F1阈值: {best_f1_threshold:.4f}")
    print(f"准确率: {accuracy_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"精确率 (胜率): {precision_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"召回率: {recall_score(y_test, y_pred_labels_best_f1):.4f}")
    print(f"F1 分数: {f1_score(y_test, y_pred_labels_best_f1):.4f}")

    return model, scaler, best_f1_threshold


# ============================================================================
# 4. 预测演示模块
# ============================================================================


def run_prediction_demo(
    raw_df, model_path, scaler_path, trend_config, look_back, threshold
):
    """
    加载已保存的模型并对最新数据进行预测演示

    参数:
        raw_df (pandas.DataFrame): 原始价格数据
        model_path (str): 训练好的模型文件路径
        scaler_path (str): 数据标准化器文件路径
        trend_config (dict): 趋势配置参数
        look_back (int): LSTM回看窗口大小
        threshold (float): 预测阈值
    """
    logger.info("\n" + "=" * 60)
    logger.info("--- 加载已保存的模型进行预测演示 ---")
    logger.info("=" * 60)

    try:
        # 加载训练好的模型和相关文件
        loaded_model = load_model(model_path)  # 加载Keras模型
        loaded_scaler = joblib.load(scaler_path)  # 加载数据标准化器
        feature_columns = joblib.load("models/feature_columns.joblib")  # 加载特征列名
        logger.info("模型、Scaler和特征列加载成功。")

        # 准备预测数据：取最新的一段数据
        prediction_end_date = raw_df.index[-1]  # 最新时间点
        prediction_start_date = prediction_end_date - pd.DateOffset(
            hours=look_back + 200  # 确保有足够的数据计算技术指标
        )
        latest_data = raw_df.loc[prediction_start_date:prediction_end_date].copy()

        # 对最新数据进行特征工程
        latest_featured = feature_engineering(latest_data)
        # 计算EMA指标（与训练时保持一致）
        latest_featured.ta.ema(length=trend_config["ema_length"], append=True)
        latest_featured.dropna(inplace=True)

        # 确保特征列与训练时一致
        latest_featured_aligned = latest_featured.reindex(
            columns=feature_columns, fill_value=0
        )
        # 使用训练时的标准化器进行数据标准化
        latest_scaled = loaded_scaler.transform(latest_featured_aligned)

        # 准备LSTM输入序列：取最后look_back个时间步
        last_sequence = latest_scaled[-look_back:]
        last_sequence = np.expand_dims(last_sequence, axis=0)  # 添加批次维度

        # 进行预测
        prediction_prob = loaded_model.predict(last_sequence)[0][0]

        # 输出预测结果
        logger.info(f"对最新的数据序列进行预测...")
        logger.info(f"模型输出的上涨趋势概率: {prediction_prob:.4f}")
        logger.info(f"将使用在测试集上计算出的最佳阈值: {threshold:.4f}")

        # 根据阈值判断趋势方向
        if prediction_prob > threshold:
            logger.info(f"预测结果: 看涨趋势 (概率 > {threshold:.4f})")
        else:
            logger.info(f"预测结果: 非看涨趋势 (概率 <= {threshold:.4f})")

    except FileNotFoundError as e:
        logger.error(f"加载文件失败: {e}. 请确保模型等文件都存在于'models/'目录下。")
    except Exception as e:
        logger.error(f"加载模型或进行预测时出错: {e}")


# ============================================================================
# 5. 主程序流程
# ============================================================================

if __name__ == "__main__":
    # 配置参数
    SYMBOL = "ETHUSDT"
    INTERVAL = "15m"
    DATA_START_DATE = "2018-01-01"
    TRAIN_START_DATE = "2019-01-01"
    TRAIN_END_DATE = "2025-09-30"
    TEST_START_DATE = "2025-10-01"
    TEST_END_DATE = "2025-10-29"
    LOOK_BACK = 60  # LSTM回看窗口大小

    # 趋势标签配置
    TREND_CONFIG = {
        "look_forward_steps": 5,  # 向前看的K线数量
        "ema_length": 8,  # EMA周期长度
    }

    # 文件路径配置
    MODEL_SAVE_PATH = "models/eth_trend_model_v1.keras"  # 模型保存路径
    SCALER_SAVE_PATH = "models/eth_trend_scaler_v1.joblib"  # 标准化器保存路径
    DATA_CACHE_PATH = f"data/{SYMBOL.lower()}_{INTERVAL}_data.csv"  # 数据缓存路径

    # 数据加载：优先使用缓存，否则从API获取
    if os.path.exists(DATA_CACHE_PATH):
        logger.info(f"从缓存文件 {DATA_CACHE_PATH} 加载数据...")
        raw_df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        logger.info(f"本地没有缓存数据，从币安API获取...")
        raw_df = fetch_binance_klines(
            s=SYMBOL, i=INTERVAL, st=DATA_START_DATE, en=TEST_END_DATE
        )
        # 保存数据到缓存文件
        if not raw_df.empty:
            os.makedirs(os.path.dirname(DATA_CACHE_PATH), exist_ok=True)
            raw_df.to_csv(DATA_CACHE_PATH)
            logger.info(f"数据已缓存到 {DATA_CACHE_PATH}")

    # 检查数据是否成功获取
    if raw_df.empty:
        logger.error("无法获取数据，程序退出。")
        exit()

    # 数据分割：按时间划分训练集和测试集
    train_df = raw_df.loc[TRAIN_START_DATE:TRAIN_END_DATE]
    test_df = raw_df.loc[TEST_START_DATE:TEST_END_DATE]

    logger.info(f"训练数据周期: {train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"测试数据周期: {test_df.index.min()} to {test_df.index.max()}")

    # 模型训练和评估
    trained_model, trained_scaler, best_f1_threshold = train_and_evaluate(
        train_df, test_df, LOOK_BACK, MODEL_SAVE_PATH, SCALER_SAVE_PATH, TREND_CONFIG
    )

    # 预测演示：如果训练成功，则进行预测演示
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
