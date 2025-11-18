import numpy as np
import pandas as pd

# --- ✅ NumPy 2.x 兼容补丁（必须在导入 pandas_ta 之前执行） ---
if not hasattr(np, "NaN"):
    np.NaN = np.nan
if not hasattr(np, "Inf"):
    np.Inf = np.inf

import pandas_ta as ta
import pandas_ta.utils as ta_utils

# 禁用 TA-Lib，强制使用 pandas_ta 的纯 Python 实现，避免 NumPy / TA-Lib 二进制不兼容问题
ta_utils.Imports["talib"] = False
import logging
import os
import joblib
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import warnings
import requests
import time

# --- PyTorch Imports ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore", category=UserWarning)

# --- 0. 设置 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 🚀 全局配置 (V16 - Causal Labels) ---
SYMBOL = "ETHUSDT"
INTERVAL = "30m"
DATA_START_DATE = "2018-01-01"
DATA_END_DATE = "2025-11-12"

RUN_MODE = "train_global"  # "train_global" 或 "backtest_global"

# 全局模型训练/回测的时间切分
GLOBAL_TRAIN_END_DATE = "2024-01-01"  # 训练数据截止时间；之后的数据用于样本外回测

# 1h 周期下的窗口设置
LOOK_BACK = 48   # 约 2 天历史窗口 (48 * 1h)
HORIZON = 12     # 约 12 根 K 线的预测窗口
ALPHA_MIN_EDGE_ATR = 1.5  # 定义“可交易”样本所需的最小波动幅度（ATR 的倍数）

# --- 交易结果标签定义 ---
# 在未来 HORIZON 根 K 线内：
#  - 最高价相对当前收盘价至少上涨 TP_TARGET_PCT（例如 1%）
#  - 且期间最大回撤不超过 MAX_DRAWDOWN_PCT
TP_TARGET_PCT = 0.01          # 目标涨幅（1%）
MAX_DRAWDOWN_PCT = 0.01       # 允许的最大回撤（1%）

# --- PyTorch & Training Hyperparameters ---
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BATCH_SIZE = 256
# 全局模型训练：最多 50 个 epoch，至少跑 MIN_EPOCHS_FOR_EARLY_STOP，再由早停控制
FINETUNE_MAX_EPOCHS = 50
MIN_EPOCHS_FOR_EARLY_STOP = 10
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 1e-4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MODEL_CONFIG = {"d_model": 32, "nhead": 4, "num_layers": 2, "dropout": 0.4}

# --- 策略优化参数 (Strategy Parameters) ---
# 1h 周期下，适度激进的默认阈值
BACKTEST_QUALITY_THRESHOLD = 0.5   # 质量阈值
BACKTEST_EDGE_THRESHOLD = 0.5      # edge 概率阈值
COST_PER_TRADE = 0.0015            # 单笔交易成本假设
COOLDOWN_PERIOD = 0                # 冷静期

# 动态阈值控制：在震荡市（区间盘整）中自动提高入场门槛，减少小亏交易
USE_DYNAMIC_THRESHOLDS = True
RANGE_PRICE_POS_LOW = 0.3
RANGE_PRICE_POS_HIGH = 0.7
RANGE_ATR_ROLLING_WINDOW = 200
RANGE_QUALITY_MULTIPLIER = 1.5
RANGE_EDGE_MULTIPLIER = 1.1

# --- 文件路径 (V16) ---
MODELS_DIR = "models_transformer_v16_causal_1h"
DATA_DIR = "data"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
DATA_CACHE_PATH = os.path.join(
    DATA_DIR, f"{SYMBOL.lower()}_{INTERVAL}_data_{DATA_START_DATE}.csv"
)


# --- 数据下载功能 ---
def fetch_binance_klines(s, i, st, en=None, l=1000):
    url = "https://api.binance.com/api/v3/klines"
    cols = [
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
    sts = int(pd.to_datetime(st).timestamp() * 1000)
    ets = int(pd.to_datetime(en).timestamp() * 1000) if en else int(time.time() * 1000)
    all_d = []
    logger.info(f"从币安下载 {s} 数据...")
    while sts < ets:
        try:
            r = requests.get(
                url,
                params={
                    "symbol": s.upper(),
                    "interval": i,
                    "startTime": sts,
                    "limit": l,
                    "endTime": ets,
                },
                timeout=15,
            )
            r.raise_for_status()
            d = r.json()
            if not d:
                break
            all_d.extend(d)
            sts = d[-1][0] + 1
        except requests.exceptions.RequestException as e:
            logger.warning(f"数据下载失败，正在重试... 错误: {e}")
            time.sleep(5)
    df = pd.DataFrame(all_d, columns=cols)[
        ["timestamp", "Open", "High", "Low", "Close", "Volume"]
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"✅ 下载了 {len(df)} 行数据。")
    return df.set_index("timestamp").sort_index()


# --- ✅ [REBUILT] 特征与标签 V16 (Causal) ---
def get_features_v16(df):
    base = df.copy()
    # Ensure talib=False to use pure Python implementation
    base.ta.rsi(length=14, append=True, talib=False)
    base.ta.macd(fast=12, slow=26, signal=9, append=True, talib=False)
    base["ATR_norm"] = (
        ta.atr(base["High"], base["Low"], base["Close"], 14, talib=False)
        / base["Close"]
    )
    ma_200 = base["Close"].rolling(window=200).mean()
    base["trend_dist_200ma"] = (base["Close"] - ma_200) / ma_200

    # 价格动量特征：不同窗口的收益
    base["ret_1"] = base["Close"].pct_change(1)
    base["ret_3"] = base["Close"].pct_change(3)
    base["ret_6"] = base["Close"].pct_change(6)
    base["ret_12"] = base["Close"].pct_change(12)

    # 实现波动率特征：短期/中期波动
    close_ret = base["Close"].pct_change()
    base["vol_12"] = close_ret.rolling(12).std()
    base["vol_48"] = close_ret.rolling(48).std()

    # 价格在近期高低区间中的相对位置
    rolling_high = base["High"].rolling(96).max()
    rolling_low = base["Low"].rolling(96).min()
    base["price_pos_96"] = (base["Close"] - rolling_low) / (
        (rolling_high - rolling_low) + 1e-6
    )

    # 时间特征：日内和周内周期
    if isinstance(base.index, pd.DatetimeIndex):
        hours = base.index.hour
        dows = base.index.dayofweek
        base["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        base["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        base["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
        base["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)

    feature_cols = [
        c for c in base.columns if c not in ["Open", "High", "Low", "Close", "Volume"]
    ]
    return base[feature_cols].ffill()


def create_targets_v16(df, horizon):
    """
    基于未来 horizon 根 K 线构造“好交易”二分类标签：
    - 在未来 horizon 根 K 内，最高价相对当前收盘价上涨至少 TP_TARGET_PCT
    - 且在同一窗口内，最低价相对当前收盘价的回撤不超过 MAX_DRAWDOWN_PCT
    满足上述条件记为 1，否则为 0。尾部不足 horizon 的样本标签记为 NaN。
    """
    df = df.copy()
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    n = len(df)

    labels = np.full(n, np.nan, dtype=float)
    for i in range(n - horizon):
        entry = close[i]
        if not np.isfinite(entry):
            continue
        window_high = high[i + 1 : i + 1 + horizon]
        window_low = low[i + 1 : i + 1 + horizon]
        if len(window_high) == 0 or len(window_low) == 0:
            continue
        max_up = window_high.max() / entry - 1.0
        max_drawdown = window_low.min() / entry - 1.0
        # 条件：未来最高价至少上涨 TP_TARGET_PCT，且最大回撤不超过 -MAX_DRAWDOWN_PCT
        if (max_up >= TP_TARGET_PCT) and (max_drawdown >= -MAX_DRAWDOWN_PCT):
            labels[i] = 1.0
        else:
            labels[i] = 0.0

    edge_score = pd.Series(labels, index=df.index)
    # 为了兼容现有接口，quality_score 与 edge_score 相同，但在损失函数中只使用 edge_score
    quality_score = edge_score.copy()
    return quality_score, edge_score


# --- ✅ [REBUILT] PyTorch 系统 V16 ---
class CausalDataset(Dataset):
    def __init__(self, features, quality_labels, edge_labels, seq_len):
        self.features, self.quality, self.edge, self.seq_len = (
            features,
            quality_labels,
            edge_labels,
            seq_len,
        )

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        end = idx + self.seq_len
        # 使用 [idx, end) 这一段历史作为输入，
        # 标签对齐到序列末端时间点 end-1，避免未来信息泄露
        return (
            torch.tensor(self.features[idx:end], dtype=torch.float32),
            torch.tensor(self.quality[end - 1], dtype=torch.float32),
            torch.tensor(self.edge[end - 1], dtype=torch.float32),
        )


class CausalTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(CausalTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout, batch_first=True, activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.quality_head = nn.Linear(d_model, 1)
        self.edge_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, src):
        encoded_seq = self.transformer_encoder(self.input_projection(src))
        context_vector = encoded_seq[:, 0, :]
        pred_quality = self.quality_head(context_vector)
        pred_edge = self.edge_head(context_vector)
        return pred_quality, pred_edge


class CombinedLossV16(nn.Module):
    """
    仅使用“好交易”标签做二分类交叉熵：
    - pred_edge: 模型输出的好交易概率
    - target_edge: 0/1 标签（上涨至少 TP_TARGET_PCT 且回撤不超过 MAX_DRAWDOWN_PCT）
    """

    def __init__(self, bce_weight=1.0):
        super(CombinedLossV16, self).__init__()
        self.bce = nn.BCELoss()
        self.bce_w = bce_weight

    def forward(self, pred_quality, pred_edge, target_quality, target_edge):
        # 仅根据 target_edge 是否为 NaN 来选择有效样本
        valid_mask = ~target_edge.isnan()
        if valid_mask.sum() == 0:
            return torch.tensor(float("nan"), device=pred_edge.device)
        pe = pred_edge[valid_mask].clamp(1e-6, 1 - 1e-6)
        te = target_edge[valid_mask].clamp(0.0, 1.0)
        loss_bce = self.bce(pe, te)
        return self.bce_w * loss_bce


def backtest_period_v16(
    model,
    scaler,
    test_df,
    quality_threshold,
    edge_threshold,
    cost_per_trade,
    cooldown_period,
):
    model.eval().to(DEVICE)
    features = get_features_v16(test_df)
    if features.empty or len(features) < LOOK_BACK:
        return {"pnl": 0, "win_rate": 0, "num_trades": 0}

    # 计算用于动态阈值的“震荡市”标记：价格在区间中部且 ATR_norm 较低
    if USE_DYNAMIC_THRESHOLDS:
        atr_series = features["ATR_norm"]
        atr_roll_med = (
            atr_series.rolling(RANGE_ATR_ROLLING_WINDOW, min_periods=50).median()
        )
        # 使用 bfill/ffill 避免 fillna(method=...) 的 FutureWarning
        atr_roll_med = atr_roll_med.bfill().ffill()
        in_mid_range = features["price_pos_96"].between(
            RANGE_PRICE_POS_LOW, RANGE_PRICE_POS_HIGH
        )
        low_atr = atr_series < atr_roll_med
        is_range_regime = (in_mid_range & low_atr).astype(bool)
    else:
        is_range_regime = pd.Series(False, index=features.index)

    scaled_features = scaler.transform(features)
    actual_pnl = test_df["Close"].pct_change(HORIZON).shift(-HORIZON)
    signals = []
    cooldown_counter = 0
    with torch.no_grad():
        for i in range(len(scaled_features) - LOOK_BACK):
            if cooldown_counter > 0:
                signals.append(0)
                cooldown_counter -= 1
                continue

            idx = LOOK_BACK + i  # 对应当前决策的时间索引
            in_range = bool(is_range_regime.iloc[idx]) if USE_DYNAMIC_THRESHOLDS else False
            # 现在只根据“好交易概率” pred_edge 做决策；quality_threshold 保留参数但不再使用
            e_th = (
                edge_threshold * RANGE_EDGE_MULTIPLIER
                if in_range
                else edge_threshold
            )
            seq = (
                torch.tensor(scaled_features[i : i + LOOK_BACK], dtype=torch.float32)
                .unsqueeze(0)
                .to(DEVICE)
            )
            pred_quality, pred_edge = model(seq)
            if pred_edge.item() > e_th:
                signals.append(1)
                cooldown_counter = cooldown_period
            else:
                signals.append(0)

    results = pd.DataFrame({"signal": signals}, index=features.index[LOOK_BACK:])
    results = results.join(actual_pnl.rename("pnl"))
    trades = results[results["signal"] == 1].dropna(subset=["pnl"])
    num_trades = len(trades)
    gross_pnl = trades["pnl"].sum()
    total_costs = num_trades * cost_per_trade
    net_pnl = gross_pnl - total_costs
    return {
        "pnl": net_pnl,
        "win_rate": (trades["pnl"] > 0).mean() if num_trades > 0 else 0,
        "num_trades": num_trades,
    }


def train_global_model_v16(full_df):
    """
    训练一个“全局单一模型”，方便实盘和快速回测。
    使用 GLOBAL_TRAIN_END_DATE 之前的数据训练，并保存到 MODELS_DIR/global_model.*。
    """
    logger.info("--- 训练全局单一模型 (V16 Causal, 1h) ---")
    features_df = get_features_v16(full_df)
    quality_s, edge_s = create_targets_v16(full_df, HORIZON)

    if features_df.empty:
        logger.error("特征为空，无法训练全局模型。")
        return

    # 仅使用 GLOBAL_TRAIN_END_DATE 之前的数据训练+验证
    idx = features_df.index
    cutoff = pd.to_datetime(GLOBAL_TRAIN_END_DATE)
    total_mask = idx < cutoff
    features_total = features_df[total_mask]
    quality_total = quality_s[total_mask]
    edge_total = edge_s[total_mask]

    if len(features_total) <= LOOK_BACK * 2:
        logger.error("全局训练数据不足，请检查 GLOBAL_TRAIN_END_DATE 和窗口设置。")
        return

    # 按时间顺序切分 80% 作为训练，20% 作为验证（仅用于监控，不做早停）
    n_total = len(features_total)
    split_idx = int(n_total * 0.8)
    ft_features = features_total.iloc[:split_idx].values
    ft_quality = quality_total.iloc[:split_idx].values
    ft_edge = edge_total.iloc[:split_idx].values

    val_features = features_total.iloc[split_idx:].values
    val_quality = quality_total.iloc[split_idx:].values
    val_edge = edge_total.iloc[split_idx:].values

    if len(ft_features) <= LOOK_BACK or len(val_features) <= LOOK_BACK:
        logger.error("全局训练/验证数据不足，请检查切分比例与窗口设置。")
        return

    scaler = RobustScaler().fit(ft_features)
    ft_features_s = scaler.transform(ft_features)
    val_features_s = scaler.transform(val_features)

    train_dataset = CausalDataset(ft_features_s, ft_quality, ft_edge, LOOK_BACK)
    val_dataset = CausalDataset(val_features_s, val_quality, val_edge, LOOK_BACK)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    input_dim = ft_features_s.shape[1]
    model = CausalTransformer(input_dim, **MODEL_CONFIG).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = CombinedLossV16()
    # 初始化早停相关变量
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(FINETUNE_MAX_EPOCHS):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{FINETUNE_MAX_EPOCHS} [Train]", leave=False
        )
        for f, t_q, t_e in pbar:
            f, t_q, t_e = f.to(DEVICE), t_q.to(DEVICE), t_e.to(DEVICE)
            optimizer.zero_grad()
            p_q, p_e = model(f)
            loss = criterion(p_q.squeeze(-1), p_e.squeeze(-1), t_q, t_e)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = (
            total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        )

        # 简单监控一下验证集，不做早停和调度
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for f, t_q, t_e in val_loader:
                f, t_q, t_e = f.to(DEVICE), t_q.to(DEVICE), t_e.to(DEVICE)
                p_q, p_e = model(f)
                loss = criterion(p_q.squeeze(-1), p_e.squeeze(-1), t_q, t_e)
                if torch.isnan(loss):
                    continue
                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = (
            total_val_loss / num_val_batches if num_val_batches > 0 else float("nan")
        )
        logger.info(
            f"   Epoch {epoch+1}/{FINETUNE_MAX_EPOCHS}, "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # 简单早停：至少训练 MIN_EPOCHS_FOR_EARLY_STOP 个 epoch，
        # 之后若验证损失在 EARLY_STOP_PATIENCE 个 epoch 内未提升则停止
        if epoch + 1 >= MIN_EPOCHS_FOR_EARLY_STOP and not torch.isnan(
            torch.tensor(avg_val_loss)
        ):
            if avg_val_loss + EARLY_STOP_MIN_DELTA < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    logger.info(
                        f"   ⏹ Early stopping at epoch {epoch+1} "
                        f"(no val improvement for {EARLY_STOP_PATIENCE} epochs)"
                    )
                    break

    global_model_path = os.path.join(MODELS_DIR, "global_model.pt")
    global_scaler_path = os.path.join(MODELS_DIR, "global_scaler.joblib")
    try:
        torch.save(model.state_dict(), global_model_path)
        joblib.dump(scaler, global_scaler_path)
        logger.info(
            f"💾 Saved global model to {global_model_path} and scaler to {global_scaler_path}"
        )
    except Exception as e:
        logger.warning(f"⚠ 保存全局模型或 scaler 失败: {e}")


def backtest_global_model_v16(full_df):
    """
    使用已训练好的全局单一模型，对 GLOBAL_TRAIN_END_DATE 之后的样本外区间进行回测。
    """
    logger.info("--- 使用全局单一模型进行回测 (V16 Causal, 1h) ---")
    global_model_path = os.path.join(MODELS_DIR, "global_model.pt")
    global_scaler_path = os.path.join(MODELS_DIR, "global_scaler.joblib")

    if not (os.path.exists(global_model_path) and os.path.exists(global_scaler_path)):
        logger.error(
            f"未找到全局模型或 scaler: {global_model_path}, {global_scaler_path}。"
        )
        return

    # 为了确定特征维度，重新计算一次特征
    features_df = get_features_v16(full_df)
    if features_df.empty:
        logger.error("特征为空，无法回测全局模型。")
        return
    input_dim = features_df.shape[1]

    model = CausalTransformer(input_dim, **MODEL_CONFIG).to(DEVICE)
    try:
        state_dict = torch.load(global_model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        logger.error(f"加载全局模型失败: {e}")
        return

    try:
        scaler = joblib.load(global_scaler_path)
    except Exception as e:
        logger.error(f"加载全局 scaler 失败: {e}")
        return

    cutoff = pd.to_datetime(GLOBAL_TRAIN_END_DATE)
    test_df = full_df[cutoff:]
    if test_df.empty:
        logger.error("样本外测试数据为空，无法回测。")
        return

    results = backtest_period_v16(
        model,
        scaler,
        test_df,
        quality_threshold=BACKTEST_QUALITY_THRESHOLD,
        edge_threshold=BACKTEST_EDGE_THRESHOLD,
        cost_per_trade=COST_PER_TRADE,
        cooldown_period=COOLDOWN_PERIOD,
    )
    logger.info("\n--- 全局模型样本外回测结果 (已扣除交易成本) ---\n")
    print(results)

if __name__ == "__main__":
    if os.path.exists(DATA_CACHE_PATH):
        logger.info(f"从缓存加载数据: {DATA_CACHE_PATH}")
        raw_df = pd.read_csv(DATA_CACHE_PATH, index_col=0, parse_dates=True)
    else:
        logger.info("缓存未找到。下载新数据...")
        raw_df = fetch_binance_klines(
            s=SYMBOL, i=INTERVAL, st=DATA_START_DATE, en=DATA_END_DATE
        )
        if not raw_df.empty:
            logger.info(f"保存下载的数据到缓存: {DATA_CACHE_PATH}")
            raw_df.to_csv(DATA_CACHE_PATH)
        else:
            logger.error("数据下载失败。程序退出。")
            exit()
    logger.info(f"使用 {DEVICE} 设备。")

    if RUN_MODE == "train_global":
        train_global_model_v16(raw_df)
    elif RUN_MODE == "backtest_global":
        backtest_global_model_v16(raw_df)
    else:
        raise ValueError(f"未知 RUN_MODE: {RUN_MODE}")
