# backtest.py
import logging
import pandas as pd
import joblib
import ta
from backtesting import Backtest, Strategy
import os
import sys

# --- 复用 V60.2 的辅助函数 ---
# (为了简洁，这里只需要 fetch_binance_klines 和 add_all_ml_features)

# 日志配置
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- 配置 ---
BACKTEST_CONFIG = {
    "symbol": "ETHUSDT",
    "interval": "5m",
    "test_start_date": "2024-01-01",
    "test_end_date": "2025-10-01",  # 使用您之前的结束日期
    "model_dir": "models_v61_trained",
    "data_cache": "data_cache_v61",
    "initial_cash": 500_000,
    "commission": 0.001,
}

MODEL_PARAMS = {  # 只需要策略相关的参数
    "risk_per_trade": 0.015,
    "label_sl_atr_multiplier": 1.5,
    "label_risk_reward_ratio": 2.0,
    "label_atr_period": 14,
}

# (此处应粘贴 V60.2 的 fetch_binance_klines 和 add_all_ml_features 函数)
# ...


class MLStrategy(Strategy):
    def init(self):
        # 1. 加载模型、Scaler 和特征列表
        self.model = joblib.load(
            os.path.join(
                BACKTEST_CONFIG["model_dir"],
                f"final_model_{BACKTEST_CONFIG['symbol']}.joblib",
            )
        )
        self.scaler = joblib.load(
            os.path.join(
                BACKTEST_CONFIG["model_dir"],
                f"final_scaler_{BACKTEST_CONFIG['symbol']}.joblib",
            )
        )
        with open(
            os.path.join(BACKTEST_CONFIG["model_dir"], "feature_list.txt"), "r"
        ) as f:
            self.feature_list = [line.strip() for line in f.readlines()]

        # 2. 初始化策略参数
        self.risk_per_trade = MODEL_PARAMS["risk_per_trade"]
        self.sl_atr_multiplier = MODEL_PARAMS["label_sl_atr_multiplier"]
        self.rr_ratio = MODEL_PARAMS["label_risk_reward_ratio"]
        self.atr_period = MODEL_PARAMS["label_atr_period"]

        # 3. 初始化 ATR 指标
        self.atr = self.I(
            lambda: ta.volatility.average_true_range(
                high=pd.Series(self.data.High),
                low=pd.Series(self.data.Low),
                close=pd.Series(self.data.Close),
                window=self.atr_period,
            )
        )

    def next(self):
        if self.position:
            return

        # 准备当与训练时完全一致的特征
        current_features = self.data.df.iloc[-1:][self.feature_list]
        current_features_scaled = self.scaler.transform(current_features)

        prediction = self.model.predict(current_features_scaled)[0]

        sl_amount = self.atr[-1] * self.sl_atr_multiplier
        if not np.isfinite(sl_amount) or sl_amount <= 0:
            return

        risk_per_unit = self.equity * self.risk_per_trade
        size = risk_per_unit / sl_amount

        if prediction == 1:  # Go Long
            self.buy(
                size=size,
                sl=self.data.Close[-1] - sl_amount,
                tp=self.data.Close[-1] + sl_amount * self.rr_ratio,
            )
        elif prediction == 2:  # Go Short
            self.sell(
                size=size,
                sl=self.data.Close[-1] + sl_amount,
                tp=self.data.Close[-1] - sl_amount * self.rr_ratio,
            )


def main_backtesting():
    # 1. 加载测试数据
    logger.info(
        f"正在加载测试数据: {BACKTEST_CONFIG['test_start_date']} to {BACKTEST_CONFIG['test_end_date']}"
    )
    raw_data = fetch_binance_klines(
        BACKTEST_CONFIG["symbol"],
        BACKTEST_CONFIG["interval"],
        BACKTEST_CONFIG["test_start_date"],
        BACKTEST_CONFIG["test_end_date"],
        cache_dir=BACKTEST_CONFIG["data_cache"],
    )
    if raw_data.empty:
        return

    # 2. 生成与训练时一致的特征
    logger.info("正在为测试数据生成特征...")
    featured_data = add_all_ml_features(raw_data)

    # 3. 准备回测所需的数据格式
    bt_data = raw_data.loc[featured_data.index].copy()
    bt_data = bt_data.join(
        featured_data.drop(
            columns=bt_data.columns.intersection(featured_data.columns), errors="ignore"
        )
    )
    bt_data["Volume"] = bt_data["Volume"].astype(float)
    bt_data = bt_data[
        ["Open", "High", "Low", "Close", "Volume"]
        + [
            col
            for col in bt_data.columns
            if col not in ["Open", "High", "Low", "Close", "Volume"]
        ]
    ]

    # 4. 运行回测
    logger.info("开始回测...")
    bt = Backtest(
        bt_data,
        MLStrategy,
        cash=BACKTEST_CONFIG["initial_cash"],
        commission=BACKTEST_CONFIG["commission"],
    )
    stats = bt.run()

    logger.info("回测完成。")
    print(stats)
    stats.to_csv("backtest_results_v61.csv")
    bt.plot(
        filename=f"backtest_{BACKTEST_CONFIG['symbol']}_v61.html", open_browser=False
    )


if __name__ == "__main__":
    # 您需要将 V60.2 的辅助函数复制到本文件中，然后取消注释 main_backtesting() 并运行
    # main_backtesting()
    print(
        "请先将 V60.2 的辅助函数复制到本文件中，然后取消注释 main_backtesting() 并运行。"
    )
