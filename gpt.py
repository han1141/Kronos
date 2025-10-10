# -*- coding: utf-8 -*-
"""
V38.1 宏观风险加权版框架
作者: ChatGPT (GPT-5)
描述: 在V38.0基础上新增宏观风险权重引擎，
     实现牛熊周期下的动态风险敞口控制。
"""

import pandas as pd
import numpy as np
import talib
import plotly.express as px
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
import logging
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 📘 日志系统
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("v38_1_log.txt", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==============================
# 🧭 宏观状态计算
# ==============================
def compute_macro_regime(df_daily):
    sma200 = df_daily['Close'].rolling(200).mean()
    macro_regime = np.where(df_daily['Close'] > sma200, 1, -1)
    df_daily['macro_regime'] = macro_regime
    return df_daily[['macro_regime']]

# ==============================
# ⚙️ 动态参数生成器
# ==============================
def generate_dynamic_params(df):
    vol = df['Close'].pct_change().std() * np.sqrt(365)
    return {
        'donchian_period': int(20 * (1 + vol)),
        'chandelier_mult': round(2.5 * (1 + vol), 2),
        'bb_std': round(2 * (1 + vol / 2), 2),
        'max_risk_pct': min(0.03 * (1 + vol), 0.05)
    }

# ==============================
# 💹 宏观风险权重引擎
# ==============================
def calculate_macro_risk_weight(macro_regime, asset_volatility, equity_curve):
    base_weight = 1.2 if macro_regime == 1 else 0.7
    vol_factor = 1 / (1 + np.log1p(asset_volatility * 10))
    perf_factor = 1.0
    if len(equity_curve) > 20:
        slope = np.polyfit(range(20), equity_curve[-20:], 1)[0]
        perf_factor = 1.1 if slope > 0 else 0.9
    weight = base_weight * vol_factor * perf_factor
    return np.clip(weight, 0.5, 1.5)

class MacroRiskManager:
    def __init__(self, max_total_risk=0.05):
        self.max_total_risk = max_total_risk

    def allocate(self, strategies, macro_regime, vol_dict, equity_curve):
        weights = {}
        macro_weight = calculate_macro_risk_weight(macro_regime, np.mean(list(vol_dict.values())), equity_curve)
        for name, vol in vol_dict.items():
            risk_unit = (1 / (1 + vol)) / len(strategies)
            weights[name] = risk_unit * macro_weight
        total = sum(weights.values())
        return {k: v / total * self.max_total_risk for k, v in weights.items()}

# ==============================
# 📈 策略定义
# ==============================
class FilteredTF_MR(Strategy):
    def init(self):
        self.macro_regime = self.data.macro_regime
        self.equity_curve = []

    def next(self):
        price = self.data.Close[-1]
        atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)[-1]
        macro_regime = int(self.macro_regime[-1])
        self.equity_curve.append(self.equity)
        risk = self._calculate_dynamic_risk(atr, macro_regime)

        # 策略方向过滤
        sma_fast = self.I(talib.SMA, self.data.Close, 20)
        sma_slow = self.I(talib.SMA, self.data.Close, 50)

        if crossover(sma_fast, sma_slow) and macro_regime == 1:
            self.buy(size=self._calc_size(risk))
        elif crossover(sma_slow, sma_fast) and macro_regime == -1:
            self.sell(size=self._calc_size(risk))

    def _calc_size(self, risk):
        return max(int(self.equity * risk / self.data.Close[-1]), 1)

    def _calculate_dynamic_risk(self, current_volatility, macro_regime):
        base_risk = min(0.02, 1.0 / np.sqrt(1 + current_volatility))
        weight = calculate_macro_risk_weight(macro_regime, current_volatility, self.equity_curve)
        return base_risk * weight

# ==============================
# 📉 宏观风险热力图
# ==============================
def plot_macro_risk_heatmap(df):
    fig = px.imshow(
        df[['macro_regime', 'macro_risk_weight']],
        color_continuous_scale='RdYlGn',
        title='📊 Macro Regime & Risk Heatmap'
    )
    fig.update_xaxes(title='Time')
    fig.update_yaxes(title='Indicator')
    fig.show()

# ==============================
# 🚀 主回测流程
# ==============================
def run_backtest(symbol, df_4h, df_1d):
    logger.info(f"开始运行 {symbol} 回测...")

    # 宏观状态同步
    df_macro = compute_macro_regime(df_1d)
    df_4h = df_4h.join(df_macro, how='left').ffill()

    # 计算动态参数
    params = generate_dynamic_params(df_4h)
    vol = df_4h['Close'].pct_change().std() * np.sqrt(365)

    # 初始化风险管理器
    risk_manager = MacroRiskManager(max_total_risk=0.05)
    vol_dict = {symbol: vol}
    equity_curve = np.linspace(10000, 12000, len(df_4h))  # 模拟权益曲线
    risk_allocation = risk_manager.allocate([symbol], df_4h['macro_regime'].iloc[-1], vol_dict, equity_curve)
    df_4h['macro_risk_weight'] = calculate_macro_risk_weight(df_4h['macro_regime'].iloc[-1], vol, equity_curve)

    # 回测
    bt = Backtest(df_4h, FilteredTF_MR, cash=10000, commission=0.001)
    stats = bt.run()
    logger.info(f"{symbol} 回测完成，年化收益率: {stats['Return [%]']:.2f}%")

    plot_macro_risk_heatmap(df_4h.tail(200))
    return stats, df_4h, risk_allocation

# ==============================
# 🧪 示例运行（请替换为真实数据）
# ==============================
if __name__ == "__main__":
    logger.info(">>> 启动 V38.1 宏观风险加权引擎")

    # 模拟数据 (你可替换为 Binance 抓取数据)
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=800, freq="4H")
    df_4h = pd.DataFrame({
        "Open": np.random.rand(len(dates)) * 100 + 20000,
        "High": np.random.rand(len(dates)) * 100 + 20100,
        "Low": np.random.rand(len(dates)) * 100 + 19900,
        "Close": np.random.rand(len(dates)) * 100 + 20000,
        "Volume": np.random.rand(len(dates)) * 10
    }, index=dates)
    df_1d = df_4h.resample('1D').agg({'Open':'first','High':'max','Low':'min','Close':'last'})

    stats, df_risk, alloc = run_backtest("BTC/USDT", df_4h, df_1d)
    print(stats)
    print("风险分配:", alloc)
