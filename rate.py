import requests
import urllib3
import time
import sys
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------- 配置区 ----------------
# 自动扫描这些端口
POSSIBLE_PORTS = [7890, 10809, 1087, 7897, 1080]
HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------- 1. 自动寻找梯子 ----------------
def find_working_proxy():
    print("🔍 正在校准瞄准镜 (扫描代理)...")
    test_url = "https://www.google.com"

    for port in POSSIBLE_PORTS:
        proxy_conf = {
            "http": f"http://127.0.0.1:{port}",
            "https": f"http://127.0.0.1:{port}",
        }
        try:
            requests.get(test_url, proxies=proxy_conf, timeout=2, verify=False)
            print(f"   ✅ 端口 {port} 锁定目标！")
            return proxy_conf
        except:
            pass
    return None


# ---------------- 2. 获取单个币种全维数据 ----------------
def analyze_coin(symbol, proxies):
    # 自动补全 USDT
    if not symbol.endswith("USDT"):
        symbol = symbol.upper() + "USDT"

    print(f"\n⚡ 正在分析目标: 【 {symbol} 】...")

    try:
        # A. 获取 24小时 价格/成交量数据
        url_ticker = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        res_ticker = requests.get(
            url_ticker,
            params={"symbol": symbol},
            headers=HEADERS,
            proxies=proxies,
            verify=False,
        ).json()

        # B. 获取 资金费率
        url_rate = "https://fapi.binance.com/fapi/v1/premiumIndex"
        res_rate = requests.get(
            url_rate,
            params={"symbol": symbol},
            headers=HEADERS,
            proxies=proxies,
            verify=False,
        ).json()

        # C. 获取 合约持仓量 (Open Interest) - 核心主力指标
        url_oi = "https://fapi.binance.com/fapi/v1/openInterest"
        res_oi = requests.get(
            url_oi,
            params={"symbol": symbol},
            headers=HEADERS,
            proxies=proxies,
            verify=False,
        ).json()

        # ---------------- 数据清洗 ----------------
        price = float(res_ticker["lastPrice"])
        price_change = float(res_ticker["priceChangePercent"])
        volume_usdt = float(res_ticker["quoteVolume"])  # 24h成交额

        funding_rate = float(res_rate["lastFundingRate"]) * 100

        oi_usdt = float(res_oi["openInterest"]) * price  # 持仓价值 (美金)

        # ---------------- 生成战报 ----------------
        print("-" * 30)
        print(
            f"💰 当前价格: ${price} ({'📈' if price_change>0 else '📉'} {price_change:.2f}%)"
        )
        print(f"📊 24h成交: ${volume_usdt/1e6:.1f} M (百万美金)")
        print(f"🏦 合约持仓: ${oi_usdt/1e6:.1f} M (百万美金) <-- 主力底牌")
        print(f"🌡️ 资金费率: {funding_rate:.4f}%")
        print("-" * 30)

        # ---------------- 智能战术分析 (AI 参谋) ----------------
        print("【战术分析】")

        # 1. 趋势判断
        trend = "震荡"
        if price_change > 5:
            trend = "强势上涨"
        elif price_change < -5:
            trend = "弱势下跌"

        # 2. 持仓量分析 (OI)
        # 简单的逻辑：持仓量巨大(相对于成交量)说明多空博弈极度激烈
        oi_vol_ratio = oi_usdt / volume_usdt

        if trend == "强势上涨":
            if funding_rate > 0.05:
                print("🚨 风险提示：价格大涨 + 费率过热。谨防庄家骗炮画门！")
            else:
                print("🚀 机会提示：上涨且费率健康。如果是“真突破”，可以顺势跟进。")

        elif trend == "弱势下跌":
            if funding_rate < -0.05:
                print(
                    "💎 机会提示：大跌且费率极负。空头太挤了，可能出现轧空反弹(Short Squeeze)。"
                )
            else:
                print("⚠️ 风险提示：阴跌不止，且费率正常。说明主力在撤退，不要接飞刀。")

        else:  # 震荡
            if oi_vol_ratio > 2:
                print(
                    "💣 变盘预警：持仓量异常高！多空都在积攒弹药，即将出大方向（暴涨或暴跌）。"
                )
            else:
                print("💤 垃圾时间：资金关注度低，建议观望。")

    except Exception as e:
        print(f"❌ 获取失败，请检查币种名称是否正确 (如 BTC, ETH): {e}")


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python rate.py BTC  （ETH、SOL 等）")
        sys.exit(1)

    target_symbol = sys.argv[1].strip().upper()

    proxies = find_working_proxy()

    if not proxies:
        print("❌ 网络不通，请检查梯子。")
        sys.exit(1)

    print(
        f"\n⏱️ 已启动，对 {target_symbol} 自动检查 (Ctrl+C 退出)"
    )

    try:
        while True:
            print(
                f"\n--- 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
            )
            analyze_coin(target_symbol, proxies)
            print("\n⏳ 等待 1 分钟后再次检查...\n")
            time.sleep(1 * 60)
    except KeyboardInterrupt:
        print("\n👋 已手动停止定时检查。")
