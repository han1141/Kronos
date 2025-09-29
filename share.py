import pandas as pd
import akshare as ak
import talib
from tqdm import tqdm
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print(
        "警告: yfinance未安装，将跳过雅虎财经数据源。可通过 'pip install yfinance' 安装。"
    )


class StockScreener:
    def __init__(
        self, mode="strict", debug=False, debug_sample_size=50, enable_macro_filter=True
    ):
        self.screening_modes = {
            "strict": {
                "MIN_ROE": 10,
                "MIN_NET_PROFIT_GROWTH": 10,
                "MIN_REVENUE_GROWTH": 5,
                "MAX_PE": 80,
                "MAX_PB": 8,
                "MAX_DEBT_RATIO": 60,
                "REQUIRE_POSITIVE_CASH_FLOW": True,
                "MIN_MARKET_CAP": 50,
                "description": "严格模式 - 高质量成长股(已优化)",
            },
            "balanced": {
                "MIN_ROE": 8,
                "MIN_NET_PROFIT_GROWTH": 5,
                "MIN_REVENUE_GROWTH": 0,
                "MAX_PE": 100,
                "MAX_PB": 10,
                "MAX_DEBT_RATIO": 70,
                "REQUIRE_POSITIVE_CASH_FLOW": True,
                "MIN_MARKET_CAP": 50,
                "description": "平衡模式 - 质量与机会并重",
            },
            "relaxed": {
                "MIN_ROE": 5,
                "MIN_NET_PROFIT_GROWTH": -50,
                "MIN_REVENUE_GROWTH": -20,
                "MAX_PE": 200,
                "MAX_PB": 15,
                "MAX_DEBT_RATIO": 80,
                "REQUIRE_POSITIVE_CASH_FLOW": False,
                "MIN_MARKET_CAP": 30,
                "description": "宽松模式 - 发现潜力股",
            },
        }
        self.weights = {"fundamental": 0.6, "technical": 0.4}
        self.current_mode = mode
        self.config = self.screening_modes[self.current_mode]
        self.debug_mode = debug
        self.debug_sample_size = debug_sample_size
        self.enable_macro_filter = enable_macro_filter
        self.financial_cache = self._load_financial_cache()

    def _get_financial_cache_filename(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        return f"financial_data_v2_{today_str}.json"

    def _load_financial_cache(self):
        cache_file = self._get_financial_cache_filename()
        if not os.path.exists(cache_file):
            return {}
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"读取财务数据缓存失败: {e}")
            return {}

    def _save_to_financial_cache(self, code, data):
        self.financial_cache[code] = data
        cache_file = self._get_financial_cache_filename()
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.financial_cache, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"保存财务数据缓存失败: {e}")

    def check_market_condition(self):
        if not self.enable_macro_filter:
            print("宏观择时过滤器已禁用。")
            return True

        print("正在进行宏观市场环境分析 (基于沪深300指数)...")
        try:
            index_code = "sz399300"
            hist = ak.index_zh_a_hist(
                symbol=index_code, period="daily", start_date="20230101"
            )

            if len(hist) < 250:
                print("警告: 指数历史数据不足250天，无法计算年线。跳过宏观择时。")
                return True

            current_price = hist["收盘"].iloc[-1]
            ma250 = talib.SMA(hist["收盘"], timeperiod=250).iloc[-1]

            print(f"当前沪深300指数: {current_price:.2f}")
            print(f"年线 (MA250): {ma250:.2f}")

            if current_price > ma250:
                print("✅ 市场环境判断: 指数位于年线之上，市场处于多头趋势。")
                return True
            else:
                print("❌ 市场环境判断: 指数位于年线之下，市场处于空头趋势。")
                print("警告: 当前市场存在系统性风险，建议谨慎操作。程序将退出。")
                return False
        except Exception as e:
            print(f"获取市场指数失败: {e}。将跳过宏观环境检查。")
            return True

    def get_a_share_list(self):
        print("正在通过API获取所有A股实时数据并进行预筛选...")
        try:
            stock_df = ak.stock_zh_a_spot_em()

            initial_count = len(stock_df)

            stock_df = stock_df[
                ~stock_df["代码"].str.startswith(("8", "688", "4", "9"))
            ]
            stock_df = stock_df[~stock_df["名称"].str.contains("ST")]

            min_market_cap_value = self.config["MIN_MARKET_CAP"] * 100000000
            stock_df = stock_df[stock_df["总市值"] > min_market_cap_value]

            final_df = stock_df[["代码", "名称", "总市值", "市净率", "所属行业"]].copy()
            final_df["总市值"] = final_df["总市值"] / 100000000

            print(f"市值与板块预筛选完成:")
            print(f"  - 原始股票数: {initial_count}")
            print(f"  - 最终保留数: {len(final_df)}")
            print(
                f"  - 过滤比例: {(initial_count - len(final_df)) / initial_count * 100:.1f}%"
            )

            return final_df

        except Exception as e:
            print(f"获取并筛选股票列表失败: {e}")
            return pd.DataFrame()

    def fetch_financial_data_from_api(self, code):
        try:
            fin_indicators = ak.stock_financial_analysis_indicator(
                symbol=code, period="年度"
            )
            if not fin_indicators.empty:
                latest_fin = fin_indicators.iloc[0]
                roe = pd.to_numeric(latest_fin.get("净资产收益率加权"), errors="coerce")
                net_profit_growth = pd.to_numeric(
                    latest_fin.get("扣除非经常性损益后的净利润同比增长率"),
                    errors="coerce",
                )
                revenue_growth = pd.to_numeric(
                    latest_fin.get("营业总收入同比增长率"), errors="coerce"
                )
                debt_ratio = pd.to_numeric(
                    latest_fin.get("资产负债率"), errors="coerce"
                )
                cash_flow_per_share = pd.to_numeric(
                    latest_fin.get("每股经营性现金流(元)"), errors="coerce"
                )

                return {
                    "roe": roe if pd.notna(roe) else 0,
                    "net_profit_growth": (
                        net_profit_growth if pd.notna(net_profit_growth) else 0
                    ),
                    "revenue_growth": revenue_growth if pd.notna(revenue_growth) else 0,
                    "debt_ratio": debt_ratio if pd.notna(debt_ratio) else 100,
                    "cash_flow": (
                        1
                        if (pd.notna(cash_flow_per_share) and cash_flow_per_share > 0)
                        else -1
                    ),
                }
        except Exception:
            return None
        return None

    def get_fundamental_score(self, stock_info):
        code = stock_info["代码"]
        if code in self.financial_cache:
            financial_data = self.financial_cache[code]
        else:
            financial_data = self.fetch_financial_data_from_api(code)
            if financial_data:
                self._save_to_financial_cache(code, financial_data)
            else:
                return 0, {}

        roe = financial_data.get("roe", 0)
        net_profit_growth = financial_data.get("net_profit_growth", 0)
        revenue_growth = financial_data.get("revenue_growth", 0)
        debt_ratio = financial_data.get("debt_ratio", 100)
        cash_flow = financial_data.get("cash_flow", -1)

        market_cap = stock_info["总市值"]
        pb_ratio = pd.to_numeric(stock_info["市净率"], errors="coerce")

        conditions = {
            "ROE": roe > self.config["MIN_ROE"],
            "净利润增长": net_profit_growth > self.config["MIN_NET_PROFIT_GROWTH"],
            "营收增长": revenue_growth > self.config["MIN_REVENUE_GROWTH"],
            "负债率": debt_ratio < self.config["MAX_DEBT_RATIO"],
            "市净率": 0 < pb_ratio < self.config["MAX_PB"],
            "现金流": not self.config["REQUIRE_POSITIVE_CASH_FLOW"] or cash_flow > 0,
        }

        conditions_met = sum(conditions.values())
        min_conditions_required = 5 if self.current_mode == "strict" else 4

        metrics = {
            "ROE(%)": round(roe, 2),
            "净利润增长(%)": round(net_profit_growth, 2),
            "营收增长(%)": round(revenue_growth, 2),
            "市净率(PB)": round(pb_ratio, 2),
            "负债率(%)": round(debt_ratio, 2),
            "总市值(亿)": round(market_cap, 2),
            "所属行业": stock_info["所属行业"],
        }

        if conditions_met < min_conditions_required:
            return 0, metrics

        score = 0
        if roe > 25:
            score += 5
        elif roe > 15:
            score += 3
        elif roe > 8:
            score += 1

        if net_profit_growth > 50:
            score += 5
        elif net_profit_growth > 20:
            score += 3
        elif net_profit_growth > 5:
            score += 1

        if revenue_growth > 50:
            score += 5
        elif revenue_growth > 20:
            score += 3
        elif revenue_growth > 5:
            score += 1

        return score, metrics

    def get_technical_score(self, code):
        try:
            start_date = (datetime.now() - pd.DateOffset(months=18)).strftime("%Y%m%d")
            hist_data = ak.stock_zh_a_hist(
                symbol=code, period="daily", start_date=start_date, adjust="qfq"
            )
            if len(hist_data) < 250:
                return 0

            close = hist_data["收盘"]
            volume = hist_data["成交量"]

            score = 0

            ma50 = talib.SMA(close, timeperiod=50).iloc[-1]
            ma200 = talib.SMA(close, timeperiod=200).iloc[-1]
            if close.iloc[-1] > ma50:
                score += 2
            if close.iloc[-1] > ma200:
                score += 2
            if ma50 > ma200:
                score += 3

            vol_ma20 = talib.SMA(volume, timeperiod=20).iloc[-1]
            if volume.iloc[-1] > vol_ma20 * 1.2 and close.iloc[-1] > close.iloc[-2]:
                score += 2

            rsi14 = talib.RSI(close, timeperiod=14).iloc[-1]
            if rsi14 > 80:
                score -= 1

            return max(0, score)  # Score cannot be negative
        except Exception:
            return 0

    def process_stock(self, stock_info):
        f_score, metrics = self.get_fundamental_score(stock_info)
        if f_score == 0:
            return None

        code = stock_info["代码"]
        name = stock_info["名称"]
        t_score = self.get_technical_score(code)

        total_score = (f_score / 15 * self.weights["fundamental"]) + (
            t_score / 10 * self.weights["technical"]
        )
        total_score = round(total_score * 100, 2)

        result = {
            "代码": code,
            "名称": name,
            "总分": total_score,
            "基本面分": f_score,
            "技术面分": t_score,
        }
        result.update(metrics)
        return result

    def run(self):
        self.display_mode_info()

        if not self.check_market_condition():
            return

        stocks_df = self.get_a_share_list()

        if stocks_df.empty:
            print("未能获取股票列表，程序退出。")
            return

        stock_list = [row for index, row in stocks_df.iterrows()]
        if self.debug_mode:
            stock_list = stock_list[: self.debug_sample_size]
            print(f"🔍 调试模式已启用，将扫描前 {len(stock_list)} 只股票")

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stock = {
                executor.submit(self.process_stock, stock_info): stock_info
                for stock_info in stock_list
            }

            for future in tqdm(
                as_completed(future_to_stock), total=len(stock_list), desc="扫描股票中"
            ):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    stock_info = future_to_stock[future]
                    print(
                        f"处理股票 {stock_info['代码']} {stock_info['名称']} 时发生错误: {e}"
                    )

        if not results:
            self.display_no_results()
            return

        results_df = (
            pd.DataFrame(results)
            .sort_values(by="总分", ascending=False)
            .reset_index(drop=True)
        )

        self.display_and_save_results(results_df)

    def display_mode_info(self):
        print(f"\n========== 当前筛选模式: {self.current_mode.upper()} ==========")
        print(f"模式描述: {self.config['description']}")
        print(f"宏观择时过滤器: {'启用' if self.enable_macro_filter else '禁用'}")
        print(f"筛选条件:")
        print(
            f"  - 最低ROE: {self.config['MIN_ROE']}% | 最低市值: {self.config['MIN_MARKET_CAP']}亿"
        )
        print(
            f"  - 最低净利润增长: {self.config['MIN_NET_PROFIT_GROWTH']}% | 最高市净率: {self.config['MAX_PB']}"
        )
        print(
            f"  - 最低营收增长: {self.config['MIN_REVENUE_GROWTH']}% | 最高负债率: {self.config['MAX_DEBT_RATIO']}%"
        )
        print(
            f"  - 现金流要求: {'经营性现金流 > 0' if self.config['REQUIRE_POSITIVE_CASH_FLOW'] else '无'}"
        )
        print("=" * 60)

    def display_and_save_results(self, results_df):
        print(
            f"\n========================= 选股评分结果 (Top 20) ========================="
        )
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 2000)

        column_order = [
            "代码",
            "名称",
            "总分",
            "基本面分",
            "技术面分",
            "ROE(%)",
            "净利润增长(%)",
            "营收增长(%)",
            "市净率(PB)",
            "负债率(%)",
            "总市值(亿)",
            "所属行业",
        ]
        print(results_df.head(20)[column_order].to_string(index=False))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"股票筛选结果_v2_{self.current_mode}_{timestamp}.csv"
        try:
            results_df[column_order].to_csv(
                csv_filename, index=False, encoding="utf-8-sig"
            )
            print(f"\n✅ 完整结果已保存到: {csv_filename}")
        except Exception as e:
            print(f"❌ 保存CSV文件失败: {e}")

        print(f"\n========== 结果统计 ==========")
        print(f"找到符合条件的股票: {len(results_df)} 只")
        if not results_df.empty:
            print(f"平均总分: {results_df['总分'].mean():.2f}")
            print(f"最高总分: {results_df['总分'].max():.2f}")
            print(f"平均ROE: {results_df['ROE(%)'].mean():.2f}%")
            print(f"平均市值: {results_df['总市值(亿)'].mean():.2f}亿")

    def display_no_results(self):
        print("\n在当前筛选模式下没有找到符合条件的股票。")
        print("\n建议:")
        print("1. 尝试切换到更宽松的筛选模式 (如 'balanced' 或 'relaxed')")
        print("2. 检查网络连接和数据源是否正常")
        print("3. 如果宏观择时过滤器开启，可能是当前市场整体趋势不佳")


if __name__ == "__main__":
    screener = StockScreener(mode="strict", debug=False, enable_macro_filter=True)
    screener.run()
