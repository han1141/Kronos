import requests
import pandas as pd
import datetime
import time


def get_binance_klines(symbol, interval, start_time, end_time):
    """
    分批获取币安K线数据。
    """
    base_url = "https://api.binance.com/api/v3/klines"
    limit = 1000
    all_klines = []

    current_start_time = start_time

    print("开始从币安获取K线数据...")

    while current_start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current_start_time),
            "endTime": int(end_time),
            "limit": limit,
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            all_klines.extend(klines)
            current_start_time = klines[-1][0] + 1

            # 显示进度信息
            utc_time = datetime.datetime.fromtimestamp(klines[-1][0] / 1000)
            print(f"已获取 {len(all_klines)} 条数据，最新时间: {utc_time.date()}")

            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"请求数据时发生错误: {e}")
            break

    return all_klines


def main():
    """主函数，执行数据获取、处理和保存"""
    # 1. 设置参数
    symbol = "BTCUSDT"
    interval = "1h"  # 获取日K数据

    # 2. 计算时间范围（最近3年）
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=22)

    start_timestamp_ms = int(start_dt.timestamp() * 1000)
    end_timestamp_ms = int(end_dt.timestamp() * 1000)

    print(f"计划获取从 {start_dt.date()} 到 {end_dt.date()} 的日K数据...")

    # 3. 获取K线数据
    klines_data = get_binance_klines(
        symbol, interval, start_timestamp_ms, end_timestamp_ms
    )

    if not klines_data:
        print("未能获取到任何数据，程序退出。")
        return

    # 4. 将数据转换为Pandas DataFrame
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines_data, columns=columns)

    df_final = df[
        ["open_time", "open", "high", "low", "close", "volume", "quote_asset_volume"]
    ]

    # 2. 将 open_time 从毫秒时间戳转换为时间格式
    df_final["open_time"] = pd.to_datetime(df_final["open_time"], unit="ms")

    # 3. 将所有数值列转换为数值类型
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
    for col in numeric_cols:
        df_final[col] = pd.to_numeric(df_final[col])

    # 4. 重命名列以匹配预测脚本的格式
    df_final = df_final.rename(
        columns={"open_time": "timestamps", "quote_asset_volume": "amount"}
    )

    # 6. 保存为CSV文件
    output_filename = "csv/btc_usdt_1d_no_time.csv"  # 更新文件名以反映内容
    df_final.to_csv(output_filename, index=False)

    print(f"\n数据处理完成！")
    print(f"总共获取了 {len(df_final)} 条日K线记录。")
    print(f"数据已成功保存到文件: {output_filename}")


if __name__ == "__main__":
    main()
