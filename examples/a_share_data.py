import akshare as ak
import pandas as pd
import datetime
import time


def get_stock_hist_data(symbol, period="daily", start_date=None, end_date=None):
    """
    使用akshare获取A股历史数据。
    
    Args:
        symbol: 股票代码，如 "000001"
        period: 数据周期，"daily"为日线数据，"weekly"为周线，"monthly"为月线
        start_date: 开始日期，格式为 "20240101"
        end_date: 结束日期，格式为 "20240131"
    """
    print(f"开始从akshare获取股票 {symbol} 的历史数据...")
    
    try:
        stock_data = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if stock_data is None or stock_data.empty:
            print(f"未能获取到股票 {symbol} 的数据")
            return None
            
        print(f"成功获取到 {len(stock_data)} 条历史数据")
        return stock_data
        
    except Exception as e:
        print(f"获取股票数据时发生错误: {e}")
        return None


def main():
    """主函数，执行数据获取、处理和保存"""
    symbol = "601138"
    period = "daily"  # 支持 "daily", "weekly", "monthly"

    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=800)

    start_date = start_dt.strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")

    period_map = {
        "daily": "日K线",
        "weekly": "周K线",
        "monthly": "月K线"
    }
    data_type = period_map.get(period, "日K线")
    print(f"计划获取股票 {symbol} 从 {start_dt.date()} 到 {end_dt.date()} 的{data_type}数据...")
    stock_data = get_stock_hist_data(symbol, period, start_date, end_date)

    if stock_data is None or stock_data.empty:
        print("未能获取到任何数据，程序退出。")
        return
    print(f"原始数据列名: {list(stock_data.columns)}")
    print(f"原始数据前5行:")
    print(stock_data.head())
    
    df_final = stock_data.copy()
    column_mapping = {
        '时间': 'timestamps',
        '日期': 'timestamps',
        'datetime': 'timestamps',
        'time': 'timestamps',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume',
        '成交额': 'amount'
    }
    available_cols = [col for col in column_mapping.keys() if col in df_final.columns]
    print(f"可用的列: {available_cols}")
    
    if available_cols:
        df_final = df_final[available_cols].rename(columns=column_mapping)
    else:
        print("未找到匹配的列名，尝试使用默认列名...")
        if len(df_final.columns) >= 6:
            df_final.columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume'] + list(df_final.columns[6:])
    if 'timestamps' in df_final.columns:
        df_final['timestamps'] = pd.to_datetime(df_final['timestamps'])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        if col in df_final.columns:
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    if 'timestamps' in df_final.columns:
        df_final = df_final.sort_values('timestamps').reset_index(drop=True)
    output_filename = "csv/history.csv"
    df_final.to_csv(output_filename, index=False)

    print(f"\n数据处理完成！")
    period_map = {
        "daily": "日K线",
        "weekly": "周K线",
        "monthly": "月K线"
    }
    data_type = period_map.get(period, "日K线")
    print(f"总共获取了 {len(df_final)} 条{data_type}记录。")
    print(f"数据已成功保存到文件: {output_filename}")
    print(f"数据列: {list(df_final.columns)}")


if __name__ == "__main__":
    main()
