import requests
import json

stock_code = "sh601208"
# 获取日线数据
scale = 240
# 获取的数据长度
datalen = 10  # 获取更多数据点以便测试

# 构建请求URL
url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={stock_code}&scale={scale}&datalen={datalen}"

# 初始化 data 变量为 None
data = None

print(f"正在从以下URL获取数据: {url}")

try:
    # 发送HTTP请求，并设置超时
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # 如果状态码不是200, 会抛出异常

    # 尝试解析返回的JSON数据
    # 新浪接口有时会返回非标准JSON，需要手动处理
    text_content = response.text
    if text_content and text_content.strip():
        data = json.loads(text_content)
    else:
        print("请求成功，但返回内容为空。")

except requests.exceptions.RequestException as e:
    print(f"HTTP请求失败: {e}")
except json.JSONDecodeError:
    # 关键的调试步骤：打印出无法解析的原始文本
    print("无法解析JSON数据。服务器返回的原始内容是:")
    print("-----------------------------------------")
    print(response.text)
    print("-----------------------------------------")


# --- 核心修正：在使用 data 之前，检查它是否有效 ---
if data:
    print("\n数据获取并解析成功，开始处理数据：")
    # 现在 data 肯定不是 None，可以安全地进行循环
    for item in data:
        # 确保 item 是字典并且包含所需键
        if isinstance(item, dict) and 'day' in item:
            print(f"日期: {item['day']}, 开盘价: {item['open']}, 收盘价: {item['close']}")
        else:
            print(f"发现无效的数据项: {item}")
else:
    print("\n未能获取到有效数据，程序结束。")