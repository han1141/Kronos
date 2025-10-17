import websocket
import json
import time
import threading

# --- 配置 ---
# OKX 公共 WebSocket 地址
WSS_URL = "wss://ws.okx.com:8443/ws/v5/public"

# 要订阅的交易对和频道
# 示例：ETH-USDT 现货的 ticker (行情数据)
SUBSCRIBE_OP = {
    "op": "subscribe",
    "args": [{"channel": "tickers", "instId": "ETH-USDT"}],
}

# --- 如果你需要代理 (例如在中国大陆) ---
# 如果不需要代理，保持为 None
# http_proxy_host = "127.0.0.1"
# http_proxy_port = 7890
http_proxy_host = None
http_proxy_port = None


def send_heartbeat(ws):
    """
    发送心跳包 (Ping) 以保持连接活跃。
    OKX 要求每 30 秒内至少发送一次。
    """
    while ws.keep_running:
        try:
            ws.send("ping")
            # print("Ping sent") # 调试用
            time.sleep(20)  # 每20秒发送一次
        except Exception as e:
            print(f"心跳发送失败: {e}")
            break


def on_open(ws):
    """连接建立后触发"""
    print("--- 连接已建立 ---")

    # 1. 启动心跳线程
    ws.keep_running = True
    threading.Thread(target=send_heartbeat, args=(ws,), daemon=True).start()

    # 2. 发送订阅请求
    json_str = json.dumps(SUBSCRIBE_OP)
    ws.send(json_str)
    print(f"已发送订阅请求: {json_str}")


def on_message(ws, message):
    """接收到服务器消息时触发"""

    # 处理心跳响应
    if message == "pong":
        # print("Received: pong") # 调试用
        return

    # 处理业务数据
    try:
        data = json.loads(message)

        # 处理订阅成功的确认消息
        if "event" in data and data["event"] == "subscribe":
            print(f"订阅成功: {data['arg']['instId']} - {data['arg']['channel']}")
            return

        # 处理实际的行情数据
        if "data" in data:
            for ticker in data["data"]:
                inst_id = ticker.get("instId")
                last_price = ticker.get("last")  # 最新成交价
                ask_price = ticker.get("askPx")  # 卖一价
                bid_price = ticker.get("bidPx")  # 买一价

                print("-" * 30)
                print(f"产品: {inst_id}")
                print(f"最新价: {last_price}")
                print(f"买一: {bid_price} | 卖一: {ask_price}")
        else:
            print(f"收到其他消息: {message}")

    except json.JSONDecodeError:
        print(f"JSON 解析错误: {message}")


def on_error(ws, error):
    """发生错误时触发"""
    print(f"--- 错误发生 --- : {error}")
    ws.keep_running = False


def on_close(ws, close_status_code, close_msg):
    """连接关闭时触发"""
    print("--- 连接已关闭 ---")
    print(f"状态码: {close_status_code}, 信息: {close_msg}")
    ws.keep_running = False


if __name__ == "__main__":
    # 打开调试信息 (可选，不需要可以注释掉)
    # websocket.enableTrace(True)

    ws = websocket.WebSocketApp(
        WSS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    print(f"正在连接到 {WSS_URL} ...")

    # 开始运行，如果有代理则配置代理
    ws.run_forever(
        http_proxy_host=http_proxy_host,
        http_proxy_port=http_proxy_port,
        ping_interval=None,  # 我们手动实现 ping，所以这里设为 None
        ping_timeout=None,
    )
