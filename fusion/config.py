# fusion/config.py (最终修正版)

# TIMEFRAMES 列表保持原样，因为它正确匹配了您的文件名
TIMEFRAMES = ["_4h"]
SYMBOL = "ETHUSDT"

# 修正模型目录的相对路径，去掉 ".."
MODEL_DIR = "models"

LOG_DIR = "logs"

# 修正权重字典的键，使其与 TIMEFRAMES 列表中的元素完全对应
# 空字符串 "" 对应不带后缀的基础模型（即15m）
INIT_WEIGHTS = {"": 0.20, "_1h": 0.30, "_4h": 0.30, "_8h": 0.20}

FUSION_THRESHOLD = 0.40