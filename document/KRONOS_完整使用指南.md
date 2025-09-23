# Kronos 金融市场基础模型 - 完整使用指南

## 📖 文档导航

本指南包含Kronos项目的完整使用说明，按照从入门到精通的顺序组织：

### 🚀 快速开始
1. [项目概述与架构](#项目概述与架构) - 了解Kronos是什么
2. [环境配置与安装](#环境配置与安装) - 搭建运行环境
3. [项目启动检查清单](#项目启动检查清单) - 验证安装成功

### 📊 基础使用
4. [基础使用流程](#基础使用流程) - 学习核心预测功能
5. [Web UI使用指南](#web-ui使用指南) - 图形化界面操作
6. [项目结构说明](#项目结构说明) - 深入了解代码结构

### 🔧 高级功能
7. [微调训练流程](#微调训练流程) - 自定义模型训练
8. [故障排除指南](#故障排除指南) - 解决常见问题

---

## 🎯 项目概述与架构

### Kronos简介

**Kronos** 是首个专为金融市场K线数据设计的开源基础模型，基于来自全球45个交易所的数据进行训练。它采用两阶段架构：分层离散化tokenizer + 大型自回归Transformer，专门处理金融市场的"语言"——K线序列数据。

### 核心特点
- 🚀 **专业性**: 专为金融K线数据优化
- 🔧 **两阶段架构**: Tokenizer + Transformer
- 📊 **多维数据**: 支持OHLCV数据处理
- 🎯 **统一框架**: 适用于多种量化任务
- 🌐 **多模型**: mini/small/base不同规模

### 模型规格

| 模型 | 参数量 | 上下文长度 | 特点 | 开源状态 |
|------|--------|------------|------|----------|
| Kronos-mini | 4.1M | 2048 | 轻量快速 | ✅ |
| Kronos-small | 24.7M | 512 | 平衡性能 | ✅ |
| Kronos-base | 102.3M | 512 | 高质量预测 | ✅ |

### 应用场景
- 📈 **金融预测**: 股票、加密货币、期货价格预测
- 🤖 **量化交易**: 信号生成、风险管理、投资组合优化
- 🔍 **研究分析**: 市场行为研究、模式识别、趋势分析

---

## 🔧 环境配置与安装

### 系统要求
- **Python**: 3.10+
- **内存**: 最少8GB RAM
- **存储**: 至少10GB可用空间
- **GPU**: 可选，推荐NVIDIA GPU或Apple Silicon

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

#### 2. 创建虚拟环境
```bash
# 使用conda（推荐）
conda create -n kronos python=3.10
conda activate kronos

# 或使用venv
python -m venv kronos_env
source kronos_env/bin/activate  # Linux/macOS
```

#### 3. 安装依赖
```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装Web UI依赖（可选）
cd webui
pip install -r requirements.txt
cd ..

# 安装微调依赖（可选）
pip install pyqlib
```

#### 4. 验证安装
```python
# 测试基础导入
from model import Kronos, KronosTokenizer, KronosPredictor
print("✅ Kronos安装成功")

# 检查GPU支持
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
```

### 常见安装问题

#### PyTorch安装失败
```bash
# 使用官方安装命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Hugging Face连接超时
```bash
# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
```

---

## ✅ 项目启动检查清单

### 环境验证脚本

创建并运行以下验证脚本：

```python
# check_environment.py
import sys
import torch
import pandas as pd

def check_environment():
    print("🔍 Kronos环境检查...")
    
    # 检查Python版本
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python版本: {version.major}.{version.minor}")
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        return False
    
    # 检查PyTorch
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    
    # 检查Kronos模块
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos模块导入成功")
    except ImportError as e:
        print(f"❌ Kronos模块导入失败: {e}")
        return False
    
    print("🎉 环境检查通过！")
    return True

if __name__ == "__main__":
    check_environment()
```

### 功能测试

```python
# test_basic_functionality.py
def test_model_loading():
    """测试模型加载"""
    try:
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("✅ 模型加载成功")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

# 运行测试
tokenizer, model = test_model_loading()
```

---

## 📊 基础使用流程

### 数据准备

#### 数据格式要求
```csv
timestamps,open,high,low,close,volume,amount
2024-01-01 09:30:00,100.0,102.5,99.8,101.2,1000000,101200000
2024-01-01 09:35:00,101.2,103.0,100.5,102.8,1200000,122400000
```

**必需列**: `open`, `high`, `low`, `close`
**可选列**: `volume`, `amount`, `timestamps`

#### 数据验证
```python
def validate_data(df):
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        return False
    
    if len(df) < 520:
        print(f"⚠️ 数据长度不足: {len(df)} < 520")
        return False
    
    print("✅ 数据格式正确")
    return True
```

### 基础预测流程

#### 1. 导入模块
```python
import pandas as pd
import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor
```

#### 2. 加载模型
```python
# 从Hugging Face加载预训练模型
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 创建预测器
predictor = KronosPredictor(
    model=model, 
    tokenizer=tokenizer, 
    device="cuda:0",  # 或 "cpu", "mps"
    max_context=512
)
```

#### 3. 准备数据
```python
# 加载数据
df = pd.read_csv("your_data.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# 设置参数
lookback = 400    # 历史数据长度
pred_len = 120    # 预测长度

# 准备输入数据
x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']]
x_timestamp = df.iloc[:lookback]['timestamps']
y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
```

#### 4. 执行预测
```python
# 进行预测
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # 温度参数
    top_p=0.9,      # 核采样概率
    sample_count=1, # 采样次数
    verbose=True
)

print("预测结果:")
print(pred_df.head())
```

#### 5. 结果可视化
```python
def plot_prediction(historical_df, pred_df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制价格
    ax1.plot(historical_df['timestamps'], historical_df['close'], 
             label='历史数据', color='blue')
    
    pred_timestamps = pd.date_range(
        start=historical_df['timestamps'].iloc[-1], 
        periods=len(pred_df)+1, freq='5T'
    )[1:]
    
    ax1.plot(pred_timestamps, pred_df['close'], 
             label='预测数据', color='red')
    ax1.set_ylabel('收盘价')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制成交量
    ax2.plot(historical_df['timestamps'], historical_df['volume'], 
             label='历史成交量', color='blue', alpha=0.7)
    ax2.plot(pred_timestamps, pred_df['volume'], 
             label='预测成交量', color='red', alpha=0.7)
    ax2.set_ylabel('成交量')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 绘制结果
plot_prediction(x_df, pred_df)
```

### 参数调优指南

#### 预测质量参数

1. **Temperature (T)**: 0.1-2.0
   - `T=0.8-1.2`: 平衡预测
   - `T<0.8`: 保守预测
   - `T>1.2`: 多样化预测

2. **Nucleus Sampling (top_p)**: 0.1-1.0
   - `top_p=0.9-1.0`: 考虑更多可能性
   - `top_p=0.7-0.9`: 平衡质量和多样性

3. **Sample Count**: 1-5
   - `sample_count=1`: 快速预测
   - `sample_count=3-5`: 更稳定结果

#### 推荐参数组合
```python
# 保守预测
conservative_params = {"T": 0.8, "top_p": 0.7, "sample_count": 3}

# 平衡预测（推荐）
balanced_params = {"T": 1.0, "top_p": 0.9, "sample_count": 2}

# 探索性预测
exploratory_params = {"T": 1.5, "top_p": 1.0, "sample_count": 5}
```

---

## 🌐 Web UI使用指南

### 启动Web UI

#### 方式1: Python脚本启动（推荐）
```bash
cd webui
python run.py
```

#### 方式2: Shell脚本启动
```bash
cd webui
chmod +x start.sh
./start.sh
```

启动成功后访问: **http://localhost:7070**

### 界面功能

#### 1. 数据管理
- **文件上传**: 支持CSV、Feather格式
- **数据验证**: 自动检查必需列和数据质量
- **信息显示**: 显示数据行数、列数、时间范围

#### 2. 模型配置
- **模型选择**: Kronos-mini/small/base
- **设备选择**: CPU/CUDA/MPS
- **状态监控**: 实时显示模型加载状态

#### 3. 参数设置
- **Temperature**: 0.1-2.0，控制预测随机性
- **Nucleus Sampling**: 0.1-1.0，控制预测多样性
- **Sample Count**: 1-5，生成样本数量

#### 4. 时间窗口
- **固定窗口**: 400历史+120预测
- **滑块选择**: 选择起始时间点
- **数据验证**: 确保数据充足性

#### 5. 结果展示
- **K线图表**: 交互式图表显示
- **数据对比**: 预测vs实际数据
- **结果下载**: JSON格式保存

### 操作流程

1. **加载数据** → 选择文件并验证格式
2. **配置模型** → 选择模型和设备
3. **设置参数** → 调整预测质量参数
4. **选择窗口** → 确定时间范围
5. **执行预测** → 开始预测并等待结果
6. **分析结果** → 查看图表和数据

---

## 🔧 微调训练流程

### 前置条件

#### 环境准备
```bash
# 安装微调依赖
pip install pyqlib
pip install comet-ml  # 可选，用于实验跟踪
```

#### 数据准备
```bash
# 下载Qlib数据（参考官方指南）
# https://github.com/microsoft/qlib
```

### 配置设置

编辑 `finetune/config.py`：

```python
class Config:
    # 必须修改的路径
    self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
    self.dataset_path = "./data/processed_datasets"
    self.save_path = "./outputs/models"
    
    # 预训练模型路径
    self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
    self.pretrained_predictor_path = "NeoQuasar/Kronos-small"
    
    # 训练参数
    self.epochs = 30
    self.batch_size = 50
    self.tokenizer_learning_rate = 2e-4
    self.predictor_learning_rate = 4e-5
```

### 微调步骤

#### 步骤1: 数据预处理
```bash
python finetune/qlib_data_preprocess.py
```

**功能**:
- 从Qlib加载A股数据
- 生成时间特征
- 创建滑动窗口样本
- 划分训练/验证/测试集

#### 步骤2: Tokenizer微调
```bash
# 多GPU训练
torchrun --standalone --nproc_per_node=2 finetune/train_tokenizer.py

# 单GPU训练
python finetune/train_tokenizer.py
```

#### 步骤3: Predictor微调
```bash
# 多GPU训练
torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py

# 单GPU训练
python finetune/train_predictor.py
```

#### 步骤4: 模型评估
```bash
python finetune/qlib_test.py --device cuda:0
```

### 训练监控

#### 关键指标
- **重构损失**: Tokenizer训练质量
- **预测损失**: Predictor训练质量
- **验证精度**: 模型泛化能力
- **回测收益**: 实际应用效果

#### Comet ML集成
```python
# 自动记录训练指标
experiment.log_metric("train_loss", loss)
experiment.log_metric("val_loss", val_loss)
experiment.log_parameters(config.__dict__)
```

---

## 📁 项目结构说明

### 目录结构
```
Kronos/
├── model/                          # 🔥 核心模型代码
│   ├── __init__.py                 # 模块导出
│   ├── kronos.py                   # 主要模型实现
│   └── module.py                   # 模型组件
├── examples/                       # 📚 使用示例
│   ├── prediction_example.py       # 基础预测示例
│   ├── prediction_batch_example.py # 批量预测示例
│   └── prediction_wo_vol_example.py # 无成交量示例
├── webui/                          # 🌐 Web界面
│   ├── app.py                      # Flask主应用
│   ├── run.py                      # 启动脚本
│   ├── templates/index.html        # 页面模板
│   └── prediction_results/         # 结果存储
├── finetune/                       # 🔧 微调训练
│   ├── config.py                   # 配置参数
│   ├── dataset.py                  # 数据集处理
│   ├── train_tokenizer.py          # Tokenizer训练
│   ├── train_predictor.py          # Predictor训练
│   └── qlib_test.py               # 回测评估
└── figures/                        # 🖼️ 项目图片
```

### 核心组件

#### 1. KronosTokenizer
- **功能**: 连续数据→离散token
- **特点**: 分层离散化，处理OHLCV
- **方法**: `encode()`, `decode()`, `from_pretrained()`

#### 2. Kronos
- **功能**: 主要预测模型
- **架构**: Decoder-only Transformer
- **方法**: `forward()`, `generate()`, `from_pretrained()`

#### 3. KronosPredictor
- **功能**: 高级预测接口
- **特点**: 完整预测流程封装
- **方法**: `predict()`, `predict_batch()`

---

## 🆘 故障排除指南

### 常见问题解决

#### 1. 模型加载失败
**问题**: `from_pretrained()` 失败

**解决方案**:
```bash
# 检查网络连接
ping huggingface.co

# 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 本地下载
huggingface-cli download NeoQuasar/Kronos-small
```

#### 2. GPU内存不足
**问题**: CUDA out of memory

**解决方案**:
```python
# 使用更小模型
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")

# 减少批大小
predictor = KronosPredictor(model, tokenizer, device="cpu")

# 清理GPU缓存
torch.cuda.empty_cache()
```

#### 3. 数据格式错误
**问题**: 数据列名或格式不正确

**解决方案**:
```python
def fix_data_format(df):
    # 检查必需列
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必需列: {col}")
    
    # 转换数据类型
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理时间戳
    if 'timestamps' not in df.columns:
        df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='5T')
    
    return df.dropna()
```

#### 4. Web UI启动失败
**问题**: Flask应用无法启动

**解决方案**:
```bash
# 检查端口占用
lsof -i :7070

# 更换端口
python -c "from app import app; app.run(port=7071)"

# 检查依赖
pip install -r webui/requirements.txt
```

### 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 检查中间结果
```python
def debug_prediction(predictor, data):
    # 检查输入数据
    print("输入数据统计:", data.describe())
    
    # 检查模型状态
    print("模型设备:", next(predictor.model.parameters()).device)
    
    # 逐步预测
    with torch.no_grad():
        processed_data = predictor.preprocess(data)
        output = predictor.model(processed_data)
        final_result = predictor.postprocess(output)
    
    return final_result
```

#### 3. 性能监控
```python
import psutil
import time

def monitor_performance():
    process = psutil.Process()
    start_time = time.time()
    
    # 执行预测
    result = predictor.predict(data)
    
    end_time = time.time()
    memory_usage = process.memory_info().rss / 1024 / 1024
    
    print(f"预测时间: {end_time - start_time:.2f}秒")
    print(f"内存使用: {memory_usage:.2f}MB")
    
    return result
```

---

## 🎯 最佳实践

### 1. 数据准备
- ✅ 确保数据质量，无缺失值
- ✅ 使用连续的时间序列
- ✅ 准备足够的历史数据（≥520个点）
- ✅ 验证数据格式和列名

### 2. 模型选择
- 🔸 **Kronos-mini**: 快速验证和原型开发
- 🔸 **Kronos-small**: 平衡性能和速度（推荐）
- 🔸 **Kronos-base**: 高质量预测需求

### 3. 参数调优
- 🎯 从默认参数开始：`T=1.0, top_p=0.9, sample_count=1`
- 🎯 根据结果质量调整温度参数
- 🎯 使用多样本平均提高稳定性

### 4. 性能优化
- ⚡ 优先使用GPU加速
- ⚡ 批量处理多个预测
- ⚡ 缓存常用模型和数据

### 5. 生产部署
- 🚀 使用模型量化减少内存
- 🚀 实现模型版本管理
- 🚀 添加监控和日志
- 🚀 考虑交易成本和风险管理

---

## 📚 学习路径

### 初学者路径
1. 📖 阅读项目概述，了解基本概念
2. 🔧 完成环境配置和安装
3. ✅ 运行启动检查清单
4. 📊 学习基础预测流程
5. 🌐 尝试Web UI界面

### 进阶用户路径
1. 📁 深入了解项目结构
2. 🔧 学习微调训练流程
3. 🛠️ 自定义数据处理
4. 📈 优化预测参数
5. 🚀 部署到生产环境

### 开发者路径
1. 🔍 研究核心模型代码
2. 🧩 扩展新功能模块
3. 🔄 贡献代码和文档
4. 📊 开发新的应用场景
5. 🌟 参与社区建设

---

## 🔗 相关资源

### 官方资源
- **GitHub仓库**: [https://github.com/shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- **论文**: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)
- **在线演示**: [Live Demo](https://shiyu-coder.github.io/Kronos-demo/)
- **模型下载**: [Hugging Face Hub](https://huggingface.co/NeoQuasar)

### 技术文档
- [PyTorch官方文档](https://pytorch.org/docs/)
- [Hugging Face文档](https://huggingface.co/docs)
- [Qlib文档](https://github.com/microsoft/qlib)
- [Flask文档](https://flask.palletsprojects.com/)

### 社区支持
- **GitHub Issues**: 报告问题和获取帮助
- **GitHub Discussions**: 技术讨论和经验分享
- **相关论坛**: 量化投资和机器学习社区

---

## 📞 获取帮助

### 问题反馈
1. 🔍 首先查看[故障排除指南](#故障排除指南)
2. 📖 搜索[GitHub Issues](https://github.com/shiyu-coder/Kronos/issues)
3. 📝 提交新Issue并提供详细信息：
   - 错误信息和堆栈跟踪
   - 系统环境信息
   - 复现步骤
   - 期望行为

### 贡献指南
- 🐛 报告Bug和问题
- 💡 提出功能建议
- 📖 改进文档
- 🔧 提交代码修复
- 🌟 分享使用经验

---

**恭喜！** 您已经掌握了Kronos的完整使用方法。现在可以开始您的金融预测之旅了！

> **免责声明**: Kronos仅用于研究和教育目的。实际投资决策应结合专业知识和风险管理策略。