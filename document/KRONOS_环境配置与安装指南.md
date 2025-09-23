# Kronos 环境配置与安装指南

## 🔧 系统要求

### 基础要求
- **Python**: 3.10+ （推荐 3.10 或 3.11）
- **操作系统**: Linux、macOS、Windows
- **内存**: 最少 8GB RAM（推荐 16GB+）
- **存储**: 至少 10GB 可用空间

### GPU要求（可选但推荐）
- **NVIDIA GPU**: 支持CUDA 11.8+
- **显存**: 最少 4GB（mini模型），推荐 8GB+（base模型）
- **Apple Silicon**: 支持MPS加速（M1/M2/M3芯片）

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

### 2. 创建虚拟环境（推荐）
```bash
# 使用 conda
conda create -n kronos python=3.10
conda activate kronos

# 或使用 venv
python -m venv kronos_env
source kronos_env/bin/activate  # Linux/macOS
# kronos_env\Scripts\activate  # Windows
```

### 3. 安装核心依赖
```bash
# 安装基础依赖
pip install -r requirements.txt
```

### 4. 验证安装
```python
# 测试基础导入
python -c "
from model import Kronos, KronosTokenizer, KronosPredictor
print('✅ Kronos 核心模块导入成功')
"
```

## 📋 依赖详解

### 核心依赖 (requirements.txt)
```
numpy                 # 数值计算基础
pandas==2.2.2        # 数据处理
torch                 # 深度学习框架
einops==0.8.1        # 张量操作
huggingface_hub==0.33.1  # 模型下载
matplotlib==3.9.3    # 图表绘制
tqdm==4.67.1         # 进度条
safetensors==0.6.2   # 安全张量存储
```

### Web UI 额外依赖 (webui/requirements.txt)
```
flask==2.3.3         # Web框架
flask-cors==4.0.0    # 跨域支持
pandas==2.2.2        # 数据处理
numpy==1.24.3        # 数值计算
plotly==5.17.0       # 交互式图表
torch>=2.1.0         # 深度学习框架
huggingface_hub==0.33.1  # 模型下载
```

### 微调训练额外依赖
```bash
# 安装 Qlib（用于A股数据处理）
pip install pyqlib

# 可选：Comet ML（实验跟踪）
pip install comet-ml
```

## 🚀 GPU 配置

### NVIDIA CUDA 配置
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应的PyTorch版本
# 访问 https://pytorch.org/get-started/locally/ 获取具体命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (MPS) 配置
```bash
# 确保使用最新的PyTorch版本
pip install torch torchvision torchaudio

# 验证MPS可用性
python -c "
import torch
print(f'MPS 可用: {torch.backends.mps.is_available()}')
print(f'MPS 已构建: {torch.backends.mps.is_built()}')
"
```

## 🔍 安装验证

### 1. 基础功能测试
```python
import torch
import pandas as pd
from model import Kronos, KronosTokenizer, KronosPredictor

# 检查设备
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"MPS 可用: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
print(f"可用设备数: {torch.cuda.device_count()}")

# 测试模型加载（需要网络连接）
try:
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    print("✅ Tokenizer 加载成功")
except Exception as e:
    print(f"❌ Tokenizer 加载失败: {e}")
```

### 2. Web UI 测试
```bash
cd webui
python app.py
```
访问 http://localhost:7070 验证Web界面

### 3. 示例代码测试
```bash
cd examples
# 需要先准备数据文件
python prediction_example.py
```

## 🛠️ 常见安装问题

### 问题1: PyTorch 安装失败
**解决方案**:
```bash
# 清理缓存
pip cache purge

# 使用官方源安装
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 问题2: Hugging Face 连接超时
**解决方案**:
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=your_proxy
export HTTPS_PROXY=your_proxy
```

### 问题3: 内存不足
**解决方案**:
- 使用更小的模型（kronos-mini）
- 减少批处理大小
- 使用CPU模式

### 问题4: 权限错误
**解决方案**:
```bash
# Linux/macOS
sudo chown -R $USER ~/.cache/huggingface

# 或使用用户安装
pip install --user -r requirements.txt
```

## 📁 目录结构检查

安装完成后，确保以下目录结构：
```
Kronos/
├── model/                 # 核心模型代码
├── examples/             # 示例代码
├── webui/               # Web界面
├── finetune/            # 微调训练
├── requirements.txt     # 依赖列表
├── README.md           # 项目说明
└── LICENSE             # 许可证
```

## 🔄 更新和维护

### 更新依赖
```bash
pip install --upgrade -r requirements.txt
```

### 清理缓存
```bash
# 清理pip缓存
pip cache purge

# 清理Hugging Face缓存
rm -rf ~/.cache/huggingface/transformers
```

### 环境重置
```bash
# 删除虚拟环境
conda env remove -n kronos
# 或
rm -rf kronos_env

# 重新创建
conda create -n kronos python=3.10
conda activate kronos
pip install -r requirements.txt
```

## 📊 性能优化建议

### 1. 内存优化
- 使用适当的批处理大小
- 启用梯度检查点（训练时）
- 使用混合精度训练

### 2. 计算优化
- 优先使用GPU
- 启用编译优化（PyTorch 2.0+）
- 使用多进程数据加载

### 3. 存储优化
- 使用SSD存储
- 预加载常用模型
- 压缩数据格式

---

**下一步**: 完成环境配置后，请参考 [基础使用流程指南] 开始使用Kronos进行金融预测。