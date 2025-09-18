# Kronos 故障排除和常见问题解答

## 🔧 安装和环境问题

### Q1: PyTorch安装失败
**问题描述**: `pip install torch` 失败或版本不兼容

**解决方案**:
```bash
# 方案1: 使用官方安装命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 方案2: 清理缓存重新安装
pip cache purge
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio

# 方案3: 使用conda安装
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**验证安装**:
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
```

### Q2: Hugging Face连接超时
**问题描述**: 下载模型时网络连接失败

**解决方案**:
```bash
# 方案1: 使用镜像源
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub

# 方案2: 设置代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port

# 方案3: 离线下载
# 手动下载模型文件到本地，然后使用本地路径
```

**验证连接**:
```python
from huggingface_hub import HfApi
api = HfApi()
try:
    models = api.list_models(author="NeoQuasar", limit=5)
    print("✅ Hugging Face连接正常")
except Exception as e:
    print(f"❌ 连接失败: {e}")
```

### Q3: 依赖版本冲突
**问题描述**: 不同包之间版本不兼容

**解决方案**:
```bash
# 方案1: 创建新的虚拟环境
conda create -n kronos-clean python=3.10
conda activate kronos-clean
pip install -r requirements.txt

# 方案2: 使用pip-tools管理依赖
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt

# 方案3: 手动解决冲突
pip install --upgrade --force-reinstall package_name
```

## 🚀 模型加载问题

### Q4: 模型下载失败
**问题描述**: `from_pretrained()` 方法失败

**解决方案**:
```python
# 方案1: 指定缓存目录
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "NeoQuasar/Kronos-small",
    cache_dir="./models_cache",
    force_download=True
)

# 方案2: 使用本地路径
# 先手动下载模型到本地
model = Kronos.from_pretrained("./local_models/Kronos-small")

# 方案3: 分步下载
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="NeoQuasar/Kronos-small",
    local_dir="./models/Kronos-small"
)
```

### Q5: GPU内存不足
**问题描述**: CUDA out of memory错误

**解决方案**:
```python
# 方案1: 减少批处理大小
predictor = KronosPredictor(
    model, tokenizer, 
    device="cuda:0", 
    max_context=256  # 减少上下文长度
)

# 方案2: 使用CPU
predictor = KronosPredictor(
    model, tokenizer, 
    device="cpu"
)

# 方案3: 清理GPU缓存
import torch
torch.cuda.empty_cache()

# 方案4: 使用更小的模型
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")  # 4.1M参数
```

### Q6: 模型加载速度慢
**问题描述**: 首次加载模型耗时很长

**解决方案**:
```python
# 方案1: 预下载模型
from huggingface_hub import snapshot_download
snapshot_download("NeoQuasar/Kronos-small", local_dir="./models")

# 方案2: 使用本地缓存
import os
os.environ['TRANSFORMERS_CACHE'] = './models_cache'

# 方案3: 并行下载
from concurrent.futures import ThreadPoolExecutor
def download_model(model_name):
    return snapshot_download(model_name)

with ThreadPoolExecutor() as executor:
    future = executor.submit(download_model, "NeoQuasar/Kronos-small")
```

## 📊 数据处理问题

### Q7: 数据格式错误
**问题描述**: 数据列名或格式不正确

**解决方案**:
```python
import pandas as pd

# 检查数据格式
def validate_data(df):
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        return False
    
    # 检查数据类型
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"❌ 列 {col} 不是数值类型")
            return False
    
    # 检查缺失值
    if df[required_cols].isnull().any().any():
        print("❌ 存在缺失值")
        return False
    
    print("✅ 数据格式正确")
    return True

# 数据清理
def clean_data(df):
    # 转换数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理时间戳
    if 'timestamps' not in df.columns:
        if 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
    
    # 删除缺失值
    df = df.dropna()
    
    return df
```

### Q8: 时间戳处理问题
**问题描述**: 时间戳格式不正确或缺失

**解决方案**:
```python
# 处理各种时间戳格式
def fix_timestamps(df):
    timestamp_cols = ['timestamps', 'timestamp', 'date', 'datetime']
    
    for col in timestamp_cols:
        if col in df.columns:
            try:
                df['timestamps'] = pd.to_datetime(df[col])
                break
            except:
                continue
    
    # 如果没有时间戳，创建一个
    if 'timestamps' not in df.columns:
        df['timestamps'] = pd.date_range(
            start='2024-01-01', 
            periods=len(df), 
            freq='5T'  # 5分钟间隔
        )
    
    # 确保时间戳排序
    df = df.sort_values('timestamps').reset_index(drop=True)
    
    return df
```

### Q9: 数据长度不足
**问题描述**: 历史数据不够400个点

**解决方案**:
```python
def check_data_length(df, lookback=400, pred_len=120):
    required_length = lookback + pred_len
    
    if len(df) < required_length:
        print(f"❌ 数据长度不足: 需要{required_length}个点，实际{len(df)}个点")
        
        # 方案1: 减少lookback
        max_lookback = len(df) - pred_len
        if max_lookback > 0:
            print(f"建议将lookback调整为: {max_lookback}")
            return max_lookback, pred_len
        
        # 方案2: 减少pred_len
        max_pred_len = len(df) - 100  # 保留最少100个历史点
        if max_pred_len > 0:
            print(f"建议将pred_len调整为: {max_pred_len}")
            return 100, max_pred_len
        
        return None, None
    
    print("✅ 数据长度充足")
    return lookback, pred_len
```

## 🌐 Web UI问题

### Q10: Web UI启动失败
**问题描述**: Flask应用无法启动

**解决方案**:
```bash
# 检查端口占用
lsof -i :7070
netstat -tulpn | grep 7070

# 更换端口
cd webui
python -c "
from app import app
app.run(debug=True, host='0.0.0.0', port=7071)
"

# 检查依赖
pip install -r webui/requirements.txt

# 权限问题
sudo chown -R $USER:$USER webui/
```

### Q11: Web UI模型加载失败
**问题描述**: 在Web界面中无法加载模型

**解决方案**:
```python
# 检查模型可用性
def check_model_availability():
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ 模型模块导入成功")
        
        # 测试模型加载
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("✅ 模型加载成功")
        
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

# 在webui目录下运行
check_model_availability()
```

### Q12: 预测结果异常
**问题描述**: Web UI预测结果不合理

**解决方案**:
```python
# 检查预测参数
def validate_prediction_params(T, top_p, sample_count):
    issues = []
    
    if T < 0.1 or T > 2.0:
        issues.append(f"Temperature {T} 超出范围 [0.1, 2.0]")
    
    if top_p < 0.1 or top_p > 1.0:
        issues.append(f"top_p {top_p} 超出范围 [0.1, 1.0]")
    
    if sample_count < 1 or sample_count > 5:
        issues.append(f"sample_count {sample_count} 超出范围 [1, 5]")
    
    if issues:
        print("❌ 参数问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ 参数设置正确")
    return True

# 推荐参数
recommended_params = {
    "conservative": {"T": 0.8, "top_p": 0.7, "sample_count": 3},
    "balanced": {"T": 1.0, "top_p": 0.9, "sample_count": 2},
    "exploratory": {"T": 1.5, "top_p": 1.0, "sample_count": 5}
}
```

## 🔧 微调训练问题

### Q13: 训练过程中断
**问题描述**: 训练过程中出现错误中断

**解决方案**:
```python
# 添加检查点恢复
def resume_training(checkpoint_path, model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从第{start_epoch}轮恢复训练")
        return start_epoch
    return 0

# 定期保存检查点
def save_checkpoint(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
```

### Q14: 训练速度慢
**问题描述**: 训练过程耗时过长

**解决方案**:
```python
# 优化训练速度
def optimize_training():
    # 1. 使用多GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # 2. 混合精度训练
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    # 3. 优化数据加载
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=4,  # 多进程加载
        pin_memory=True,  # 固定内存
        persistent_workers=True
    )
    
    # 4. 编译模型（PyTorch 2.0+）
    model = torch.compile(model)
    
    return model, scaler, dataloader
```

### Q15: 内存溢出
**问题描述**: 训练时出现OOM错误

**解决方案**:
```python
# 内存优化策略
def optimize_memory():
    # 1. 减少批大小
    batch_size = 16  # 从50减少到16
    
    # 2. 梯度累积
    accumulation_steps = 4  # 有效批大小 = 16 * 4 = 64
    
    # 3. 梯度检查点
    model.gradient_checkpointing_enable()
    
    # 4. 清理缓存
    torch.cuda.empty_cache()
    
    # 5. 使用CPU卸载
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    model = FSDP(model, cpu_offload=True)
    
    return batch_size, accumulation_steps
```

## 📈 性能优化问题

### Q16: 预测速度慢
**问题描述**: 单次预测耗时过长

**解决方案**:
```python
# 预测速度优化
def optimize_inference():
    # 1. 模型量化
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # 2. TorchScript编译
    model = torch.jit.script(model)
    
    # 3. 批量预测
    def batch_predict(data_list):
        with torch.no_grad():
            results = []
            for batch in data_list:
                pred = model(batch)
                results.append(pred)
        return results
    
    # 4. 缓存常用数据
    @lru_cache(maxsize=128)
    def cached_predict(data_hash):
        return model.predict(data)
    
    return model
```

### Q17: 内存使用过高
**问题描述**: 程序占用内存过多

**解决方案**:
```python
import gc
import psutil

def monitor_memory():
    # 监控内存使用
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"内存使用: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 优化数据加载
    def efficient_data_loading(file_path):
        # 分块读取大文件
        chunk_size = 10000
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            yield chunk
    
    return memory_info
```

## 🔍 调试技巧

### Q18: 如何调试预测结果
**问题描述**: 预测结果不符合预期

**调试方法**:
```python
def debug_prediction(predictor, data):
    # 1. 检查输入数据
    print("输入数据统计:")
    print(data.describe())
    
    # 2. 检查模型状态
    print(f"模型设备: {next(predictor.model.parameters()).device}")
    print(f"模型模式: {'训练' if predictor.model.training else '评估'}")
    
    # 3. 逐步预测
    with torch.no_grad():
        # 数据预处理
        processed_data = predictor.preprocess(data)
        print(f"预处理后数据形状: {processed_data.shape}")
        
        # 模型推理
        output = predictor.model(processed_data)
        print(f"模型输出形状: {output.shape}")
        
        # 后处理
        final_result = predictor.postprocess(output)
        print(f"最终结果形状: {final_result.shape}")
    
    return final_result

# 4. 可视化中间结果
def visualize_debug(data, prediction):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 原始数据
    axes[0,0].plot(data['close'])
    axes[0,0].set_title('原始数据')
    
    # 预测结果
    axes[0,1].plot(prediction['close'])
    axes[0,1].set_title('预测结果')
    
    # 数据分布
    axes[1,0].hist(data['close'], bins=50)
    axes[1,0].set_title('原始数据分布')
    
    axes[1,1].hist(prediction['close'], bins=50)
    axes[1,1].set_title('预测数据分布')
    
    plt.tight_layout()
    plt.show()
```

## 📞 获取帮助

### 官方资源
- **GitHub Issues**: [https://github.com/shiyu-coder/Kronos/issues](https://github.com/shiyu-coder/Kronos/issues)
- **论文**: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)
- **在线演示**: [Live Demo](https://shiyu-coder.github.io/Kronos-demo/)

### 社区支持
- **讨论区**: GitHub Discussions
- **技术交流**: 相关技术论坛
- **文档反馈**: 通过Issues提交文档改进建议

### 日志收集
```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kronos_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('kronos')

# 在关键位置添加日志
logger.info("开始加载模型")
logger.debug(f"数据形状: {data.shape}")
logger.error(f"预测失败: {error}")
```

---

**提示**: 如果以上解决方案都无法解决您的问题，请收集详细的错误信息和环境信息，通过GitHub Issues寻求帮助。