# Kronos 项目启动检查清单

## 📋 快速启动检查清单

### ✅ 环境准备阶段

#### 1. 系统要求检查
- [ ] Python 3.10+ 已安装
- [ ] 可用内存 ≥ 8GB
- [ ] 可用存储空间 ≥ 10GB
- [ ] 网络连接正常（用于下载模型）

#### 2. 项目获取
- [ ] 克隆项目仓库
  ```bash
  git clone https://github.com/shiyu-coder/Kronos.git
  cd Kronos
  ```
- [ ] 检查项目结构完整性
- [ ] 确认所有必要文件存在

#### 3. 虚拟环境设置
- [ ] 创建虚拟环境
  ```bash
  conda create -n kronos python=3.10
  conda activate kronos
  ```
- [ ] 激活虚拟环境
- [ ] 验证Python版本

#### 4. 依赖安装
- [ ] 安装核心依赖
  ```bash
  pip install -r requirements.txt
  ```
- [ ] 验证PyTorch安装
  ```python
  import torch
  print(torch.__version__)
  ```
- [ ] 检查CUDA可用性（如有GPU）
  ```python
  print(torch.cuda.is_available())
  ```

### ✅ 基础功能验证

#### 5. 模型导入测试
- [ ] 测试核心模块导入
  ```python
  from model import Kronos, KronosTokenizer, KronosPredictor
  print("✅ 模块导入成功")
  ```
- [ ] 验证Hugging Face连接
- [ ] 测试模型下载（可选）

#### 6. 数据准备
- [ ] 准备测试数据文件
- [ ] 验证数据格式（包含open, high, low, close列）
- [ ] 检查数据长度（≥520个数据点）
- [ ] 确认时间戳格式正确

### ✅ 基础预测测试

#### 7. 简单预测测试
- [ ] 运行基础预测示例
  ```bash
  cd examples
  python prediction_example.py
  ```
- [ ] 检查预测结果输出
- [ ] 验证图表生成

#### 8. 无成交量预测测试
- [ ] 运行无成交量示例
  ```bash
  python prediction_wo_vol_example.py
  ```
- [ ] 确认预测正常完成

### ✅ Web UI功能验证

#### 9. Web UI启动
- [ ] 安装Web UI依赖
  ```bash
  cd webui
  pip install -r requirements.txt
  ```
- [ ] 启动Web服务
  ```bash
  python run.py
  ```
- [ ] 访问 http://localhost:7070
- [ ] 确认界面正常显示

#### 10. Web UI功能测试
- [ ] 数据文件加载功能
- [ ] 模型加载功能
- [ ] 参数配置功能
- [ ] 预测执行功能
- [ ] 结果可视化功能

### ✅ 高级功能验证（可选）

#### 11. 批量预测测试
- [ ] 运行批量预测示例
  ```bash
  python prediction_batch_example.py
  ```
- [ ] 验证多序列预测结果

#### 12. 微调环境准备（可选）
- [ ] 安装Qlib
  ```bash
  pip install pyqlib
  ```
- [ ] 配置微调参数
- [ ] 准备训练数据

## 🔧 详细检查步骤

### 环境验证脚本

创建并运行以下验证脚本：

```python
# check_environment.py
import sys
import torch
import pandas as pd
import numpy as np

def check_python_version():
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        return False

def check_pytorch():
    try:
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
        return True
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")
        return False

def check_dependencies():
    try:
        import pandas
        import numpy
        import matplotlib
        import tqdm
        import einops
        import huggingface_hub
        print("✅ 所有依赖包导入成功")
        return True
    except ImportError as e:
        print(f"❌ 依赖包缺失: {e}")
        return False

def check_kronos_import():
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ Kronos模块导入失败: {e}")
        return False

def main():
    print("🔍 Kronos环境检查开始...")
    print("=" * 50)
    
    checks = [
        ("Python版本", check_python_version),
        ("PyTorch", check_pytorch),
        ("依赖包", check_dependencies),
        ("Kronos模块", check_kronos_import)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n检查 {name}:")
        result = check_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 所有检查通过！环境配置正确。")
    else:
        print("⚠️  部分检查失败，请参考错误信息进行修复。")

if __name__ == "__main__":
    main()
```

### 数据验证脚本

```python
# check_data.py
import pandas as pd
import numpy as np

def validate_data_file(file_path):
    """验证数据文件格式"""
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        print(f"✅ 成功读取数据文件: {file_path}")
        print(f"   数据行数: {len(df)}")
        print(f"   数据列数: {len(df.columns)}")
        
        # 检查必需列
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ 缺少必需列: {missing_cols}")
            return False
        else:
            print("✅ 包含所有必需列")
        
        # 检查数据类型
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"❌ 列 {col} 不是数值类型")
                return False
        print("✅ 数据类型正确")
        
        # 检查数据长度
        if len(df) < 520:
            print(f"⚠️  数据长度不足: {len(df)} < 520")
            print("   建议准备至少520个数据点用于完整预测")
        else:
            print("✅ 数据长度充足")
        
        # 检查缺失值
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            print(f"⚠️  存在缺失值: {null_counts.to_dict()}")
        else:
            print("✅ 无缺失值")
        
        # 检查时间戳
        timestamp_cols = ['timestamps', 'timestamp', 'date']
        has_timestamp = any(col in df.columns for col in timestamp_cols)
        if has_timestamp:
            print("✅ 包含时间戳列")
        else:
            print("⚠️  无时间戳列，将自动生成")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据文件验证失败: {e}")
        return False

# 使用示例
# validate_data_file("./data/your_data.csv")
```

### 功能测试脚本

```python
# test_basic_functionality.py
import pandas as pd
import numpy as np
from model import Kronos, KronosTokenizer, KronosPredictor

def test_model_loading():
    """测试模型加载"""
    try:
        print("🔄 正在加载模型...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        print("✅ 模型加载成功")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def test_prediction(tokenizer, model):
    """测试基础预测功能"""
    try:
        # 创建预测器
        predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
        print("✅ 预测器创建成功")
        
        # 生成测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=600, freq='5T')
        
        # 生成模拟K线数据
        base_price = 100
        prices = []
        for i in range(600):
            if i == 0:
                open_price = base_price
            else:
                open_price = prices[-1]['close']
            
            high_price = open_price * (1 + np.random.uniform(0, 0.02))
            low_price = open_price * (1 - np.random.uniform(0, 0.02))
            close_price = open_price + np.random.uniform(-0.5, 0.5)
            volume = np.random.randint(1000, 10000)
            
            prices.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(prices)
        df['timestamps'] = dates
        
        print("✅ 测试数据生成成功")
        
        # 执行预测
        lookback = 400
        pred_len = 120
        
        x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']]
        x_timestamp = df.iloc[:lookback]['timestamps']
        y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1
        )
        
        print(f"✅ 预测完成，生成 {len(pred_df)} 个预测点")
        print("预测结果示例:")
        print(pred_df.head())
        
        return True
        
    except Exception as e:
        print(f"❌ 预测测试失败: {e}")
        return False

def main():
    print("🧪 Kronos功能测试开始...")
    print("=" * 50)
    
    # 测试模型加载
    tokenizer, model = test_model_loading()
    if tokenizer is None or model is None:
        print("❌ 模型加载失败，无法继续测试")
        return
    
    # 测试预测功能
    prediction_success = test_prediction(tokenizer, model)
    
    print("\n" + "=" * 50)
    if prediction_success:
        print("🎉 所有功能测试通过！")
    else:
        print("⚠️  部分功能测试失败")

if __name__ == "__main__":
    main()
```

## 📊 性能基准测试

### 预测性能测试
```python
# benchmark_prediction.py
import time
import torch
from model import Kronos, KronosTokenizer, KronosPredictor

def benchmark_prediction_speed():
    """测试预测速度"""
    # 加载模型
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    
    # 测试不同设备
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    
    for device in devices:
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
        
        # 生成测试数据
        test_data = generate_test_data(520)
        
        # 预热
        _ = predictor.predict(
            df=test_data.iloc[:400][['open', 'high', 'low', 'close', 'volume']],
            x_timestamp=test_data.iloc[:400]['timestamps'],
            y_timestamp=test_data.iloc[400:520]['timestamps'],
            pred_len=120
        )
        
        # 性能测试
        start_time = time.time()
        for _ in range(5):
            _ = predictor.predict(
                df=test_data.iloc[:400][['open', 'high', 'low', 'close', 'volume']],
                x_timestamp=test_data.iloc[:400]['timestamps'],
                y_timestamp=test_data.iloc[400:520]['timestamps'],
                pred_len=120
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 5
        print(f"设备 {device}: 平均预测时间 {avg_time:.2f} 秒")
```

## 🚀 启动成功标志

当您完成所有检查项目后，应该能够：

1. ✅ **成功导入Kronos模块**
2. ✅ **加载预训练模型**
3. ✅ **执行基础预测**
4. ✅ **启动Web UI界面**
5. ✅ **查看预测结果图表**

## 🔗 下一步行动

完成检查清单后，您可以：

1. 📖 阅读 [基础使用流程指南](KRONOS_基础使用流程指南.md)
2. 🌐 尝试 [Web UI使用指南](KRONOS_WebUI启动和使用指南.md)
3. 🔧 探索 [微调训练流程](KRONOS_微调训练流程说明.md)
4. 🆘 参考 [故障排除指南](KRONOS_故障排除和常见问题解答.md)

## 📞 获取帮助

如果在检查过程中遇到问题：

1. 查看 [故障排除指南](KRONOS_故障排除和常见问题解答.md)
2. 检查 [GitHub Issues](https://github.com/shiyu-coder/Kronos/issues)
3. 提交新的Issue并附上详细的错误信息

---

**恭喜！** 完成此检查清单后，您就可以开始使用Kronos进行金融预测了！