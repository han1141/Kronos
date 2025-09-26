# 使用微调模型进行BTC预测指南

## 概述

本指南详细说明如何使用outputs目录中的微调后模型进行BTC价格预测。

## 微调模型结构

根据配置文件 `finetune/btc_config.py`，微调训练会生成两个模型：

1. **微调后的Tokenizer** (`btc_finetune_tokenizer`)
   - 路径: `outputs/models/btc_finetune_tokenizer/checkpoints/best_model/`
   - 作用: 将BTC价格数据转换为模型可理解的token序列
   - 状态: ✅ 已训练完成

2. **微调后的Predictor** (`btc_finetune_predictor`)
   - 路径: `outputs/models/btc_finetune_predictor/checkpoints/best_model/`
   - 作用: 基于token序列进行价格预测
   - 状态: ❌ 尚未训练（需要先完成tokenizer训练）

## 当前可用的使用方式

### 方式1: 使用微调后的Tokenizer + 预训练Predictor

这是目前推荐的方式，因为只有tokenizer完成了微调训练。

```python
# 使用示例
from model import Kronos, KronosTokenizer, KronosPredictor

# 加载微调后的tokenizer
tokenizer = KronosTokenizer.from_pretrained("../outputs/models/btc_finetune_tokenizer/checkpoints/best_model")

# 加载预训练的predictor
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 创建预测器
predictor = KronosPredictor(
    model=model,
    tokenizer=tokenizer,
    device=device,
)
```

### 方式2: 完全微调模型（需要先训练predictor）

如果你已经完成了predictor的微调训练，可以使用：

```python
# 加载完全微调后的模型
tokenizer = KronosTokenizer.from_pretrained("../outputs/models/btc_finetune_tokenizer/checkpoints/best_model")
model = Kronos.from_pretrained("../outputs/models/btc_finetune_predictor/checkpoints/best_model")
```

## 使用步骤

### 1. 运行预测脚本

我已经为你创建了一个专门的脚本 `examples/bitcoin_forecast_finetuned.py`，它会：

- 自动检测可用的微调模型
- 智能选择最佳的模型组合
- 提供详细的模型信息

```bash
cd examples
python bitcoin_forecast_finetuned.py
```

### 2. 脚本功能特点

- **智能模型检测**: 自动检查微调后的predictor是否存在
- **灵活加载策略**: 
  - 如果predictor存在：使用完全微调模型
  - 如果predictor不存在：使用微调tokenizer + 预训练predictor
- **详细信息输出**: 显示使用的模型类型和状态
- **结果文件**: 输出到 `csv/kronos_btc_finetuned.csv`

### 3. 预期输出

运行脚本后，你会看到类似以下输出：

```
🔄 正在加载微调后的模型...
⚠️  未找到微调后的predictor模型，使用微调后的tokenizer + 预训练的predictor
✅ 成功加载微调后的tokenizer和预训练的predictor
🔮 开始进行BTC价格预测...
...
✅ 已成功导出合并数据到: csv/kronos_btc_finetuned.csv

📊 使用的模型信息:
   - Tokenizer: 微调后的BTC专用tokenizer
   - Predictor: 预训练的通用predictor
   - 模型状态: 部分微调（仅tokenizer）
   - 设备: mps
   - 历史窗口: 400 天
   - 预测窗口: 120 天
```

## 微调模型的优势

### 微调后的Tokenizer优势：

1. **专门针对BTC数据优化**: 更好地理解BTC价格模式
2. **改进的数据编码**: 针对加密货币特有的波动性进行优化
3. **更准确的特征提取**: 能够捕捉BTC特有的市场行为

### 完全微调模型的优势（当predictor也完成训练时）：

1. **端到端优化**: 整个模型都针对BTC数据进行了优化
2. **更好的预测精度**: 理论上应该提供更准确的预测结果
3. **领域特化**: 专门为加密货币市场设计

## 如何训练Predictor模型

如果你想要完成predictor的微调训练，可以运行：

```bash
# 使用分布式训练（推荐）
torchrun --standalone --nproc_per_node=1 finetune/train_predictor.py

# 或者根据你的GPU数量调整
torchrun --standalone --nproc_per_node=2 finetune/train_predictor.py
```

## 性能对比建议

建议你对比以下几种模型的预测效果：

1. **原始预训练模型** (`examples/bitcoin_forecast.py`)
2. **微调tokenizer + 预训练predictor** (`examples/bitcoin_forecast_finetuned.py`)
3. **完全微调模型** (当predictor训练完成后)

通过对比不同模型的预测结果，你可以评估微调的效果。

## 注意事项

1. **路径问题**: 确保脚本从正确的目录运行，路径设置正确
2. **数据要求**: 确保有足够的历史数据（至少400天）
3. **设备兼容**: 脚本会自动检测并使用最佳设备（MPS/CPU）
4. **内存使用**: 微调后的模型可能需要更多内存

## 故障排除

如果遇到问题，请检查：

1. **模型文件是否存在**: 确认 `outputs/models/btc_finetune_tokenizer/checkpoints/best_model/` 目录存在
2. **依赖包版本**: 确保所有依赖包版本兼容
3. **数据文件**: 确认 `csv/btc_usdt_1d_no_time.csv` 文件存在且格式正确
4. **设备支持**: 确认你的设备支持所选的计算设备

## 总结

目前你可以使用微调后的tokenizer结合预训练的predictor进行BTC预测，这已经能够提供比纯预训练模型更好的效果。如果需要进一步提升性能，可以继续训练predictor模型以获得完全微调的解决方案。