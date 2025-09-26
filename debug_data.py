import pickle
import numpy as np

# 检查训练数据的格式
with open('./data/processed_datasets/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"数据类型: {type(data)}")
print(f"数据长度: {len(data)}")

if len(data) > 0:
    sample = data[0]
    print(f"第一个样本类型: {type(sample)}")
    print(f"第一个样本内容: {sample}")
    
    if isinstance(sample, dict):
        print(f"字典键: {sample.keys()}")
    elif isinstance(sample, (list, tuple)):
        print(f"列表/元组长度: {len(sample)}")
        if len(sample) > 0:
            print(f"第一个元素类型: {type(sample[0])}")
            print(f"第一个元素: {sample[0]}")
    elif isinstance(sample, np.ndarray):
        print(f"数组形状: {sample.shape}")
        print(f"数组内容: {sample}")