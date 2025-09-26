#!/usr/bin/env python3
"""
单进程BTC tokenizer训练脚本 - 适用于Mac环境
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.btc_config import BTCConfig
from finetune.dataset import QlibDataset
from model.kronos import KronosTokenizer

def create_dataloaders(config):
    """创建数据加载器"""
    print("创建数据加载器...")
    try:
        print("正在创建训练数据集...")
        train_dataset = QlibDataset('train', config=config)
        print(f"训练数据集创建成功，大小: {len(train_dataset)}")
        
        print("正在创建验证数据集...")
        valid_dataset = QlibDataset('val', config=config)
        print(f"验证数据集创建成功，大小: {len(valid_dataset)}")
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise e

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,  # Mac上设为False
        drop_last=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=False
    )
    print(f"数据加载器创建完成。训练步数/epoch: {len(train_loader)}, 验证步数: {len(val_loader)}")
    return train_loader, val_loader, train_dataset, valid_dataset

def train_model(model, device, config, save_dir):
    """训练模型"""
    start_time = time.time()
    print(f"批次大小: {config.batch_size}")
    
    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        pct_start=0.03,
        div_factor=10
    )
    
    best_val_loss = float('inf')
    batch_idx_global_train = 0
    
    for epoch_idx in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        
        # 设置数据集种子
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        valid_dataset.set_epoch_seed(0)
        
        print(f"\n=== Epoch {epoch_idx + 1}/{config.epochs} ===")
        
        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
            
            # 梯度累积循环
            current_batch_total_loss = 0.0
            for j in range(config.accumulation_steps):
                start_idx = j * (ori_batch_x.shape[0] // config.accumulation_steps)
                end_idx = (j + 1) * (ori_batch_x.shape[0] // config.accumulation_steps)
                batch_x = ori_batch_x[start_idx:end_idx]
                
                # 前向传播
                try:
                    zs, bsq_loss, _, _ = model(batch_x)
                    z_pre, z = zs
                    
                    # 损失计算
                    recon_loss_pre = F.mse_loss(z_pre, batch_x)
                    recon_loss_all = F.mse_loss(z, batch_x)
                    recon_loss = recon_loss_pre + recon_loss_all
                    loss = (recon_loss + bsq_loss) / 2
                    
                    loss_scaled = loss / config.accumulation_steps
                    current_batch_total_loss += loss.item()
                    loss_scaled.backward()
                    
                except Exception as e:
                    print(f"前向传播错误: {e}")
                    continue
            
            # 优化器步骤
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 记录日志
            if (batch_idx_global_train + 1) % config.log_interval == 0:
                avg_loss = current_batch_total_loss / config.accumulation_steps
                print(
                    f"[Epoch {epoch_idx + 1}/{config.epochs}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}"
                )
            
            batch_idx_global_train += 1
        
        # 验证循环
        model.eval()
        tot_val_loss = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                try:
                    zs, _, _, _ = model(ori_batch_x)
                    _, z = zs
                    val_loss_item = F.mse_loss(z, ori_batch_x)
                    
                    tot_val_loss += val_loss_item.item() * ori_batch_x.size(0)
                    val_sample_count += ori_batch_x.size(0)
                except Exception as e:
                    print(f"验证错误: {e}")
                    continue
        
        avg_val_loss = tot_val_loss / val_sample_count if val_sample_count > 0 else float('inf')
        
        # Epoch总结
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print(f"\n--- Epoch {epoch_idx + 1}/{config.epochs} 总结 ---")
        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"本轮用时: {epoch_time:.2f}秒")
        print(f"总用时: {total_time:.2f}秒")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model"
            os.makedirs(save_path, exist_ok=True)
            try:
                model.save_pretrained(save_path)
                print(f"最佳模型已保存到 {save_path} (验证损失: {best_val_loss:.4f})")
            except Exception as e:
                print(f"模型保存失败: {e}")
    
    return {'best_val_loss': best_val_loss}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='btc_config', help='配置模块名')
    parser.add_argument('--test', action='store_true', help='测试模式')
    args = parser.parse_args()
    
    # 加载配置
    config = BTCConfig()
    
    if args.test:
        print("=== 测试模式 ===")
        print("配置加载成功")
        print(f"预训练模型路径: {config.pretrained_tokenizer_path}")
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置保存目录
    save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    
    print("开始单进程训练...")
    print("注意: 这是一个简化版本，用于Mac环境测试")
    
    try:
        # 尝试加载预训练模型
        print(f"尝试加载预训练模型: {config.pretrained_tokenizer_path}")
        
        # 检查是否为本地路径
        if os.path.exists(config.pretrained_tokenizer_path):
            print("检测到本地模型路径")
            model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
        else:
            print("检测到Hugging Face模型标识符，尝试从Hub下载...")
            try:
                model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
            except Exception as e:
                print(f"⚠️ 从Hugging Face加载失败: {e}")
                print("使用默认参数初始化新的KronosTokenizer...")
                
                # 使用默认参数初始化模型
                model = KronosTokenizer(
                    d_in=6,  # BTC特征数量 (open, high, low, close, volume, amount)
                    d_model=512,
                    n_heads=8,
                    ff_dim=2048,
                    n_enc_layers=6,
                    n_dec_layers=6,
                    ffn_dropout_p=0.1,
                    attn_dropout_p=0.1,
                    resid_dropout_p=0.1,
                    s1_bits=8,
                    s2_bits=8,
                    beta=0.25,
                    gamma0=1.0,
                    gamma=0.99,
                    zeta=1e-4,
                    group_size=1
                )
                print("✅ 使用默认参数初始化模型成功")
        
        model.to(device)
        print("✅ 模型加载/初始化成功")
        
        # 开始训练
        result = train_model(model, device, config, save_dir)
        print(f"训练完成！最佳验证损失: {result['best_val_loss']:.4f}")
        
    except Exception as e:
        print(f"❌ 模型加载或训练失败: {e}")
        print("请检查:")
        print("1. 预训练模型路径是否正确")
        print("2. 模型文件是否完整")
        print("3. 依赖库是否正确安装")

if __name__ == '__main__':
    main()