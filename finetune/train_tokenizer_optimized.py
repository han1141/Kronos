#!/usr/bin/env python3
"""
优化版BTC tokenizer训练脚本
包含早停机制、增强监控和数据增强功能
"""
import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.btc_config_optimized import BTCConfigOptimized
from finetune.dataset import QlibDataset
from model.kronos import KronosTokenizer


class EarlyStopping:
    """早停机制实现"""
    def __init__(self, patience=5, min_delta=1e-5, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class TrainingMonitor:
    """训练监控器"""
    def __init__(self, config):
        self.config = config
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'reconstruction_loss': [],
            'quantization_loss': [],
            'gradient_norm': []
        }
    
    def log_metrics(self, epoch, metrics):
        """记录训练指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # 打印详细指标
        print(f"\n--- Epoch {epoch} 详细指标 ---")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    
    def save_metrics(self, save_path):
        """保存训练指标到文件"""
        metrics_file = os.path.join(save_path, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"训练指标已保存到: {metrics_file}")


def apply_data_augmentation(batch_x, config):
    """应用数据增强"""
    if not config.data_augmentation['enabled']:
        return batch_x
    
    augmented_x = batch_x.clone()
    
    # 添加高斯噪声
    if config.data_augmentation['noise_std'] > 0:
        noise = torch.randn_like(augmented_x) * config.data_augmentation['noise_std']
        augmented_x += noise
    
    # 随机缩放
    scale_range = config.data_augmentation['scale_factor_range']
    scale_factor = torch.uniform(scale_range[0], scale_range[1], (1,)).item()
    augmented_x *= scale_factor
    
    return augmented_x


def create_optimized_scheduler(optimizer, config, steps_per_epoch):
    """创建优化的学习率调度器"""
    scheduler_config = config.get_scheduler_config(optimizer, steps_per_epoch)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=scheduler_config['max_lr'],
        steps_per_epoch=steps_per_epoch,
        epochs=config.epochs,
        pct_start=scheduler_config['pct_start'],
        div_factor=scheduler_config['div_factor'],
        final_div_factor=scheduler_config['final_div_factor'],
        anneal_strategy=scheduler_config['anneal_strategy']
    )
    
    return scheduler


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
        pin_memory=False,
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
    """优化版训练函数"""
    start_time = time.time()
    print(f"批次大小: {config.batch_size}")
    
    # 打印优化配置摘要
    config.print_optimization_summary()
    
    train_loader, val_loader, train_dataset, valid_dataset = create_dataloaders(config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )
    
    # 使用优化的学习率调度器
    scheduler = create_optimized_scheduler(optimizer, config, len(train_loader))
    
    # 初始化早停机制
    early_stopping = None
    if config.early_stopping_enabled:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min'
        )
        print(f"早停机制已启用: patience={config.early_stopping_patience}")
    
    # 初始化训练监控器
    monitor = TrainingMonitor(config)
    
    best_val_loss = float('inf')
    batch_idx_global_train = 0
    saved_models = []  # 保存的模型列表
    
    for epoch_idx in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        
        # 设置数据集种子
        train_dataset.set_epoch_seed(epoch_idx * 10000)
        valid_dataset.set_epoch_seed(0)
        
        print(f"\n=== Epoch {epoch_idx + 1}/{config.epochs} ===")
        
        # 训练循环
        epoch_train_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_quant_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0
        
        for i, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
            
            # 应用数据增强
            if config.data_augmentation['enabled']:
                ori_batch_x = apply_data_augmentation(ori_batch_x, config)
            
            # 梯度累积循环
            current_batch_total_loss = 0.0
            current_batch_recon_loss = 0.0
            current_batch_quant_loss = 0.0
            
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
                    current_batch_recon_loss += recon_loss.item()
                    current_batch_quant_loss += bsq_loss.item()
                    loss_scaled.backward()
                    
                except Exception as e:
                    print(f"前向传播错误: {e}")
                    continue
            
            # 计算梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config.gradient_clip_norm
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 累积统计
            epoch_train_loss += current_batch_total_loss / config.accumulation_steps
            epoch_recon_loss += current_batch_recon_loss / config.accumulation_steps
            epoch_quant_loss += current_batch_quant_loss / config.accumulation_steps
            epoch_grad_norm += grad_norm.item()
            num_batches += 1
            
            # 记录日志
            if (batch_idx_global_train + 1) % config.log_interval == 0:
                avg_loss = current_batch_total_loss / config.accumulation_steps
                print(
                    f"[Epoch {epoch_idx + 1}/{config.epochs}, Step {i + 1}/{len(train_loader)}] "
                    f"LR {optimizer.param_groups[0]['lr']:.6f}, Loss: {avg_loss:.4f}, "
                    f"Recon: {current_batch_recon_loss/config.accumulation_steps:.4f}, "
                    f"Quant: {current_batch_quant_loss/config.accumulation_steps:.4f}, "
                    f"GradNorm: {grad_norm:.4f}"
                )
            
            batch_idx_global_train += 1
        
        # 计算平均训练指标
        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0
        avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0
        avg_quant_loss = epoch_quant_loss / num_batches if num_batches > 0 else 0
        avg_grad_norm = epoch_grad_norm / num_batches if num_batches > 0 else 0
        
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
        
        # 记录训练指标
        current_lr = optimizer.param_groups[0]['lr']
        metrics = {
            'epoch': epoch_idx + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': current_lr,
            'reconstruction_loss': avg_recon_loss,
            'quantization_loss': avg_quant_loss,
            'gradient_norm': avg_grad_norm
        }
        monitor.log_metrics(epoch_idx + 1, metrics)
        
        # Epoch总结
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        print(f"\n--- Epoch {epoch_idx + 1}/{config.epochs} 总结 ---")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"学习率: {current_lr:.6f}")
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
                
                # 管理保存的模型数量
                saved_models.append((avg_val_loss, save_path))
                saved_models.sort(key=lambda x: x[0])  # 按损失排序
                
                # 只保留前k个最佳模型
                if len(saved_models) > config.validation_config['save_top_k']:
                    saved_models = saved_models[:config.validation_config['save_top_k']]
                
            except Exception as e:
                print(f"模型保存失败: {e}")
        
        # 早停检查
        if early_stopping is not None:
            if early_stopping(avg_val_loss):
                print(f"\n🛑 早停触发！验证损失连续{config.early_stopping_patience}轮未改善")
                print(f"最佳验证损失: {best_val_loss:.4f}")
                break
    
    # 保存训练指标
    monitor.save_metrics(save_dir)
    
    return {
        'best_val_loss': best_val_loss,
        'total_epochs': epoch_idx + 1,
        'early_stopped': early_stopping.early_stop if early_stopping else False
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='btc_config_optimized', help='配置模块名')
    parser.add_argument('--test', action='store_true', help='测试模式')
    args = parser.parse_args()
    
    # 加载优化配置
    config = BTCConfigOptimized()
    
    if args.test:
        print("=== 测试模式 ===")
        print("优化配置加载成功")
        config.print_optimization_summary()
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置保存目录
    save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    
    print("开始优化版训练...")
    print("新增功能: 早停机制、数据增强、增强监控")
    
    try:
        # 尝试加载预训练模型
        print(f"尝试加载预训练模型: {config.pretrained_tokenizer_path}")
        
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
                
                model = KronosTokenizer(
                    d_in=6,
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
        
        print(f"\n🎉 训练完成！")
        print(f"最佳验证损失: {result['best_val_loss']:.4f}")
        print(f"实际训练轮数: {result['total_epochs']}/{config.epochs}")
        if result['early_stopped']:
            print("✅ 通过早停机制提前结束训练")
        
    except Exception as e:
        print(f"❌ 模型加载或训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()