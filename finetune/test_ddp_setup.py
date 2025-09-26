#!/usr/bin/env python3
"""
测试分布式训练设置的简单脚本
"""
import os
import sys
import torch
import torch.distributed as dist

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.utils.training_utils import setup_ddp, cleanup_ddp, set_seed

def test_ddp():
    """测试DDP设置"""
    print("=== 测试分布式训练设置 ===")
    
    try:
        # 设置DDP
        rank, world_size, local_rank = setup_ddp()
        
        # 设置设备
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            print(f"[Rank {rank}] 使用GPU设备: {device}")
        else:
            device = torch.device("cpu")
            print(f"[Rank {rank}] 使用CPU设备: {device}")
        
        # 设置随机种子
        set_seed(42, rank)
        print(f"[Rank {rank}] 随机种子设置完成")
        
        # 创建一个简单的测试张量
        test_tensor = torch.randn(3, 4).to(device)
        print(f"[Rank {rank}] 测试张量形状: {test_tensor.shape}")
        
        # 测试all_reduce操作
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] All-reduce操作完成")
        
        # 同步所有进程
        dist.barrier()
        print(f"[Rank {rank}] 进程同步完成")
        
        if rank == 0:
            print("✅ 分布式训练设置测试成功！")
        
    except Exception as e:
        print(f"❌ 分布式训练设置测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    if "WORLD_SIZE" not in os.environ:
        print("错误: 此脚本必须使用 torchrun 启动")
        print("使用方法: torchrun --standalone --nproc_per_node=1 finetune/test_ddp_setup.py")
        exit(1)
    
    test_ddp()