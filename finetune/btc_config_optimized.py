import os

class BTCConfigOptimized:
    """
    BTC数据微调优化配置类
    基于训练效果分析的优化版本
    """
    def __init__(self):
        # =================================================================
        # 数据和特征参数
        # =================================================================
        self.data_source = "csv"
        self.csv_data_path = "./csv/btc_usdt_1d_no_time.csv"
        self.dataset_begin_time = "2022-09-27"
        self.dataset_end_time = "2025-09-26"
        
        # 时间窗口参数
        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512
        
        # BTC特征列
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        # 数据集路径
        self.dataset_path = "./data/processed_datasets"
        
        # =================================================================
        # 训练超参数 - 优化版本
        # =================================================================
        self.clip = 5.0
        
        self.epochs = 30
        self.log_interval = 100
        self.batch_size = 64  # 优化: 从50增加到64
        
        # 每个"epoch"的训练/验证样本数
        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size
        
        # 优化: 降低学习率提高稳定性
        self.tokenizer_learning_rate = 1e-4  # 从2e-4降至1e-4
        self.predictor_learning_rate = 4e-5
        
        # 梯度累积步数
        self.accumulation_steps = 1
        
        # AdamW优化器参数 - 优化版本
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.05  # 优化: 从0.1降至0.05
        
        # 其他参数
        self.seed = 100
        self.num_workers = 2
        
        # =================================================================
        # 新增: 早停机制配置
        # =================================================================
        self.early_stopping_patience = 5  # 验证损失5轮不改善则停止
        self.early_stopping_min_delta = 1e-5  # 最小改善阈值
        self.early_stopping_enabled = True
        
        # =================================================================
        # 新增: 学习率调度优化配置
        # =================================================================
        self.scheduler_config = {
            'type': 'OneCycleLR',
            'pct_start': 0.1,        # 优化: 从0.03增加到0.1
            'div_factor': 25,        # 优化: 从10增加到25
            'final_div_factor': 100, # 新增: 最终衰减因子
            'anneal_strategy': 'cos'
        }
        
        # =================================================================
        # 新增: 梯度裁剪优化
        # =================================================================
        self.gradient_clip_norm = 1.0  # 优化: 从2.0降至1.0
        
        # =================================================================
        # 新增: 数据增强配置
        # =================================================================
        self.data_augmentation = {
            'enabled': True,
            'noise_std': 0.01,      # 高斯噪声标准差
            'dropout_rate': 0.1,    # 随机特征丢弃率
            'time_shift_max': 2,    # 最大时间偏移
            'scale_factor_range': (0.95, 1.05)  # 缩放因子范围
        }
        
        # =================================================================
        # 新增: 验证策略优化
        # =================================================================
        self.validation_config = {
            'interval': 1,          # 每轮都验证
            'save_top_k': 3,        # 保存前3个最佳模型
            'monitor_metric': 'val_loss',
            'mode': 'min'
        }
        
        # =================================================================
        # 新增: 监控指标配置
        # =================================================================
        self.monitoring_metrics = {
            'reconstruction_loss': True,
            'quantization_loss': True,
            'learning_rate': True,
            'gradient_norm': True,
            'model_weights_norm': True
        }
        
        # =================================================================
        # 实验记录和保存
        # =================================================================
        self.use_comet = False
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-BTC-Finetune-Optimized",
            "workspace": "your_comet_workspace"
        }
        self.comet_tag = 'btc_finetune_optimized'
        self.comet_name = 'btc_finetune_optimized'
        
        # 模型检查点和结果保存目录
        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = 'btc_finetune_tokenizer_optimized'
        self.predictor_save_folder_name = 'btc_finetune_predictor_optimized'
        self.backtest_save_folder_name = 'btc_finetune_backtest_optimized'
        
        # 回测结果路径
        self.backtest_result_path = "./outputs/backtest_results"
        
        # =================================================================
        # 模型和检查点路径
        # =================================================================
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"
        
        # 微调后的模型路径
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        
        # =================================================================
        # 新增: 混合精度训练配置 (需要GPU支持)
        # =================================================================
        self.mixed_precision = {
            'enabled': False,  # CPU环境下设为False
            'opt_level': 'O1',
            'loss_scale': 'dynamic'
        }
        
        # =================================================================
        # 新增: 分布式训练配置
        # =================================================================
        self.distributed_training = {
            'enabled': False,  # 单机训练设为False
            'backend': 'nccl',
            'world_size': 1,
            'rank': 0
        }
        
        # =================================================================
        # 回测参数
        # =================================================================
        self.backtest_n_symbol_hold = 1
        self.backtest_n_symbol_drop = 0
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = "BTCUSDT"
        
    def get_scheduler_config(self, optimizer, steps_per_epoch):
        """
        获取优化后的学习率调度器配置
        """
        if self.scheduler_config['type'] == 'OneCycleLR':
            return {
                'scheduler': 'OneCycleLR',
                'optimizer': optimizer,
                'max_lr': self.tokenizer_learning_rate,
                'steps_per_epoch': steps_per_epoch,
                'epochs': self.epochs,
                'pct_start': self.scheduler_config['pct_start'],
                'div_factor': self.scheduler_config['div_factor'],
                'final_div_factor': self.scheduler_config['final_div_factor'],
                'anneal_strategy': self.scheduler_config['anneal_strategy']
            }
        else:
            raise ValueError(f"不支持的调度器类型: {self.scheduler_config['type']}")
    
    def print_optimization_summary(self):
        """
        打印优化配置摘要
        """
        print("=== BTC微调优化配置摘要 ===")
        print(f"学习率: {self.tokenizer_learning_rate} (优化: 从2e-4降至1e-4)")
        print(f"批次大小: {self.batch_size} (优化: 从50增至64)")
        print(f"权重衰减: {self.adam_weight_decay} (优化: 从0.1降至0.05)")
        print(f"梯度裁剪: {self.gradient_clip_norm} (优化: 从2.0降至1.0)")
        print(f"早停机制: {'启用' if self.early_stopping_enabled else '禁用'}")
        print(f"数据增强: {'启用' if self.data_augmentation['enabled'] else '禁用'}")
        print(f"学习率预热: {self.scheduler_config['pct_start']} (优化: 从0.03增至0.1)")
        print("=" * 40)