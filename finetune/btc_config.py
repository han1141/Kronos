import os

class BTCConfig:
    """
    BTC数据微调配置类
    """
    def __init__(self):
        # =================================================================
        # 数据和特征参数
        # =================================================================
        self.data_source = "csv"
        self.csv_data_path = "./csv/btc_usdt_1d_no_time.csv"
        self.dataset_begin_time = "2022-09-27"  # 根据实际数据调整
        self.dataset_end_time = "2025-09-26"
        
        # 时间窗口参数
        self.lookback_window = 90   # 历史窗口长度
        self.predict_window = 10    # 预测窗口长度
        self.max_context = 512      # 模型最大上下文
        
        # BTC特征列
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        # 数据集路径
        self.dataset_path = "./data/processed_datasets"
        
        # =================================================================
        # 训练超参数
        # =================================================================
        self.clip = 5.0  # 标准化数据的裁剪值
        
        self.epochs = 30
        self.log_interval = 100  # 每N个批次记录训练状态
        self.batch_size = 50  # 每个GPU的批次大小
        
        # 每个"epoch"的训练/验证样本数
        self.n_train_iter = 2000 * self.batch_size
        self.n_val_iter = 400 * self.batch_size
        
        # 不同模型组件的学习率
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5
        
        # 梯度累积步数
        self.accumulation_steps = 1
        
        # AdamW优化器参数
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        
        # 其他参数
        self.seed = 100  # 全局随机种子
        self.num_workers = 2  # 数据加载器工作进程数
        
        # =================================================================
        # 实验记录和保存
        # =================================================================
        self.use_comet = False  # BTC微调暂时不使用Comet ML
        self.comet_config = {
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-BTC-Finetune",
            "workspace": "your_comet_workspace"
        }
        self.comet_tag = 'btc_finetune'
        self.comet_name = 'btc_finetune'
        
        # 模型检查点和结果保存目录
        self.save_path = "./outputs/models"
        self.tokenizer_save_folder_name = 'btc_finetune_tokenizer'
        self.predictor_save_folder_name = 'btc_finetune_predictor'
        self.backtest_save_folder_name = 'btc_finetune_backtest'
        
        # 回测结果路径
        self.backtest_result_path = "./outputs/backtest_results"
        
        # =================================================================
        # 模型和检查点路径
        # =================================================================
        # Hugging Face 预训练模型路径
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"
        
        # 微调后的模型路径
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"
        
        # =================================================================
        # 回测参数
        # =================================================================
        self.backtest_n_symbol_hold = 1  # BTC只有一个符号
        self.backtest_n_symbol_drop = 0
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = "BTCUSDT"  # BTC基准