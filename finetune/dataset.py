import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from config import Config
except ImportError:
    Config = None
try:
    from btc_config import BTCConfig
except ImportError:
    BTCConfig = None


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.
        config (object, optional): Configuration object. If None, will try to load Config or BTCConfig.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train', config=None):
        if config is not None:
            self.config = config
        elif BTCConfig is not None:
            self.config = BTCConfig()
        elif Config is not None:
            self.config = Config()
        else:
            raise ImportError("Neither Config nor BTCConfig could be imported")
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        print(f"正在加载数据文件: {self.data_path}")
        with open(self.data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        print(f"数据类型: {type(raw_data)}")
        
        # 处理不同的数据格式
        if isinstance(raw_data, dict):
            print(f"数据包含 {len(raw_data)} 个符号")
            self.data = raw_data
            self.symbols = list(self.data.keys())
        elif isinstance(raw_data, list):
            print(f"数据是列表格式，包含 {len(raw_data)} 个样本")
            # 将列表格式转换为字典格式，使用单一符号 'BTC'
            self.data = {'BTC': raw_data}
            self.symbols = ['BTC']
        else:
            print(f"❌ 数据格式错误: 不支持的数据类型 {type(raw_data)}")
            raise TypeError(f"数据格式错误: 不支持的数据类型 {type(raw_data)}")

        self.window = self.config.lookback_window + self.config.predict_window + 1
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        
        for symbol in self.symbols:
            data_item = self.data[symbol]
            
            # 处理列表格式的数据
            if isinstance(data_item, list):
                print(f"处理符号 {symbol} 的列表数据，包含 {len(data_item)} 个样本")
                # 假设每个样本已经是处理好的格式
                for i, sample in enumerate(data_item):
                    self.indices.append((symbol, i))
            else:
                # 处理DataFrame格式的数据（原有逻辑）
                df = data_item.reset_index()
                series_len = len(df)
                num_samples = series_len - self.window + 1

                if num_samples > 0:
                    # Generate time features and store them directly in the dataframe.
                    df['minute'] = df['datetime'].dt.minute
                    df['hour'] = df['datetime'].dt.hour
                    df['weekday'] = df['datetime'].dt.weekday
                    df['day'] = df['datetime'].dt.day
                    df['month'] = df['datetime'].dt.month
                    # Keep only necessary columns to save memory.
                    self.data[symbol] = df[self.feature_list + self.time_feature_list]

                    # Add all valid starting indices for this symbol to the global list.
                    for i in range(num_samples):
                        self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`. This
        ensures random sampling over the entire dataset for each call.

        Args:
            idx (int): Ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
                - x_stamp_tensor (torch.Tensor): The time feature tensor.
        """
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, sample_idx = self.indices[random_idx]

        data_item = self.data[symbol]
        
        # 处理列表格式的数据
        if isinstance(data_item, list):
            # 直接从列表中获取样本
            sample = data_item[sample_idx]
            
            # 处理BTC数据格式
            if isinstance(sample, dict) and 'input_data' in sample:
                # 使用input_data作为特征
                x = sample['input_data'].astype(np.float32)
                
                # 从时间戳生成时间特征
                timestamps = sample['input_timestamps']
                x_stamp = []
                for ts in timestamps:
                    time_features = [
                        ts.minute,
                        ts.hour,
                        ts.weekday(),
                        ts.day,
                        ts.month
                    ]
                    x_stamp.append(time_features)
                x_stamp = np.array(x_stamp, dtype=np.float32)
            else:
                # 如果样本格式不明确，尝试直接使用
                x = np.array(sample, dtype=np.float32)
                # 创建虚拟时间特征
                seq_len = x.shape[0] if x.ndim > 1 else 1
                x_stamp = np.zeros((seq_len, len(self.time_feature_list)), dtype=np.float32)
        else:
            # 处理DataFrame格式的数据（原有逻辑）
            df = data_item
            end_idx = sample_idx + self.window
            win_df = df.iloc[sample_idx:end_idx]

            # Separate main features and time features.
            x = win_df[self.feature_list].values.astype(np.float32)
            x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Perform instance-level normalization.
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")
