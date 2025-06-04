from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.augmentation import run_augmentation_single
import os
import pandas as pd
import numpy as np
import torch
class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.assets = []
        self.channels = []
        self.n_assetes = 0
        self.n_channels = 0
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_parquet(os.path.join(self.root_path,
                                          self.data_path))
        self.assets = df_raw.columns[1:].get_level_values(0).unique().values  
        self.n_assetes = len(self.assets)
        self.channels = df_raw.columns[1:].get_level_values(1).unique()
        self.n_channels = len(self.channels)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns)
        # cols.remove(self.target)
        # cols.remove('date')
        # df_raw = df_raw[['date'] + cols + [self.target]]
        df_raw[('date','')] = df_raw.index
        new_columns = [('date','')] + list(df_raw.columns[:-1])
        df_raw = df_raw[new_columns]
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.03)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        self.timestamp = df_stamp.copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # 使用矢量化操作提取时间特征
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).to_numpy()  # 使用 to_numpy() 替代 values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        timestamp = self.timestamp.iloc[s_end].values[0]
        #timestamp = np.datetime64(timestamp).astype('datetime64[ms]').astype(np.int64)
        #reshape seq_x to [seq_len, n_assets, n_channels]
        seq_x = seq_x.reshape(self.seq_len, self.n_assetes, self.n_channels)
        #permute seq_x to [n_channels, seq_len, n_assets]
        seq_x = np.transpose(seq_x, (2, 0, 1))
        seq_y = seq_y.reshape(self.label_len + self.pred_len, self.n_assetes, self.n_channels)
        seq_y = np.transpose(seq_y, (2, 0, 1))
        return seq_x, seq_y, timestamp, self.assets, seq_x_mark, seq_y_mark #, self.channels  

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def custom_collate_fn(batch):
    # 解包 batch
    X_batch, y_batch, timestamps_batch, assets, X_mark, y_mark = zip(*batch)
    
    # 将 X 和 y 转换为张量
    X_batch = torch.stack([torch.as_tensor(x) for x in X_batch])
    y_batch = torch.stack([torch.as_tensor(y) for y in y_batch])
    
    # 将 timestamps 转换为 Unix 时间戳 (int64) 或保留为列表
    timestamps_batch = [ts for ts in timestamps_batch]
    
    X_mark = torch.stack([torch.as_tensor(xm) for xm in X_mark])
    y_mark = torch.stack([torch.as_tensor(ym) for ym in y_mark])
    
    # assets 保持为列表或元组（假设为非张量数据，如字符串）
    return X_batch, y_batch, timestamps_batch, assets, X_mark, y_mark