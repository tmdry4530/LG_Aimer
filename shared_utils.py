# shared_utils.py
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

# --- 분산 학습 설정 ---
def setup_distributed(backend='nccl'):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup():
    dist.destroy_process_group()

# --- 재현성 설정 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 특성 공학 ---
@torch.no_grad()
def get_korean_holidays(years):
    holidays = {
        '2023': ['2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24', '2023-03-01', '2023-05-05', '2023-05-27', '2023-06-06', '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-03', '2023-10-09', '2023-12-25'],
        '2024': ['2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12', '2024-03-01', '2024-04-10', '2024-05-05', '2024-05-06', '2024-05-15', '2024-06-06', '2024-08-15', '2024-09-16', '2024-09-17', '2024-09-18', '2024-10-03', '2024-10-09', '2024-12-25']
    }
    all_holidays = []
    for year in years:
        if str(year) in holidays:
            all_holidays.extend(holidays[str(year)])
    return pd.to_datetime(all_holidays)

def feature_engineering(df):
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['dayofweek'] = df['영업일자'].dt.dayofweek
    df['month'] = df['영업일자'].dt.month
    df['year'] = df['영업일자'].dt.year
    df['dayofyear'] = df['영업일자'].dt.dayofyear
    df['weekofyear'] = df['영업일자'].dt.isocalendar().week.astype(int)
    holidays = get_korean_holidays(df['year'].unique())
    df['is_holiday'] = df['영업일자'].isin(holidays).astype(int)
    df = df.sort_values(by=['영업장명_메뉴명', '영업일자'])
    grouped = df.groupby('영업장명_메뉴명')['매출수량']
    df['lag_7'] = grouped.shift(7).fillna(0)
    df['rolling_mean_7'] = grouped.rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0)
    df['rolling_std_7'] = grouped.rolling(window=7, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    return df

# --- 데이터셋 클래스 ---
class SalesDataset(Dataset):
    def __init__(self, df, item_encoder, scalers, lookback, predict_days, is_train=True):
        self.lookback = lookback
        self.predict_days = predict_days
        self.data = []
        self.time_features = ['dayofweek', 'month', 'year', 'dayofyear', 'weekofyear', 'is_holiday']
        self.numeric_features = ['매출수량', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
        
        total_len = lookback + predict_days if is_train else lookback

        for item_id, group in df.groupby('item_id'):
            if item_id not in scalers: continue
            
            group_scaled = scalers[item_id].transform(group[self.numeric_features])
            all_features = np.hstack([group_scaled, group[self.time_features].values, group[['item_id']].values])

            if is_train:
                for i in range(len(all_features) - total_len + 1):
                    x = all_features[i : i + lookback]
                    y = all_features[i + lookback : i + total_len, 0]
                    self.data.append((x, y))
            else: # For prediction
                if len(all_features) >= lookback:
                    x = all_features[-lookback:]
                    self.data.append((x, item_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]