# train_nbeats.py
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.preprocessing import MinMaxScaler

# 공유 유틸리티 임포트
from shared_utils import setup_distributed, cleanup, set_seed

# --- 하이퍼파라미터 ---
LOOKBACK_DAYS = 28
PREDICT_DAYS = 7
BATCH_SIZE = 256
EPOCHS = 100
HIDDEN_UNITS = 256
NUM_BLOCKS = 3
NUM_LAYERS = 4
THETA_DIM = 32
LEARNING_RATE = 1e-3

# --- N-BEATS 모델 아키텍처 ---
class NBEATSBlock(nn.Module):
    def __init__(self, input_size, theta_dim, hidden_units, num_layers, output_size):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_units, hidden_units), nn.ReLU()])
        self.layers = nn.Sequential(*layers)
        self.theta_f = nn.Linear(hidden_units, theta_dim, bias=False)
        self.theta_b = nn.Linear(hidden_units, theta_dim, bias=False)
        self.forecast_layer = nn.Linear(theta_dim, output_size)
        self.backcast_layer = nn.Linear(theta_dim, input_size)

    def forward(self, x):
        hidden = self.layers(x)
        theta_f = self.theta_f(hidden)
        theta_b = self.theta_b(hidden)
        return self.backcast_layer(theta_b), self.forecast_layer(theta_f)

class NBEATS(nn.Module):
    def __init__(self, num_blocks, num_layers, input_size, output_size, hidden_units, theta_dim):
        super().__init__()
        self.blocks = nn.ModuleList([NBEATSBlock(input_size, theta_dim, hidden_units, num_layers, output_size) for _ in range(num_blocks)])

    def forward(self, x):
        residuals = x
        forecast = torch.zeros_like(x[:, :PREDICT_DAYS])
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast

# --- N-BEATS 전용 데이터셋 ---
class NBEATSDataset(Dataset):
    def __init__(self, df, lookback, predict_days, is_train=True):
        self.data = []
        self.scalers = {}
        
        for item, group in df.groupby('영업장명_메뉴명'):
            scaler = MinMaxScaler()
            sales = group['매출수량'].values.reshape(-1, 1)
            sales_scaled = scaler.fit_transform(sales)
            self.scalers[item] = scaler
            
            if is_train:
                for i in range(len(sales_scaled) - (lookback + predict_days) + 1):
                    x = sales_scaled[i : i + lookback].flatten()
                    y = sales_scaled[i + lookback : i + lookback + predict_days].flatten()
                    self.data.append({'x': x, 'y': y, 'item': item})
            else:
                if len(sales_scaled) >= lookback:
                    x = sales_scaled[-lookback:].flatten()
                    self.data.append({'x': x, 'item': item})

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def main():
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    set_seed(42)

    if rank == 0: print("N-BEATS: Loading and preprocessing data...")
    train_df = pd.read_csv('./data/train/train.csv')
    train_dataset = NBEATSDataset(train_df, LOOKBACK_DAYS, PREDICT_DAYS, is_train=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)

    model = NBEATS(NUM_BLOCKS, NUM_LAYERS, LOOKBACK_DAYS, PREDICT_DAYS, HIDDEN_UNITS, THETA_DIM).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if rank == 0: print("N-BEATS: Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        for data in train_loader:
            x = data['x'].float().to(device)
            y = data['y'].float().to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"N-BEATS Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    if rank == 0: print("N-BEATS: Starting prediction...")
    model.eval()
    all_predictions = []
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))
    files_per_rank = [f for i, f in enumerate(test_files) if i % world_size == rank]

    with torch.no_grad():
        for file_path in files_per_rank:
            test_df = pd.read_csv(file_path)
            test_prefix = os.path.basename(file_path).split('.')[0]
            test_dataset = NBEATSDataset(test_df, LOOKBACK_DAYS, PREDICT_DAYS, is_train=False)
            
            for data in test_dataset:
                x_test = torch.FloatTensor(data['x']).unsqueeze(0).to(device)
                item = data['item']
                scaler = test_dataset.scalers[item]
                
                pred_scaled = model.module(x_test).squeeze().cpu().numpy()
                pred_denorm = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                pred_final = np.maximum(0, pred_denorm).round()

                for i in range(PREDICT_DAYS):
                    all_predictions.append({
                        '영업일자': f"{test_prefix}+{i+1}일",
                        '영업장명_메뉴명': item,
                        '매출수량': pred_final[i]
                    })

    pd.DataFrame(all_predictions).to_csv(f'./predictions_nbeats_rank_{rank}.csv', index=False)
    dist.barrier()

    if rank == 0:
        print("N-BEATS: Aggregating results...")
        df_list = [pd.read_csv(f) for f in glob.glob('./predictions_nbeats_rank_*.csv')]
        combined_df = pd.concat(df_list)
        
        sample_sub = pd.read_csv('./data/sample_submission.csv')
        pivot = combined_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
        submission = pd.merge(sample_sub[['영업일자']], pivot, on='영업일자', how='left').fillna(0)
        submission.to_csv('submission_nbeats.csv', index=False)
        for f in glob.glob('./predictions_nbeats_rank_*.csv'): os.remove(f)
        print("N-BEATS: Submission file created.")

    cleanup()

if __name__ == '__main__':
    main()