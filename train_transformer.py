# train_transformer.py
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# 공유 유틸리티 임포트
from shared_utils import setup_distributed, cleanup, set_seed, feature_engineering, SalesDataset

# --- 하이퍼파라미터 ---
LOOKBACK_DAYS = 28
PREDICT_DAYS = 7
BATCH_SIZE = 128
EPOCHS = 120
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 6
DROPOUT = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# --- Transformer 모델 아키텍처 ---
class SalesTransformer(nn.Module):
    def __init__(self, num_items, d_model, n_head, n_layers, dropout, num_time_features, num_numeric_features):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, d_model)
        input_dim = d_model + num_time_features + num_numeric_features
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, LOOKBACK_DAYS, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, PREDICT_DAYS))

    def forward(self, x):
        numeric_data = x[:, :, :4]
        time_data = x[:, :, 4:-1]
        item_ids = x[:, :, -1][:, 0].long()
        item_embed = self.item_embedding(item_ids).unsqueeze(1).repeat(1, LOOKBACK_DAYS, 1)
        combined_features = torch.cat([numeric_data, time_data, item_embed], dim=-1)
        projected_input = self.input_projection(combined_features)
        encoder_input = projected_input + self.pos_encoder
        encoder_output = self.transformer_encoder(encoder_input)
        output = self.output_layer(encoder_output[:, -1, :])
        return output

def main():
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    set_seed(42)

    if rank == 0: print("Transformer: Loading and preprocessing data...")
    train_df = pd.read_csv('./data/train/train.csv')
    train_df = feature_engineering(train_df)
    item_encoder = {item: i for i, item in enumerate(train_df['영업장명_메뉴명'].unique())}
    train_df['item_id'] = train_df['영업장명_메뉴명'].map(item_encoder)
    num_items = len(item_encoder)

    scalers = {}
    for item_id, group in train_df.groupby('item_id'):
        scaler = MinMaxScaler()
        scalers[item_id] = scaler.fit(group[['매출수량', 'lag_7', 'rolling_mean_7', 'rolling_std_7']])

    train_dataset = SalesDataset(train_df, item_encoder, scalers, LOOKBACK_DAYS, PREDICT_DAYS, is_train=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)

    model = SalesTransformer(num_items, D_MODEL, N_HEAD, N_LAYERS, DROPOUT, 6, 4).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.L1Loss() # MAE Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if rank == 0: print("Transformer: Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Transformer Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    if rank == 0: print("Transformer: Starting prediction...")
    model.eval()
    all_predictions = []
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))
    files_per_rank = [f for i, f in enumerate(test_files) if i % world_size == rank]
    
    with torch.no_grad():
        for file_path in files_per_rank:
            test_df = pd.read_csv(file_path)
            test_prefix = os.path.basename(file_path).split('.')[0]
            test_df = feature_engineering(test_df)
            test_df['item_id'] = test_df['영업장명_메뉴명'].map(item_encoder)
            
            test_dataset = SalesDataset(test_df, item_encoder, scalers, LOOKBACK_DAYS, PREDICT_DAYS, is_train=False)
            
            for x_test, item_id in test_dataset:
                x_test_tensor = torch.FloatTensor(x_test).unsqueeze(0).to(device)
                pred_scaled = model.module(x_test_tensor).squeeze().cpu().numpy()
                
                scaler = scalers[item_id]
                temp_pred = np.zeros((len(pred_scaled), 4))
                temp_pred[:, 0] = pred_scaled
                pred_denorm = scaler.inverse_transform(temp_pred)[:, 0]
                pred_final = np.maximum(0, pred_denorm).round()

                for i in range(PREDICT_DAYS):
                    all_predictions.append({
                        '영업일자': f"{test_prefix}+{i+1}일",
                        '영업장명_메뉴명': list(item_encoder.keys())[list(item_encoder.values()).index(item_id)],
                        '매출수량': pred_final[i]
                    })

    pd.DataFrame(all_predictions).to_csv(f'./predictions_transformer_rank_{rank}.csv', index=False)
    dist.barrier()

    if rank == 0:
        print("Transformer: Aggregating results...")
        df_list = [pd.read_csv(f) for f in glob.glob('./predictions_transformer_rank_*.csv')]
        combined_df = pd.concat(df_list)
        
        sample_sub = pd.read_csv('./data/sample_submission.csv')
        pivot = combined_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
        submission = pd.merge(sample_sub[['영업일자']], pivot, on='영업일자', how='left').fillna(0)
        submission.to_csv('submission_transformer.csv', index=False)
        for f in glob.glob('./predictions_transformer_rank_*.csv'): os.remove(f)
        print("Transformer: Submission file created.")

    cleanup()

if __name__ == '__main__':
    main()