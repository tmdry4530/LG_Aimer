```python
# 기본 라이브러리
import os
import random
import glob
import re
from datetime import datetime

# 데이터 처리
import pandas as pd
import numpy as np

# 데이터 전처리 및 모델링
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 재현성을 위한 시드 고정
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

set_seed(42)

# --- 하이퍼파라미터 설정 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOOKBACK_DAYS = 28
PREDICT_DAYS = 7
BATCH_SIZE = 256
EPOCHS = 100
D_MODEL = 128
N_HEAD = 8
N_LAYERS = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
HIGH_WEIGHT_STORES = ['담하', '미라시아']
STORE_WEIGHT = 2.0

print(f"사용 디바이스: {DEVICE}")
print(f"중요 가중치 적용 매장: {HIGH_WEIGHT_STORES} (가중치: {STORE_WEIGHT})")

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

print("학습 데이터 로드 및 특성 엔지니어링...")
train_df = pd.read_csv('./data/train/train.csv')
train_df = feature_engineering(train_df)

item_encoder = {item: i for i, item in enumerate(train_df['영업장명_메뉴명'].unique())}
train_df['item_id'] = train_df['영업장명_메뉴명'].map(item_encoder)
num_items = len(item_encoder)

scalers = {}
numeric_features = ['매출수량', 'lag_7', 'rolling_mean_7', 'rolling_std_7']

for item_id, group in tqdm(train_df.groupby('item_id'), desc="아이템별 스케일러 학습"):
    scaler = MinMaxScaler()
    scaler.fit(group[numeric_features])
    scalers[item_id] = scaler

print(f"총 {len(item_encoder)}개의 아이템과 {len(scalers)}개의 스케일러가 준비되었습니다.")

# --- 수정 --- : 스케일러 파라미터를 PyTorch 텐서로 미리 변환
scaler_params = {}
for item_id, scaler in scalers.items():
    min_val = scaler.min_[0]  # 매출수량의 min
    scale_val = scaler.scale_[0] # 매출수량의 scale
    scaler_params[item_id] = {
        'min': torch.tensor(min_val, device=DEVICE, dtype=torch.float32),
        'scale_inv': torch.tensor(1.0 / scale_val if scale_val != 0 else 1.0, device=DEVICE, dtype=torch.float32) # scale_은 1/(max-min)이므로 역수(max-min)를 저장
    }

class SalesDataset(Dataset):
    def __init__(self, df, item_encoder, scalers, lookback, predict_days):
        self.df = df
        self.item_encoder = item_encoder
        self.scalers = scalers
        self.lookback = lookback
        self.predict_days = predict_days
        self.total_len = lookback + predict_days

        self.data = []
        self.time_features = ['dayofweek', 'month', 'year', 'dayofyear', 'weekofyear', 'is_holiday']
        self.numeric_features = ['매출수량', 'lag_7', 'rolling_mean_7', 'rolling_std_7']

        for item_id, group in tqdm(df.groupby('item_id'), desc="데이터 시퀀스 생성"):
            if item_id not in self.scalers: continue

            group_scaled = self.scalers[item_id].transform(group[self.numeric_features])

            all_features = np.hstack([
                group_scaled,
                group[self.time_features].values,
                group[['item_id']].values
            ])

            for i in range(len(all_features) - self.total_len + 1):
                x = all_features[i : i + self.lookback]
                y = all_features[i + self.lookback : i + self.total_len, 0]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

train_dataset = SalesDataset(train_df, item_encoder, scalers, LOOKBACK_DAYS, PREDICT_DAYS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2)

class SalesTransformer(nn.Module):
    def __init__(self, num_items, d_model, n_head, n_layers, dropout, num_time_features, num_numeric_features):
        super(SalesTransformer, self).__init__()
        self.item_embedding = nn.Embedding(num_items, d_model)
        input_dim = d_model + num_time_features + num_numeric_features
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Linear(d_model, PREDICT_DAYS)
        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        numeric_data = x[:, :, :4]
        time_data = x[:, :, 4:-1]
        item_ids = x[:, :, -1][:, 0].long()
        item_embed = self.item_embedding(item_ids).unsqueeze(1).repeat(1, LOOKBACK_DAYS, 1)
        combined_features = torch.cat([numeric_data, time_data, item_embed], dim=-1)
        projected_input = self.input_projection(combined_features)
        pe = torch.zeros_like(projected_input)
        position = torch.arange(0, LOOKBACK_DAYS, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-np.log(10000.0) / D_MODEL))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        encoder_input = projected_input + pe.to(DEVICE)
        encoder_output = self.transformer_encoder(encoder_input)
        output = self.output_layer(encoder_output[:, -1, :])
        return output

# --- 변경 --- : Loss 함수가 스케일러 파라미터를 받도록 수정
class WeightedSMAPELoss(nn.Module):
    def __init__(self, high_weight_stores, item_encoder, scaler_params, weight=2.0, epsilon=1e-8):
        super(WeightedSMAPELoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight
        self.scaler_params = scaler_params
        self.high_weight_ids = {item_encoder[item] for item in item_encoder if item.split('_')[0] in high_weight_stores}
        print(f"가중치 적용 Item ID 개수: {len(self.high_weight_ids)}")

    def forward(self, y_pred_scaled, y_true_scaled, item_ids):
        # --- 수정 --- : 배치 내 각 아이템에 맞는 스케일러 파라미터를 가져옴
        batch_mins = torch.stack([self.scaler_params[i.item()]['min'] for i in item_ids])
        batch_scale_invs = torch.stack([self.scaler_params[i.item()]['scale_inv'] for i in item_ids])

        # unsqueeze를 통해 [batch_size, 1] 형태로 만들어 브로드캐스팅이 가능하게 함
        batch_mins = batch_mins.unsqueeze(1)
        batch_scale_invs = batch_scale_invs.unsqueeze(1)

        # --- 수정 --- : PyTorch 연산으로 역정규화 수행 (연산 그래프 유지)
        y_pred_denorm = (y_pred_scaled * batch_scale_invs) + batch_mins
        y_true_denorm = (y_true_scaled * batch_scale_invs) + batch_mins

        mask = y_true_denorm != 0

        y_pred_masked = y_pred_denorm[mask]
        y_true_masked = y_true_denorm[mask]

        numerator = 2 * torch.abs(y_pred_masked - y_true_masked)
        denominator = torch.abs(y_pred_masked) + torch.abs(y_true_masked) + self.epsilon

        smape = numerator / denominator

        weights = torch.ones_like(smape)
        batch_item_ids = item_ids.unsqueeze(1).repeat(1, y_true_scaled.shape[1])[mask]

        for i, item_id in enumerate(batch_item_ids):
            if item_id.item() in self.high_weight_ids:
                weights[i] = self.weight

        weighted_smape = smape * weights

        return torch.mean(weighted_smape)

model = SalesTransformer(
    num_items=num_items,
    d_model=D_MODEL,
    n_head=N_HEAD,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
    num_time_features=len(train_dataset.time_features),
    num_numeric_features=len(train_dataset.numeric_features)
).to(DEVICE)

# --- 변경 --- : Loss 함수에 스케일러 파라미터 전달
criterion = WeightedSMAPELoss(HIGH_WEIGHT_STORES, item_encoder, scaler_params, weight=STORE_WEIGHT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

print("모델 학습을 시작합니다...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0
    for i, (x_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        item_ids = x_batch[:, 0, -1].long()

        optimizer.zero_grad()

        predictions_scaled = model(x_batch)

        # --- 수정 --- : 복잡한 역정규화 루프를 제거하고 Loss 함수에 스케일된 값 그대로 전달
        loss = criterion(predictions_scaled, y_batch, item_ids)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.6f}, LR: {scheduler.get_last_lr()[0]:.7f}")

print("\n예측을 시작합니다...")
model.eval()
all_predictions = []
test_files = sorted(glob.glob('./data/test/TEST_*.csv'))

item_decoder = {i: item for item, i in item_encoder.items()}

with torch.no_grad():
    for file_path in tqdm(test_files, desc="테스트 파일 처리"):
        test_df = pd.read_csv(file_path)
        test_prefix = os.path.basename(file_path).split('.')[0]

        test_df = feature_engineering(test_df)

        for store_menu, group in test_df.groupby('영업장명_메뉴명'):
            if store_menu not in item_encoder: continue

            item_id = item_encoder[store_menu]
            scaler = scalers[item_id]

            group = group.sort_values('영업일자').tail(LOOKBACK_DAYS)
            if len(group) < LOOKBACK_DAYS: continue

            group['item_id'] = item_id

            group_scaled_numeric = scaler.transform(group[numeric_features])

            x_test = np.hstack([
                group_scaled_numeric,
                group[train_dataset.time_features].values,
                group[['item_id']].values
            ])

            x_test_tensor = torch.FloatTensor(x_test).unsqueeze(0).to(DEVICE)

            prediction_scaled = model(x_test_tensor).squeeze().cpu().numpy()

            temp_pred = np.zeros((len(prediction_scaled), len(numeric_features)))
            temp_pred[:, 0] = prediction_scaled
            prediction_denorm = scaler.inverse_transform(temp_pred)[:, 0]

            prediction_final = np.maximum(0, prediction_denorm).round()

            for i in range(PREDICT_DAYS):
                date_str = f"{test_prefix}+{i+1}일"
                all_predictions.append({
                    '영업일자': date_str,
                    '영업장명_메뉴명': store_menu,
                    '매출수량': prediction_final[i]
                })

submission_df = pd.read_csv('./data/sample_submission.csv')
pred_df = pd.DataFrame(all_predictions)

submission_pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()

final_submission = pd.merge(submission_df[['영업일자']], submission_pivot, on='영업일자', how='left')
final_submission = final_submission[submission_df.columns]
final_submission = final_submission.fillna(0)

final_submission.to_csv('transformer_submission.csv', index=False, encoding='utf-8-sig')
print("\n제출 파일 'transformer_submission.csv' 생성이 완료되었습니다.")
```
