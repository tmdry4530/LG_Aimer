# [Baseline] LSTM을 활용한 메뉴별 매출 수량 예측

이 코드는 LSTM 딥러닝 모델을 사용하여 각 영업장의 메뉴별 매출 수량을 예측하는 베이스라인 모델입니다.

## 1. 필요한 라이브러리 Import

```python
# 기본 라이브러리
import os
import random
import glob
import re

# 데이터 처리
import pandas as pd
import numpy as np

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler

# 딥러닝 모델
import torch
import torch.nn as nn
from tqdm import tqdm
```

## 2. 실험 환경 설정 및 하이퍼파라미터

```python
def set_seed(seed=42):
    """
    재현 가능한 결과를 위한 랜덤 시드 고정

    Args:
        seed (int): 설정할 랜덤 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # GPU 사용 시 추가 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 시드 고정
set_seed(42)

# 하이퍼파라미터 설정
LOOKBACK_DAYS = 28      # 과거 몇 일의 데이터를 사용할지
PREDICT_DAYS = 7        # 앞으로 몇 일을 예측할지
BATCH_SIZE = 16         # 배치 크기
EPOCHS = 50             # 학습 에포크 수

# 사용할 디바이스 설정 (GPU 우선)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {DEVICE}")
```

## 3. 데이터 로드

```python
# 학습 데이터 로드
print("학습 데이터를 로드하는 중...")
train_data = pd.read_csv('./train/train.csv')
print(f"학습 데이터 형태: {train_data.shape}")
print(f"영업장-메뉴 조합 수: {train_data['영업장명_메뉴명'].nunique()}")
```

## 4. LSTM 모델 정의

```python
class MultiOutputLSTM(nn.Module):
    """
    여러 일수를 한 번에 예측하는 LSTM 모델

    과거 LOOKBACK_DAYS 일의 매출 데이터를 입력받아
    앞으로 PREDICT_DAYS 일의 매출을 예측합니다.
    """

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=7):
        """
        Args:
            input_dim (int): 입력 특성 수 (매출수량 1개)
            hidden_dim (int): LSTM 은닉층 차원
            num_layers (int): LSTM 레이어 수
            output_dim (int): 출력 차원 (예측할 일수)
        """
        super(MultiOutputLSTM, self).__init__()

        # LSTM 레이어: 시퀀스 데이터 처리
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 완전연결층: LSTM 출력을 예측값으로 변환
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        순전파 함수

        Args:
            x (torch.Tensor): 입력 시퀀스 (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: 예측값 (batch_size, output_dim)
        """
        # LSTM 처리 (마지막 타임스텝의 출력만 사용)
        lstm_out, _ = self.lstm(x)

        # 마지막 타임스텝의 출력을 완전연결층에 통과
        prediction = self.fc(lstm_out[:, -1, :])

        return prediction
```

## 5. 모델 학습 함수

```python
def train_lstm_model(train_df):
    """
    각 영업장-메뉴 조합별로 개별 LSTM 모델을 학습합니다.

    Args:
        train_df (pd.DataFrame): 학습 데이터

    Returns:
        dict: 학습된 모델들과 전처리 객체들을 담은 딕셔너리
    """
    trained_models = {}

    print("영업장-메뉴별 모델 학습을 시작합니다...")

    # 각 영업장-메뉴 조합별로 개별 모델 학습
    for store_menu, group_data in tqdm(train_df.groupby(['영업장명_메뉴명']),
                                      desc='모델 학습 진행'):

        # 데이터 정렬 및 복사
        store_data = group_data.sort_values('영업일자').copy()

        # 데이터가 충분한지 확인 (최소 학습+예측 일수 필요)
        min_required_days = LOOKBACK_DAYS + PREDICT_DAYS
        if len(store_data) < min_required_days:
            print(f"데이터 부족으로 {store_menu} 건너뜀 (필요: {min_required_days}일, 실제: {len(store_data)}일)")
            continue

        # 특성 추출 및 정규화
        features = ['매출수량']
        scaler = MinMaxScaler()
        store_data[features] = scaler.fit_transform(store_data[features])
        sales_values = store_data[features].values  # shape: (날짜수, 1)

        # 시퀀스 데이터 생성
        X_sequences, y_targets = [], []

        # 슬라이딩 윈도우 방식으로 학습 데이터 생성
        for i in range(len(sales_values) - LOOKBACK_DAYS - PREDICT_DAYS + 1):
            # 입력: 과거 LOOKBACK_DAYS 일의 매출
            input_sequence = sales_values[i:i+LOOKBACK_DAYS]
            # 타겟: 다음 PREDICT_DAYS 일의 매출
            target_sequence = sales_values[i+LOOKBACK_DAYS:i+LOOKBACK_DAYS+PREDICT_DAYS, 0]

            X_sequences.append(input_sequence)
            y_targets.append(target_sequence)

        # 텐서로 변환 및 GPU 이동
        X_train = torch.tensor(X_sequences).float().to(DEVICE)
        y_train = torch.tensor(y_targets).float().to(DEVICE)

        # 모델 초기화
        model = MultiOutputLSTM(
            input_dim=1,
            output_dim=PREDICT_DAYS
        ).to(DEVICE)

        # 옵티마이저 및 손실 함수 설정
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # 모델 학습
        model.train()
        for epoch in range(EPOCHS):
            # 데이터 셔플링
            indices = torch.randperm(len(X_train))

            # 미니배치 학습
            for i in range(0, len(X_train), BATCH_SIZE):
                batch_indices = indices[i:i+BATCH_SIZE]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                # 순전파
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 학습된 모델 및 관련 정보 저장
        trained_models[store_menu] = {
            'model': model.eval(),  # 평가 모드로 전환
            'scaler': scaler,       # 정규화 객체
            'last_sequence': sales_values[-LOOKBACK_DAYS:]  # 예측에 사용할 마지막 시퀀스
        }

    print(f"총 {len(trained_models)}개의 모델이 학습되었습니다.")
    return trained_models

# 모델 학습 실행
trained_models = train_lstm_model(train_data)
```

## 6. 예측 함수

```python
def predict_sales(test_df, trained_models, test_prefix):
    """
    학습된 모델을 사용하여 매출을 예측합니다.

    Args:
        test_df (pd.DataFrame): 테스트 데이터
        trained_models (dict): 학습된 모델들
        test_prefix (str): 테스트 파일 접두어 (예: 'TEST_00')

    Returns:
        pd.DataFrame: 예측 결과
    """
    prediction_results = []

    print(f"{test_prefix} 데이터에 대한 예측을 시작합니다...")

    # 각 영업장-메뉴별로 예측 수행
    for store_menu, test_group in test_df.groupby(['영업장명_메뉴명']):

        # 해당 조합의 학습된 모델이 있는지 확인
        if store_menu not in trained_models:
            continue

        # 모델 및 전처리 객체 로드
        model_info = trained_models[store_menu]
        model = model_info['model']
        scaler = model_info['scaler']

        # 테스트 데이터 정렬
        test_sorted = test_group.sort_values('영업일자')

        # 최근 LOOKBACK_DAYS 일의 데이터 추출
        recent_sales = test_sorted['매출수량'].values[-LOOKBACK_DAYS:]

        # 데이터가 충분한지 확인
        if len(recent_sales) < LOOKBACK_DAYS:
            continue

        # 데이터 정규화
        recent_sales_normalized = scaler.transform(recent_sales.reshape(-1, 1))

        # 텐서로 변환 및 배치 차원 추가
        input_tensor = torch.tensor([recent_sales_normalized]).float().to(DEVICE)

        # 예측 수행
        with torch.no_grad():
            predictions_scaled = model(input_tensor).squeeze().cpu().numpy()

        # 예측값 역정규화
        final_predictions = []
        for i in range(PREDICT_DAYS):
            # 임시 배열 생성 (MinMaxScaler의 inverse_transform을 위해)
            temp_array = np.zeros((1, 1))
            temp_array[0, 0] = predictions_scaled[i]

            # 역정규화 수행
            denormalized_value = scaler.inverse_transform(temp_array)[0, 0]

            # 음수 방지 (매출은 0 이상)
            final_predictions.append(max(denormalized_value, 0))

        # 예측 날짜 생성 (TEST_00+1일, TEST_00+2일, ...)
        prediction_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT_DAYS)]

        # 결과 저장
        for date, predicted_sales in zip(prediction_dates, final_predictions):
            prediction_results.append({
                '영업일자': date,
                '영업장명_메뉴명': store_menu,
                '매출수량': predicted_sales
            })

    return pd.DataFrame(prediction_results)

# 모든 테스트 파일에 대한 예측 수행
all_predictions = []

print("테스트 파일들을 찾는 중...")
test_file_paths = sorted(glob.glob('./test/TEST_*.csv'))
print(f"총 {len(test_file_paths)}개의 테스트 파일을 찾았습니다.")

# 각 테스트 파일별로 예측 수행
for file_path in test_file_paths:
    # 테스트 데이터 로드
    test_data = pd.read_csv(file_path)

    # 파일명에서 접두어 추출 (예: 'TEST_00')
    filename = os.path.basename(file_path)
    test_prefix = re.search(r'(TEST_\d+)', filename).group(1)

    print(f"{test_prefix} 처리 중... (데이터 크기: {test_data.shape})")

    # 예측 수행
    predictions = predict_sales(test_data, trained_models, test_prefix)
    all_predictions.append(predictions)

# 모든 예측 결과 합치기
final_predictions_df = pd.concat(all_predictions, ignore_index=True)
print(f"총 예측 결과 수: {len(final_predictions_df)}")
```

## 7. 제출 파일 생성

```python
def create_submission_file(predictions_df, sample_submission_df):
    """
    예측 결과를 제출 형식에 맞게 변환합니다.

    Args:
        predictions_df (pd.DataFrame): 예측 결과 데이터프레임
        sample_submission_df (pd.DataFrame): 샘플 제출 파일

    Returns:
        pd.DataFrame: 제출 형식에 맞는 데이터프레임
    """
    print("제출 파일 형식으로 변환 중...")

    # 예측 결과를 딕셔너리로 변환 (빠른 조회를 위해)
    # 키: (영업일자, 영업장명_메뉴명), 값: 매출수량
    prediction_dict = dict(zip(
        zip(predictions_df['영업일자'], predictions_df['영업장명_메뉴명']),
        predictions_df['매출수량']
    ))

    print(f"예측 딕셔너리 크기: {len(prediction_dict)}")

    # 샘플 제출 파일 형식 복사
    submission_df = sample_submission_df.copy()

    # 각 행(날짜)과 열(메뉴)에 대해 예측값 채우기
    filled_count = 0
    for row_idx in submission_df.index:
        date = submission_df.loc[row_idx, '영업일자']

        # 첫 번째 열(영업일자)을 제외한 모든 메뉴 열 처리
        for menu_column in submission_df.columns[1:]:
            # 해당 (날짜, 메뉴) 조합의 예측값 조회
            prediction_key = (date, menu_column)
            if prediction_key in prediction_dict:
                submission_df.loc[row_idx, menu_column] = prediction_dict[prediction_key]
                filled_count += 1
            else:
                # 예측값이 없는 경우 0으로 설정
                submission_df.loc[row_idx, menu_column] = 0

    print(f"총 {filled_count}개의 예측값이 채워졌습니다.")
    return submission_df

# 샘플 제출 파일 로드
print("샘플 제출 파일을 로드하는 중...")
sample_submission = pd.read_csv('./sample_submission.csv')
print(f"제출 파일 형태: {sample_submission.shape}")

# 제출 파일 생성
final_submission = create_submission_file(final_predictions_df, sample_submission)

# CSV 파일로 저장
output_filename = 'baseline_lstm_submission.csv'
final_submission.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"제출 파일이 '{output_filename}'로 저장되었습니다.")
print("베이스라인 LSTM 모델 실행이 완료되었습니다!")
```

## 모델 성능 개선 아이디어

1. **특성 엔지니어링**: 요일, 월, 계절성 등의 추가 특성 활용
2. **모델 앙상블**: 여러 모델의 예측을 결합
3. **하이퍼파라미터 튜닝**: Grid Search나 Optuna를 활용한 최적화
4. **정규화 기법**: Dropout, L2 regularization 추가
5. **다른 시계열 모델**: GRU, Transformer, Prophet 등 시도
