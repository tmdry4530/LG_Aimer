# train_lightgbm.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import glob
from tqdm import tqdm
import os

def create_lgbm_features(df):
    df['영업일자'] = pd.to_datetime(df['영업일자'])
    df['day'] = df['영업일자'].dt.day
    df['month'] = df['영업일자'].dt.month
    df['year'] = df['영업일자'].dt.year
    df['weekday'] = df['영업일자'].dt.dayofweek
    df['weekofyear'] = df['영업일자'].dt.isocalendar().week.astype(int)
    df['dayofyear'] = df['영업일자'].dt.dayofyear
    
    for lag in [7, 14, 21, 28]:
        df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag).fillna(0)
        
    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).mean().fillna(0)
        df[f'rolling_std_{window}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(window).std().fillna(0)
        
    return df

def main():
    print("LightGBM: Loading and preprocessing data...")
    train_df = pd.read_csv('./data/train/train.csv')
    train_df = create_lgbm_features(train_df)

    categorical_features = ['weekday', 'month']
    train_df['영업장명_메뉴명_cat'] = train_df['영업장명_메뉴명'].astype('category').cat.codes
    categorical_features.append('영업장명_메뉴명_cat')

    features = [col for col in train_df.columns if col not in ['영업일자', '매출수량', '영업장명_메뉴명']]
    
    params = {
        'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2500,
        'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'num_leaves': 63,
        'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
        'device': 'gpu', 'gpu_device_id': 0 # CUDA_VISIBLE_DEVICES에 따라 0, 1로 매핑됨
    }

    print("LightGBM: Training model...")
    model = lgb.LGBMRegressor(**params)
    model.fit(train_df[features], train_df['매출수량'], categorical_feature=categorical_features)

    print("LightGBM: Starting prediction...")
    all_predictions = []
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))

    for file_path in tqdm(test_files, desc="LGBM Predicting"):
        test_df = pd.read_csv(file_path)
        test_prefix = os.path.basename(file_path).split('.')[0]
        
        history = test_df.copy()
        
        for day in range(1, 8):
            pred_date = pd.to_datetime(history['영업일자'].max()) + pd.Timedelta(days=1)
            
            # 예측할 날짜의 데이터프레임 생성
            future_df = history.groupby('영업장명_메뉴명').tail(1).copy()
            future_df['영업일자'] = pred_date
            
            # 미래 시점의 특성 생성
            temp_df = pd.concat([history, future_df], ignore_index=True)
            temp_df = create_lgbm_features(temp_df)
            temp_df['영업장명_메뉴명_cat'] = temp_df['영업장명_메뉴명'].astype('category').cat.codes
            
            # 예측할 부분만 추출
            predict_data = temp_df[temp_df['영업일자'] == pred_date]
            
            # 예측
            predictions = model.predict(predict_data[features])
            predictions = np.maximum(0, predictions).round()
            
            # 예측 결과를 history에 추가하여 다음 날 예측에 사용
            predict_data['매출수량'] = predictions
            history = pd.concat([history, predict_data[['영업일자', '영업장명_메뉴명', '매출수량']]], ignore_index=True)
            
            # 최종 제출 형식으로 저장
            for idx, row in predict_data.iterrows():
                all_predictions.append({
                    '영업일자': f"{test_prefix}+{day}일",
                    '영업장명_메뉴명': row['영업장명_메뉴명'],
                    '매출수량': row['매출수량']
                })

    pred_df = pd.DataFrame(all_predictions)
    sample_sub = pd.read_csv('./data/sample_submission.csv')
    pivot = pred_df.pivot(index='영업일자', columns='영업장명_메뉴명', values='매출수량').reset_index()
    submission = pd.merge(sample_sub[['영업일자']], pivot, on='영업일자', how='left').fillna(0)
    submission.to_csv('submission_lightgbm.csv', index=False)
    print("LightGBM: Submission file created.")

if __name__ == '__main__':
    main()