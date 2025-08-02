# create_ensemble.py
import pandas as pd

print("Creating final ensemble submission...")

try:
    sub_transformer = pd.read_csv("submission_transformer.csv")
    sub_nbeats = pd.read_csv("submission_nbeats.csv")
    sub_lgbm = pd.read_csv("submission_lightgbm.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure all model training scripts ran successfully.")
    exit(1)

# 가중치 설정 (성능 검증 후 조정 필요)
# Transformer에 가장 높은 가중치, 그 다음으로 LGBM, N-BEATS 순으로 설정
w_transformer = 0.50
w_lgbm = 0.30
w_nbeats = 0.20

print(f"Ensemble weights: Transformer={w_transformer}, LGBM={w_lgbm}, N-BEATS={w_nbeats}")

# 제출 파일 형식 복사
submission = sub_transformer.copy()

# 가중 평균 계산
for col in submission.columns:
    if col != '영업일자':
        submission[col] = (sub_transformer[col] * w_transformer +
                           sub_lgbm[col] * w_lgbm +
                           sub_nbeats[col] * w_nbeats)

# 결과는 0 이상 정수여야 함
submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(0).round().astype(int)

submission.to_csv("submission_ensemble.csv", index=False, encoding='utf-8-sig')

print("\nEnsemble submission file 'submission_ensemble.csv' created successfully!")
print("This is the final file to submit.")