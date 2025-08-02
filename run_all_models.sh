#!/bin/bash

echo "======================================================"
echo "  STARTING 3-MODEL HYBRID ENSEMBLE TRAINING"
echo "  GOAL: Achieve SMAPE Score < 0.1"
echo "  Utilizing 8 GPUs in parallel."
echo "======================================================"
START_TIME=$(date)

# 로그 디렉토리 생성
mkdir -p logs

# --- 1. 각 모델 학습을 백그라운드에서 동시 실행 ---

echo "[$(date)] Starting Transformer Training on GPU 0,1,2..."
# torchrun을 사용하여 3개의 GPU에서 Transformer 학습
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train_transformer.py > logs/transformer.log 2>&1 &
PID_TRANSFORMER=$!

echo "[$(date)] Starting N-BEATS Training on GPU 3,4,5..."
# torchrun을 사용하여 3개의 GPU에서 N-BEATS 학습
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 train_nbeats.py > logs/nbeats.log 2>&1 &
PID_NBEATS=$!

echo "[$(date)] Starting LightGBM Training on GPU 6,7..."
# 2개의 GPU를 사용하도록 LightGBM 학습 (스크립트 내에서 GPU ID 지정)
CUDA_VISIBLE_DEVICES=6,7 python train_lightgbm.py > logs/lightgbm.log 2>&1 &
PID_LGBM=$!

# --- 2. 모든 학습이 완료될 때까지 대기 ---

echo ""
echo "All models are training in the background..."
echo "  - Transformer PID: $PID_TRANSFORMER (Logs: logs/transformer.log)"
echo "  - N-BEATS PID:     $PID_NBEATS (Logs: logs/nbeats.log)"
echo "  - LightGBM PID:    $PID_LGBM (Logs: logs/lightgbm.log)"
echo ""
echo "Waiting for all models to finish training. This may take a long time."
echo ""

wait $PID_TRANSFORMER
echo "[$(date)] Transformer training finished."

wait $PID_NBEATS
echo "[$(date)] N-BEATS training finished."

wait $PID_LGBM
echo "[$(date)] LightGBM training finished."

echo ""
echo "All training processes are complete."
echo "======================================================"


# --- 3. 앙상블 스크립트 실행 ---

echo "[$(date)] Creating final ensemble submission..."
python create_ensemble.py

echo ""
echo "======================================================"
echo "  ENSEMBLE PROCESS COMPLETE"
echo "  Start Time: $START_TIME"
echo "  End Time:   $(date)"
echo "  Final submission file: submission_ensemble.csv"
echo "======================================================"