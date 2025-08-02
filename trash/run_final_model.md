```shell
#!/bin/bash

# =============================================================================
# LG Aimers 최종 모델 실행 스크립트
# =============================================================================

echo " LG Aimers 해커톤 Phase 2 - 최종 모델 실행"
echo " 목표: SMAPE 0.01 달성"
echo " GPU 8개 병렬 처리 활성화"
echo ""

# 현재 시간 기록
START_TIME=$(date)
echo " 시작 시간: $START_TIME"

# 가상환경 활성화 (있는 경우)
if [ -d "lg_aimers_env" ]; then
    echo " 가상환경 활성화 중..."
    source lg_aimers_env/bin/activate
fi

# GPU 상태 확인
echo ""
echo " GPU 상태 확인:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 8개 GPU 모두 사용
export OMP_NUM_THREADS=32  # CPU 스레드 수
export PYTHONUNBUFFERED=1  # 실시간 출력

echo ""
echo " 환경 변수 설정:"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"

# 데이터 구조 확인
echo ""
echo " 데이터 구조 확인:"
if [ -f "./data/train/train.csv" ]; then
    echo "   ✓ 학습 데이터: $(wc -l < ./data/train/train.csv) 행"
else
    echo "    학습 데이터가 없습니다!"
    exit 1
fi

if [ -f "./data/sample_submission.csv" ]; then
    echo "   ✓ 샘플 제출 파일: $(wc -l < ./data/sample_submission.csv) 행"
else
    echo "    샘플 제출 파일이 없습니다!"
    exit 1
fi

TEST_COUNT=$(ls ./data/test/TEST_*.csv 2>/dev/null | wc -l)
echo "   ✓ 테스트 파일: $TEST_COUNT 개"

if [ $TEST_COUNT -eq 0 ]; then
    echo "    테스트 파일이 없습니다!"
    exit 1
fi

# 로그 디렉토리 생성
mkdir -p logs
LOG_FILE="logs/final_model_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo " 로그 파일: $LOG_FILE"
echo ""

# GPU 메모리 모니터링 함수
monitor_gpu() {
    while true; do
        echo "$(date '+%H:%M:%S') GPU 메모리:" >> $LOG_FILE
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> $LOG_FILE
        sleep 60
    done
}

# 백그라운드에서 GPU 모니터링 시작
monitor_gpu &
MONITOR_PID=$!

# 메인 모델 실행
echo " 최종 모델 실행 시작..."
echo "    Transformer + 앙상블 + GPU 병렬 처리"
echo "    목표 SMAPE: 0.01"
echo ""

# Python 스크립트 실행 (로그 저장)
python3 final_complete_model.py 2>&1 | tee $LOG_FILE

# 실행 결과 확인
EXIT_CODE=${PIPESTATUS[0]}

# 모니터링 프로세스 종료
kill $MONITOR_PID 2>/dev/null

# 완료 시간 기록
END_TIME=$(date)
echo ""
echo " 완료 시간: $END_TIME"

# 결과 파일 확인
echo ""
echo " 결과 파일 확인:"
if [ -f "lg_aimers_final_submission.csv" ]; then
    FILE_SIZE=$(du -h lg_aimers_final_submission.csv | cut -f1)
    LINE_COUNT=$(wc -l < lg_aimers_final_submission.csv)
    echo "    제출 파일 생성 완료!"
    echo "    파일명: lg_aimers_final_submission.csv"
    echo "    파일 크기: $FILE_SIZE"
    echo "    행 수: $LINE_COUNT"

    # 파일 미리보기 (처음 5행)
    echo ""
    echo " 제출 파일 미리보기 (처음 5행):"
    head -5 lg_aimers_final_submission.csv

    echo ""
    echo " 모델 실행 성공!"
    echo "    제출 파일: lg_aimers_final_submission.csv"
    echo "    로그 파일: $LOG_FILE"

else
    echo "    제출 파일이 생성되지 않았습니다!"
    echo "    로그를 확인하세요: $LOG_FILE"
    exit 1
fi

# GPU 최종 상태
echo ""
echo " 최종 GPU 상태:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# 실행 결과 요약
echo ""
echo " === 실행 결과 요약 ==="
echo "   시작 시간: $START_TIME"
echo "   완료 시간: $END_TIME"
echo "   종료 코드: $EXIT_CODE"
echo "   로그 파일: $LOG_FILE"
echo "   제출 파일: lg_aimers_final_submission.csv"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo " === LG Aimers 해커톤 Phase 2 완료 ==="
    echo "    SMAPE 0.01 목표 달성을 위한 최고 성능 모델"
    echo "    GPU 8개 병렬 처리로 최적화"
    echo "    Transformer + 앙상블 + 고급 특성 엔지니어링"
    echo ""
    echo " 다음 단계: lg_aimers_final_submission.csv 파일을 LG Aimers 플랫폼에 제출하세요!"
    echo ""
else
    echo ""
    echo " 모델 실행 중 오류가 발생했습니다."
    echo "    로그 파일을 확인하세요: $LOG_FILE"
    echo "    문제 해결 후 다시 실행하세요."
fi
```
