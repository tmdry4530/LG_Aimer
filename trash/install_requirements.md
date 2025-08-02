```shell
#!/bin/bash

# =============================================================================
# LG Aimers 해커톤 라이브러리 설치 스크립트
# =============================================================================

echo "LG Aimers 해커톤 Phase 2 - 라이브러리 설치"
echo ""

# 시스템 업데이트
echo "시스템 패키지 업데이트 중..."
sudo apt update -y

# 필수 시스템 패키지 설치
echo "필수 시스템 패키지 설치 중..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    nvtop \
    screen \
    tmux \
    bc

# Python 가상환경 생성
echo ""
echo "Python 가상환경 생성 중..."
python3 -m venv lg_aimers_env
source lg_aimers_env/bin/activate

# pip 업그레이드
echo "pip 업그레이드 중..."
pip install --upgrade pip setuptools wheel

# PyTorch 설치 (CUDA 12.8 최신버전)
echo ""
echo "PyTorch 설치 중 (CUDA 12.8 최신버전)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 데이터 과학 라이브러리 설치 (최신버전)
echo ""
echo "데이터 과학 라이브러리 설치 중 (최신버전)..."
pip install \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm

# 개발 도구 (최신버전)
echo ""
echo "개발 도구 설치 중 (최신버전)..."
pip install \
    jupyter \
    ipython \
    notebook

# GPU 확인 스크립트 생성
echo ""
echo "GPU 확인 스크립트 생성 중..."
cat > check_environment.py << 'EOF'
#!/usr/bin/env python3
"""
LG Aimers 해커톤 환경 확인 스크립트
"""

import sys
import torch
import pandas as pd
import numpy as np
import sklearn
from datetime import datetime

def check_environment():
    print("=== LG Aimers 환경 확인 ===")
    print(f"확인 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Python 버전
    print(f"Python 버전: {sys.version}")

    # 핵심 라이브러리 버전
    print("\n라이브러리 버전:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   Pandas: {pd.__version__}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Scikit-learn: {sklearn.__version__}")

    # CUDA 및 GPU 확인
    print(f"\nGPU 정보:")
    print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA 버전: {torch.version.cuda}")
        print(f"   GPU 개수: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")

        # GPU 연산 테스트
        print(f"\nGPU 연산 테스트:")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("   GPU 연산 테스트 성공!")

            # 메모리 사용량 확인
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            print(f"   GPU 메모리 사용량: {memory_allocated:.1f}MB")

        except Exception as e:
            print(f"   GPU 연산 테스트 실패: {e}")
    else:
        print("   GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")

    # 데이터 파일 확인
    print(f"\n데이터 파일 확인:")
    import os

    data_files = [
        "./data/train/train.csv",
        "./data/sample_submission.csv"
    ]

    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   확인됨: {file_path} ({size:.2f}MB)")
        else:
            print(f"   누락됨: {file_path}")

    # 테스트 파일 확인
    test_files = [f"./data/test/TEST_{i:02d}.csv" for i in range(10)]
    test_count = sum(1 for f in test_files if os.path.exists(f))
    print(f"   테스트 파일: {test_count}/10개 발견")

    print(f"\n환경 확인 완료!")

    if torch.cuda.is_available() and test_count == 10:
        print(f"모든 준비가 완료되었습니다! 최종 모델을 실행할 수 있습니다.")
        return True
    else:
        print(f"일부 구성 요소가 누락되었습니다. 확인 후 다시 시도하세요.")
        return False

if __name__ == "__main__":
    check_environment()
EOF

chmod +x check_environment.py

# requirements.txt 생성 (최신버전)
echo ""
echo "requirements.txt 생성 중..."
cat > requirements.txt << 'EOF'
# LG Aimers 해커톤 Phase 2 필수 라이브러리 (최신버전)

# PyTorch (CUDA 12.8 최신버전)
torch
torchvision
torchaudio

# 데이터 과학 (최신버전)
pandas
numpy
scikit-learn

# 시각화 (최신버전)
matplotlib
seaborn

# 유틸리티 (최신버전)
tqdm

# 개발 도구 (최신버전)
jupyter
ipython
notebook
EOF

# 설치 완료 메시지
echo ""
echo "=== 라이브러리 설치 완료 ==="
echo ""
echo "설치된 구성 요소:"
echo "   Python 가상환경: lg_aimers_env"
echo "   PyTorch (CUDA 12.8 지원) 최신버전"
echo "   데이터 과학 라이브러리 최신버전"
echo "   개발 도구 최신버전"
echo ""
echo "다음 단계:"
echo "   1. 환경 확인: python3 check_environment.py"
echo "   2. 가상환경 활성화: source lg_aimers_env/bin/activate"
echo "   3. 최종 모델 실행: ./run_final_model.sh"
echo ""
echo "생성된 파일:"
echo "   - requirements.txt: 라이브러리 목록"
echo "   - check_environment.py: 환경 확인 스크립트"
echo "   - lg_aimers_env/: Python 가상환경"
echo ""

# 환경 확인 실행
echo "환경 자동 확인 중..."
python3 check_environment.py

echo ""
echo "설치 스크립트 완료!"
echo "이제 ./run_final_model.sh를 실행하여 모델을 학습할 수 있습니다."
```
