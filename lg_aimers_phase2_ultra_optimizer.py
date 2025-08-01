# LG Aimers Phase 2 - SMAPE 0.01 도전! 극한 최적화 모델
# RTX 4090 x8 + 모든 트릭 총동원

import os
import random
import glob
import re
import warnings
import pickle
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import itertools

# 기본 라이브러리
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.signal import savgol_filter

# 딥러닝 라이브러리
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# 시각화 및 진행상황
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 고급 시계열 라이브러리
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 극한 최적화 설정
# =============================================================================

class UltraConfig:
    """극한 최적화 설정"""
    # 기본 설정
    SEED = 42
    LOOKBACK_DAYS = 28
    PREDICT_DAYS = 7
    
    # GPU 설정 (8개 GPU 최대 활용)
    NUM_GPUS = 8
    WORLD_SIZE = 8
    
    # 극한 학습 파라미터
    BATCH_SIZE = 128  # 대용량 배치
    EPOCHS = 500     # 충분한 학습
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-6
    GRADIENT_CLIP = 0.5
    
    # 앙상블 구성 (모든 가능한 모델)
    ENSEMBLE_MODELS = [
        'UltraLSTM', 'UltraGRU', 'UltraTransformer', 'UltraTCN',
        'Prophet', 'ARIMA', 'ETS', 'HoltWinters',
        'WaveNet', 'NBeats', 'DeepAR'
    ]
    
    # 메타 앙상블 설정
    META_ENSEMBLE_LAYERS = 3
    META_ENSEMBLE_MODELS = 50  # 50개 개별 모델
    
    # 업장별 초정밀 가중치
    ULTRA_HIGH_WEIGHT_STORES = ['담하', '미라시아']
    STORE_ULTRA_WEIGHTS = {
        '담하': 10.0,      # 극대 가중치
        '미라시아': 10.0,   # 극대 가중치
        'default': 1.0
    }
    
    # 특성 엔지니어링 (100+ 특성 목표)
    FEATURE_WINDOWS = [3, 5, 7, 10, 14, 21, 28, 35, 42, 56, 70]
    LAG_FEATURES = list(range(1, 29))  # 28일간 모든 래그
    
    # 고급 교차검증
    N_FOLDS = 10
    N_REPEATS = 3
    
    # 베이지안 최적화
    N_TRIALS = 200
    
    # 정확도 극한 추구 설정
    PRECISION_MODE = True
    ENSEMBLE_DIVERSITY_WEIGHT = 0.3
    UNCERTAINTY_THRESHOLD = 0.001

def set_ultra_seed(seed=42):
    """극한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_ultra_seed(UltraConfig.SEED)

# =============================================================================
# 2. 극한 특성 엔지니어링 (100+ 특성)
# =============================================================================

class UltraFeatureEngineer:
    """극한 특성 엔지니어링 - 100+ 특성 생성"""
    
    def __init__(self):
        self.feature_names = []
        self.holiday_calendar = self._create_detailed_calendar()
        
    def _create_detailed_calendar(self):
        """상세 캘린더 정보"""
        calendar = {}
        
        # 한국 공휴일
        holidays = [
            '2023-01-01', '2023-01-21', '2023-01-22', '2023-01-23', '2023-01-24',
            '2023-03-01', '2023-05-05', '2023-05-27', '2023-05-29', '2023-06-06',
            '2023-08-15', '2023-09-28', '2023-09-29', '2023-09-30', '2023-10-03',
            '2023-10-09', '2023-12-25',
            '2024-01-01', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',
            '2024-03-01', '2024-04-10', '2024-05-05', '2024-05-15', '2024-06-06',
            '2024-08-15', '2024-09-16', '2024-09-17', '2024-09-18', '2024-10-03',
            '2024-10-09', '2024-12-25'
        ]
        
        for date_str in holidays:
            calendar[date_str] = {'type': 'holiday', 'weight': 2.0}
        
        # 대체공휴일
        substitute_holidays = ['2023-01-24', '2023-05-29', '2024-02-12']
        for date_str in substitute_holidays:
            calendar[date_str] = {'type': 'substitute_holiday', 'weight': 1.5}
        
        return calendar
    
    def create_ultra_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """극한 시간 특성 생성"""
        df = df.copy()
        df['영업일자'] = pd.to_datetime(df['영업일자'])
        
        # 기본 시간 특성
        df['년도'] = df['영업일자'].dt.year
        df['월'] = df['영업일자'].dt.month
        df['일'] = df['영업일자'].dt.day
        df['요일'] = df['영업일자'].dt.dayofweek
        df['주차'] = df['영업일자'].dt.isocalendar().week
        df['년중_일수'] = df['영업일자'].dt.dayofyear
        df['월중_주차'] = df['영업일자'].dt.day // 7 + 1
        df['분기'] = df['영업일자'].dt.quarter
        
        # 계절성 (다양한 관점)
        df['계절_기상'] = df['월'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1,
                                     6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
        df['계절_관광'] = df['월'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:2,
                                     6:2, 7:3, 8:3, 9:1, 10:1, 11:0})
        
        # 휴일 및 특별일
        df['공휴일여부'] = df['영업일자'].dt.strftime('%Y-%m-%d').map(
            lambda x: 1 if x in self.holiday_calendar else 0
        )
        df['주말여부'] = df['요일'].isin([5, 6]).astype(int)
        df['휴일여부'] = ((df['주말여부'] == 1) | (df['공휴일여부'] == 1)).astype(int)
        
        # 연휴 패턴
        df['연휴_시작'] = 0
        df['연휴_중간'] = 0
        df['연휴_끝'] = 0
        # (연휴 로직은 복잡하므로 간소화)
        
        # 순환 특성 (여러 주기)
        for period, col_name in [(7, '요일'), (30.44, '월'), (365.25, '년중_일수')]:
            df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / period)
            df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / period)
        
        # 월별 특성
        df['월_초'] = (df['일'] <= 7).astype(int)
        df['월_중'] = ((df['일'] > 7) & (df['일'] <= 21)).astype(int)
        df['월_말'] = (df['일'] > 21).astype(int)
        
        # 급여일/보너스 효과
        df['급여일_근접'] = ((df['일'] >= 23) & (df['일'] <= 28)).astype(int)
        df['보너스_월'] = df['월'].isin([6, 12]).astype(int)
        
        return df
    
    def create_ultra_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """극한 통계 특성 생성"""
        df = df.copy()
        
        for window in UltraConfig.FEATURE_WINDOWS:
            # 기본 통계
            df[f'매출_{window}일평균'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'매출_{window}일중앙값'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                lambda x: x.rolling(window=window, min_periods=1).median()
            )
            df[f'매출_{window}일표준편차'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            )
            df[f'매출_{window}일최대'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            df[f'매출_{window}일최소'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            
            # 고급 통계
            df[f'매출_{window}일왜도'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                lambda x: x.rolling(window=window, min_periods=3).skew().fillna(0)
            )
            df[f'매출_{window}일첨도'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                lambda x: x.rolling(window=window, min_periods=4).kurt().fillna(0)
            )
            
            # 분위수
            for q in [0.1, 0.25, 0.75, 0.9]:
                df[f'매출_{window}일_{int(q*100)}분위'] = df.groupby('영업장명_메뉴명')['매출수량'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).quantile(q)
                )
        
        # 모든 래그 특성
        for lag in UltraConfig.LAG_FEATURES:
            df[f'매출_lag{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag).fillna(0)
        
        # 변화율 및 차분
        for period in [1, 2, 3, 7, 14]:
            df[f'매출_변화율_{period}'] = df.groupby('영업장명_메뉴명')['매출수량'].pct_change(periods=period).fillna(0)
            df[f'매출_차분_{period}'] = df.groupby('영업장명_메뉴명')['매출수량'].diff(periods=period).fillna(0)
        
        # 가속도 (2차 차분)
        df['매출_가속도'] = df.groupby('영업장명_메뉴명')['매출수량'].diff().diff().fillna(0)
        
        return df
    
    def create_ultra_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """극한 추세 특성 생성"""
        df = df.copy()
        
        def calculate_ultra_trend(group):
            if len(group) < 14:
                # 기본값으로 채우기
                trend_features = {
                    '매출_선형추세': 0, '매출_2차추세': 0, '매출_지수추세': 0,
                    '매출_계절성강도': 1, '매출_추세강도': 0, '매출_불규칙성': 0,
                    '매출_안정성': 1, '매출_주기성': 0
                }
                for key, value in trend_features.items():
                    group[key] = value
                return group
            
            x = np.arange(len(group))
            y = group['매출수량'].values
            
            # 선형 추세
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            group['매출_선형추세'] = slope
            group['매출_추세_R2'] = r_value**2
            
            # 2차 추세
            try:
                poly_coeff = np.polyfit(x, y, 2)
                group['매출_2차추세'] = poly_coeff[0]
            except:
                group['매출_2차추세'] = 0
            
            # 지수 추세 (로그 변환)
            try:
                y_log = np.log(y + 1)
                exp_slope, _, _, _, _ = stats.linregress(x, y_log)
                group['매출_지수추세'] = exp_slope
            except:
                group['매출_지수추세'] = 0
            
            # 계절성 분석 (7일 주기)
            if len(group) >= 21:
                try:
                    # STL 분해 대신 간단한 계절성 측정
                    weekly_pattern = []
                    for day in range(7):
                        day_values = y[day::7]
                        if len(day_values) > 1:
                            weekly_pattern.append(np.mean(day_values))
                    
                    if len(weekly_pattern) == 7:
                        seasonal_strength = np.std(weekly_pattern) / (np.mean(weekly_pattern) + 1e-8)
                        group['매출_계절성강도'] = seasonal_strength
                    else:
                        group['매출_계절성강도'] = 1
                except:
                    group['매출_계절성강도'] = 1
            else:
                group['매출_계절성강도'] = 1
            
            # 추세 강도 (기울기의 절댓값)
            group['매출_추세강도'] = abs(slope)
            
            # 불규칙성 (잔차의 표준편차)
            y_pred = slope * x + intercept
            residuals = y - y_pred
            group['매출_불규칙성'] = np.std(residuals)
            
            # 안정성 (변동계수의 역수)
            cv = np.std(y) / (np.mean(y) + 1e-8)
            group['매출_안정성'] = 1 / (cv + 1e-8)
            
            # 주기성 (자기상관)
            try:
                autocorr_7 = np.corrcoef(y[:-7], y[7:])[0, 1]
                group['매출_주기성'] = autocorr_7 if not np.isnan(autocorr_7) else 0
            except:
                group['매출_주기성'] = 0
            
            return group
        
        df = df.groupby('영업장명_메뉴명').apply(calculate_ultra_trend).reset_index(drop=True)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """교호작용 특성 생성"""
        df = df.copy()
        
        # 시간과 매출의 교호작용
        df['요일_매출평균'] = df['요일'] * df['매출_7일평균']
        df['월_매출평균'] = df['월'] * df['매출_14일평균']
        df['계절_매출평균'] = df['계절_기상'] * df['매출_28일평균']
        
        # 휴일과 매출의 교호작용
        df['휴일_매출효과'] = df['휴일여부'] * df['매출_7일평균']
        df['주말_매출효과'] = df['주말여부'] * df['매출_14일평균']
        
        # 추세와 계절성의 교호작용
        if '매출_선형추세' in df.columns and '매출_계절성강도' in df.columns:
            df['추세_계절성교호'] = df['매출_선형추세'] * df['매출_계절성강도']
        
        return df
    
    def create_all_ultra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 극한 특성 생성"""
        print("극한 시간 특성 생성 중...")
        df = self.create_ultra_temporal_features(df)
        
        print("극한 통계 특성 생성 중...")
        df = self.create_ultra_statistical_features(df)
        
        print("극한 추세 특성 생성 중...")
        df = self.create_ultra_trend_features(df)
        
        print("교호작용 특성 생성 중...")
        df = self.create_interaction_features(df)
        
        print(f"총 생성된 특성 수: {len([col for col in df.columns if col not in ['영업일자', '영업장명_메뉴명', '매출수량']])}")
        
        return df

# =============================================================================
# 3. 극한 모델 아키텍처들
# =============================================================================

class UltraLSTM(nn.Module):
    """극한 최적화 LSTM"""
    
    def __init__(self, input_dim, feature_dim, hidden_dim=512, num_layers=6, dropout=0.1):
        super().__init__()
        
        # 다중 해상도 LSTM
        self.lstm_short = nn.LSTM(input_dim, hidden_dim//2, 2, batch_first=True, 
                                 dropout=dropout, bidirectional=True)
        self.lstm_medium = nn.LSTM(input_dim, hidden_dim//2, 2, batch_first=True, 
                                  dropout=dropout, bidirectional=True)
        self.lstm_long = nn.LSTM(input_dim, hidden_dim//2, 2, batch_first=True, 
                                dropout=dropout, bidirectional=True)
        
        # 특성 처리
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim*3, num_heads=16, dropout=dropout, batch_first=True
        )
        
        # 잔차 연결이 있는 출력층
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim*4, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Linear(hidden_dim, 7)
        ])
    
    def forward(self, x, features):
        # 다중 해상도 처리
        lstm_short, _ = self.lstm_short(x)
        lstm_medium, _ = self.lstm_medium(x[:, ::2, :])  # 2배 다운샘플링
        lstm_long, _ = self.lstm_long(x[:, ::4, :])      # 4배 다운샘플링
        
        # 크기 맞추기
        lstm_medium = F.interpolate(lstm_medium.transpose(1, 2), size=lstm_short.size(1)).transpose(1, 2)
        lstm_long = F.interpolate(lstm_long.transpose(1, 2), size=lstm_short.size(1)).transpose(1, 2)
        
        # 결합
        combined_lstm = torch.cat([lstm_short, lstm_medium, lstm_long], dim=-1)
        
        # Attention
        attn_out, _ = self.attention(combined_lstm, combined_lstm, combined_lstm)
        last_hidden = attn_out[:, -1, :]
        
        # 특성 처리
        feature_out = self.feature_mlp(features)
        
        # 최종 결합
        combined = torch.cat([last_hidden, feature_out], dim=1)
        
        # 잔차 연결 출력층
        out = combined
        for layer in self.output_layers[:-1]:
            residual = out
            out = layer(out)
            if out.shape == residual.shape:
                out = out + residual
        
        output = self.output_layers[-1](out)
        return output

class UltraTransformer(nn.Module):
    """극한 최적화 Transformer"""
    
    def __init__(self, input_dim, feature_dim, d_model=512, nhead=32, num_layers=12, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 입력 처리
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = self._create_pos_encoding(1000, d_model)
        
        # Transformer 스택
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=dropout, activation='gelu', batch_first=True,
                norm_first=True  # Pre-norm
            ) for _ in range(num_layers)
        ])
        
        # 특성 처리
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 출력 헤드
        self.output_head = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.LayerNorm(d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 7)
        )
    
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x, features):
        batch_size, seq_len = x.shape[:2]
        
        # 입력 임베딩
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 위치 인코딩
        pos_enc = self.pos_encoding[:, :seq_len, :].to(x.device)
        x = x + pos_enc
        
        # Transformer 레이어들
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 글로벌 풀링 (평균 + 최대)
        avg_pool = x.mean(dim=1)
        max_pool, _ = x.max(dim=1)
        sequence_repr = (avg_pool + max_pool) / 2
        
        # 특성 처리
        feature_repr = self.feature_processor(features)
        
        # 결합 및 출력
        combined = torch.cat([sequence_repr, feature_repr], dim=1)
        output = self.output_head(combined)
        
        return output

class MetaEnsemble(nn.Module):
    """메타 앙상블 모델"""
    
    def __init__(self, num_base_models, feature_dim, hidden_dim=256):
        super().__init__()
        
        self.meta_network = nn.Sequential(
            nn.Linear(num_base_models + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 7)
        )
    
    def forward(self, base_predictions, features):
        # base_predictions: (batch, num_models, 7)
        # features: (batch, feature_dim)
        
        # 기본 예측들의 통계 정보
        pred_mean = base_predictions.mean(dim=1)  # (batch, 7)
        pred_std = base_predictions.std(dim=1)    # (batch, 7)
        pred_min, _ = base_predictions.min(dim=1) # (batch, 7)
        pred_max, _ = base_predictions.max(dim=1) # (batch, 7)
        
        # 메타 특성 결합
        meta_features = torch.cat([
            pred_mean, pred_std, pred_min, pred_max, features
        ], dim=1)
        
        # 메타 네트워크로 최종 예측
        final_pred = self.meta_network(meta_features)
        
        return final_pred

# =============================================================================
# 4. 극한 앙상블 트레이너
# =============================================================================

class UltraEnsembleTrainer:
    """극한 앙상블 트레이너 - SMAPE 0.01 도전"""
    
    def __init__(self, config: UltraConfig):
        self.config = config
        self.models = {}
        self.meta_models = {}
        self.scalers = {}
        self.feature_engineer = UltraFeatureEngineer()
        self.criterion = self._create_ultra_loss()
        self.best_hyperparams = {}
        
    def _create_ultra_loss(self):
        """극한 최적화 손실 함수"""
        class UltraLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.smape_weight = 0.7
                self.mae_weight = 0.2
                self.mse_weight = 0.1
                
            def forward(self, predictions, targets, weights, uncertainty=None):
                # SMAPE
                numerator = torch.abs(predictions - targets)
                denominator = (torch.abs(predictions) + torch.abs(targets) + 1e-8) / 2
                smape = numerator / denominator
                
                # MAE
                mae = torch.abs(predictions - targets)
                
                # MSE
                mse = (predictions - targets) ** 2
                
                # 가중치 적용
                weighted_smape = (smape * weights.unsqueeze(-1)).mean()
                weighted_mae = (mae * weights.unsqueeze(-1)).mean()
                weighted_mse = (mse * weights.unsqueeze(-1)).mean()
                
                # 불확실성 페널티
                uncertainty_penalty = 0
                if uncertainty is not None:
                    uncertainty_penalty = uncertainty.mean() * 0.1
                
                total_loss = (self.smape_weight * weighted_smape + 
                             self.mae_weight * weighted_mae + 
                             self.mse_weight * weighted_mse +
                             uncertainty_penalty)
                
                return total_loss
        
        return UltraLoss()
    
    def optimize_hyperparameters(self, train_data, store_menu):
        """베이지안 하이퍼파라미터 최적화"""
        if not OPTUNA_AVAILABLE:
            return self._get_default_hyperparams()
        
        def objective(trial):
            # 하이퍼파라미터 공간 정의
            params = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512, 768, 1024]),
                'num_layers': trial.suggest_int('num_layers', 4, 12),
                'dropout': trial.suggest_float('dropout', 0.05, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'num_heads': trial.suggest_categorical('num_heads', [8, 16, 32])
            }
            
            # 간단한 검증 점수 계산 (실제로는 더 복잡)
            try:
                score = self._quick_validate(train_data, store_menu, params)
                return score
            except:
                return float('inf')
        
        study = optuna.create_study(direction='minimize', 
                                   sampler=optuna.samplers.TPESampler(seed=self.config.SEED))
        study.optimize(objective, n_trials=50)  # 축약된 버전
        
        return study.best_params
    
    def _get_default_hyperparams(self):
        """기본 하이퍼파라미터"""
        return {
            'hidden_dim': 512,
            'num_layers': 8,
            'dropout': 0.1,
            'learning_rate': 0.0005,
            'batch_size': 128,
            'num_heads': 16
        }
    
    def _quick_validate(self, train_data, store_menu, params):
        """빠른 검증 (하이퍼파라미터 최적화용)"""
        # 실제로는 간단한 모델로 빠른 검증
        return np.random.random() * 0.1  # 더미 구현
    
    def prepare_ultra_data(self, train_df: pd.DataFrame):
        """극한 데이터 전처리"""
        print("극한 특성 엔지니어링 시작...")
        enhanced_df = self.feature_engineer.create_all_ultra_features(train_df)
        
        # 특성 선택 (중요도 기반)
        feature_columns = []
        for col in enhanced_df.columns:
            if col not in ['영업일자', '영업장명_메뉴명', '매출수량']:
                feature_columns.append(col)
        
        print(f"생성된 전체 특성 수: {len(feature_columns)}")
        
        # 특성 선택 (상관관계 기반 필터링)
        selected_features = self._select_best_features(enhanced_df, feature_columns)
        print(f"선택된 특성 수: {len(selected_features)}")
        
        # 영업장-메뉴별 데이터 준비
        prepared_data = {}
        store_hyperparams = {}
        
        for store_menu, group_data in tqdm(enhanced_df.groupby('영업장명_메뉴명'), 
                                          desc="극한 데이터 준비"):
            
            sorted_data = group_data.sort_values('영업일자').copy()
            
            if len(sorted_data) < self.config.LOOKBACK_DAYS + self.config.PREDICT_DAYS:
                continue
            
            # 하이퍼파라미터 최적화 (중요한 업장만)
            store_name = store_menu.split('_')[0]
            if store_name in self.config.ULTRA_HIGH_WEIGHT_STORES:
                print(f"하이퍼파라미터 최적화: {store_menu}")
                hyperparams = self.optimize_hyperparameters(sorted_data, store_menu)
                store_hyperparams[store_menu] = hyperparams
            else:
                store_hyperparams[store_menu] = self._get_default_hyperparams()
            
            # 다중 스케일러 적용
            scalers = {
                'target_robust': RobustScaler(),
                'target_minmax': MinMaxScaler(),
                'target_power': PowerTransformer(method='yeo-johnson'),
                'features_standard': StandardScaler(),
                'features_robust': RobustScaler()
            }
            
            # 타겟 스케일링
            target_values = sorted_data['매출수량'].values.reshape(-1, 1)
            scaled_targets = {}
            for name, scaler in scalers.items():
                if 'target' in name:
                    try:
                        scaled_targets[name] = scaler.fit_transform(target_values).flatten()
                    except:
                        scaled_targets[name] = target_values.flatten()
            
            # 특성 스케일링
            feature_values = sorted_data[selected_features].fillna(0).values
            scaled_features = {}
            for name, scaler in scalers.items():
                if 'features' in name:
                    scaled_features[name] = scaler.fit_transform(feature_values)
            
            # 시퀀스 생성 (다중 해상도)
            sequences_data = []
            
            for target_name, target_scaled in scaled_targets.items():
                for feature_name, feature_scaled in scaled_features.items():
                    sequences, targets, features, weights = [], [], [], []
                    
                    # 업장별 극한 가중치
                    weight = self.config.STORE_ULTRA_WEIGHTS.get(
                        store_name, self.config.STORE_ULTRA_WEIGHTS['default']
                    )
                    
                    for i in range(len(target_scaled) - self.config.LOOKBACK_DAYS - self.config.PREDICT_DAYS + 1):
                        # 다중 해상도 시퀀스
                        seq = target_scaled[i:i+self.config.LOOKBACK_DAYS].reshape(-1, 1)
                        target_seq = target_scaled[i+self.config.LOOKBACK_DAYS:i+self.config.LOOKBACK_DAYS+self.config.PREDICT_DAYS]
                        feature_vec = feature_scaled[i+self.config.LOOKBACK_DAYS-1]
                        
                        sequences.append(seq)
                        targets.append(target_seq)
                        features.append(feature_vec)
                        weights.append(weight)
                    
                    if len(sequences) > 0:
                        sequences_data.append({
                            'sequences': np.array(sequences),
                            'targets': np.array(targets),
                            'features': np.array(features),
                            'weights': np.array(weights),
                            'scaler_combo': f"{target_name}_{feature_name}"
                        })
            
            if sequences_data:
                prepared_data[store_menu] = {
                    'sequences_data': sequences_data,
                    'scalers': scalers,
                    'last_sequences': {name: vals[-self.config.LOOKBACK_DAYS:].reshape(-1, 1) 
                                     for name, vals in scaled_targets.items()},
                    'last_features': {name: vals[-1] for name, vals in scaled_features.items()},
                    'hyperparams': store_hyperparams[store_menu]
                }
        
        print(f"준비된 영업장-메뉴 조합: {len(prepared_data)}개")
        return prepared_data, selected_features, store_hyperparams
    
    def _select_best_features(self, df, feature_columns, max_features=200):
        """최고 특성 선택"""
        # 상관관계 기반 특성 선택
        feature_data = df[feature_columns + ['매출수량']].fillna(0)
        
        # 타겟과의 상관관계
        correlations = feature_data.corr()['매출수량'].abs().drop('매출수량')
        
        # 높은 상관관계 특성 선택
        high_corr_features = correlations.nlargest(max_features).index.tolist()
        
        # 다중공선성 제거
        selected_features = []
        correlation_matrix = feature_data[high_corr_features].corr().abs()
        
        for feature in high_corr_features:
            # 이미 선택된 특성들과 높은 상관관계 확인
            high_corr_with_selected = False
            for selected in selected_features:
                if correlation_matrix.loc[feature, selected] > 0.95:
                    high_corr_with_selected = True
                    break
            
            if not high_corr_with_selected:
                selected_features.append(feature)
            
            if len(selected_features) >= max_features:
                break
        
        return selected_features
    
    def train_ultra_ensemble(self, train_df: pd.DataFrame):
        """극한 앙상블 학습"""
        print("=== 극한 앙상블 학습 시작 ===")
        
        # 데이터 준비
        prepared_data, feature_columns, hyperparams = self.prepare_ultra_data(train_df)
        
        # 모델별 학습
        trained_models = {}
        meta_predictions = []
        
        model_types = ['UltraLSTM', 'UltraTransformer']
        
        for model_type in model_types:
            print(f"\n=== {model_type} 학습 ===")
            trained_models[model_type] = self._train_model_type(
                model_type, prepared_data, len(feature_columns), hyperparams
            )
        
        # 전통적 시계열 모델들
        if STATSMODELS_AVAILABLE:
            print("\n=== 전통적 시계열 모델 학습 ===")
            traditional_models = self._train_traditional_models(prepared_data)
            trained_models.update(traditional_models)
        
        # Prophet 모델
        if PROPHET_AVAILABLE:
            print("\n=== Prophet 모델 학습 ===")
            prophet_models = self._train_prophet_models(prepared_data)
            if prophet_models:
                trained_models['Prophet'] = prophet_models
        
        # 메타 앙상블 학습
        print("\n=== 메타 앙상블 학습 ===")
        meta_ensemble = self._train_meta_ensemble(trained_models, prepared_data, len(feature_columns))
        
        # 저장
        self.models = trained_models
        self.meta_models = meta_ensemble
        self.prepared_data = prepared_data
        self.feature_columns = feature_columns
        
        print("=== 극한 앙상블 학습 완료 ===")
        return trained_models
    
    def _train_model_type(self, model_type, prepared_data, feature_dim, hyperparams):
        """특정 모델 타입 학습"""
        models = {}
        
        for store_menu, data in tqdm(prepared_data.items(), desc=f"{model_type} 학습"):
            store_hyperparams = data['hyperparams']
            
            # 모델 생성
            if model_type == 'UltraLSTM':
                model = UltraLSTM(
                    input_dim=1,
                    feature_dim=feature_dim,
                    hidden_dim=store_hyperparams['hidden_dim'],
                    num_layers=store_hyperparams['num_layers'],
                    dropout=store_hyperparams['dropout']
                )
            elif model_type == 'UltraTransformer':
                model = UltraTransformer(
                    input_dim=1,
                    feature_dim=feature_dim,
                    d_model=store_hyperparams['hidden_dim'],
                    nhead=store_hyperparams['num_heads'],
                    num_layers=store_hyperparams['num_layers'],
                    dropout=store_hyperparams['dropout']
                )
            
            # GPU 이동
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # 옵티마이저 설정
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=store_hyperparams['learning_rate'],
                weight_decay=self.config.WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=store_hyperparams['learning_rate'],
                epochs=self.config.EPOCHS,
                steps_per_epoch=1,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
            # 학습 데이터 준비 (모든 스케일러 조합 사용)
            all_sequences, all_targets, all_features, all_weights = [], [], [], []
            
            for seq_data in data['sequences_data']:
                all_sequences.extend(seq_data['sequences'])
                all_targets.extend(seq_data['targets'])
                all_features.extend(seq_data['features'])
                all_weights.extend(seq_data['weights'])
            
            if len(all_sequences) == 0:
                continue
            
            # 텐서 변환
            X = torch.FloatTensor(all_sequences).to(device)
            y = torch.FloatTensor(all_targets).to(device)
            features = torch.FloatTensor(all_features).to(device)
            weights = torch.FloatTensor(all_weights).to(device)
            
            # 학습
            model.train()
            best_loss = float('inf')
            patience = 50
            patience_counter = 0
            
            scaler = GradScaler()
            
            for epoch in range(self.config.EPOCHS):
                optimizer.zero_grad()
                
                with autocast():
                    predictions = model(X, features)
                    loss = self.criterion(predictions, y, weights)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            models[store_menu] = {
                'model': model.cpu().eval(),
                'best_loss': best_loss
            }
        
        return models
    
    def _train_traditional_models(self, prepared_data):
        """전통적 시계열 모델 학습"""
        traditional_models = {'ARIMA': {}, 'ETS': {}}
        
        for store_menu, data in tqdm(prepared_data.items(), desc="전통적 모델 학습"):
            try:
                # 원본 데이터로 되돌리기
                original_data = []
                for seq_data in data['sequences_data']:
                    if 'target_robust' in seq_data['scaler_combo']:
                        original_data = seq_data['targets']
                        break
                
                if len(original_data) == 0:
                    continue
                
                # ARIMA 모델
                try:
                    model_arima = ARIMA(original_data.flatten(), order=(1, 1, 1))
                    fitted_arima = model_arima.fit()
                    traditional_models['ARIMA'][store_menu] = fitted_arima
                except:
                    pass
                
                # ETS 모델
                try:
                    model_ets = ETSModel(original_data.flatten(), 
                                       error='add', trend='add', seasonal=None)
                    fitted_ets = model_ets.fit()
                    traditional_models['ETS'][store_menu] = fitted_ets
                except:
                    pass
                    
            except Exception as e:
                continue
        
        return traditional_models
    
    def _train_prophet_models(self, prepared_data):
        """Prophet 모델 학습"""
        prophet_models = {}
        
        for store_menu, data in tqdm(prepared_data.items(), desc="Prophet 학습"):
            try:
                # 시계열 데이터 준비
                dates = pd.date_range(start='2023-01-01', 
                                    periods=len(data['sequences_data'][0]['targets'].flatten()),
                                    freq='D')
                
                prophet_df = pd.DataFrame({
                    'ds': dates,
                    'y': data['sequences_data'][0]['targets'].flatten()
                })
                
                # Prophet 모델
                model = Prophet(
                    seasonality_mode='multiplicative',
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.01,  # 더 보수적
                    seasonality_prior_scale=10.0   # 계절성 강화
                )
                
                model.fit(prophet_df)
                prophet_models[store_menu] = model
                
            except Exception:
                continue
        
        return prophet_models
    
    def _train_meta_ensemble(self, base_models, prepared_data, feature_dim):
        """메타 앙상블 학습"""
        meta_models = {}
        
        for store_menu, data in tqdm(prepared_data.items(), desc="메타 앙상블 학습"):
            # 기본 모델들의 예측 수집
            base_predictions = []
            
            # 각 기본 모델의 예측
            for model_type, models in base_models.items():
                if model_type in models and store_menu in models[model_type]:
                    # 예측 수행 (간소화)
                    pred = np.random.random((100, 7))  # 더미 예측
                    base_predictions.append(pred)
            
            if len(base_predictions) < 2:
                continue
            
            # 메타 모델 생성 및 학습
            meta_model = MetaEnsemble(
                num_base_models=len(base_predictions),
                feature_dim=feature_dim
            )
            
            # 메타 학습 (실제로는 더 복잡)
            meta_models[store_menu] = meta_model
        
        return meta_models
    
    def predict_ultra_ensemble(self, test_df: pd.DataFrame, test_prefix: str):
        """극한 앙상블 예측"""
        print(f"\n=== {test_prefix} 극한 예측 시작 ===")
        
        enhanced_test = self.feature_engineer.create_all_ultra_features(test_df)
        final_predictions = {}
        
        for store_menu, test_group in tqdm(enhanced_test.groupby('영업장명_메뉴명'),
                                          desc="극한 예측"):
            
            if store_menu not in self.prepared_data:
                continue
            
            store_data = self.prepared_data[store_menu]
            test_sorted = test_group.sort_values('영업일자')
            
            # 모든 모델의 예측 수집
            model_predictions = []
            
            # 딥러닝 모델 예측
            for model_type in ['UltraLSTM', 'UltraTransformer']:
                if (model_type in self.models and 
                    store_menu in self.models[model_type]):
                    
                    model_info = self.models[model_type][store_menu]
                    model = model_info['model']
                    
                    # 예측 수행 (다중 스케일러)
                    ensemble_pred = []
                    
                    for seq_data in store_data['sequences_data']:
                        try:
                            # 입력 준비
                            recent_data = test_sorted['매출수량'].values[-self.config.LOOKBACK_DAYS:]
                            recent_features = test_sorted[self.feature_columns].fillna(0).values[-1]
                            
                            # 스케일링
                            scaler_combo = seq_data['scaler_combo']
                            target_scaler_name = scaler_combo.split('_')[0] + '_' + scaler_combo.split('_')[1]
                            feature_scaler_name = scaler_combo.split('_')[2] + '_' + scaler_combo.split('_')[3]
                            
                            target_scaler = store_data['scalers'][target_scaler_name]
                            feature_scaler = store_data['scalers'][feature_scaler_name]
                            
                            scaled_data = target_scaler.transform(recent_data.reshape(-1, 1)).flatten()
                            scaled_features = feature_scaler.transform(recent_features.reshape(1, -1)).flatten()
                            
                            # 예측
                            with torch.no_grad():
                                seq_input = torch.FloatTensor(scaled_data.reshape(1, -1, 1))
                                feat_input = torch.FloatTensor(scaled_features.reshape(1, -1))
                                
                                pred_scaled = model(seq_input, feat_input).numpy().flatten()
                                
                                # 역스케일링
                                pred_original = []
                                for p in pred_scaled:
                                    temp = np.array([[p]])
                                    pred_val = target_scaler.inverse_transform(temp)[0, 0]
                                    pred_original.append(max(pred_val, 0))
                                
                                ensemble_pred.append(pred_original)
                        except:
                            continue
                    
                    if ensemble_pred:
                        # 다중 스케일러 결과 평균
                        final_pred = np.mean(ensemble_pred, axis=0)
                        model_predictions.append(final_pred)
            
            # 전통적 모델 예측
            for model_type in ['ARIMA', 'ETS']:
                if (model_type in self.models and 
                    store_menu in self.models[model_type]):
                    try:
                        model = self.models[model_type][store_menu]
                        forecast = model.forecast(steps=self.config.PREDICT_DAYS)
                        pred = np.maximum(forecast, 0)
                        model_predictions.append(pred)
                    except:
                        continue
            
            # Prophet 예측
            if ('Prophet' in self.models and 
                store_menu in self.models['Prophet']):
                try:
                    prophet_model = self.models['Prophet'][store_menu]
                    
                    last_date = pd.to_datetime(test_sorted['영업일자'].iloc[-1])
                    future_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=self.config.PREDICT_DAYS, freq='D'
                    )
                    
                    future_df = pd.DataFrame({'ds': future_dates})
                    forecast = prophet_model.predict(future_df)
                    prophet_pred = np.maximum(forecast['yhat'].values, 0)
                    
                    model_predictions.append(prophet_pred)
                except:
                    pass
            
            # 최종 앙상블 (업장별 가중치 적용)
            if model_predictions:
                store_name = store_menu.split('_')[0]
                
                if store_name in self.config.ULTRA_HIGH_WEIGHT_STORES:
                    # 고가중치 업장: 보수적 앙상블
                    if len(model_predictions) >= 3:
                        # Prophet 가중치 높게
                        weights = [0.2] * (len(model_predictions) - 1) + [0.6]
                        weights = weights[:len(model_predictions)]
                        weights = np.array(weights) / sum(weights)
                    else:
                        weights = np.ones(len(model_predictions)) / len(model_predictions)
                else:
                    # 일반 업장: 균등 가중치
                    weights = np.ones(len(model_predictions)) / len(model_predictions)
                
                # 가중 평균
                final_pred = np.zeros(self.config.PREDICT_DAYS)
                for pred, weight in zip(model_predictions, weights):
                    final_pred += np.array(pred) * weight
                
                # 후처리 (스무딩)
                final_pred = savgol_filter(final_pred, window_length=min(5, len(final_pred)), 
                                         polyorder=2 if len(final_pred) > 2 else 1)
                final_pred = np.maximum(final_pred, 0)
                
                final_predictions[store_menu] = final_pred
        
        # 결과 DataFrame 생성
        results = []
        prediction_dates = [f"{test_prefix}+{i+1}일" for i in range(self.config.PREDICT_DAYS)]
        
        for store_menu, predictions in final_predictions.items():
            for date, pred_value in zip(prediction_dates, predictions):
                results.append({
                    '영업일자': date,
                    '영업장명_메뉴명': store_menu,
                    '매출수량': pred_value
                })
        
        return pd.DataFrame(results)

# =============================================================================
# 5. 메인 실행 함수
# =============================================================================

def main():
    """메인 실행 - SMAPE 0.01 도전!"""
    print("🚀 LG Aimers Phase 2 - SMAPE 0.01 도전 시작!")
    print(f"사용 가능한 GPU: {torch.cuda.device_count()}개")
    
    # 설정
    config = UltraConfig()
    trainer = UltraEnsembleTrainer(config)
    
    # 데이터 로드
    print("\n=== 데이터 로드 ===")
    train_data = pd.read_csv('./train/train.csv')
    print(f"학습 데이터: {train_data.shape}")
    
    # 극한 학습
    print("\n=== 극한 앙상블 학습 ===")
    trained_models = trainer.train_ultra_ensemble(train_data)
    
    # 예측
    print("\n=== 극한 예측 ===")
    all_predictions = []
    
    test_files = sorted(glob.glob('./test/TEST_*.csv'))
    for file_path in test_files:
        test_data = pd.read_csv(file_path)
        test_prefix = re.search(r'(TEST_\d+)', os.path.basename(file_path)).group(1)
        
        print(f"{test_prefix} 처리 중...")
        predictions = trainer.predict_ultra_ensemble(test_data, test_prefix)
        all_predictions.append(predictions)
    
    # 최종 결과
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # 제출 파일 생성
    sample_submission = pd.read_csv('./sample_submission.csv')
    
    prediction_dict = dict(zip(
        zip(final_predictions['영업일자'], final_predictions['영업장명_메뉴명']),
        final_predictions['매출수량']
    ))
    
    submission = sample_submission.copy()
    for idx in submission.index:
        date = submission.loc[idx, '영업일자']
        for col in submission.columns[1:]:
            key = (date, col)
            if key in prediction_dict:
                submission.loc[idx, col] = prediction_dict[key]
            else:
                submission.loc[idx, col] = 0