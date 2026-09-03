import tensorflow as tf
import numpy as np
from sac_algorithm import SAC
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
parent_dir = current_file.parents[3]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

DIR = str(parent_dir / "features" / "data_preprocessing" / "vor_sd") 
from features.data_preprocessing.vor_sd.rl_env_voronoi_mw import TrafficRLEnvMW

from neural_networks import GaussianPolicy, QNetwork
from replay_buffer import ReplayBuffer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import sys

# 로그 파일 설정
log_file = open('log.txt', 'w', buffering=1)  # buffering=1: 한 줄씩 즉시 저장

class Logger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file
    
    def write(self, message):
        self.terminal.write(message)  # 터미널에도 출력
        self.file.write(message)      # 파일에도 저장
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()

sys.stdout = Logger(log_file)
# =====================
# 환경 설정 (당신의 2D 맵)
# =====================
class SimpleEnv:
    """예제용 간단한 환경"""
    #이거 수정해야함
    def __init__(self):
        self.state_dim = 4  # x, y, vx, vy
        self.action_dim = 2  # ax, ay
        self.reset()
    
    def reset(self):
        self.state = np.random.uniform(-1, 1, self.state_dim).astype('float32')
        return self.state
    
    def step(self, action):
        # 1. 현재 상태 분리 (위치 x, y / 속도 vx, vy)
        x, y, vx, vy = self.state
        
        # 2. 액션 분리 (가속도 ax, ay)
        ax, ay = action
        
        # 3. 물리 법칙 적용 (가속도로 속도 변경 -> 속도로 위치 변경)
        dt = 0.1
        next_vx = vx + ax * dt
        next_vy = vy + ay * dt
        next_x = x + next_vx * dt
        next_y = y + next_vy * dt
        
        # 4. 새로운 상태 결합 및 Clip (경계 제한)
        self.state = np.array([next_x, next_y, next_vx, next_vy], dtype=np.float32)
        self.state = np.clip(self.state, -10, 10).astype('float32')
        
        # 보상: 원점(위치 x, y)에 가까울수록 높음
        # (전체 state가 아니라 위치인 x, y만 가지고 보상을 계산하는 것이 더 직관적입니다)
        reward = -np.sum(self.state[:2] ** 2)
        done = False
        
        return self.state, reward, done, {}
# =====================
# 하이퍼파라미터
# =====================
HIDDEN_SIZE = 256
BUFFER_SIZE = 100000
BATCH_SIZE = 256
NUM_EPISODES = 100
MAX_STEPS = 1000
SEGMENTS_FILE = f"{DIR}/outputs/pems_d07_segments.csv"
SITES_FILE    = f"{DIR}/outputs/pems_d07_sites.csv"
META_FILE     = f"{DIR}/d07_text_meta_2018_10_13.txt"

env = TrafficRLEnvMW(
    segments_csv=SEGMENTS_FILE, 
    sites_csv=SITES_FILE, 
    meta_txt=META_FILE
)

it=100000
mnval = 9999999
res=0
a=[]
for i in range(it):
    env.reset()
    val, ans=(env.evaluate(env.a))
    if val < mnval:n  
        mnval = val
        res=ans
    a.append(val)
    

import matplotlib.pyplot as plt
a.sort()
plt.hist(a, bins=30, color='green', edgecolor='black')
plt.title('분포도 히스토그램')
plt.savefig("variation histogram_3", dpi=150, bbox_inches="tight")
plt.close()

print(mnval,res)