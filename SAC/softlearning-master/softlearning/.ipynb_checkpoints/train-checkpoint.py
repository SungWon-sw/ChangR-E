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
STATE_DIM = env.K
ACTION_DIM = 4
# =====================
# 네트워크 초기화
# =====================
policy = GaussianPolicy(STATE_DIM, ACTION_DIM, hidden_size=HIDDEN_SIZE)
q1 = QNetwork(hidden_size=HIDDEN_SIZE)
q2 = QNetwork(hidden_size=HIDDEN_SIZE)

# SAC 초기화
sac = SAC(policy, q1, q2, STATE_DIM, ACTION_DIM,
          policy_lr=1e-4,
          q_lr=1e-4,
          alpha_lr=1e-4)
# 리플레이 버퍼
buffer = ReplayBuffer(max_size=BUFFER_SIZE, state_dim=STATE_DIM, action_dim=ACTION_DIM)

# 환경
try:
    print("\n--- [학습 루프 테스트: 동적 도로 분할 연산 시뮬레이션] ---")
    
    uniform_weights = np.zeros(env.K)
    tst1, tst2, tst3, tst4 = env.step(uniform_weights)
    print(f"[Test 1] 균등 가중치(유클리드) 적용 시 평균 표준편차: {-tst2:.8f}")
    
except FileNotFoundError as e:
    print(f"\n[오류] 데이터 파일을 찾을 수 없습니다: {e}")

    
import json
import os

# =====================
# 학습 루프
# =====================
print("학습 시작...")
WARMUP_STEPS = 5000  # 추가

for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_reward = 0
    
    for step in range(MAX_STEPS):
        if buffer.size < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=(ACTION_DIM,)).astype('float32')
        else:
            action, _ = policy(tf.expand_dims(state, 0))
            action = action[0].numpy()
        
        # 환경과 상호작용
        next_state, reward, done, _ = env.step(action)
        
        # 리플레이 버퍼에 저장
        buffer.add(state, action, reward, next_state, float(done))
        episode_reward += reward
        
        # 학습 (버퍼에 충분한 데이터가 있으면)
        if buffer.size > WARMUP_STEPS:
            batch = buffer.sample(BATCH_SIZE)
            
            sac.update_critic(batch)
            sac.update_actor(batch)
            sac.update_alpha(batch)
            sac.update_target_networks()
        
        state = next_state
        
        if done:
            break
    
    if (episode + 1) % 1 == 0:
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Buffer Size: {buffer.size}")

print("학습 완료!")
print(env.finalW)


# # =====================
# # 모델 저장
# # =====================
# save_dir = 'saved_model'
# os.makedirs(save_dir, exist_ok=True)

# # 가중치 저장
# policy.save_weights(f'{save_dir}/policy.weights.h5')
# q1.save_weights(f'{save_dir}/q1.weights.h5')
# q2.save_weights(f'{save_dir}/q2.weights.h5')

# # 하이퍼파라미터 + 학습 결과 저장
# config = {
#     "STATE_DIM":    STATE_DIM,
#     "ACTION_DIM":   ACTION_DIM,
#     "HIDDEN_SIZE":  HIDDEN_SIZE,
#     "BUFFER_SIZE":  BUFFER_SIZE,
#     "BATCH_SIZE":   BATCH_SIZE,
#     "NUM_EPISODES": NUM_EPISODES,
#     "MAX_STEPS":    MAX_STEPS,
#     "WARMUP_STEPS": WARMUP_STEPS,
#     "policy_lr":    1e-4,
#     "q_lr":         1e-4,
#     "alpha_lr":     1e-4,
#     "log_alpha_final": float(sac.log_alpha.numpy()),
#     "alpha_final":     float(sac.alpha.numpy()),
# }
# with open(f'{save_dir}/config.json', 'w') as f:
#     json.dump(config, f, indent=2)

# print(f"모델 저장 완료 → {save_dir}/")
# print(f"최종 alpha: {config['alpha_final']:.4f}")