import tensorflow as tf
import numpy as np
from sac_algorithm import SAC평가함수
from neural_networks import GaussianPolicy, QNetwork
from replay_buffer import ReplayBuffer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

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
        self.state = np.random.randn(self.state_dim).astype('float32')
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
STATE_DIM = 4
ACTION_DIM = 2
HIDDEN_SIZE = 256
BUFFER_SIZE = 100000
BATCH_SIZE = 256
NUM_EPISODES = 100
MAX_STEPS = 1000

# =====================
# 네트워크 초기화
# =====================
policy = GaussianPolicy(STATE_DIM, ACTION_DIM, hidden_size=HIDDEN_SIZE)
q1 = QNetwork(hidden_size=HIDDEN_SIZE)
q2 = QNetwork(hidden_size=HIDDEN_SIZE)

# SAC 초기화
sac = SAC(policy, q1, q2, STATE_DIM, ACTION_DIM)

# 리플레이 버퍼
buffer = ReplayBuffer(max_size=BUFFER_SIZE, state_dim=STATE_DIM, action_dim=ACTION_DIM)

# 환경
env = SimpleEnv()

# =====================
# 학습 루프
# =====================
print("학습 시작...")
WARMUP_STEPS = 1000  # 추가

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
        if buffer.size > BATCH_SIZE:
            batch = buffer.sample(BATCH_SIZE)
            
            sac.update_critic(batch)
            sac.update_actor(batch)
            sac.update_alpha(batch)
            sac.update_target_networks()
        
        state = next_state
        
        if done:
            break
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Buffer Size: {buffer.size}")

print("학습 완료!")