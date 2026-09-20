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
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)

HIDDEN_SIZE = 256
BUFFER_SIZE = 200000
BATCH_SIZE = 256
NUM_EPISODES = 2000        # 짧은 에피소드 -> 개수 늘림 (2000*64 = 128k step)
MAX_STEPS = 64             # reset -> 몇 스텝 refine. 1000 은 return 이 상수에 묻힘
WARMUP_STEPS = 2000
REWARD_SCALE = 250.0       # 개선량 보상 r=J_prev-J 는 스텝당 ~1e-3 -> 키운다.
                           # 5000은 너무 컸음: 1500ep 실측 로그에서 alpha~0.04,
                           # log_probs~+25일 때 actor loss의 alpha*log_probs 항은
                           # ~1인데 Q항은 ~590 (500배 차이) — 엔트로피 보너스가
                           # Q에 완전히 묻혀서 target_entropy 오토튜닝이 policy
                           # gradient에 실질적 영향을 못 줌. Q는 reward에 선형
                           # 비례(감가 누적)하므로 250 = 5000*(30/590) 목표: Q를
                           # 수십 단위로 낮춰 alpha*log_probs와 비슷한 자릿수로.
EVAL_EVERY = 100           # N 에피소드마다 결정론 롤아웃 평가
RHO_MAX = 5.0              # within 은 큰 rho_max 에서 셀 굶기기로 뚫린다. 3~5 권장.
OBJECTIVE = "within"       # (근본 수정은 objective 재설계: 조각수 가중 within + 굶은셀 페널티 + min_len>=50)
WINDOW_MIN = 90            # pems_pipeline.py --span 기본값. 아래 PHF 계산에 쓰임
PHF = 60.0 / WINDOW_MIN    # 시간창 파일의 vol_day_veh 는 '창 동안의 통과 대수'다.
                           # 이 값을 넘겨야 lam = vol/(span*60) 인 실제 창 도착률이 된다.
                           # 기본값(0.15)을 쓰면 창 카운트를 일 총량으로 오해해 ~4.4배 과소평가.
SEGMENTS_FILE = f"{DIR}/outputs_2008/pems_d07_segments_0841_after.csv"
SITES_FILE    = f"{DIR}/outputs_2008/pems_d07_sites.csv"
META_FILE     = f"{DIR}/d07_text_meta_2018_10_13.txt"

env = TrafficRLEnvMW(
    segments_csv=SEGMENTS_FILE,
    sites_csv=SITES_FILE,
    meta_txt=META_FILE,
    rho_max=RHO_MAX,
    objective=OBJECTIVE,
    phf=PHF,
)
STATE_DIM = env.obs_dim      # [a(K), 셀별 std(K), 굶은셀 마스크(K), J(1)]
ACTION_DIM = env.K           # 사이트별 Δa
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
    
    _, _, _, info0 = env.step(np.zeros(ACTION_DIM))   # 무영향 액션 -> baseline J
    print(f"[Test 1] 균등 가중치 baseline J: {info0['J']:.8f}")
    
except FileNotFoundError as e:
    print(f"\n[오류] 데이터 파일을 찾을 수 없습니다: {e}")

    
import json
import os

# =====================
# 학습 루프
# =====================
print("학습 시작...")


def eval_policy(n_steps=MAX_STEPS):
    """결정론 롤아웃 (tanh(mu)). 정책이 실제로 뭘 배웠는지 본다."""
    s = np.asarray(env.reset(), np.float32)
    best = np.inf
    info = {"J": np.nan}
    for _ in range(n_steps):
        a, _ = policy(tf.expand_dims(s, 0), deterministic=True)
        s, _, _, info = env.step(a[0].numpy())
        s = np.asarray(s, np.float32)
        best = min(best, info["J"])
    return best, info["J"]


global_step = 0
for episode in range(NUM_EPISODES):
    state = np.asarray(env.reset(), np.float32)
    episode_reward = 0.0
    q1_loss = policy_loss = alpha_loss = 0.0

    for step in range(MAX_STEPS):
        if buffer.size < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=(ACTION_DIM,)).astype('float32')
        else:
            action, _ = policy(tf.expand_dims(state, 0))
            action = action[0].numpy()
        
        # 환경과 상호작용
        next_state, reward, done, info = env.step(action)
        next_state = np.asarray(next_state, np.float32)
        reward *= REWARD_SCALE

        # 리플레이 버퍼에 저장
        buffer.add(state, action, reward, next_state, float(done))
        episode_reward += reward
        
        # 학습 (버퍼에 충분한 데이터가 있으면)
        if buffer.size > WARMUP_STEPS:
            batch = buffer.sample(BATCH_SIZE)
            
            # critic -> actor -> alpha -> target 을 그래프 하나로 실행 (@tf.function)
            q1_loss, q2_loss, policy_loss, alpha_loss = sac.train_step(batch)

        # ---- 계측 ----
        if global_step % 200 == 0:
            sat = float(np.mean(np.abs(action) > 0.99))
            da  = float(env.a.max() - env.a.min())
            print(f"[dbg] ep{episode} gs{global_step} J={info['J']:.5f} "
                  f"a_range={da:.3f} sat={sat:.2f} alpha={float(sac.alpha):.3f} "
                  f"q1={float(q1_loss):.3f} pi={float(policy_loss):.3f}")
        if buffer.size > WARMUP_STEPS and global_step % 1000 == 0:
            b = buffer.sample(BATCH_SIZE)
            na, nlp = policy(b[3])
            mq = tf.minimum(q1(tf.concat([b[3], na], -1)), q2(tf.concat([b[3], na], -1)))
            qt = b[2] + (1.0 - b[4]) * 0.99 * (mq - sac.alpha * nlp)
            print(f"[dbg]   next_logp={float(tf.reduce_mean(nlp)):+.2f} "
                  f"q_target_mean={float(tf.reduce_mean(qt)):+.2f} "
                  f"q_target_std={float(tf.math.reduce_std(qt)):.2f}")

        state = next_state
        global_step += 1

        if done:
            break
    
    msg = f"Episode {episode + 1}, Reward(scaled): {episode_reward:.2f}, Buffer: {buffer.size}"
    if (episode + 1) % EVAL_EVERY == 0 and buffer.size > WARMUP_STEPS:
        b_best, b_last = eval_policy()
        msg += f"  | eval best J={b_best:.5f} last J={b_last:.5f}"
    print(msg)

print("학습 완료!")
print("finalA (best J 방문):", env.finalA)
print("finalW:", env.finalW)


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