"""
Simulated Annealing for TrafficRLEnvMW.

목표: env.evaluate(a) 가 돌려주는 목적함수값 J 를 최소화하는 log-가중치 벡터 a 를 찾는다.
  - a: 길이 K(=분기점 수) 벡터, 평균 0, 각 성분 |a_i| <= a_bound (= ln(rho_max)/2).
  - J = env.evaluate(a)[0]  (objective="global" -> 도로 조각별 blocking 확률의 표준편차)

접근:
  - 상태를 a 로 두고 이웃 이동(4종) 후 Metropolis 기준으로 수락/거부.
  - 이동은 "국소적/공간적으로 매끄러운" 것 위주 (MW 경계는 r/d = sigmoid(a_i - a_j) 라
    이웃 사이트 간 '차이'가 목적함수를 좌우한다).
  - 초기온도 T0 는 랜덤 이동들의 |ΔJ| 로 자동 보정(초기 수락률 ~0.5).
  - 기하냉각 + best 로 복귀하는 재가열(reanneal) 사이클.
  - 이동 크기(scale)는 최근 수락률로 적응.
  - 매 사이클마다 best 를 디스크에 저장(중간에 죽어도 결과 확보).
"""

import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------
# 경로 / 환경 설정
# ----------------------------------------------------------------------
current_file = Path(__file__).resolve()
parent_dir = current_file.parents[0]                      # .../project3
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
    
from features.data_preprocessing.vor_sd.rl_env_voronoi_mw import TrafficRLEnvMW
DIR = str(parent_dir / "features" / "data_preprocessing" / "vor_sd")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")

# 로그 파일 (터미널 + 파일 동시 출력)
log_file = open("sa_log.txt", "w", buffering=1, encoding="utf-8")


class Logger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


sys.stdout = Logger(log_file)

SEGMENTS_FILE = f"{DIR}/outputs/pems_d07_segments.csv"
SITES_FILE = f"{DIR}/outputs/pems_d07_sites.csv"
META_FILE = f"{DIR}/d07_text_meta_2018_10_13.txt"

OBJECTIVE = "global"          # "global" | "within" | "mixed"

env = TrafficRLEnvMW(
    segments_csv=SEGMENTS_FILE,
    sites_csv=SITES_FILE,
    meta_txt=META_FILE,
    objective=OBJECTIVE,
)

# ----------------------------------------------------------------------
# SA 하이퍼파라미터 (평가 1회 ~ 80ms 기준. MAX_EVALS 만 조절하면 실행시간이 정해진다)
# ----------------------------------------------------------------------
MAX_EVALS = 12000            # 총 env.evaluate 호출 예산 (~17분). 여유 있으면 30000~60000 권장.
RESTARTS = 2                 # 서로 다른 시작점에서 독립 체인 수
CYCLES = 3                   # 체인당 재가열 사이클 수
CALIB_SAMPLES = 80           # 체인마다 T0 보정용 샘플 수
TARGET_ACCEPT0 = 0.5         # 초기 목표 수락률
T_END_FRAC = 5e-4            # 사이클 종료온도 = 사이클 시작온도 * 이 값
REHEAT_FRAC = [1.0, 0.35, 0.15]   # 사이클별 시작온도 배수 (T0 기준)
SEED = 20260903

K = env.K
A_BOUND = float(env.a_bound)
COORDS = np.asarray(env.site_coords, float)               # (K, 2)
SPACING = float(env.spacing)
# 사이트 간 거리행렬 (공간 bump 이동에 사용)
DIST = np.linalg.norm(COORDS[:, None, :] - COORDS[None, :, :], axis=-1)

steps_per_chain = MAX_EVALS // RESTARTS - CALIB_SAMPLES
steps_per_cycle = max(1, steps_per_chain // CYCLES)

EVAL_COUNT = 0
T_START = time.time()


# ----------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------
def project(a):
    """평균 0 + 클리핑 (env.step 과 동일한 게이지 고정)."""
    a = a - a.mean()
    np.clip(a, -A_BOUND, A_BOUND, out=a)
    a -= a.mean()
    return a


def objective(a):
    """env.evaluate 래퍼. 퇴화(조각 없음)나 예외는 +inf 로 처리."""
    global EVAL_COUNT
    EVAL_COUNT += 1
    try:
        val, _ = env.evaluate(a)
    except Exception:
        return np.inf
    val = float(val)
    if not np.isfinite(val) or val <= 0.0:
        return np.inf
    return val


def propose(a, scale, rng):
    """이웃 상태 생성. scale 이 클수록 큰 이동."""
    b = a.copy()
    r = rng.random()
    if r < 0.45:
        # (1) 공간 bump: 임의의 사이트를 중심으로 가우시안 형태의 매끄러운 변형
        c = rng.integers(K)
        ell = SPACING * rng.uniform(0.4, 3.0)
        amp = scale * A_BOUND * rng.standard_normal()
        b += amp * np.exp(-(DIST[c] ** 2) / (2.0 * ell * ell))
    elif r < 0.75:
        # (2) 단일 사이트 킥: 해당 사이트의 모든 경계를 한 번에 움직임
        i = rng.integers(K)
        b[i] += 1.5 * scale * A_BOUND * rng.standard_normal()
    elif r < 0.92:
        # (3) 희소 다중 사이트 킥
        m = int(rng.integers(2, max(3, K // 6) + 1))
        idx = rng.choice(K, size=m, replace=False)
        b[idx] += scale * A_BOUND * rng.standard_normal(m)
    else:
        # (4) 저주파 필드: 여러 bump 의 합 (전역적이지만 매끄러운 재배치)
        for _ in range(int(rng.integers(2, 5))):
            c = rng.integers(K)
            ell = SPACING * rng.uniform(0.5, 2.5)
            amp = 0.5 * scale * A_BOUND * rng.standard_normal()
            b += amp * np.exp(-(DIST[c] ** 2) / (2.0 * ell * ell))
    return project(b)


def elapsed():
    return time.time() - T_START


# ----------------------------------------------------------------------
# 한 체인의 SA
# ----------------------------------------------------------------------
def anneal(a0, rng, chain_tag):
    a = project(np.asarray(a0, float).copy())
    J = objective(a)
    best_a, best_J = a.copy(), J

    # --- T0 보정: 현재 지점에서 랜덤 이동들의 |ΔJ| 평균으로 초기온도 설정 ---
    scale = 0.4
    deltas = []
    for _ in range(CALIB_SAMPLES):
        Jb = objective(propose(a, scale, rng))
        if np.isfinite(Jb):
            deltas.append(abs(Jb - J))
    mean_d = float(np.mean(deltas)) if deltas else 1e-4
    T0 = max(mean_d / (-np.log(TARGET_ACCEPT0)), 1e-9)
    print(f"[{chain_tag}] start J={J:.8f}  T0={T0:.3e}  "
          f"(mean|dJ|={mean_d:.3e}, evals={EVAL_COUNT}, {elapsed():.0f}s)")

    win = deque(maxlen=100)          # 최근 수락 여부
    step = 0
    for cyc in range(CYCLES):
        if cyc > 0:
            a, J = best_a.copy(), best_J        # best 로 복귀 후 재가열
        T_hi = T0 * (REHEAT_FRAC[cyc] if cyc < len(REHEAT_FRAC) else REHEAT_FRAC[-1])
        T_lo = T0 * T_END_FRAC
        alpha = (T_lo / T_hi) ** (1.0 / steps_per_cycle)
        T = T_hi

        for _ in range(steps_per_cycle):
            b = propose(a, scale, rng)
            Jb = objective(b)
            dJ = Jb - J
            if dJ <= 0.0 or rng.random() < np.exp(-dJ / max(T, 1e-12)):
                a, J = b, Jb
                win.append(1)
            else:
                win.append(0)

            if J < best_J - 1e-12:
                best_a, best_J = a.copy(), J

            T *= alpha
            step += 1

            # 이동 크기 적응 (최근 100스텝 수락률 기준)
            if len(win) == 100 and step % 40 == 0:
                ar = float(np.mean(win))
                if ar > 0.55:
                    scale = min(scale * 1.15, 3.0)
                elif ar < 0.18:
                    scale = max(scale * 0.85, 0.02)

            if step % 500 == 0:
                print(f"[{chain_tag}] step {step:5d}  cyc {cyc}  "
                      f"T={T:.2e}  scale={scale:.3f}  acc={float(np.mean(win)):.2f}  "
                      f"J={J:.8f}  best={best_J:.8f}  "
                      f"evals={EVAL_COUNT}  {elapsed():.0f}s")

        print(f"[{chain_tag}] cycle {cyc} done -> best={best_J:.8f}")
        np.save("sa_best_a.npy", best_a)        # 중간 저장

    return best_a, best_J


# ----------------------------------------------------------------------
# 드라이버: 여러 시작점에서 체인 실행, 전역 best 유지
# ----------------------------------------------------------------------
def main():
    master = np.random.default_rng(SEED)

    base_J = objective(np.zeros(K))
    print(f"[baseline] J(a=0) = {base_J:.8f}   "
          f"budget: {MAX_EVALS} evals, {RESTARTS} chains x {CYCLES} cycles x "
          f"{steps_per_cycle} steps")

    global_best_a = np.zeros(K)
    global_best_J = base_J
    visited = []

    for c in range(RESTARTS):
        rng = np.random.default_rng(master.integers(1 << 63))
        if c == 0:
            a0 = np.zeros(K)                    # 균등 가중치에서 출발 (알려진 기준점)
        else:
            env.reset()
            a0 = env.a.copy()                   # 환경이 주는 무작위 시작점
        ba, bJ = anneal(a0, rng, chain_tag=f"chain{c}")
        visited.append(bJ)
        if bJ < global_best_J:
            global_best_a, global_best_J = ba.copy(), bJ
            np.save("sa_best_a.npy", global_best_a)

    # 최종 검증
    final_J, final_stds = env.evaluate(global_best_a)
    improve = (base_J - final_J) / base_J * 100.0
    print("\n" + "=" * 60)
    print(f"BEST J        = {final_J:.8f}")
    print(f"baseline J    = {base_J:.8f}")
    print(f"improvement   = {improve:.2f}%  ({base_J:.6f} -> {final_J:.6f})")
    print(f"rho used      = {np.exp(global_best_a.max() - global_best_a.min()):.4f} "
          f"(<= {env.rho_max})")
    print(f"total evals   = {EVAL_COUNT},  wall = {elapsed():.0f}s")
    print(f"best a saved  -> sa_best_a.npy")
    print("a =", np.array2string(global_best_a, precision=5, max_line_width=120))
    print("=" * 60)

    # 그림: 체인별 도달 best 분포
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.axhline(base_J, color="red", ls="--", label=f"baseline {base_J:.5f}")
        plt.plot(range(RESTARTS), visited, "o-", color="green", label="chain best")
        plt.axhline(final_J, color="blue", ls=":", label=f"global best {final_J:.5f}")
        plt.xlabel("chain")
        plt.ylabel("objective J")
        plt.title("SA: per-chain best objective")
        plt.legend()
        plt.savefig("sa_result.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("plot saved -> sa_result.png")
    except Exception as e:
        print("plot skipped:", e)

    return global_best_a, final_J


if __name__ == "__main__":
    main()
