"""
Simple SA, matched 1:1 to train.py's env config (same files / rho_max / phf / objective),
for a fair ceiling comparison against the SAC run's plateau (eval best J ~0.025-0.029).

No cycles/reheat/restarts machinery (that's Comp_SA.py) - just single-chain
geometric-cooling SA with per-site random-walk proposals, same move scale as
the SAC action budget (_act_scale) so it's exploring a comparable neighborhood.
"""
import sys
import time
from pathlib import Path

import numpy as np

current_file = Path(__file__).resolve()
parent_dir = current_file.parents[3]  # .../project3
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from features.data_preprocessing.vor_sd.rl_env_voronoi_mw import TrafficRLEnvMW

DIR = str(parent_dir / "features" / "data_preprocessing" / "vor_sd")

# ---- exactly train.py's env construction ----
RHO_MAX = 5.0
OBJECTIVE = "within"
WINDOW_MIN = 90
PHF = 60.0 / WINDOW_MIN
SEGMENTS_FILE = f"{DIR}/outputs_2008/pems_d07_segments_0841_after.csv"
SITES_FILE = f"{DIR}/outputs_2008/pems_d07_sites.csv"
META_FILE = f"{DIR}/d07_text_meta_2018_10_13.txt"

env = TrafficRLEnvMW(
    segments_csv=SEGMENTS_FILE,
    sites_csv=SITES_FILE,
    meta_txt=META_FILE,
    rho_max=RHO_MAX,
    objective=OBJECTIVE,
    phf=PHF,
)

K = env.K
A_BOUND = float(env.a_bound)
SEED = 20260917
MAX_EVALS = 20000
T_END_FRAC = 1e-3
CALIB_SAMPLES = 60
TARGET_ACCEPT0 = 0.5

rng = np.random.default_rng(SEED)


def objective_fn(a):
    try:
        val, _ = env.evaluate(a)
    except Exception:
        return np.inf
    val = float(val)
    return val if np.isfinite(val) and val > 0.0 else np.inf


def project(a):
    a = a - a.mean()
    np.clip(a, -A_BOUND, A_BOUND, out=a)
    return a - a.mean()


def propose(a, scale):
    b = a.copy()
    r = rng.random()
    if r < 0.6:
        # single-site kick (SAC's per-site action space, same order of magnitude as _act_scale)
        i = rng.integers(K)
        b[i] += scale * A_BOUND * rng.standard_normal()
    else:
        # sparse multi-site kick
        m = int(rng.integers(2, max(3, K // 6) + 1))
        idx = rng.choice(K, size=m, replace=False)
        b[idx] += scale * A_BOUND * rng.standard_normal(m)
    return project(b)


t0 = time.time()
base_J = objective_fn(np.zeros(K))
print(f"[baseline] J(a=0) = {base_J:.8f}  (K={K}, a_bound={A_BOUND:.4f}, rho_max={RHO_MAX})")

a = np.zeros(K)
J = base_J
best_a, best_J = a.copy(), J

scale = 0.4
deltas = []
for _ in range(CALIB_SAMPLES):
    Jb = objective_fn(propose(a, scale))
    if np.isfinite(Jb):
        deltas.append(abs(Jb - J))
mean_d = float(np.mean(deltas)) if deltas else 1e-4
T0 = max(mean_d / (-np.log(TARGET_ACCEPT0)), 1e-9)
T_end = T0 * T_END_FRAC
n_steps = MAX_EVALS - CALIB_SAMPLES
cool = (T_end / T0) ** (1.0 / n_steps)
T = T0
print(f"T0={T0:.3e}  T_end={T_end:.3e}  steps={n_steps}  evals_budget={MAX_EVALS}")

from collections import deque
win = deque(maxlen=100)

for step in range(n_steps):
    b = propose(a, scale)
    Jb = objective_fn(b)
    dJ = Jb - J
    if dJ <= 0.0 or rng.random() < np.exp(-dJ / max(T, 1e-12)):
        a, J = b, Jb
        win.append(1)
    else:
        win.append(0)
    if J < best_J - 1e-12:
        best_a, best_J = a.copy(), J
    T *= cool

    if len(win) == 100 and step % 40 == 0:
        ar = float(np.mean(win))
        if ar > 0.55:
            scale = min(scale * 1.15, 3.0)
        elif ar < 0.18:
            scale = max(scale * 0.85, 0.02)

    if step % 1000 == 0:
        print(f"step {step:6d}  T={T:.2e}  scale={scale:.3f}  "
              f"acc={float(np.mean(win)) if win else 0:.2f}  "
              f"J={J:.6f}  best={best_J:.6f}  {time.time()-t0:.0f}s")

improve = (base_J - best_J) / base_J * 100.0
print("=" * 60)
print(f"baseline J   = {base_J:.6f}")
print(f"SA best J    = {best_J:.6f}")
print(f"improvement  = {improve:.2f}%")
print(f"SAC eval best J was ~0.0247-0.0291 (no clear downward trend)")
print(f"total evals  = {MAX_EVALS}  wall = {time.time()-t0:.0f}s")
np.save("sa_simple_best_a.npy", best_a)
print("=" * 60)
