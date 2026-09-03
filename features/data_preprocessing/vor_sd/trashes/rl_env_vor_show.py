import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 이 파일(trashes/rl_env_vor_show.py) 기준으로 저장소 루트를 sys.path 에 추가한다.
# rl_env_voronoi_mw.py 가 절대 패키지 경로(features.data_preprocessing.vor_sd...)로
# 자기 모듈들을 import 하기 때문에, 루트가 sys.path 에 있어야 한다.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.data_preprocessing.vor_sd.rl_env_voronoi_mw import TrafficRLEnvMW, VF_MS
from features.data_preprocessing.vor_sd.mw_cut import cut_segments_fast
from features.data_preprocessing.vor_sd.mg_cc_batch import blocking_probability_batch


def compute_mw_stats(env, a):
    """
    evaluate()와 동일한 MW(곱셈가중) 절단/차단확률 로직을 재현하되,
    (1) 세그먼트별 대표 소유 사이트/확률, (2) 사이트(셀)별 평균/표준편차를 모두 반환한다.
    폴리곤을 만들지 않는 MW 버전에서는 셀 좌표 자체가 존재하지 않는다.
    """
    w = env.weights(a)

    Lmat = cut_segments_fast(env.P, env.Q, env.site_coords, w, min_len=env.min_len)
    seg_i, cell_i = np.nonzero(Lmat)
    L_eff = Lmat[seg_i, cell_i]

    lam_eff = env.lam[seg_i]
    if env.lam_scaling == "length":
        lam_eff = lam_eff * (L_eff / env.seg_len[seg_i])
    probs, _ = blocking_probability_batch(L_eff, VF_MS, lam_eff, env.lanes[seg_i])

    # 세그먼트별 대표 소유 사이트 = 그 세그먼트 위에서 가장 긴 조각을 차지한 셀
    assigned_site = np.full(env.N, -1, dtype=int)
    seg_prob = np.full(env.N, np.nan)
    if len(seg_i) > 0:
        order = np.lexsort((-L_eff, seg_i))  # seg_i 오름차순, 그 안에서 L_eff 내림차순
        dominant = order[np.concatenate(([True], seg_i[order][1:] != seg_i[order][:-1]))]
        assigned_site[seg_i[dominant]] = cell_i[dominant]
        seg_prob[seg_i[dominant]] = probs[dominant]

    # 셀(사이트)별 평균/표준편차
    cnt = np.bincount(cell_i, minlength=env.K)
    s1 = np.bincount(cell_i, weights=probs, minlength=env.K)
    s2 = np.bincount(cell_i, weights=probs ** 2, minlength=env.K)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
    cell_mean_prob = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
    cell_std = np.where(cnt >= env.min_pieces, np.sqrt(var), 0.0)

    return assigned_site, seg_prob, cell_mean_prob, cell_std


def rasterize_owner_grid(env, w, grid_res=300, pad=4000):
    """
    MW(Apollonius) 셀은 비볼록/비연결일 수 있어 폴리곤으로 그리기 어렵다.
    대신 평면을 촘촘한 격자로 샘플링해 각 점의 소유 사이트(d_i = |x-p_i|/w_i 최소)를 구해
    래스터(pcolormesh)로 영역을 채운다.
    """
    x_min, y_min = env.site_coords.min(axis=0) - pad
    x_max, y_max = env.site_coords.max(axis=0) + pad

    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    d = np.linalg.norm(pts[:, None, :] - env.site_coords[None, :, :], axis=2) / w[None, :]
    owner = d.argmin(1).reshape(grid_res, grid_res)

    return XX, YY, owner, (x_min, y_min, x_max, y_max)


def plot_traffic_voronoi(env, a, save_filename="traffic_visualization.png"):
    """
    MW(곱셈가중) 보로노이 다이어그램을 래스터로 채워서 시각화한다.
    왼쪽: site별 영역 분할, 오른쪽: 영역별 평균 M/G/c/c 차단확률(+개별 도로 산점도).
    """
    w = env.weights(a)
    seg_coords = 0.5 * (env.P + env.Q)

    assigned_site, seg_prob, cell_mean_prob, cell_std = compute_mw_stats(env, a)
    XX, YY, owner_grid, (x_min, y_min, x_max, y_max) = rasterize_owner_grid(env, w)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1)

    # --------------------------------------------------------------------------
    # [왼쪽 플롯] MW 보로노이 영역 분할 (래스터 채우기)
    # --------------------------------------------------------------------------
    ax1 = axes[0]

    ax1.pcolormesh(XX, YY, owner_grid, cmap="tab20", vmin=0, vmax=max(env.K - 1, 1),
                   alpha=0.55, shading="auto")

    unassigned = assigned_site == -1
    ax1.scatter(seg_coords[unassigned, 0], seg_coords[unassigned, 1],
                c="dimgray", s=4, alpha=0.5, label="Unassigned road")
    ax1.scatter(seg_coords[~unassigned, 0], seg_coords[~unassigned, 1],
                c="black", s=2, alpha=0.35, label="Road segments")

    ax1.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                c="red", marker="^", s=100, edgecolor="black", linewidth=1.2, label="Junction Sites")
    for k in range(env.K):
        ax1.text(env.site_coords[k, 0] + 300, env.site_coords[k, 1] + 300,
                  f"#{k}\n(σ:{cell_std[k]:.3f})", fontsize=8, weight="bold",
                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))

    rho = w.max() / w.min()
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal")
    ax1.set_title(f"1. MW-Voronoi Area Partitions (ρ = w_max/w_min: {rho:.2f})", fontsize=14, weight="bold")
    ax1.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (meters)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")

    # --------------------------------------------------------------------------
    # [오른쪽 플롯] 영역별 평균 차단확률 (래스터 채우기) + 개별 도로 산점도
    # --------------------------------------------------------------------------
    ax2 = axes[1]

    prob_grid = np.nan_to_num(cell_mean_prob, nan=0.0)[owner_grid]
    mesh2 = ax2.pcolormesh(XX, YY, prob_grid, cmap="YlOrRd", vmin=0.0, vmax=1.0,
                            alpha=0.85, shading="auto")

    valid_seg = ~np.isnan(seg_prob)
    ax2.scatter(seg_coords[valid_seg, 0], seg_coords[valid_seg, 1],
                c=seg_prob[valid_seg], cmap="YlOrRd", s=5, alpha=0.9,
                vmin=0.0, vmax=1.0, edgecolors="black", linewidths=0.15)

    ax2.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                c="black", marker="o", s=40, edgecolor="white", linewidth=0.8)

    cbar = fig.colorbar(mesh2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("M/G/c/c Blocking Probability $P(c)$ (영역 평균)", fontsize=12, weight="bold")

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect("equal")
    ax2.set_title("2. Spatial Distribution of Blocking Probabilities (Rasterized)", fontsize=14, weight="bold")
    ax2.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)

    J, _ = env.evaluate(a)
    fig.suptitle(
        f"PeMS D07 Traffic RL Environment State Analysis (MW Voronoi)\n"
        f"Objective [{env.objective}]: {J:.6f}",
        fontsize=16, weight="bold", y=0.98
    )

    os.makedirs(os.path.dirname(save_filename) if os.path.dirname(save_filename) else ".", exist_ok=True)
    plt.savefig(save_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[성공] 시각화 플롯이 '{save_filename}'에 저장되었습니다.")


def random_search_best_a(env, iters=2000, seed=42):
    """하드코딩된 옛 power-diagram 가중치 예시를 대체:
    현재 MW 환경(a-space, |a|<=a_bound)에서 무작위 탐색으로 목적함수가 가장 낮은 a를 찾는다."""
    rng = np.random.default_rng(seed)
    best_J, best_a = np.inf, np.zeros(env.K)
    for _ in range(iters):
        a = rng.uniform(-env.a_bound, env.a_bound, env.K)
        a -= a.mean()
        J, _ = env.evaluate(a)
        if J < best_J:
            best_J, best_a = J, a
    return best_a, best_J


if __name__ == "__main__":
    DIR = str(REPO_ROOT / "features" / "data_preprocessing" / "vor_sd")

    SEGMENTS_FILE = f"{DIR}/outputs/pems_d07_segments.csv"
    SITES_FILE = f"{DIR}/outputs/pems_d07_sites.csv"
    META_FILE = f"{DIR}/d07_text_meta_2018_10_13.txt"

    print("[Visualizer] 환경 데이터 로드 및 초기화 중...")
    env = TrafficRLEnvMW(
        segments_csv=SEGMENTS_FILE,
        sites_csv=SITES_FILE,
        meta_txt=META_FILE
    )

    print("\n[시나리오 1] 균등 가중치(Uniform, a=0) 시각화 생성 중...")
    uniform_a = np.zeros(env.K)
    plot_traffic_voronoi(env, uniform_a, save_filename=f"{DIR}/outputs/vis_uniform_weights.png")

    print("\n[시나리오 2] 무작위 가중치(Random a) 시각화 생성 중...")
    rng = np.random.default_rng(42)
    random_a = rng.uniform(-env.a_bound, env.a_bound, env.K)
    random_a -= random_a.mean()
    plot_traffic_voronoi(env, random_a, save_filename=f"{DIR}/outputs/vis_random_weights.png")

    print("\n[시나리오 3] 탐색된(무작위 서치 최적) 가중치 시각화 생성 중...")
    best_a, best_J = random_search_best_a(env, iters=2000)
    print(f"  -> 탐색된 최적 목적함수 값: {best_J:.6f}")
    plot_traffic_voronoi(env, best_a, save_filename=f"{DIR}/outputs/vis_anal_weights.png")
