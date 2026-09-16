import os
import sys
import colorsys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize

# 기본 폰트(DejaVu Sans)는 한글 글리프가 없어 제목/축/컬러바의 한글이
# 네모(tofu)로 깨진다. Windows에 기본 내장된 맑은 고딕으로 지정한다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 이 파일(vor_sd/rl_env_vor_show.py) 기준으로 저장소 루트를 sys.path 에 추가한다.
# rl_env_voronoi_mw.py 가 절대 패키지 경로(features.data_preprocessing.vor_sd...)로
# 자기 모듈들을 import 하기 때문에, 루트가 sys.path 에 있어야 한다.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.data_preprocessing.vor_sd.rl_env_voronoi_mw import TrafficRLEnvMW, VF_MS
from features.data_preprocessing.vor_sd.mw_cut import graph_cut_segments_fast, graph_cut_segments_pieces
from features.data_preprocessing.vor_sd.mg_cc_batch import blocking_probability_batch


def distinct_cmap(n):
    """
    tab20은 20색까지만 구분되고 그 이상은 색이 반복돼 인접 사이트끼리 헷갈린다.
    황금각(golden angle)으로 색상환을 순회해 n개(여기선 K=56) 모두 서로
    뚜렷이 구분되는 색을 만든다.
    """
    hues = (np.arange(n) * 0.6180339887498949) % 1.0
    colors = [colorsys.hsv_to_rgb(h, 0.65, 0.92) for h in hues]
    return ListedColormap(colors)


def compute_mw_stats(env, a):
    """
    evaluate()와 동일한 MW(곱셈가중) 절단/차단확률 로직을 재현해
    (세그먼트,셀)별 유효길이 Lmat 와 그 자리의 차단확률 prob_mat, 그리고
    사이트(셀)별 표준편차를 반환한다.
    """
    w = env.weights(a)

    Lmat = graph_cut_segments_fast(
        env.P, env.Q, env.edge_u, env.edge_v, env.site_node,
        env.node_dist, w, min_len=env.min_len)
    seg_i, cell_i = np.nonzero(Lmat)
    L_eff = Lmat[seg_i, cell_i]

    lam_eff = env.lam[seg_i]
    if env.lam_scaling == "length":
        lam_eff = lam_eff * (L_eff / env.seg_len[seg_i])
    probs, _ = blocking_probability_batch(L_eff, VF_MS, lam_eff, env.lanes[seg_i])

    prob_mat = np.full_like(Lmat, np.nan)
    prob_mat[seg_i, cell_i] = probs

    # 셀(사이트)별 표준편차
    cnt = np.bincount(cell_i, minlength=env.K)
    s1 = np.bincount(cell_i, weights=probs, minlength=env.K)
    s2 = np.bincount(cell_i, weights=probs ** 2, minlength=env.K)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
    cell_std = np.where(cnt >= env.min_pieces, np.sqrt(var), 0.0)

    return Lmat, prob_mat, cell_std, w


def build_piece_geometry(env, w, Lmat, prob_mat):
    """
    실제 그래프 보로노이 경계를 도로 위에 정확히 그리기 위한 조각별 기하.
    평면 전체를 격자로 채워 근사하던 예전 래스터 대신, graph_cut_segments_pieces
    (도로그래프 최단거리 절단, 세그먼트 내부에서 소유 셀이 바뀌는 조각까지 보존)
    로 얻은 조각들을 실제 위치(t_lo,t_hi) 그대로 선분으로 그린다.
    조각 하나의 owned 길이가 min_len 미만이면(Lmat==0) objective에서도 빠지는
    슬리버이므로 회색(미배정)으로 표시한다.
    """
    seg_i, cell_i, t_lo, t_hi = graph_cut_segments_pieces(
        env.P, env.Q, env.edge_u, env.edge_v, env.site_node, env.node_dist, w)

    u = env.Q[seg_i] - env.P[seg_i]
    p0 = env.P[seg_i] + t_lo[:, None] * u
    p1 = env.P[seg_i] + t_hi[:, None] * u
    lines = np.stack([p0, p1], axis=1)  # (M, 2, 2)

    owned = Lmat[seg_i, cell_i] > 0
    return lines, cell_i, prob_mat[seg_i, cell_i], owned


def plot_traffic_voronoi(env, a, save_filename="traffic_visualization.png"):
    """
    도로 그래프(정점=사이트/분기점, 간선=도로 선분) 위에 실제 MW 그래프-보로노이
    절단 조각을 그 위치 그대로 그린다. 평면 전체를 래스터로 채우던 예전 방식은
    도로 밖 빈 공간까지 "영역"으로 보여줘 오해를 줄 수 있었다 — 이 환경이 실제로
    다루는 것은 연속된 2D 평면이 아니라 도로망(그래프) 그 자체이므로, 소유권이
    바뀌는 지점을 도로 위에서 정확히 잘라 색칠하는 쪽이 모델과 일치한다.
    왼쪽: 조각별 소유 사이트, 오른쪽: 조각별 M/G/c/c 차단확률.
    """
    Lmat, prob_mat, cell_std, w = compute_mw_stats(env, a)
    lines, owner_cell, piece_prob, owned = build_piece_geometry(env, w, Lmat, prob_mat)

    pad = 4000
    x_min, y_min = env.site_coords.min(axis=0) - pad
    x_max, y_max = env.site_coords.max(axis=0) + pad

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1)

    # --------------------------------------------------------------------------
    # [왼쪽 플롯] 그래프 보로노이 — 조각별 소유 사이트
    # --------------------------------------------------------------------------
    ax1 = axes[0]

    cmap1 = distinct_cmap(env.K)
    cmap1.set_bad("lightgray")
    cell_arr = np.ma.masked_array(owner_cell.astype(float), mask=~owned)
    lc1 = LineCollection(lines, cmap=cmap1, norm=Normalize(0, max(env.K - 1, 1)),
                          linewidths=2.5)
    lc1.set_array(cell_arr)
    ax1.add_collection(lc1)

    ax1.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                c="red", marker="^", s=100, edgecolor="black", linewidth=1.2,
                label="Junction Sites", zorder=3)
    ax1.plot([], [], color="lightgray", linewidth=3,
             label=f"Excluded sliver (< {env.min_len:.0f} m)")
    for k in range(env.K):
        ax1.text(env.site_coords[k, 0] + 300, env.site_coords[k, 1] + 300,
                  f"#{k}\n(σ:{cell_std[k]:.3f})", fontsize=8, weight="bold",
                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))

    rho = w.max() / w.min()
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal")
    ax1.set_title(f"1. MW Graph-Voronoi Partitions (ρ = w_max/w_min: {rho:.2f})", fontsize=14, weight="bold")
    ax1.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (meters)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")

    # --------------------------------------------------------------------------
    # [오른쪽 플롯] 조각별 M/G/c/c 차단확률
    # --------------------------------------------------------------------------
    ax2 = axes[1]

    cmap2 = plt.get_cmap("YlOrRd").copy()
    cmap2.set_bad("lightgray")
    prob_arr = np.ma.masked_invalid(piece_prob)
    lc2 = LineCollection(lines, cmap=cmap2, norm=Normalize(0.0, 1.0), linewidths=2.5)
    lc2.set_array(prob_arr)
    ax2.add_collection(lc2)

    ax2.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                c="black", marker="o", s=40, edgecolor="white", linewidth=0.8, zorder=3)

    cbar = fig.colorbar(lc2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("M/G/c/c Blocking Probability $P(c)$ (조각별)", fontsize=12, weight="bold")

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_aspect("equal")
    ax2.set_title("2. Spatial Distribution of Blocking Probabilities (Graph Edges)", fontsize=14, weight="bold")
    ax2.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)

    J, _ = env.evaluate(a)
    fig.suptitle(
        f"PeMS D07 Traffic RL Environment State Analysis (MW Graph Voronoi)\n"
        f"Objective [{env.objective}]: {J:.6f}",
        fontsize=16, weight="bold", y=0.98
    )

    os.makedirs(os.path.dirname(save_filename) if os.path.dirname(save_filename) else ".", exist_ok=True)
    plt.savefig(save_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[성공] 시각화 플롯이 '{save_filename}'에 저장되었습니다.")


def random_search_best_a(env, iters=200, seed=42):
    """SAC 로 실제 학습해 찾은 log-가중치(a) 결과값 (main 의 최신 학습 결과).
    (iters/seed 인자는 더 이상 쓰이지 않음 — 과거 랜덤서치 인터페이스와의 호환을 위해 남겨둠.)

    주의: 이 값은 '직선거리' MW 보로노이로 학습한 결과다. 이 브랜치의 도로 그래프
    최단거리 기준에서는 최적이 아니므로, 그래프 버전으로 재학습해 갱신해야 한다.
    또 값의 범위(±0.81)는 rho_max=5 (a_bound=ln5/2≈0.805) 학습 결과라,
    아래 __main__ 이 쓰는 기본 rho_max=2 (a_bound≈0.347) 보다 넓다."""
    x = [-0.79426757 , 0.66747757,  0.20282636, -0.06653599,  0.80158351,  0.49406705,
  0.69944818 ,-0.57269679 , 0.80158351,  0.80158351, -0.42666669 , 0.63910317,
  0.49580206 ,-0.01356709 , 0.27130863,  0.37320597,  0.03198263 ,-0.8078544,
 -0.50651805 , 0.32704624 ,-0.25084041,  0.5406728 , -0.69678164 ,-0.11670329,
  0.46863978 ,-0.01871506 ,-0.8078544 ,  0.31275646, -0.60338359 ,-0.8078544,
  0.46585443 ,-0.33263067 ,-0.60464305, -0.31266501, -0.6548776  ,-0.8078544,
  0.4664359  ,-0.02178422  ,0.72828239, -0.20276264,  0.39298267 ,-0.49520908,
 -0.69331466 ,-0.8078544   ,0.80158351 , 0.41831631,  0.41805437 , 0.09270113,
 -0.49532246 , 0.41860901 , 0.5022547 , -0.8078544 , -0.11662466 ,-0.8078544,
  0.40383831 , 0.61349091]

    return np.array(x)


if __name__ == "__main__":
    DIR = str(REPO_ROOT / "features" / "data_preprocessing" / "vor_sd")

    SEGMENTS_FILE = f"{DIR}/outputs_2008/pems_d07_segments"
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

    print("\n[시나리오 3] SAC로 학습된 가중치 시각화 생성 중...")
    best_a = random_search_best_a(env)
    plot_traffic_voronoi(env, best_a, save_filename=f"{DIR}/outputs/vis_anal_weights_after.png")
