import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon

from rl_env_voronoi import TrafficRLEnv, VF_MS
from weighted_voronoi import all_power_cells, blocking_probability


def compute_cells_and_segment_stats(env, weights):
    """
    evaluate()와 동일한 멱 다이어그램/차단확률 로직을 재현하되,
    (1) 폴리곤 좌표(cells) 자체, (2) 셀별 평균/표준편차, (3) 세그먼트별 상세값을 모두 반환한다.
    """
    cells = all_power_cells(env.site_coords, weights * env.spacing2, env.bbox)

    N = env.N
    assigned_site = np.full(N, -1, dtype=int)
    seg_prob = np.full(N, np.nan)
    best_len = np.zeros(N)

    stds = np.zeros(env.K)
    cell_mean_prob = np.full(env.K, np.nan)

    for k, poly_coords in enumerate(cells):
        if len(poly_coords) < 3:
            continue

        poly = Polygon(poly_coords)
        idx_hits = env.tree.query(poly)

        cell_probs = []
        for idx in idx_hits:
            line = env.seg_lines[idx]
            if not poly.intersects(line):
                continue

            inter = poly.intersection(line)
            L_eff = inter.length
            if L_eff < 1.0:
                continue

            prob, _, _ = blocking_probability(
                length_m=L_eff, v_free_ms=VF_MS,
                lam=env.lam[idx], lanes=env.lanes[idx],
            )
            cell_probs.append(prob)

            if L_eff > best_len[idx]:
                best_len[idx] = L_eff
                assigned_site[idx] = k
                seg_prob[idx] = prob

        if cell_probs:
            stds[k] = np.std(cell_probs) if len(cell_probs) > 1 else 0.0
            cell_mean_prob[k] = np.mean(cell_probs)

    return cells, assigned_site, seg_prob, stds, cell_mean_prob


def plot_traffic_voronoi(env, weights, save_filename="traffic_visualization.png"):
    """
    멱 다이어그램(power diagram) 자체를 면적으로 채워서 시각화한다.
    왼쪽: site별 영역 분할, 오른쪽: 영역별 평균 M/G/c/c 차단확률(+개별 도로 산점도).
    """
    seg_coords = env.seg_df[["mid_x", "mid_y"]].values
    cells, assigned_site, seg_prob, stds, cell_mean_prob = compute_cells_and_segment_stats(env, weights)

    valid_idx = [k for k, poly in enumerate(cells) if len(poly) >= 3]
    verts = [cells[k] for k in valid_idx]

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1)

    # --------------------------------------------------------------------------
    # [왼쪽 플롯] 멱 다이어그램 영역 분할 (면적 채우기)
    # --------------------------------------------------------------------------
    ax1 = axes[0]

    area1 = PolyCollection(
        verts, array=np.array(valid_idx, dtype=float), cmap="tab20",
        edgecolors="white", linewidths=0.8, alpha=0.55
    )
    area1.set_clim(0, max(env.K - 1, 1))
    ax1.add_collection(area1)

    unassigned = assigned_site == -1
    ax1.scatter(seg_coords[unassigned, 0], seg_coords[unassigned, 1],
                c="dimgray", s=4, alpha=0.5, label="Unassigned road")
    ax1.scatter(seg_coords[~unassigned, 0], seg_coords[~unassigned, 1],
                c="black", s=2, alpha=0.35, label="Road segments")

    ax1.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                c="red", marker="^", s=100, edgecolor="black", linewidth=1.2, label="Junction Sites")
    for k in range(env.K):
        ax1.text(env.site_coords[k, 0] + 300, env.site_coords[k, 1] + 300,
                  f"#{k}\n(σ:{stds[k]:.3f})", fontsize=8, weight="bold",
                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))

    ax1.set_xlim(env.bbox[0], env.bbox[2])
    ax1.set_ylim(env.bbox[1], env.bbox[3])
    ax1.set_aspect("equal")
    ax1.set_title(f"1. Power-Diagram Area Partitions (Weights Scale: {np.mean(weights):.2f})", fontsize=14, weight="bold")
    ax1.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (meters)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")

    # --------------------------------------------------------------------------
    # [오른쪽 플롯] 영역별 평균 차단확률 (면적 채우기) + 개별 도로 산점도
    # --------------------------------------------------------------------------
    ax2 = axes[1]

    area2 = PolyCollection(
        verts, array=cell_mean_prob[valid_idx], cmap="YlOrRd",
        edgecolors="white", linewidths=0.8, alpha=0.85
    )
    area2.set_clim(0.0, 1.0)
    ax2.add_collection(area2)

    valid_seg = ~np.isnan(seg_prob)
    ax2.scatter(seg_coords[valid_seg, 0], seg_coords[valid_seg, 1],
                c=seg_prob[valid_seg], cmap="YlOrRd", s=5, alpha=0.9,
                vmin=0.0, vmax=1.0, edgecolors="black", linewidths=0.15)

    ax2.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                c="black", marker="o", s=40, edgecolor="white", linewidth=0.8)

    cbar = fig.colorbar(area2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("M/G/c/c Blocking Probability $P(c)$ (영역 평균)", fontsize=12, weight="bold")

    ax2.set_xlim(env.bbox[0], env.bbox[2])
    ax2.set_ylim(env.bbox[1], env.bbox[3])
    ax2.set_aspect("equal")
    ax2.set_title("2. Spatial Distribution of Blocking Probabilities (Area-filled)", fontsize=14, weight="bold")
    ax2.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.3)

    mean_std = np.mean(stds)
    fig.suptitle(
        f"PeMS D07 Traffic RL Environment State Analysis\nGlobal Objective (Mean of Stds): {mean_std:.6f}",
        fontsize=16, weight="bold", y=0.98
    )

    os.makedirs(os.path.dirname(save_filename) if os.path.dirname(save_filename) else ".", exist_ok=True)
    plt.savefig(save_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[성공] 시각화 플롯이 '{save_filename}'에 저장되었습니다.")


if __name__ == "__main__":
    DIR = "."  # vor_sd 폴더 안에서 실행한다고 가정

    SEGMENTS_FILE = f"{DIR}/outputs/pems_d07_segments.csv"
    SITES_FILE = f"{DIR}/outputs/pems_d07_sites.csv"
    META_FILE = f"{DIR}/d07_text_meta_2018_10_13.txt"

    print("[Visualizer] 환경 데이터 로드 및 초기화 중...")
    env = TrafficRLEnv(
        segments_csv=SEGMENTS_FILE,
        sites_csv=SITES_FILE,
        meta_txt=META_FILE
    )

    print("\n[시나리오 1] 균등 가중치(Uniform Weights) 시각화 생성 중...")
    uniform_weights = np.zeros(env.K)
    plot_traffic_voronoi(env, uniform_weights, save_filename=f"{DIR}/outputs/vis_uniform_weights.png")

    print("\n[시나리오 2] 무작위 가중치(Random Weights) 시각화 생성 중...")
    np.random.seed(42)
    random_weights = np.random.uniform(-50, 50, size=env.K)
    plot_traffic_voronoi(env, random_weights, save_filename=f"{DIR}/outputs/vis_random_weights.png")

    print("\n[시나리오 3] 학습된(도출) 가중치 시각화 생성 중...")
    anal_weights = np.array([
        49.96741934, 47.13893601, -42.7545594, -1.65908331, 47.85887268,
        -42.21540915, 35.87545882, -30.28017296, -21.80077422, 1.26895474,
        -50.92952524, -4.88680289, -6.7142753, -42.04246368, -49.18016608,
        50.90372189, -15.26008708, 46.59871147, -11.98695063, -50.9681357,
        29.33884171, -50.03240965, 0.95607672, -47.12416277, -11.85760138,
        -50.9743623, -16.31290461, 22.13128088, 4.78380801, -50.97372816,
        -0.5530032, -50.97250603, -23.77873442, -50.94838635, 11.48741087,
        -45.07208194, 36.80758185, 50.21449381, -36.55854074, 0.91187941,
        12.49942578, 22.49752191, -48.85336488, 26.8287647, 51.02502568,
        51.02502914, 50.88768059, 33.66862827, 49.52608593, 35.32594693,
        50.99385174, 18.34145188, -50.93979422, 19.3288309, 35.1415751,
        12.29671952,
    ])
    plot_traffic_voronoi(env, anal_weights, save_filename=f"{DIR}/outputs/vis_anal_weights.png")
