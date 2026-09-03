import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rl_env_voronoi import TrafficRLEnv  # 기존 환경 파일에서 클래스 임포트

def plot_traffic_voronoi(env, weights, save_filename="traffic_visualization.png"):
    """
    에이전트의 가중치(weights)에 따른 보로노이 구역 할당 및 도로 차단확률 분포를 
    2개의 서브플롯으로 나누어 시각화합니다.
    """
    # 1. 환경의 할당 연산 로직 재현
    safe_weights = np.clip(weights, 1e-5, None)
    weighted_dist = env.base_dist_matrix / safe_weights
    assigned_sites = np.argmin(weighted_dist, axis=1)
    
    # 각 분기점(Site)별 표준편차 계산
    _, stds = env.step(weights)

    # 2. 캔버스 생성 (1행 2열 구조)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.1)
    
    # --------------------------------------------------------------------------
    # [왼쪽 플롯] 가중 보로노이 셀 분할 현황 (Junction Assignment Map)
    # --------------------------------------------------------------------------
    ax1 = axes[0]
    # 도로 구간들(Segment Midpoints)을 할당된 분기점 ID에 따라 고유 색상으로 플롯
    scatter1 = ax1.scatter(
        env.seg_coords[:, 0], env.seg_coords[:, 1], 
        c=assigned_sites, cmap="tab20", s=6, alpha=0.5, edgecolors='none'
    )
    
    # 기준 분기점(Sites) 위치 표시
    ax1.scatter(
        env.site_coords[:, 0], env.site_coords[:, 1], 
        c="red", marker="^", s=100, edgecolor="black", linewidth=1.2, label="Junction Sites"
    )
    
    # 분기점 위에 텍스트로 인덱스 고유 번호(0~K-1) 및 표준편차 표기
    for k in range(env.K):
        ax1.text(
            env.site_coords[k, 0] + 300, env.site_coords[k, 1] + 300, 
            f"#{k}\n(σ:{stds[k]:.3f})", fontsize=8, weight="bold",
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )
        
    ax1.set_title(f"1. Weighted Voronoi Partitions (Weights Scale: {np.mean(weights):.2f})", fontsize=14, weight="bold")
    ax1.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (meters)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # --------------------------------------------------------------------------
    # [오른쪽 플롯] M/G/c/c 실제 차단확률 공간 분포 (Traffic Bottleneck Map)
    # --------------------------------------------------------------------------
    ax2 = axes[1]
    # 도로 구간들을 실제 계산된 차단확률(0.0 ~ 1.0)에 따라 Reds 컬러맵으로 플롯
    # 극단적인 0과 1 분포를 보기 쉽게 표출하기 위해 vmin, vmax 지정
    scatter2 = ax2.scatter(
        env.seg_coords[:, 0], env.seg_coords[:, 1], 
        c=env.cached_mgcc, cmap="YlOrRd", s=6, alpha=0.7, vmin=0.0, vmax=1.0
    )
    
    # 분기점 위치 표시 (오른쪽 플롯은 단순 위치 레이블링용 검은 원 마커)
    ax2.scatter(
        env.site_coords[:, 0], env.site_coords[:, 1], 
        c="black", marker="o", s=40, edgecolor="white", linewidth=0.8
    )
    
    # 우측 플롯 전용 컬러바 추가 (차단확률 강도 표시)
    cbar = fig.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("M/G/c/c Blocking Probability $P(c)$", fontsize=12, weight="bold")
    
    ax2.set_title("2. Spatial Distribution of M/G/c/c Blocking Probabilities", fontsize=14, weight="bold")
    ax2.set_xlabel("X Coordinate (meters)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. 전체 메인 타이틀 및 이미지 저장
    mean_std = np.mean(stds)
    fig.suptitle(
        f"PeMS D07 Traffic RL Environment State Analysis\nGlobal Objective (Mean of Stds): {mean_std:.6f}", 
        fontsize=16, weight="bold", y=0.98
    )
    
    # 출력 디렉토리 확인 및 저장
    os.makedirs(os.path.dirname(save_filename) if os.path.dirname(save_filename) else ".", exist_ok=True)
    plt.savefig(save_filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[성공] 시각화 플롯이 '{save_filename}'에 저장되었습니다.")


# ==============================================================================
# 테스트 실행 스크립트
# ==============================================================================
if __name__ == "__main__":
    DIR = "features/data_preprocessing/vor_sd"
    SEGMENTS_FILE = f"{DIR}/outputs/pems_d07_segments.csv"
    SITES_FILE    = f"{DIR}/outputs/pems_d07_sites.csv"
    META_FILE     = f"{DIR}/d07_text_meta_2018_10_13.txt"
    
    print("[Visualizer] 환경 데이터 로드 및 초기화 중...")
    env = TrafficRLEnv(
        segments_csv=SEGMENTS_FILE, 
        sites_csv=SITES_FILE, 
        meta_txt=META_FILE
    )
    
    # 시나리오 1: 모든 분기점의 가중치가 1.0으로 균등할 때 (일반 유클리드 보로노이)
    print("\n[시나리오 1] 균등 가중치(Uniform Weights) 시각화 생성 중...")
    uniform_weights = np.ones(env.K)
    plot_traffic_voronoi(env, uniform_weights, save_filename=f"{DIR}outputs/vis_uniform_weights.png")
    
    # 시나리오 2: 임의의 무작위 가중치 변화가 생겼을 때 (영역이 찌그러지거나 확장됨)
    print("\n[시나리오 2] 무작위 가중치(Random Weights) 시각화 생성 중...")
    np.random.seed(42)  # 재현성을 위한 시드 고정
    random_weights = np.random.uniform(0.1, 5.0, size=env.K)
    plot_traffic_voronoi(env, random_weights, save_filename=f"{DIR}outputs/vis_random_weights.png")