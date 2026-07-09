import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree
import sys

# [수정] 차단확률 연산과 보로노이 분할 함수를 코어 모듈(weighted_voronoi.py)에서 직접 가져옵니다.
sys.path.insert(0, "/home/claude")
from weighted_voronoi import all_power_cells, blocking_probability

# ==============================================================================
# 글로벌 보정 파라미터 및 상수 설정
# ==============================================================================
VF_MPH_DEFAULT = 65.0      # 자유흐름 속도 기본값 (mph)
MPH2MS = 0.44704           # mph -> m/s 변환 상수
VF_MS = VF_MPH_DEFAULT * MPH2MS  
PHF = 0.15                 # 첨두시간 계수 (하루 교통량의 15%가 1시간에 몰린다고 가정)
DIR = "features/data_preprocessing/vor_sd"

class TrafficRLEnv:
    def __init__(self, segments_csv, sites_csv, meta_txt):
        print("[RL Env] 데이터를 로드하고 R-Tree 인덱스를 구성합니다...")
        
        # 1. 데이터 로드
        self.sites_df = pd.read_csv(sites_csv)
        self.seg_df = pd.read_csv(segments_csv)
        
        meta = pd.read_csv(meta_txt, sep="\t")
        meta = meta.dropna(subset=["ID", "Lanes"])[["ID", "Lanes"]]
        self.seg_df = pd.merge(self.seg_df, meta, left_on="sta_from", right_on="ID", how="inner")
        
        # 2. 사이트 정보 및 도로 속성 세팅
        self.site_coords = self.sites_df[['x_m', 'y_m']].values  # (K, 2)
        self.K = len(self.site_coords)
        self.N = len(self.seg_df)
        
        self.lanes = self.seg_df["Lanes"].values
        self.vols = self.seg_df["vol_day_veh"].values
        self.lam = (self.vols * PHF) / 3600.0  # 초당 첨두 도착률 [veh/s]
        
        # 3. 보로노이 생성 시 자를 Bounding Box 계산
        pad = 4000
        self.bbox = (
            self.seg_df[['x1', 'x2']].values.min() - pad,
            self.seg_df[['y1', 'y2']].values.min() - pad,
            self.seg_df[['x1', 'x2']].values.max() + pad,
            self.seg_df[['y1', 'y2']].values.max() + pad
        )
        
        # 4. 도로의 양 끝점을 활용하여 LineString 객체 및 공간 인덱스(STRtree) 구축
        self.seg_lines = []
        for _, r in self.seg_df.iterrows():
            self.seg_lines.append(LineString([(r.x1, r.y1), (r.x2, r.y2)]))
        
        self.tree = STRtree(self.seg_lines)
        print(f"[RL Env] 초기화 완료: 분기점 {self.K}개, 도로 선분 {self.N}개")

    def step(self, weights):
        """
        매 에피소드마다 에이전트의 가중치를 받아 멱 다이어그램을 형성하고,
        도로를 물리적으로 절단한 뒤 동적으로 M/G/c/c 차단확률을 계산합니다.
        """
        if len(weights) != self.K:
            raise ValueError(f"가중치 개수({len(weights)})가 분기점 개수({self.K})와 다릅니다.")
            
        # 나눗셈 로직 제거 -> 수학적으로 올바른 가중 보로노이(멱 다이어그램) 코어 함수 직접 호출
        safe_weights = np.array(weights) 
        cells = all_power_cells(self.site_coords, safe_weights, self.bbox)
        
        stds = np.zeros(self.K)
        
        for k, poly_coords in enumerate(cells):
            if len(poly_coords) < 3:
                stds[k] = 0.0
                continue
                
            poly = Polygon(poly_coords)
            idx_hits = self.tree.query(poly)  # 해당 다각형과 겹칠 가능성이 있는 선분만 1차 필터링
            
            cell_probs = []
            for idx in idx_hits:
                line = self.seg_lines[idx]
                if poly.intersects(line):
                    inter = poly.intersection(line)
                    L_eff = inter.length
                    
                    if L_eff < 1.0: # 1m 이하의 자투리는 노이즈로 간주하여 연산에서 제외
                        continue
                        
                    # [수정] weighted_voronoi.py 의 진짜 M/G/c/c 함수를 호출
                    prob, _, _ = blocking_probability(
                        length_m=L_eff,
                        v_free_ms=VF_MS,
                        lam=self.lam[idx],
                        lanes=self.lanes[idx]
                    )
                    cell_probs.append(prob)
                    
            if len(cell_probs) > 1:
                stds[k] = np.std(cell_probs)
            else:
                stds[k] = 0.0
                
        mean_std = np.mean(stds)
        return mean_std, stds

# ==============================================================================
# 학습 루프 사전 테스트 실행부
# ==============================================================================
if __name__ == "__main__":
    SEGMENTS_FILE = f"{DIR}/outputs/pems_d07_segments.csv"
    SITES_FILE    = f"{DIR}/outputs/pems_d07_sites.csv"
    META_FILE     = f"{DIR}/d07_text_meta_2018_10_13.txt"
    
    try:
        env = TrafficRLEnv(
            segments_csv=SEGMENTS_FILE, 
            sites_csv=SITES_FILE, 
            meta_txt=META_FILE
        )
        print("\n--- [학습 루프 테스트: 동적 도로 분할 연산 시뮬레이션] ---")
        
        uniform_weights = np.zeros(env.K)
        mean_std_uni, all_stds_uni = env.step(uniform_weights)
        print(f"[Test 1] 균등 가중치(유클리드) 적용 시 평균 표준편차: {mean_std_uni:.8f}")
        
    except FileNotFoundError as e:
        print(f"\n[오류] 데이터 파일을 찾을 수 없습니다: {e}")
