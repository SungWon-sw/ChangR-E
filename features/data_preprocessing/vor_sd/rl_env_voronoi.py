import numpy as np
import pandas as pd

# ==============================================================================
# 1. 글로벌 보정 파라미터 및 상수 설정
# ==============================================================================
KJ = 0.15                  # 차선당 정체밀도 (veh/m)
VF_MPH_DEFAULT = 65.0      # 자유흐름 속도 기본값 (mph)
MPH2MS = 0.44704           # mph -> m/s 변환 상수
VF_MS = VF_MPH_DEFAULT * MPH2MS  # 자유흐름 속도 (m/s)

PHF = 0.15                 # ① 첨두시간 계수 (Peak Hour Factor: 하루 교통량의 15%가 1시간에 몰린다고 가정)

DIR = "features/data_preprocessing/vor_sd"

# ==============================================================================
# 2. M/G/c/c 차단확률 연산 모듈 (그린쉴즈 선형 추세선)
# ==============================================================================
def blocking_probability(L_m, lanes, Vf, vol_day, kj=KJ, phf=PHF):
    """
    그린쉴즈 선형 추세선(f(n) = v_n / v_f)을 적용한 실제 차단확률 P(c)를 반환합니다.
    연산 중 오버플로우 방지를 위해 내부적으로 Log 스케일을 사용하지만 반환값은 원래의 확률입니다.
    """
    # 도로 길이 클리핑 (수용량 c 폭발 방지)
    L_eff = L_m
    N_max = kj * lanes * L_eff
    c = max(1, int(round(N_max)))
    
    # 첨두시간 도착률(lambda) 현실화: 가장 붐비는 1시간의 초당 도착률 [veh/s]
    lam = (vol_day * phf) / 3600.0
    
    if lam <= 0 or L_eff <= 0 or Vf <= 0:
        return 0.0  # 교통량이 없는 곳의 차단 확률은 0
        
    a1 = L_eff / Vf
    log_lam_a1 = np.log(lam * a1)
    
    log_t = np.full(c + 1, -np.inf)
    log_t[0] = 0.0
    
    cum = 0.0
    for n in range(1, c + 1):
        # 상태별 속도 v_n (선형 추세선 적용)
        v_n = Vf * (1.0 - (n / N_max))
        v_n = max(0.001 * Vf, v_n)  # 제로 디비전 방지용 하한선
        
        # 서비스율 감소 보정 계수 f(n) = v_n / v_f
        f_n = v_n / Vf
        
        # 로그 영역에서 누적 점화식 계산
        cum += log_lam_a1 - np.log(n) - np.log(f_n)
        log_t[n] = cum
        
    # Log-Sum-Exp 트릭을 통해 로그 차단확률 도출 후 지수(exp) 취해 원래 확률 P(c)로 복원
    m_val = np.max(log_t)
    log_prob = (log_t[c] - m_val) - np.log(np.sum(np.exp(log_t - m_val)))
    
    return float(np.exp(log_prob))


# ==============================================================================
# 3. 강화학습 환경용 평가기 클래스 (RL Agent 연동부)
# ==============================================================================
class TrafficRLEnv:
    def __init__(self, segments_csv, sites_csv, meta_txt):
        """
        초기화 단계에서 데이터를 로드하고, 변하지 않는 환경의 실제 차단확률을
        사전에 일괄 계산(Caching)하여 RL 학습 속도를 최적화합니다.
        """
        print("[RL Env] 데이터를 로드하고 물리적 거리를 사전 연산합니다...")
        
        # 1. 데이터 로드
        self.sites_df = pd.read_csv(sites_csv)
        self.seg_df = pd.read_csv(segments_csv)
        
        # 메타데이터 로드 및 차선(Lanes), 길이(Length) 정보 병합
        meta = pd.read_csv(meta_txt, sep="\t")
        meta = meta.dropna(subset=["ID", "Lanes", "Length"])[["ID", "Lanes", "Length"]]
        
        self.seg_df = pd.merge(self.seg_df, meta, left_on="sta_from", right_on="ID", how="inner")
        
        # 2. M/G/c/c 실제 차단확률(P_c) 사전 계산 및 캐싱
        mgcc_list = []
        for _, row in self.seg_df.iterrows():
            L_m = row["Length"] * 1609.34      # 마일 -> 미터 변환
            lanes = row["Lanes"]
            vol_day = row["vol_day_veh"]
            
            p_c = blocking_probability(L_m, lanes, VF_MS, vol_day)
            mgcc_list.append(p_c)
            
        self.cached_mgcc = np.array(mgcc_list)
        
        # 데이터 검증
        print(f"[검증] Cached Prob -> Max: {np.max(self.cached_mgcc):.4e}, Min: {np.min(self.cached_mgcc):.4e}")
        
        # 3. 사이트와 세그먼트 좌표 추출
        self.site_coords = self.sites_df[['x_m', 'y_m']].values  # (K, 2)
        self.seg_coords = self.seg_df[['mid_x', 'mid_y']].values # (N, 2)
        
        self.K = len(self.site_coords)
        self.N = len(self.seg_coords)
        
        # 4. 물리적 유클리드 거리 행렬 사전 계산 (N x K)
        diff_x = self.seg_coords[:, 0][:, np.newaxis] - self.site_coords[:, 0]
        diff_y = self.seg_coords[:, 1][:, np.newaxis] - self.site_coords[:, 1]
        self.base_dist_matrix = np.hypot(diff_x, diff_y)
        
        print(f"[RL Env] 초기화 완료: 분기점 {self.K}개, 분석 가능 도로 구간 {self.N}개")

    def step(self, weights):
        """
        [RL Agent 호출용 핵심 함수] 
        매 에피소드 스텝마다 에이전트의 가중치를 받아 즉각적으로 보상 지표를 반환합니다.
        
        :param weights: 모델이 출력한 가중치 배열 (길이 K, 1차원)
        :return: (평균 표준편차, 각 셀별 표준편차 배열) -> 원래 확률값 기준
        """
        if len(weights) != self.K:
            raise ValueError(f"입력된 가중치의 개수({len(weights)})가 분기점 개수({self.K})와 다릅니다.")
            
        safe_weights = np.clip(weights, 1e-0, None)
        
        # [곱셈형 가중 보로노이] 거리를 가중치로 나눔
        weighted_dist = self.base_dist_matrix / safe_weights
        assigned_sites = np.argmin(weighted_dist, axis=1)
        
        # 분기점별 원래 확률(P_c)의 표준편차 집계
        stds = np.zeros(self.K)
        for k in range(self.K):
            group_vals = self.cached_mgcc[assigned_sites == k]
            # print(np.average(group_vals),np.std(group_vals))
            if len(group_vals) > 1:
                stds[k] = np.std(group_vals)
            else:
                stds[k] = 0.0
        # print()
        # 표준편차들의 평균값 계산
        mean_std = np.mean(stds)
        
        return mean_std, stds


# ==============================================================================
# 4. 학습 루프 사전 테스트 실행부
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
        
        print("\n--- [학습 루프 테스트: 가중치 변동 시뮬레이션] ---")
        
        uniform_weights = np.ones(env.K)
        mean_std_uni, all_stds_uni = env.step(uniform_weights)
        print(f"[Test 1] 균등 가중치 적용 시 평균 표준편차: {mean_std_uni:.8f}")
        print(f"         각 구역 표준편차(앞 5개): {all_stds_uni[:5].round(8)}")
        
        extreme_weights = np.random.uniform(0.1, 10.0, size=env.K)
        mean_std_ext, all_stds_ext = env.step(extreme_weights)
        print(f"[Test 2] 극단적 가중치 변경 시 평균 표준편차: {mean_std_ext:.8f}")
        print(f"         각 구역 표준편차(앞 5개): {all_stds_ext[:5].round(8)}")
        
        diff = abs(mean_std_uni - mean_std_ext)
        print(f"-> 가중치 변화에 따른 지표 변동폭: {diff:.8e}")
        
    except FileNotFoundError as e:
        print(f"\n[오류] 데이터 파일을 찾을 수 없습니다: {e}")