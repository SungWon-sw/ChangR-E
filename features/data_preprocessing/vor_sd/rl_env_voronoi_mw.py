"""
TrafficRLEnv — MW(곱셈가중) 보로노이 버전.

기존 power(멱) 다이어그램 버전과 인터페이스는 동일하다. 바뀐 것:

  1) 상태변수가 w 가 아니라 a = log w  이다.
       - MW 는 스케일 불변(모든 w 에 상수배 -> 동일 다이어그램)이므로
         a 의 '평균 0' 이 기존 w 의 '평균 0' 에 대응하는 올바른 게이지 고정.
       - w = exp(a) > 0 이 자동 보장된다.
       - spacing2 곱셈은 삭제. MW 가중치는 무차원이라 단위 보정이 필요 없다.
         대신 a 의 범위가 곧 w_max/w_min 비율을 결정한다:  rho = exp(max a - min a).

  2) 폴리곤을 만들지 않는다. MW 셀은 비볼록/비연결/구멍 가능이라
     Shapely Polygon 으로 표현하기 까다롭다. 대신 도로 선분을 직접
     이차방정식으로 잘라 (선분 x 셀) 유효길이 행렬을 정확히 얻는다.
     -> STRtree / Polygon / intersection 전부 불필요.

  3) M/G/c/c 를 배치로 계산한다 (수학적으로 동일, ~30x 빠름).
"""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from features.data_preprocessing.vor_sd.mw_cut import cut_segments_fast
from features.data_preprocessing.vor_sd.mg_cc_batch import blocking_probability_batch

from math import sqrt
from scipy.stats import norm

VF_MPH_DEFAULT = 65.0
MPH2MS = 0.44704
VF_MS = VF_MPH_DEFAULT * MPH2MS
PHF = 0.15


class TrafficRLEnvMW:
    def __init__(self, segments_csv, sites_csv, meta_txt,
                 rho_max=2.0,          # 허용 가중치 비 w_max/w_min
                 min_len=1.0,          # 자투리 컷 [m]
                 min_pieces=2,         # 이보다 조각이 적은 셀은 목적함수에서 제외
                 objective="within",   # "global" | "within" | "mixed"
                 lam_scaling="none"):  # "none" | "length"  (아래 설명 참조)
        self.sites_df = pd.read_csv(sites_csv)
        self.seg_df = pd.read_csv(segments_csv)
        
        
        meta = pd.read_csv(meta_txt, sep="\t")
        meta = meta.dropna(subset=["ID", "Lanes"])[["ID", "Lanes"]]
        meta = meta.drop_duplicates(subset=["ID"])          # 중복 ID 로 인한 행 증식 방지
        self.seg_df = pd.merge(self.seg_df, meta,
                               left_on="sta_from", right_on="ID", how="inner")

        self.site_coords = self.sites_df[["x_m", "y_m"]].values.astype(float)
        self.K = len(self.site_coords)
        self.N = len(self.seg_df)

        self.lanes = self.seg_df["Lanes"].values.astype(float)
        self.vols = self.seg_df["vol_day_veh"].values.astype(float)
        self.lam = (self.vols * PHF) / 3600.0
        self.lam_scaling = lam_scaling

        self.P = self.seg_df[["x1", "y1"]].values.astype(float)
        self.Q = self.seg_df[["x2", "y2"]].values.astype(float)
        self.seg_len = np.linalg.norm(self.Q - self.P, axis=1)

        # bbox 는 더 이상 필요 없다 (셀을 자르지 않고 선분만 다루므로).
        self.rho_max = float(rho_max)
        self.a_bound = np.log(self.rho_max) / 2.0
        self.min_len = float(min_len)
        self.min_pieces = int(min_pieces)
        self.objective = objective

        # 참고용: 액션 스케일 감각. MW 에서 두 사이트를 잇는 선분 위의 경계는
        #   r/d = sigmoid(a_i - a_j)  위치에 온다. 즉 Δa=1 이면 경계가 약 0.23d 이동.
        nn = KDTree(self.site_coords).query(self.site_coords, k=2)[0][:, 1]
        self.spacing = float(np.median(nn))

        self.a = np.zeros(self.K)
        self.finalW = np.zeros(self.K)
        self.finalA = 99999999
        print(f"[MW Env] 분기점 {self.K}개, 도로 선분 {self.N}개, "
              f"이웃간격 중앙값 {self.spacing:.0f} m, rho_max={self.rho_max}")
        
        self.x_min, self.y_min = self.site_coords.min(axis=0)
        self.x_max, self.y_max = self.site_coords.max(axis=0)
        self._obs_a_scale = 1.0 / max(float(self.a_bound), 1e-6)
        self._act_scale = 0.3 * float(self.a_bound)   # 스텝당 사이트별 최대 이동폭
        self._J_prev = None

    # ------------------------------------------------------------------
    def weights(self, a=None):
        a = self.a if a is None else np.asarray(a, float)
        return np.exp(a - a.mean())

    def evaluate(self, a, return_cells=False):
        """log-가중치 a -> (목적함수값 J, 셀별 std).

        return_cells=True 면 (J, stds, cnt, valid) 를 반환한다 (RL 관측 조립용).
        퇴화(도로 한 조각도 못 잡음)는 J=1.0 (정상값 ~0.1 대비 큰 페널티).
        """
        a = np.asarray(a, float)
        if len(a) != self.K:
            raise ValueError(f"가중치 개수({len(a)}) != 분기점 개수({self.K})")
        w = self.weights(a)

        # 1) 선분을 MW 셀 경계로 정확히 절단 -> (N, K) 유효길이
        Lmat, info = cut_segments_fast(self.P, self.Q, self.site_coords, w,
                                       min_len=self.min_len, return_info=True)
        seg_i, cell_i = np.nonzero(Lmat)
        if len(seg_i) == 0:
            z = np.zeros(self.K)
            return (1.0, z, z.copy(), z.copy()) if return_cells else (1.0, z)
        L_eff = Lmat[seg_i, cell_i]

        # 2) 배치 M/G/c/c
        lam_eff = self.lam[seg_i]
        if self.lam_scaling == "length":
            # 조각이 원 선분의 일부만 차지하면 도착률도 비례 축소 (선택 사항)
            lam_eff = lam_eff * (L_eff / self.seg_len[seg_i])
        probs, cap = blocking_probability_batch(
            L_eff, VF_MS, lam_eff, self.lanes[seg_i])

        # 3) 셀별 집계
        cnt = np.bincount(cell_i, minlength=self.K)
        s1 = np.bincount(cell_i, weights=probs, minlength=self.K)
        s2 = np.bincount(cell_i, weights=probs ** 2, minlength=self.K)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
        stds = np.where(cnt >= self.min_pieces, np.sqrt(var), np.nan)

        valid = cnt >= self.min_pieces
        within = float(np.sqrt(np.nanmean(stds[valid] ** 2))) if valid.any() else 0.
        glob = float(probs.std())

        if self.objective == "within":
            J = within
        elif self.objective == "global":
            J = glob
        else:                                   # mixed
            J = within + glob

        info = dict(info, n_pieces=len(probs), n_valid_cells=int(valid.sum()),
                    within=within, global_std=glob,
                    mean_prob=float(probs.mean()), rho=float(w.max() / w.min()))
        stds = np.nan_to_num(stds)
        if return_cells:
            return J, stds, cnt.astype(float), valid.astype(float)
        return J, stds

    # ------------------------------------------------------------------
    @property
    def obs_dim(self):
        # [ a/a_bound (K), stds*10 (K), valid (K), J*10 (1) ]
        return 3 * self.K + 1

    def _make_obs(self, J, stds, valid):
        """정책이 보는 관측 벡터. 블록별로 대략 O(1) 스케일로 맞춘다."""
        return np.concatenate([
            self.a * self._obs_a_scale,          # 현재 로그가중치
            np.asarray(stds, float) * 10.0,      # 셀별 blocking std = 불균형 지도
            np.asarray(valid, float),            # 굶은 셀 마스크 (0=조각<min_pieces)
            [float(J) * 10.0],                   # 현재 목적함수값
        ]).astype(np.float32)

    def reset(self):
        self.a = np.random.uniform(-0.1, 0.1, self.K)
        self.a -= self.a.mean()
        J, stds, cnt, valid = self.evaluate(self.a, return_cells=True)
        self._J_prev = J                      # 개선량 보상용
        return self._make_obs(J, stds, valid)

    def step(self, action):
        # 액션 = 사이트별 Δa (길이 K, 각 성분 (-1,1)). 한 스텝에 각 사이트를
        # 최대 _act_scale 만큼 민다 (iterative refinement 용).
        act = np.clip(np.asarray(action, float).ravel(), -1.0, 1.0)
        self.a = self.a + act * self._act_scale

        self.a -= self.a.mean()
        self.a = np.clip(self.a, -self.a_bound, self.a_bound)
        self.a -= self.a.mean()          # 클리핑 후 재중심화

        J, stds, cnt, valid = self.evaluate(self.a, return_cells=True)  # 스텝당 1회
        # 개선량 보상: r = J_prev - J  (줄이면 +, 늘리면 -). 상수 오프셋 제거.
        if self._J_prev is None:          # reset 없이 step 호출된 경우
            self._J_prev = J
        reward = float(self._J_prev - J)
        self._J_prev = J
        if J < self.finalA:
            self.finalA = J
            self.finalW = self.a.copy()
        obs = self._make_obs(J, stds, valid)
        info = {"stds": stds, "J": J, "cnt": cnt, "n_valid": int(valid.sum())}
        return obs, reward, False, info