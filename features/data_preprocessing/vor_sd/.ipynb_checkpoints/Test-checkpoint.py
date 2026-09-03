
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from mw_cut import cut_segments_fast
from mg_cc_batch import blocking_probability_batch

from math import sqrt
from scipy.stats import norm

VF_MPH_DEFAULT = 65.0
MPH2MS = 0.44704
VF_MS = VF_MPH_DEFAULT * MPH2MS
PHF = 0.15

class TrafficRLEnvMW:
    def __init__(self, segments_csv, sites_csv, meta_txt,
                 rho_max=50.0,          # 허용 가중치 비 w_max/w_min
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

    # ------------------------------------------------------------------
    def weights(self, a=None):
        a = self.a if a is None else np.asarray(a, float)
        return np.exp(a - a.mean())

    def evaluate(self, a):
        """log-가중치 a 를 받아 (목적함수값, 셀별 지표, 진단정보) 반환."""
        a = np.asarray(a, float)
        if len(a) != self.K:
            raise ValueError(f"가중치 개수({len(a)}) != 분기점 개수({self.K})")
        w = self.weights(a)

        # 1) 선분을 MW 셀 경계로 정확히 절단 -> (N, K) 유효길이
        Lmat, info = cut_segments_fast(self.P, self.Q, self.site_coords, w,
                                       min_len=self.min_len, return_info=True)
        seg_i, cell_i = np.nonzero(Lmat)
        if len(seg_i) == 0:
            return 0.0, np.zeros(self.K), info
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
        within = float(np.nanmean(stds[valid])) if valid.any() else 0.0
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
        return J, np.nan_to_num(stds)

    # ------------------------------------------------------------------
    def reset(self):
        self.a = np.random.uniform(-0.1, 0.1, self.K)
        self.a -= self.a.mean()
        return self.a.copy()

    def step(self, action):
        # 액션을 a 공간에서 그대로 더한다. a_bound 가 곧 rho_max 제약.
        A = action [0]
        xpos = self.x_min + (action[1] + 1) / 2 * (self.x_max - self.x_min)
        ypos = self.y_min + (action[2] + 1) / 2 * (self.y_max - self.y_min)
        mu   = max(1e-5, (action[3] + 1) / 2 * self.spacing)
        # tanh 스케일링 반영함
        for i in range(0,len(self.a)):
            uclid_dist = sqrt((xpos-self.site_coords[i][0])**2 + (ypos-self.site_coords[i][1])**2)
            kernel = np.exp(-0.5 * (uclid_dist / mu) ** 2)   # dist=0일 때 항상 1, mu와 무관
            self.a[i] += A * kernel
        # self.a += action
        self.a -= self.a.mean()
        self.a = np.clip(self.a, -self.a_bound, self.a_bound)
        self.a -= self.a.mean()          # 클리핑 후 재중심화

        mean_std, stds = self.evaluate(self.a)
        reward = -mean_std
        if(mean_std < self.finalA):
            self.finalA=mean_std
            self.finalW = self.a.copy()
        J, stds = self.evaluate(self.a)
        
        reward = -J
        return self.a.copy(), reward, False, stds

env=TrafficRLEnvMW()

it=100000
for i in range(it):
    env.reset()
    print(env.evaluate(env.a))