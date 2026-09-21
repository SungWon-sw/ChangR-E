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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from features.data_preprocessing.vor_sd.mw_cut import graph_cut_segments_fast
from features.data_preprocessing.vor_sd.mg_cc_batch import blocking_probability_batch
from features.data_preprocessing.vor_sd.ff_links import load_ff_links

from scipy.stats import norm

VF_MPH_DEFAULT = 65.0
MPH2MS = 0.44704
VF_MS = VF_MPH_DEFAULT * MPH2MS
PHF = 60/90


class TrafficRLEnvMW:
    def __init__(self, segments_csv, sites_csv, meta_txt,
                 rho_max=2.0,          # 허용 가중치 비 w_max/w_min
                 min_len=1.0,          # 자투리 컷 [m]
                 min_pieces=2,         # 이보다 조각이 적은 셀은 목적함수에서 제외
                 objective="within",   # "global" | "within" | "mixed"
                 lam_scaling="none",   # "none" | "length"  (아래 설명 참조)
                 phf=PHF,              # vol_day_veh -> 시간당 도착률 환산계수
                 connector_tol=300.0): # 대롱대롱(degree-1) 끝점 스냅 허용거리 [m] (아래 설명 참조)
        """segments/sites/meta CSV 를 읽어 도로망 + 도로 그래프(_build_road_graph)를
        구성하고, RL 상태(log-가중치 a)와 목적함수 설정을 초기화한다.

        phf 는 lam = vol_day_veh * phf / 3600 [veh/s] 로 쓰인다.
        - 종일 파일(vol_day_veh = 일 총량)이면 기본값 0.15 (첨두시간 계수).
        - pems_pipeline.py --pivot/--span 으로 만든 시간창 파일은 vol_day_veh 가
          '그 창 동안의 통과 대수'라서 phf = 60/span 을 넘겨야 실제 창 도착률
          (= vol / (span*60)) 이 된다. 기본값 그대로 두면 창 카운트를 일 총량으로
          오해해 도착률이 크게 과소평가된다.
        """
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
        # 관측 데이터가 없는 선분(vol NaN)은 NaN 이 차단확률 -> 셀 σ 로 번지므로
        # 관측된 선분들의 평균 교통량으로 보간한다.
        nan_vol = np.isnan(self.vols)
        if nan_vol.any():
            self.vols[nan_vol] = np.nanmean(self.vols)
            print(f"[MW Env] vol_day_veh 결측 {int(nan_vol.sum())}/{len(self.vols)}개 "
                  f"-> 평균 {self.vols[nan_vol][0]:.0f} 로 보간")
        self.phf = float(phf)
        self.lam = (self.vols * self.phf) / 3600.0
        self.lam_scaling = lam_scaling

        self.P = self.seg_df[["x1", "y1"]].values.astype(float)
        self.Q = self.seg_df[["x2", "y2"]].values.astype(float)
        self.seg_len = np.linalg.norm(self.Q - self.P, axis=1)
        self.connector_tol = float(connector_tol)
        self._meta_txt = meta_txt
        self._build_road_graph()

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

        self._obs_a_scale = 1.0 / max(float(self.a_bound), 1e-6)
        self._act_scale = 0.3 * float(self.a_bound)   # 스텝당 사이트별 최대 이동폭
        self._J_prev = None

    def _build_road_graph(self):
        """도로 선분을 길이 가중 무방향 그래프로 만들고 사이트를 노드에 매핑한다.

        세그먼트 끝점이 좌표 완전일치로만 병합되면, 같은 물리적 교차점인데도
        세그먼트마다 독립적으로 지오코딩돼 좌표가 몇 m~수백 m씩 어긋난 지점들이
        전부 별개 노드로 남아 도로망 전체가 수십 개의 고립된 섬(경로)으로
        쪼개진다(실측: 47개 컴포넌트, degree>=3인 진짜 분기 노드가 하나도 없이
        전부 degree 1(끝점)/2(통과)뿐 — 47개의 단순 경로, (고속도로,방향) 조합
        하나당 사슬 하나). 그 결과 그래프 최단거리 기반 소유권 판정이 "실제로
        더 가까운 사이트"가 아니라 "우연히 같은 컴포넌트에 있는 사이트"로
        결정돼 버린다.

        두 단계로 연결자 간선을 추가해 복구한다(둘 다 물리 세그먼트 self.P/Q,
        self.edge_u/edge_v — 실제 교통량이 흐르는 도로 조각 — 는 건드리지
        않는다. 연결자는 Dijkstra 그래프 전용):

          1) **FF 관측소 Name 기반(우선, 근거 있음)** — PeMS 메타데이터의
             Type=="FF"(고속도로-고속도로 연결) 관측소는 Name 필드에
             "SB 110 TO EB 105"처럼 정확히 어느 두 (고속도로,방향) 사슬을
             잇는지 문자열로 적혀 있다(ff_links.py 가 파싱). 이건 좌표
             추측이 아니라 데이터에 적힌 사실이라, 두 사슬에서 그 FF 지점에
             가장 가까운 노드를 찾아 간선으로 잇는다.
          2) **좌표 근접 휴리스틱(폴백)** — 1)로도 안 이어진 나머지에 한해,
             대롱대롱 매달린(degree-1) 끝점들 중 서로가 서로의 최근접
             끝점이면서(상호 최근접) 거리가 connector_tol 이내인 쌍만
             연결자로 추가한다. (District 경계에서 실제로 지도 밖으로
             빠져나가 안 이어지는 사슬도 있다 — 전부 다 이어야 하는 게
             아니다.)
        """
        coords = np.unique(np.vstack([self.P, self.Q]), axis=0)
        self.graph_coords = coords
        self.edge_u = np.argmin(np.linalg.norm(self.P[:, None] - coords[None, :], axis=2), axis=1)
        self.edge_v = np.argmin(np.linalg.norm(self.Q[:, None] - coords[None, :], axis=2), axis=1)
        rows = np.r_[self.edge_u, self.edge_v]
        cols = np.r_[self.edge_v, self.edge_u]
        data = np.r_[self.seg_len, self.seg_len]
        n = len(coords)
        graph0 = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        n_before = connected_components(graph0, directed=False)[0]

        # ---- 1) FF 관측소 Name 기반 연결자 -------------------------------
        ff_rows, ff_cols, ff_data = [], [], []
        n_ff_connectors = 0
        fwy_arr = self.seg_df["fwy"].values
        dir_arr = self.seg_df["dir"].values
        chain_nodes = {}
        for i in range(self.N):
            key = (fwy_arr[i], dir_arr[i])
            chain_nodes.setdefault(key, set()).update((int(self.edge_u[i]), int(self.edge_v[i])))
        chain_keys = set(chain_nodes.keys())

        try:
            links = load_ff_links(self._meta_txt, chain_keys)
        except Exception as e:
            links = []
            print(f"[MW Env] FF 연결자 파싱 실패({e}) — 좌표 휴리스틱만 사용")

        if links:
            meta_all = pd.read_csv(self._meta_txt, sep="\t").dropna(subset=["Latitude", "Longitude"])
            lat0, lon0 = meta_all.Latitude.mean(), meta_all.Longitude.mean()
            for side1, side2, lat, lon in links:
                x = (lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
                y = (lat - lat0) * 111320.0
                idx1 = np.fromiter(chain_nodes[side1], int)
                idx2 = np.fromiter(chain_nodes[side2], int)
                n1 = idx1[np.argmin(np.linalg.norm(coords[idx1] - [x, y], axis=1))]
                n2 = idx2[np.argmin(np.linalg.norm(coords[idx2] - [x, y], axis=1))]
                if n1 == n2:
                    continue
                d = float(np.linalg.norm(coords[n1] - [x, y]) + np.linalg.norm(coords[n2] - [x, y]))
                ff_rows += [int(n1), int(n2)]
                ff_cols += [int(n2), int(n1)]
                ff_data += [d, d]
                n_ff_connectors += 1

        if ff_rows:
            rows = np.r_[rows, ff_rows]
            cols = np.r_[cols, ff_cols]
            data = np.r_[data, ff_data]
        graph1 = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        n_mid, labels = connected_components(graph1, directed=False)

        # ---- 2) 좌표 근접 휴리스틱(폴백) ----------------------------------
        deg = np.asarray((graph1 > 0).sum(axis=1)).ravel()
        dangling = np.where(deg == 1)[0]

        conn_rows, conn_cols, conn_data = [], [], []
        n_connectors = 0
        if len(dangling) >= 2:
            tree = KDTree(coords[dangling])
            nn_dist, nn_idx = tree.query(coords[dangling], k=2)
            nearest_local = nn_idx[:, 1]                 # 자기 자신 제외 최근접(로컬 인덱스)
            nearest_dist = nn_dist[:, 1]
            mutual = nearest_local[nearest_local] == np.arange(len(dangling))
            within_tol = nearest_dist <= self.connector_tol
            i_local = np.where(mutual & within_tol)[0]
            pairs = {tuple(sorted((int(i), int(nearest_local[i])))) for i in i_local}

            for li, lj in pairs:
                ni, nj = int(dangling[li]), int(dangling[lj])
                if labels[ni] == labels[nj]:
                    continue                              # 이미 같은 컴포넌트(자기 루프 방지)
                d = float(np.linalg.norm(coords[ni] - coords[nj]))
                conn_rows += [ni, nj]
                conn_cols += [nj, ni]
                conn_data += [d, d]
                n_connectors += 1

        if conn_rows:
            rows = np.r_[rows, conn_rows]
            cols = np.r_[cols, conn_cols]
            data = np.r_[data, conn_data]
        graph = coo_matrix((data, (rows, cols)), shape=(n, n))

        site_to_node = np.linalg.norm(self.site_coords[:, None] - coords[None, :], axis=2)
        self.site_node = site_to_node.argmin(axis=1)
        self.site_graph_distance = site_to_node.min(axis=1)
        self.node_dist = dijkstra(graph.tocsr(), directed=False,
                                  indices=self.site_node)
        n_after = connected_components(graph.tocsr(), directed=False)[0]
        print(f"[MW Env] 그래프 노드 {len(coords)}개, 사이트-도로 노드 매핑 최대오차 "
              f"{self.site_graph_distance.max():.1f} m")
        print(f"[MW Env] 연결성 보정: 컴포넌트 {n_before}개 "
              f"-(FF Name 연결자 {n_ff_connectors}개)-> {n_mid}개 "
              f"-(좌표 휴리스틱 {n_connectors}개, tol={self.connector_tol:.0f}m)-> {n_after}개")

    # ------------------------------------------------------------------
    def weights(self, a=None):
        """로그-가중치 a (생략 시 self.a) 로부터 실제 MW 가중치 w = exp(a - mean(a)) 를 구한다.
        평균을 빼는 것이 스케일 불변(w -> cw 는 동일 다이어그램) 게이지 고정이다."""
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
        Lmat, info = graph_cut_segments_fast(
            self.P, self.Q, self.edge_u, self.edge_v, self.site_node,
            self.node_dist, w, min_len=self.min_len, return_info=True)
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
        """log-가중치 a 를 [-0.1, 0.1] 범위의 무작위값(평균 0)으로 재초기화하고,
        관측 벡터(obs_dim = 3K+1: 정규화된 a·셀별 std·굶은 셀 마스크·J)를 반환한다."""
        self.a = np.random.uniform(-0.1, 0.1, self.K)
        self.a -= self.a.mean()
        J, stds, cnt, valid = self.evaluate(self.a, return_cells=True)
        self._J_prev = J                      # 개선량 보상용
        return self._make_obs(J, stds, valid)

    def step(self, action):
        """action 은 사이트(그래프 정점)별 Δa (길이 K, 각 성분 (-1,1))다. 공간 커널로
        퍼뜨리지 않고 각 정점에 직접 대응시킨다 — 그래프 보로노이는 이산 구조(정점=사이트,
        간선=도로, 간선 가중치=길이)이므로 연속 평면 위의 가우시안 범프로 스무딩할 이유가
        없다: 도로망 자체의 정점/간선 가중치는 evaluate() 의 graph_cut_segments_fast 가
        이미 최단거리로 반영한다. 한 스텝에 각 사이트를 최대 _act_scale 만큼만 밀어
        (iterative refinement), 평균 0 게이지로 재정규화한 뒤 a_bound(rho_max 제약)로
        클리핑한다. 보상은 개선량 r = J_prev - J (줄이면 +, 늘리면 -). (obs, reward,
        done(항상 False), info={stds,J,cnt,n_valid}) 튜플을 반환한다."""
        act = np.clip(np.asarray(action, float).ravel(), -1.0, 1.0)
        self.a = self.a + act * self._act_scale

        self.a -= self.a.mean()
        self.a = np.clip(self.a, -self.a_bound, self.a_bound)
        self.a -= self.a.mean()          # 클리핑 후 재중심화

        J, stds, cnt, valid = self.evaluate(self.a, return_cells=True)  # 스텝당 1회
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