"""
pems_pipeline.py — PeMS D07(2018-10-13) 실데이터로 3개 작업 수행:
  (1) 분기점(FF 인터체인지) + 본선 기하학적 교차 보완 -> 보로노이 사이트로
  (2) MW(곱셈가중) 보로노이 분할 + 선분 정확 절단으로 수요 할당
  (3) 5분 데이터 전처리 -> 본선 구간별 교통량(volume)·속도(speed)

Task 2 는 power(멱) 다이어그램에서 MW(곱셈가중) 보로노이로 교체되었다.
rl_env_voronoi_mw.py 와 동일한 엔진(mw_cut.cut_segments_fast)을 쓰므로,
이 파이프라인이 만드는 CSV 와 RL 환경이 보는 셀 분할이 일치한다.

바뀐 점:
  1) 상태변수가 w 가 아니라 a = log w.  MW 는 스케일 불변(모든 w 에 상수배
     -> 동일 다이어그램)이라 a 의 '평균 0' 이 올바른 게이지 고정이고,
     w = exp(a) > 0 이 자동 보장된다.  power 가중치의 spacing2 단위보정은
     불필요해져 삭제했다.  대신 a 의 범위가 곧 rho = w_max/w_min 을 정한다.
  2) 폴리곤을 만들지 않는다.  MW 셀은 비볼록/비연결/구멍이 가능해 Shapely
     Polygon 으로 표현하기 까다롭다.  대신 선분을 이차방정식으로 직접 잘라
     (선분 x 셀) 유효길이 행렬을 얻는다 -> STRtree/Polygon/intersection 불필요.
"""
import numpy as np, pandas as pd, sys, os
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# 스크립트로 직접 실행하면 sys.path[0] 이 이 파일의 디렉터리라 저장소 루트를
# 찾지 못한다. __file__ 기준으로 루트(= features/ 의 부모)를 얹어 준다.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from features.data_preprocessing.vor_sd.mw_cut import cut_segments_fast

UP   = "features/data_preprocessing/vor_sd"
META = f"{UP}/d07_text_meta_2018_10_13.txt"
FIVE = f"{UP}/d07_text_station_5min_2018_10_13.csv"
OUT  = f"{UP}/outputs"; os.makedirs(OUT, exist_ok=True)

# ======================================================================
# Task 1 — 메타데이터 -> 분기점 사이트 + 본선 도로 (기하 교차점 보완 포함)
# ======================================================================
m = pd.read_csv(META, sep="\t").dropna(subset=["Latitude", "Longitude"])
lat0, lon0 = m.Latitude.mean(), m.Longitude.mean()
def project(lat, lon):
    return ((lon - lon0) * np.cos(np.radians(lat0)) * 111320.0,
            (lat - lat0) * 111320.0)
m["x"], m["y"] = project(m.Latitude.values, m.Longitude.values)

# 1. FF 기반 기본 사이트 추출
ff = m[m.Type == "FF"]
P = ff[["x", "y"]].values
pairs = cKDTree(P).query_pairs(r=800, output_type="ndarray")
g = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
ncomp, lab = connected_components(g + g.T, directed=False)
ff_sites = [P[lab == k].mean(0) for k in range(ncomp)]

# 2. 본선(ML) 간 기하학적 교차 분석 (숨겨진 인터체인지 탐지)
ml = m[m.Type == "ML"].copy()
ml_by_fwy = [grp for _, grp in ml.groupby("Fwy")]
hidden_crossings = []

for i in range(len(ml_by_fwy)):
    for j in range(i + 1, len(ml_by_fwy)):
        c1 = ml_by_fwy[i][["x", "y"]].values
        c2 = ml_by_fwy[j][["x", "y"]].values
        tree = cKDTree(c2)
        dists, idxs = tree.query(c1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 500:
            p1 = c1[min_idx]; p2 = c2[idxs[min_idx]]
            hidden_crossings.append((p1 + p2) / 2.0)

# 3. 공백 병합
final_sites = list(ff_sites)
if hidden_crossings:
    ff_tree = cKDTree(np.array(ff_sites))
    for pt in hidden_crossings:
        dist, _ = ff_tree.query(pt)
        if dist >= 1500:
            is_dup = False
            for ext_pt in final_sites[len(ff_sites):]:
                if np.hypot(*(pt - ext_pt)) < 1000:
                    is_dup = True; break
            if not is_dup:
                final_sites.append(pt)

sites = np.array(final_sites)
n_sites = len(sites)

# 본선 구간(Segment) 생성
seg_mid, seg_xy, seg_fw = [], [], []
for (fw, d), grp in ml.groupby(["Fwy", "Dir"]):
    grp = grp.sort_values("Abs_PM")
    xy = grp[["x", "y"]].values
    for k in range(len(grp) - 1):
        a, b = xy[k], xy[k + 1]
        if np.hypot(*(a - b)) < 6000:
            seg_xy.append((a, b)); seg_mid.append((a + b) / 2)
            seg_fw.append((fw, d, grp.ID.values[k], grp.ID.values[k + 1]))
seg_mid = np.array(seg_mid)
print(f"[Task1] 분기점 사이트 {n_sites}개 | 본선 구간 {len(seg_xy)}개")

# ======================================================================
# Task 3 — 5분 데이터 -> 본선 스테이션별 교통량·속도
# ======================================================================
df = pd.read_csv(FIVE, header=None, usecols=[0, 1, 5, 8, 9, 10, 11],
                 names=["ts", "station", "ltype", "pct_obs", "flow", "occ", "speed"])
d = df[df.ltype == "ML"].copy()
d["hour"] = pd.to_datetime(d.ts, format="%m/%d/%Y %H:%M:%S").dt.hour

def fw_speed(grp):
    sp, fl = grp.speed.values, grp.flow.values
    ok = ~np.isnan(sp)
    if ok.sum() == 0: return np.nan
    w = fl[ok]
    return np.average(sp[ok], weights=w) if np.nansum(w) > 0 else np.nanmean(sp[ok])

agg = d.groupby("station").apply(
    lambda grp: pd.Series({"vol_day": np.nansum(grp.flow.values),
                           "speed_day": fw_speed(grp)})).reset_index()
pm = d[d.hour.between(15, 18)].groupby("station").apply(fw_speed)
agg = agg.merge(pm.rename("speed_pm"), on="station", how="left")
ml = ml.merge(agg, left_on="ID", right_on="station", how="left")

id2spd_pm = dict(zip(ml.ID, ml.speed_pm))
id2vol = dict(zip(ml.ID, ml.vol_day))
seg_speed_pm = np.array([np.nanmean([id2spd_pm.get(a, np.nan), id2spd_pm.get(b, np.nan)]) for (_, _, a, b) in seg_fw])
seg_vol = np.array([np.nanmean([id2vol.get(a, np.nan), id2vol.get(b, np.nan)]) for (_, _, a, b) in seg_fw])

# ======================================================================
# Task 2 — MW(곱셈가중) 보로노이 + 선분 정확 절단을 통한 셀 배정 최적화
# ======================================================================
RHO_MAX  = 50.0                       # 허용 가중치 비 w_max / w_min
A_BOUND  = np.log(RHO_MAX) / 2.0      # a 의 허용 범위 (-A_BOUND, +A_BOUND)
MAX_CUTS = 512                        # 선분당 절단점 상한. 실측 최대 166 이므로
                                      # 여유를 둔다. 부족하면 긴 선분의 꼬리가
                                      # 통째로 한 셀에 오배정된다.

# bbox / spacing2 는 더 이상 필요 없다 (폴리곤을 만들지 않고, MW 가중치는 무차원).
segP = np.array([s[0] for s in seg_xy], float)
segQ = np.array([s[1] for s in seg_xy], float)
seg_lengths = np.linalg.norm(segQ - segP, axis=1)
mass = np.nan_to_num(seg_vol)

n_degen = int((seg_lengths < 1e-9).sum())
if n_degen:
    print(f"[Task2] !! 길이 0 인 선분 {n_degen}개 - 수요 배정에서 제외됩니다")
_den = np.where(seg_lengths > 1e-9, seg_lengths, 1.0)   # 0 나눗셈 가드

def mw_weights(a):
    """log-가중치 a -> MW 가중치 w. 평균 0 게이지로 고정한다."""
    a = np.asarray(a, float)
    return np.exp(a - a.mean())

def get_captured_demand_clip(a):
    """MW 셀 경계로 선분을 정확히 잘라, 유효길이 비율만큼 수요를 배분.

    Lmat[n, k] = 선분 n 중 셀 k 가 소유하는 길이 [m] 이므로
    (Lmat / seg_lengths) 는 선분별 소유 비율이고, 이를 mass 로 가중합하면
    셀별 흡수 교통량이 된다. 기존 poly.intersection(line).length 비례분할과
    같은 의미이되, 폴리곤 없이 정확하다.
    """
    Lmat = cut_segments_fast(segP, segQ, sites, mw_weights(a),
                             max_cuts=MAX_CUTS, min_len=0.0)
    return (Lmat / _den[:, None]).T @ mass

def assign_midpoints(points, a):
    """기록/시각화용: 중점 기준 MW 거리 d = |x - p| / w 의 argmin."""
    w = mw_weights(a)
    d = np.linalg.norm(points[:, None, :] - sites[None, :, :], axis=2) / w[None, :]
    return d.argmin(1)

# (a) 유클리드 (a = 0 -> 모든 w 가 같음 = 일반 보로노이)
a_eucl = np.zeros(n_sites)
owner_eucl = assign_midpoints(seg_mid, a_eucl)
cap0 = get_captured_demand_clip(a_eucl)

# (b) 균형화 최적화 (결정론적)
def balance_weights(iters=120, lr=0.30, decay=0.95):
    """a 공간에서 직접 갱신한다.

    cap < target (덜 먹은 셀) 이면 a 를 키운다. MW 에서 w 가 크면 d = |x-p|/w
    가 작아져 셀이 커지므로 부호가 맞다. 가중치가 무차원이라 power 버전의
    spacing2 배율은 필요 없고, A_BOUND 클리핑이 곧 rho_max 제약이다.

    lr 과 decay 는 반드시 필요하다. (target-cap)/target 은 실측 범위가
    [-2.5, +0.94] 라서 power 버전의 lr=0.7 을 그대로 쓰면 스텝당 Δa 가 최대
    1.7 이 된다. MW 에서 Δa=1 은 두 사이트 사이 경계를 간격의 약 23% 나
    옮기므로 곧바로 발산한다(실측 CV 0.72 -> 1.51). decay 로 스텝을 어닐링해야
    수렴한다 (0.95^120 ~ 0.002).

    최적값을 따로 보관해 돌려준다. 진동이 남아도 결과가 a=0 보다 나빠지지
    않음을 보장한다 (a=0 이 탐색공간에 포함되므로).
    """
    a = np.zeros(n_sites)
    target = mass.sum() / n_sites
    best_cv, best_a = np.inf, a.copy()
    for t in range(iters):
        cap = get_captured_demand_clip(a)
        cv = cap.std() / cap.mean()
        if cv < best_cv:
            best_cv, best_a = cv, a.copy()
        a = a + lr * (decay ** t) * (target - cap) / target
        a -= a.mean()
        a = np.clip(a, -A_BOUND, A_BOUND)
        a -= a.mean()                     # 클리핑 후 재중심화
    cap = get_captured_demand_clip(a)     # 마지막 갱신분도 후보에 넣는다
    if cap.std() / cap.mean() < best_cv:
        best_a = a.copy()
    return best_a

a_bal = balance_weights()
owner_bal = assign_midpoints(seg_mid, a_bal)
cap1 = get_captured_demand_clip(a_bal)
w_bal = mw_weights(a_bal)
print(f"[Task2] MW 분할 완료 | 흡수교통량 CV: 유클리드 {cap0.std()/cap0.mean():.3f} "
      f"-> 균형화 {cap1.std()/cap1.mean():.3f} | rho={w_bal.max()/w_bal.min():.2f}")

# ======================================================================
# 시각화 및 전처리 산출물 저장
# ======================================================================
lat_s = lat0 + sites[:, 1] / 111320.0
lon_s = lon0 + sites[:, 0] / (np.cos(np.radians(lat0)) * 111320.0)
site_types = ["FF_Based"] * len(ff_sites) + ["Geometric_Fallback"] * (n_sites - len(ff_sites))

# weight_balanced 는 이제 MW 곱셈가중치 w (>0, 평균 0 게이지) 이다.
# power 버전의 가법 가중치(길이^2 단위, 값이 ±1e7 규모)와는 의미가 다르다.
# RL 환경(rl_env_voronoi_mw.py)이 쓰는 상태변수는 a = log w 이므로 함께 저장한다.
pd.DataFrame({"site_id": np.arange(n_sites), "type": site_types, "x_m": sites[:, 0], "y_m": sites[:, 1],
              "lat": lat_s, "lon": lon_s, "weight_balanced": w_bal, "logw_balanced": a_bal,
              "captured_vol_eucl": cap0, "captured_vol_weighted": cap1}).to_csv(
    f"{OUT}/pems_d07_sites.csv", index=False)

# RL 환경에서 선분을 재구성할 수 있도록 양 끝점 좌표(x1, y1, x2, y2) 명시적 저장
pd.DataFrame({"fwy": [s[0] for s in seg_fw], "dir": [s[1] for s in seg_fw],
              "sta_from": [s[2] for s in seg_fw], "sta_to": [s[3] for s in seg_fw],
              "mid_x": seg_mid[:, 0], "mid_y": seg_mid[:, 1],
              "x1": [s[0][0] for s in seg_xy], "y1": [s[0][1] for s in seg_xy],
              "x2": [s[1][0] for s in seg_xy], "y2": [s[1][1] for s in seg_xy],
              "speed_pm_mph": seg_speed_pm, "vol_day_veh": seg_vol,
              "cell_euclidean_mid": owner_eucl, "cell_weighted_mid": owner_bal}).to_csv(
    f"{OUT}/pems_d07_segments.csv", index=False)
print("saved CSVs")
