"""
pems_pipeline.py — PeMS D07(2018-10-13) 실데이터로 3개 작업 수행:
  (1) 분기점(FF 인터체인지) + 본선 기하학적 교차 보완 -> 보로노이 사이트로
  (2) 가중 보로노이 분할 및 도로 선분 클리핑(Line Clipping) 적용하여 수요 할당
  (3) 5분 데이터 전처리 -> 본선 구간별 교통량(volume)·속도(speed)
"""
import numpy as np, pandas as pd, sys, os
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString, Polygon
from shapely.strtree import STRtree
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, "/home/claude")
from weighted_voronoi import all_power_cells

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
                           "speed_day": fw_speed(grp)}), include_groups=False).reset_index()
pm = d[d.hour.between(15, 18)].groupby("station").apply(fw_speed, include_groups=False)
agg = agg.merge(pm.rename("speed_pm"), on="station", how="left")
ml = ml.merge(agg, left_on="ID", right_on="station", how="left")

id2spd_pm = dict(zip(ml.ID, ml.speed_pm))
id2vol = dict(zip(ml.ID, ml.vol_day))
seg_speed_pm = np.array([np.nanmean([id2spd_pm.get(a, np.nan), id2spd_pm.get(b, np.nan)]) for (_, _, a, b) in seg_fw])
seg_vol = np.array([np.nanmean([id2vol.get(a, np.nan), id2vol.get(b, np.nan)]) for (_, _, a, b) in seg_fw])

# ======================================================================
# Task 2 — 선분 클리핑(Line Clipping)을 통한 셀 배정 최적화
# ======================================================================
pad = 4000
all_x = np.array([p[0] for seg in seg_xy for p in seg])
all_y = np.array([p[1] for seg in seg_xy for p in seg])
bbox = (all_x.min() - pad, all_y.min() - pad, all_x.max() + pad, all_y.max() + pad)

nbr = cKDTree(sites).query(sites, k=2)[0][:, 1]
spacing2 = np.median(nbr) ** 2
mass = np.nan_to_num(seg_vol)

# 속도 최적화를 위한 R-Tree 및 선분 배열 생성
seg_lines = [LineString([a, b]) for a, b in seg_xy]
seg_lengths = np.array([line.length for line in seg_lines])
tree = STRtree(seg_lines)

def get_captured_demand_clip(weights):
    """다각형과 선분의 실제 교차 길이를 구하여 수요를 정확히 비례 분할"""
    cells = all_power_cells(sites, weights, bbox)
    cap = np.zeros(n_sites)
    for i, poly_coords in enumerate(cells):
        if len(poly_coords) < 3: continue
        poly = Polygon(poly_coords)
        idx_hits = tree.query(poly)
        for idx in idx_hits:
            line = seg_lines[idx]
            if poly.intersects(line):
                inter = poly.intersection(line)
                cap[i] += mass[idx] * (inter.length / seg_lengths[idx])
    return cap

def assign_midpoints(points, weights):
    """시각화 플롯 색상 지정을 위해 중점 기준으로 멱 거리 계산"""
    d2 = ((points[:, None, :] - sites[None, :, :]) ** 2).sum(-1) - weights[None, :]
    return d2.argmin(1)

# (a) 유클리드(가중치 0)
w_eucl = np.zeros(n_sites)
owner_eucl = assign_midpoints(seg_mid, w_eucl)
cap0 = get_captured_demand_clip(w_eucl)

# (b) 균형화 최적화 (결정론적)
def balance_weights(iters=120, lr=0.7):
    w = np.zeros(n_sites)
    target = mass.sum() / n_sites
    for _ in range(iters):
        cap = get_captured_demand_clip(w)
        w += lr * spacing2 * (target - cap) / target
        w = np.clip(w - w.mean(), -8 * spacing2, 8 * spacing2)
    return w

w_bal = balance_weights()
owner_bal = assign_midpoints(seg_mid, w_bal)
cap1 = get_captured_demand_clip(w_bal)
print(f"[Task2] 분할 완료 | 흡수교통량 CV: 유클리드 {cap0.std()/cap0.mean():.3f} -> 균형화 {cap1.std()/cap1.mean():.3f}")

# ======================================================================
# 시각화 및 전처리 산출물 저장
# ======================================================================
lat_s = lat0 + sites[:, 1] / 111320.0
lon_s = lon0 + sites[:, 0] / (np.cos(np.radians(lat0)) * 111320.0)
site_types = ["FF_Based"] * len(ff_sites) + ["Geometric_Fallback"] * (n_sites - len(ff_sites))

pd.DataFrame({"site_id": np.arange(n_sites), "type": site_types, "x_m": sites[:, 0], "y_m": sites[:, 1],
              "lat": lat_s, "lon": lon_s, "weight_balanced": w_bal,
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
