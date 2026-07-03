"""
pems_pipeline.py — PeMS D07(2018-10-13) 실데이터로 3개 작업 수행:
  (1) 분기점(FF 인터체인지) + 본선 기하학적 교차 보완 -> 보로노이 사이트로
  (2) 가중치를 입력받아 가중 보로노이(멱 다이어그램) 형성 -> 본선 도로를 셀로 절단
  (3) 5분 데이터 전처리 -> 본선 구간별 교통량(volume)·속도(speed)
"""
import numpy as np, pandas as pd, sys, os
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
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
        
        # 두 고속도로 간 최단거리 탐색
        tree = cKDTree(c2)
        dists, idxs = tree.query(c1)
        min_idx = np.argmin(dists)
        
        if dists[min_idx] < 500: # 500m 이내로 교차 시 실제 인터체인지로 간주
            p1 = c1[min_idx]
            p2 = c2[idxs[min_idx]]
            hidden_crossings.append((p1 + p2) / 2.0)

# 3. 발견된 기하 교차점 중 FF 사이트와 1.5km 이상 떨어진 '실제 공백' 병합
final_sites = list(ff_sites)
if hidden_crossings:
    ff_tree = cKDTree(np.array(ff_sites))
    for pt in hidden_crossings:
        dist, _ = ff_tree.query(pt)
        if dist >= 1500: # 기존 FF 권역 밖인 경우
            # 새로 추가된 보완 노드들끼리의 중복 검사 (1km)
            is_dup = False
            for ext_pt in final_sites[len(ff_sites):]:
                if np.hypot(*(pt - ext_pt)) < 1000:
                    is_dup = True
                    break
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
print(f"[Task1] 분기점 사이트 {n_sites}개 (FF 기반 {len(ff_sites)}개 + 기하 보완 {n_sites-len(ff_sites)}개)")
print(f"        본선 구간 {len(seg_xy)}개 | 고속도로 {ml.Fwy.nunique()}개")

# ======================================================================
# Task 3 — 5분 데이터 -> 본선 스테이션별 교통량·속도 (일/오후첨두)
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
# 오후 첨두(15-19시) 스테이션별 속도
pm = d[d.hour.between(15, 18)].groupby("station").apply(fw_speed, include_groups=False)
agg = agg.merge(pm.rename("speed_pm"), on="station", how="left")
ml = ml.merge(agg, left_on="ID", right_on="station", how="left")

hourly = d.groupby("hour").apply(fw_speed, include_groups=False)
slow_hour = int(hourly.idxmin())
print(f"[Task3] ML {agg.station.nunique()}개 집계 | 일평균속도 {ml.speed_day.mean():.1f} mph | "
      f"일총교통량 중앙값 {ml.vol_day.median():.0f} veh | 최혼잡 {slow_hour}시({hourly.min():.1f} mph)")

id2spd_day = dict(zip(ml.ID, ml.speed_day)); id2spd_pm = dict(zip(ml.ID, ml.speed_pm))
id2vol = dict(zip(ml.ID, ml.vol_day))
seg_speed_pm = np.array([np.nanmean([id2spd_pm.get(a, np.nan), id2spd_pm.get(b, np.nan)]) for (_, _, a, b) in seg_fw])
seg_vol = np.array([np.nanmean([id2vol.get(a, np.nan), id2vol.get(b, np.nan)]) for (_, _, a, b) in seg_fw])

# ======================================================================
# Task 2 — 가중치 입력 -> 가중 보로노이 -> 도로 절단(셀 배정)
# ======================================================================
def assign_to_cells(points, sites, weights):
    d2 = ((points[:, None, :] - sites[None, :, :]) ** 2).sum(-1) - weights[None, :]
    return d2.argmin(1)

pad = 4000
bbox = (seg_mid[:, 0].min() - pad, seg_mid[:, 1].min() - pad,
        seg_mid[:, 0].max() + pad, seg_mid[:, 1].max() + pad)
nbr = cKDTree(sites).query(sites, k=2)[0][:, 1]
spacing2 = np.median(nbr) ** 2
mass = np.nan_to_num(seg_vol)

# (a) 유클리드(가중치 0)
w_eucl = np.zeros(n_sites)
owner_eucl = assign_to_cells(seg_mid, sites, w_eucl)
cap0 = np.zeros(n_sites); np.add.at(cap0, owner_eucl, mass)

# (b) 균형화 최적화(결정론적, RL 자리 대체): 셀별 흡수교통량 균일화
def balance_weights(iters=120, lr=0.7):
    w = np.zeros(n_sites)
    target = mass.sum() / n_sites
    for _ in range(iters):
        owner = assign_to_cells(seg_mid, sites, w)
        cap = np.zeros(n_sites); np.add.at(cap, owner, mass)
        w += lr * spacing2 * (target - cap) / target
        w = np.clip(w - w.mean(), -8 * spacing2, 8 * spacing2)
    return w
w_bal = balance_weights()
owner_bal = assign_to_cells(seg_mid, sites, w_bal)
cap1 = np.zeros(n_sites); np.add.at(cap1, owner_bal, mass)
print(f"[Task2] 도로 절단 | 흡수교통량 CV: 유클리드 {cap0.std()/cap0.mean():.3f} -> 균형화 {cap1.std()/cap1.mean():.3f}")

# ======================================================================
# 시각화
# ======================================================================
def draw_partition(ax, weights, owner, title):
    for poly in all_power_cells(sites, weights, bbox):
        if len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], facecolor="none", edgecolor="0.62", lw=0.6)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    ax.add_collection(LineCollection(seg_xy, colors=colors[owner % 20], lw=1.7))
    ax.scatter(sites[:, 0], sites[:, 1], s=16, c="black", zorder=5, edgecolor="white", lw=0.5)
    
    # 보완 노드(마지막 3개) 시각적 차이 부여 (옵션)
    if n_sites > len(ff_sites):
        ax.scatter(sites[len(ff_sites):, 0], sites[len(ff_sites):, 1], 
                   s=30, c="cyan", zorder=6, edgecolor="black", lw=0.8, marker="*")
        
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

def draw_traffic(ax, title):
    norm = plt.Normalize(25, 70)
    lc = LineCollection(seg_xy, cmap="RdYlGn", norm=norm,
                        lw=np.clip(seg_vol / np.nanmax(seg_vol) * 4.5, 0.7, 4.5))
    lc.set_array(seg_speed_pm)
    ax.add_collection(lc)
    ax.scatter(sites[:, 0], sites[:, 1], s=14, c="black", zorder=5, edgecolor="white", lw=0.5)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(bbox[0], bbox[2]); ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return lc

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
draw_partition(axes[0], w_eucl, owner_eucl,
               f"(1)+(2) Euclidean Voronoi  |  {n_sites} interchanges, {len(seg_xy)} segments cut\n"
               f"captured-volume CV = {cap0.std()/cap0.mean():.2f}")
draw_partition(axes[1], w_bal, owner_bal,
               f"(2) Weighted (power) Voronoi  |  balanced weights (RL slot)\n"
               f"captured-volume CV = {cap1.std()/cap1.mean():.2f}")
lc = draw_traffic(axes[2], f"(3) Traffic preprocessing: PM-peak speed (color) x daily volume (width)")
cb = fig.colorbar(lc, ax=axes[2], fraction=0.046, pad=0.02); cb.set_label("PM-peak mean speed (mph)")
fig.suptitle("PeMS District 7 (Los Angeles), 2018-10-13 (Sat) — weighted Voronoi over freeway interchanges",
             fontsize=13, y=0.99)
fig.savefig(f"{OUT}/pems_d07_voronoi.png", dpi=120, bbox_inches="tight")
print("saved figure")

# ======================================================================
# 전처리 산출물 저장
# ======================================================================
lat_s = lat0 + sites[:, 1] / 111320.0
lon_s = lon0 + sites[:, 0] / (np.cos(np.radians(lat0)) * 111320.0)
site_types = ["FF_Based"] * len(ff_sites) + ["Geometric_Fallback"] * (n_sites - len(ff_sites))

pd.DataFrame({"site_id": np.arange(n_sites), "type": site_types, "x_m": sites[:, 0], "y_m": sites[:, 1],
              "lat": lat_s, "lon": lon_s, "weight_balanced": w_bal,
              "captured_vol_eucl": cap0, "captured_vol_weighted": cap1}).to_csv(
    f"{OUT}/pems_d07_sites.csv", index=False)
pd.DataFrame({"fwy": [s[0] for s in seg_fw], "dir": [s[1] for s in seg_fw],
              "sta_from": [s[2] for s in seg_fw], "sta_to": [s[3] for s in seg_fw],
              "mid_x": seg_mid[:, 0], "mid_y": seg_mid[:, 1],
              "speed_pm_mph": seg_speed_pm, "vol_day_veh": seg_vol,
              "cell_euclidean": owner_eucl, "cell_weighted": owner_bal}).to_csv(
    f"{OUT}/pems_d07_segments.csv", index=False)
print("saved CSVs")