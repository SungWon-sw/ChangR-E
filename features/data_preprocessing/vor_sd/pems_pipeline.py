"""
pems_pipeline.py — PeMS D07(2018-10-13) 실데이터로 3개 작업 수행:
  (1) 분기점(FF 인터체인지) + 본선 기하학적 교차 보완 -> 보로노이 사이트로
  (2) 가중 보로노이 분할 및 도로 선분 클리핑(Line Clipping) 적용하여 수요 할당
  (3) 5분 데이터 전처리 -> 본선 구간별 교통량(volume)·속도(speed)

사용법
------
# 종일 (기존 동작 그대로). SA.py / train.py 가 읽는 파일명 그대로 생성.
python features/data_preprocessing/vor_sd/pems_pipeline.py

# 특정 시각 전후로 인접한 두 시간창을 만들어 교통량을 창별로 집계.
# 지오메트리(사이트·선분)는 두 창이 공유하고 vol_day_veh / speed_pm_mph 만 창값.
python features/data_preprocessing/vor_sd/pems_pipeline.py --pivot 08:41 --span 90
  -> outputs/pems_d07_segments_0841_before.csv   [07:11, 08:41)
     outputs/pems_d07_segments_0841_after.csv    [08:41, 10:11)
     (pems_d07_sites.csv 는 창과 무관 -> 없을 때만 새로 씀)

시간창 모드에서 vol_day_veh 는 "그 창 동안의 실제 통과 대수 합"(정직한 카운트)이다.
TrafficRLEnvMW 는 lam = vol_day_veh * PHF / 3600 으로 도착률을 만드므로,
창 실제 도착률 lam = vol / (span*60) 을 얻으려면 env 에 phf = 60/span 을 넘겨야 한다
(스크립트가 실행 끝에 정확한 값을 출력한다). 그냥 기본 phf=0.15 로 두면
vol 을 "일 총량"으로 해석하게 된다.
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_META = os.path.join(HERE, "d07_text_meta_2018_10_13.txt")
DEFAULT_FIVE = os.path.join(HERE, "d07_text_station_5min_2008_10_07.txt")
DEFAULT_OUT = os.path.join(HERE, "outputs")

DAY_MIN = 24 * 60
BIN_MIN = 5  # PeMS station_5min 집계 간격


# ======================================================================
# CLI
# ======================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PeMS D07 전처리 파이프라인 (선택적 시간창 분할 지원)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--meta", default=DEFAULT_META,
                   help="스테이션 메타데이터 txt (탭 구분)")
    p.add_argument("--five", default=DEFAULT_FIVE,
                   help="5분 station CSV (헤더 없음). 미지정 시 알려진 위치를 순서대로 탐색")
    p.add_argument("--out", default=DEFAULT_OUT, help="산출물 디렉터리")
    p.add_argument("--pivot", default=None, metavar="HH:MM",
                   help="이 시각을 경계로 앞/뒤 두 시간창을 만든다. 미지정 시 종일 1개.")
    p.add_argument("--span", type=int, default=90, metavar="MIN",
                   help="pivot 기준 한쪽 창의 길이(분). 창 A=[pivot-span, pivot), B=[pivot, pivot+span)")
    p.add_argument("--tag", default=None,
                   help="출력 파일명 태그 (기본: pivot 에서 자동, 예 08:41 -> 0841)")
    bal = p.add_mutually_exclusive_group()
    bal.add_argument("--balance", dest="balance", action="store_true",
                     help="Task2(균형화 가중치 최적화) 수행. 기본: 종일 모드에서만 on")
    bal.add_argument("--no-balance", dest="balance", action="store_false",
                     help="Task2 생략 (시간창 여러 개를 빠르게 뽑을 때)")
    p.set_defaults(balance=None)
    return p.parse_args(argv)

def hhmm_to_min(s):
    t = datetime.strptime(s.strip(), "%H:%M")
    return t.hour * 60 + t.minute


# ======================================================================
# Task 1 — 메타데이터 -> 분기점 사이트 + 본선 도로 (기하 교차점 보완 포함)
# ======================================================================
def build_geometry(meta_path):
    m = pd.read_csv(meta_path, sep="\t").dropna(subset=["Latitude", "Longitude"])
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
                p1 = c1[min_idx]
                p2 = c2[idxs[min_idx]]
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
                seg_xy.append((a, b))
                seg_mid.append((a + b) / 2)
                seg_fw.append((fw, d, grp.ID.values[k], grp.ID.values[k + 1]))
    seg_mid = np.array(seg_mid)
    print(f"[Task1] 분기점 사이트 {n_sites}개 | 본선 구간 {len(seg_xy)}개 | FF기반 {len(ff_sites)}개")
    return dict(sites=sites, n_sites=n_sites, n_ff=len(ff_sites),
                seg_xy=seg_xy, seg_mid=seg_mid, seg_fw=seg_fw,
                lat0=lat0, lon0=lon0, ml=ml)


# ======================================================================
# Task 3 — 5분 데이터 -> 스테이션별 교통량·속도 (시간창 파라미터화)
# ======================================================================
def load_five(five_path):
    df = pd.read_csv(five_path, header=None, usecols=[0, 1, 5, 8, 9, 10, 11],
                     names=["ts", "station", "ltype", "pct_obs", "flow", "occ", "speed"])
    d = df[df.ltype == "ML"].copy()
    dt = pd.to_datetime(d.ts, format="%m/%d/%Y %H:%M:%S")
    d["tod"] = dt.dt.hour * 60 + dt.dt.minute          # 자정 이후 분
    n_days = dt.dt.normalize().nunique()
    if n_days > 1:
        print(f"[경고] 5분 파일에 {n_days}일치 데이터가 섞여 있음 — 시간창 필터는 "
              f"time-of-day 기준이라 여러 날을 합칩니다.")
    return d


def _fw_speed(grp):
    """flow 를 가중치로 한 평균 속도 (NaN 안전)."""
    sp, fl = grp.speed.values, grp.flow.values
    ok = ~np.isnan(sp)
    if ok.sum() == 0:
        return np.nan
    w = fl[ok]
    return np.average(sp[ok], weights=w) if np.nansum(w) > 0 else np.nanmean(sp[ok])


def station_traffic(d, lo_min, hi_min):
    """[lo_min, hi_min) 분 구간의 스테이션별 (통과대수 합 vol, flow-가중 속도 speed).

    반환: (DataFrame[station, vol, speed], 데이터가 있었던 스테이션 수).
    """
    sub = d[(d.tod >= lo_min) & (d.tod < hi_min)]
    if len(sub) == 0:
        return pd.DataFrame(columns=["station", "vol", "speed"]), 0
    g = sub.groupby("station")
    vol = g.flow.apply(lambda s: np.nansum(s.values))
    spd = g[["speed", "flow"]].apply(_fw_speed)      # 열 선택 -> groupby 열 경고 회피
    res = pd.DataFrame({"vol": vol, "speed": spd}).reset_index()
    return res, int(sub.station.nunique())


def seg_traffic(seg_fw, st):
    """구간별 (속도, 교통량) = 양 끝 스테이션 값의 평균 (원 파이프라인과 동일)."""
    import warnings
    id2vol = dict(zip(st.station, st.vol))
    id2spd = dict(zip(st.station, st.speed))
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)   # 양 끝 다 NaN -> nanmean 빈 슬라이스
        seg_spd = np.array([np.nanmean([id2spd.get(a, np.nan), id2spd.get(b, np.nan)])
                            for (_, _, a, b) in seg_fw])
        seg_vol = np.array([np.nanmean([id2vol.get(a, np.nan), id2vol.get(b, np.nan)])
                            for (_, _, a, b) in seg_fw])
    return seg_spd, seg_vol


# ======================================================================
# Task 2 — 선분 클리핑을 통한 셀 배정 + 균형화 가중치 (선택)
# ======================================================================
def assign_midpoints(points, sites, weights):
    """중점 기준 멱 거리 argmin (플롯/참조 컬럼용)."""
    d2 = ((points[:, None, :] - sites[None, :, :]) ** 2).sum(-1) - weights[None, :]
    return d2.argmin(1)


def run_balance(sites, seg_xy, seg_vol):
    """원 파이프라인 Task2: 흡수 교통량을 균등화하는 결정론적 가중치 탐색.
    무거우므로 (power cell 을 120회 재계산) 시간창 모드에서는 기본 생략된다.
    """
    try:
        sys.path.insert(0, "/home/claude")
        sys.path.insert(0, HERE)
        sys.path.insert(0, os.path.join(HERE, "trashes"))
        from weighted_voronoi import all_power_cells
    except ImportError:
        from trashes.weighted_voronoi import all_power_cells
    from shapely.geometry import LineString, Polygon
    from shapely.strtree import STRtree

    n_sites = len(sites)
    pad = 4000
    all_x = np.array([p[0] for seg in seg_xy for p in seg])
    all_y = np.array([p[1] for seg in seg_xy for p in seg])
    bbox = (all_x.min() - pad, all_y.min() - pad, all_x.max() + pad, all_y.max() + pad)

    nbr = cKDTree(sites).query(sites, k=2)[0][:, 1]
    spacing2 = np.median(nbr) ** 2
    mass = np.nan_to_num(seg_vol)

    seg_lines = [LineString([a, b]) for a, b in seg_xy]
    seg_lengths = np.array([line.length for line in seg_lines])
    tree = STRtree(seg_lines)

    def captured(weights):
        cells = all_power_cells(sites, weights, bbox)
        cap = np.zeros(n_sites)
        for i, poly_coords in enumerate(cells):
            if len(poly_coords) < 3:
                continue
            poly = Polygon(poly_coords)
            for idx in tree.query(poly):
                line = seg_lines[idx]
                if poly.intersects(line):
                    inter = poly.intersection(line)
                    cap[i] += mass[idx] * (inter.length / seg_lengths[idx])
        return cap

    w_eucl = np.zeros(n_sites)
    cap0 = captured(w_eucl)

    def balance_weights(iters=120, lr=0.7):
        w = np.zeros(n_sites)
        target = mass.sum() / n_sites
        for _ in range(iters):
            cap = captured(w)
            w += lr * spacing2 * (target - cap) / target
            w = np.clip(w - w.mean(), -8 * spacing2, 8 * spacing2)
        return w

    w_bal = balance_weights()
    cap1 = captured(w_bal)
    cv0 = cap0.std() / cap0.mean()
    cv1 = cap1.std() / cap1.mean()
    print(f"[Task2] 분할 완료 | 흡수교통량 CV: 유클리드 {cv0:.3f} -> 균형화 {cv1:.3f}")
    return w_bal, cap0, cap1


# ======================================================================
# 산출물 저장
# ======================================================================
def write_sites(path, geo, w_bal, cap0, cap1):
    sites = geo["sites"]
    n_sites, n_ff = geo["n_sites"], geo["n_ff"]
    lat0, lon0 = geo["lat0"], geo["lon0"]
    lat_s = lat0 + sites[:, 1] / 111320.0
    lon_s = lon0 + sites[:, 0] / (np.cos(np.radians(lat0)) * 111320.0)
    site_types = ["FF_Based"] * n_ff + ["Geometric_Fallback"] * (n_sites - n_ff)
    pd.DataFrame({
        "site_id": np.arange(n_sites), "type": site_types,
        "x_m": sites[:, 0], "y_m": sites[:, 1], "lat": lat_s, "lon": lon_s,
        "weight_balanced": w_bal,
        "captured_vol_eucl": cap0, "captured_vol_weighted": cap1,
    }).to_csv(path, index=False)
    print(f"  saved {path}")


def write_segments(path, geo, seg_spd, seg_vol, owner_eucl, owner_bal):
    seg_fw, seg_mid, seg_xy = geo["seg_fw"], geo["seg_mid"], geo["seg_xy"]
    pd.DataFrame({
        "fwy": [s[0] for s in seg_fw], "dir": [s[1] for s in seg_fw],
        "sta_from": [s[2] for s in seg_fw], "sta_to": [s[3] for s in seg_fw],
        "mid_x": seg_mid[:, 0], "mid_y": seg_mid[:, 1],
        "x1": [s[0][0] for s in seg_xy], "y1": [s[0][1] for s in seg_xy],
        "x2": [s[1][0] for s in seg_xy], "y2": [s[1][1] for s in seg_xy],
        "speed_pm_mph": seg_spd, "vol_day_veh": seg_vol,
        "cell_euclidean_mid": owner_eucl, "cell_weighted_mid": owner_bal,
    }).to_csv(path, index=False)
    n_nan = int(np.isnan(seg_vol).sum())
    print(f"  saved {path}  (구간 {len(seg_fw)} | vol NaN {n_nan} | vol 합 {np.nansum(seg_vol):,.0f})")


# ======================================================================
# main
# ======================================================================
def main(argv=None):
    args = parse_args(argv)
    five_path = args.five
    os.makedirs(args.out, exist_ok=True)
    do_balance = args.balance if args.balance is not None else (args.pivot is None)

    print(f"[입력] meta = {args.meta}")
    print(f"[입력] five = {five_path}")

    geo = build_geometry(args.meta)
    d = load_five(five_path)
    sites = geo["sites"]
    n_sites = geo["n_sites"]
    owner_eucl = assign_midpoints(geo["seg_mid"], sites, np.zeros(n_sites))

    # 종일 스테이션 교통량 — Task2 mass 와 종일 모드에서 공용
    vd_all, _ = station_traffic(d, 0, DAY_MIN)

    # ---- Task 2 (선택) ----
    if do_balance:
        _, seg_vol_all = seg_traffic(geo["seg_fw"], vd_all)
        w_bal, cap0, cap1 = run_balance(sites, geo["seg_xy"], seg_vol_all)
        owner_bal = assign_midpoints(geo["seg_mid"], sites, w_bal)
    else:
        w_bal = np.zeros(n_sites)
        cap0 = cap1 = np.full(n_sites, np.nan)
        owner_bal = owner_eucl
        print("[Task2] 생략 (--no-balance / 시간창 기본)")

    # ---- Task 3 + 저장 ----
    if args.pivot is None:
        # 종일: 교통량=종일 합, 속도=15~18시 flow-가중 (원 파이프라인과 동일)
        sp, _ = station_traffic(d, 15 * 60, 19 * 60)
        st = vd_all[["station", "vol"]].merge(sp[["station", "speed"]], on="station", how="outer")
        seg_spd, seg_vol = seg_traffic(geo["seg_fw"], st)
        write_sites(os.path.join(args.out, "pems_d07_sites.csv"), geo, w_bal, cap0, cap1)
        write_segments(os.path.join(args.out, "pems_d07_segments.csv"),
                       geo, seg_spd, seg_vol, owner_eucl, owner_bal)
        print("saved CSVs")
        return

    # ---- 시간창 모드 ----
    pv = hhmm_to_min(args.pivot)
    span = args.span
    tag = args.tag or args.pivot.replace(":", "")
    lo_b, hi_b = pv - span, pv
    lo_a, hi_a = pv, pv + span
    if lo_b < 0:
        print(f"[경고] before 창 시작 {lo_b}분 < 00:00 -> 클램프")
        lo_b = 0
    if hi_a > DAY_MIN:
        print(f"[경고] after 창 끝 {hi_a}분 > 24:00 -> 클램프")
        hi_a = DAY_MIN

    def fmt(mn):
        return f"{mn // 60:02d}:{mn % 60:02d}"

    sites_path = os.path.join(args.out, "pems_d07_sites.csv")
    if os.path.exists(sites_path):
        print(f"[sites] {sites_path} 이미 있음 -> 유지 (창과 무관)")
    else:
        write_sites(sites_path, geo, w_bal, cap0, cap1)

    print(f"[window] pivot={args.pivot} span={span}min  "
          f"bin={BIN_MIN}min -> 창당 최대 {span // BIN_MIN} bin")
    for label, lo, hi in [("before", lo_b, hi_b), ("after", lo_a, hi_a)]:
        st, n_st = station_traffic(d, lo, hi)
        seg_spd, seg_vol = seg_traffic(geo["seg_fw"], st)
        out = os.path.join(args.out, f"pems_d07_segments_{tag}_{label}.csv")
        print(f"  [{label}] [{fmt(lo)}, {fmt(hi)})  데이터 있는 스테이션 {n_st}개")
        write_segments(out, geo, seg_spd, seg_vol, owner_eucl, owner_bal)

    span_s = span * 60
    phf = 60.0 / span
    print("\n[env] 시간창 vol_day_veh 는 '그 창 동안의 통과대수 합'입니다.")
    print(f"      창 실제 도착률을 쓰려면 TrafficRLEnvMW(..., phf={phf:.4f}) 로 실행")
    print(f"        -> lam = vol / {span_s}s  (veh/s)")
    print("      기본 phf=0.15 로 두면 vol 을 '일 총량'으로 해석합니다.")


if __name__ == "__main__":
    main()
