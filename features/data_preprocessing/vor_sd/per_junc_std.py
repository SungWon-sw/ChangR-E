import numpy as np
import pandas as pd
from math import lgamma

VALUE = "blocking"            # "blocking" | "delay" | "flow"

# ---- 보정 상수 ----
KJ    = 0.15                  # 차선당 정체밀도 (veh/m)
BETA  = 20.0                  # 속도-밀도 곡선 파라미터
GAMMA = 1.0
VF_MPH_DEFAULT = 65.0         # 자유흐름 속도 기본값(mph)
MPH2MS = 0.44704
MI2M   = 1609.34
R_EARTH = 6371000.0           # 지구 반지름 (m)

# ---------------------------------------------------------------------------
def blocking_probability(L, lanes, Vf, lam, kj=KJ, beta=BETA, gamma=GAMMA):
    """M/G/c/c 상태의존 차단확률(Cheah-Smith). 수치적 안정성을 위해 로그 영역 연산."""
    c = max(1, int(round(kj * lanes * L)))
    if lam <= 0 or L <= 0 or Vf <= 0:
        return 0.0
    a1 = L / Vf
    logr = np.log(lam * a1)
    log_t = np.full(c + 1, -np.inf)
    log_t[0] = 0.0
    cum = 0.0
    for n in range(1, c + 1):
        if n >= 2:
            cum += -(((n - 1) / beta) ** gamma)
        log_t[n] = n * logr - lgamma(n + 1) + cum
    m = np.max(log_t)
    w = np.exp(log_t - m)
    return float(w[c] / w.sum())

def _col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None

def load_meta(path):
    m = pd.read_csv(path, sep="\t")
    m = m.rename(columns={c: c.strip() for c in m.columns})
    return m.set_index(_col(m, "ID") or m.columns[0])

# ---------------------------------------------------------------------------
# [보완 로직 1] 메타데이터 기하 분석을 통한 숨겨진 교차로(인터체인지) 자동 탐지
# ---------------------------------------------------------------------------
def discover_and_fill_sites(meta_df):
    """
    FF 검지기가 누락되었으나 실제 본선(ML)끼리 만나 병목을 유발하는 
    기하학적 교차점을 찾아내어 기존 사이트 목록을 보완합니다.
    """
    # 1. 기존 FF(Freeway-Freeway) 검지기 기반 사이트 추출 및 클러스터링 (기본 54개)
    ff_stations = meta_df[meta_df['Type'] == 'FF']
    ff_sites = []
    
    if len(ff_stations) > 0:
        # 간단한 800m 거리 기반 클러스터링으로 대표 노드 산출
        coords = ff_stations[['Latitude', 'Longitude']].values
        used = np.zeros(len(coords), dtype=bool)
        for i, (lat, lon) in enumerate(coords):
            if used[i]: continue
            # 현재 점 기준 근접한 FF 검지기들을 묶음
            dists = np.hypot((coords[:,0]-lat)*111000, (coords[:,1]-lon)*111000*np.cos(np.radians(lat)))
            close_idx = dists < 800
            used[close_idx] = True
            ff_sites.append({
                'id': int(ff_stations.iloc[i].name),
                'lat': float(ff_stations.iloc[close_idx]['Latitude'].mean()),
                'lon': float(ff_stations.iloc[close_idx]['Longitude'].mean()),
                'source': 'FF_Detector'
            })
    
    # 2. 본선(ML) 기하 분석: 다른 고속도로인데 공간상 거리가 500m 이내로 교차하는 지점 추적
    ml_stations = meta_df[meta_df['Type'] == 'ML']
    ml_by_fwy = [g for _, g in ml_stations.groupby('Fwy')]
    
    hidden_crossings = []
    for i in range(len(ml_by_fwy)):
        for j in range(i + 1, len(ml_by_fwy)):
            fwy1, fwy2 = ml_by_fwy[i], ml_by_fwy[j]
            # 두 고속도로의 모든 검지기 쌍 간 거리 계산
            c1 = fwy1[['Latitude', 'Longitude']].values
            c2 = fwy2[['Latitude', 'Longitude']].values
            
            for p1 in c1:
                # 대략적인 평면 거리 계산
                dists = np.hypot((c2[:,0]-p1[0])*111000, (c2[:,1]-p1[1])*111000*np.cos(np.radians(p1[0])))
                min_idx = np.argmin(dists)
                
                if dists[min_idx] < 500: # 500m 이내로 교차하는 진짜 인프라 접점 발견
                    mid_lat = (p1[0] + c2[min_idx, 0]) / 2.0
                    mid_lon = (p1[1] + c2[min_idx, 1]) / 2.0
                    hidden_crossings.append((mid_lat, mid_lon))
                    
    # 3. 발견된 기하 교차점 중 기존 FF 사이트와 1.5km 이상 떨어진 '실제 공백 지점' 필터링 및 병합
    final_sites = list(ff_sites)
    new_id_counter = 9901 # 기하 보완 노드는 9901번부터 부여
    
    for lat, lon in hidden_crossings:
        # 기존 사이트들과의 최소 거리 계산
        if final_sites:
            existing_coords = np.array([[s['lat'], s['lon']] for s in final_sites])
            dists = np.hypot((existing_coords[:,0]-lat)*111000, (existing_coords[:,1]-lon)*111000*np.cos(np.radians(lat)))
            if np.min(dists) < 1500: 
                continue # 이미 기존 FF 사이트 권역에 포함되어 있다면 패스
                
        # 새로 발견된 기하 교차점들끼리 중복 제거
        is_dup = False
        for s in final_sites:
            if s['source'] == 'Geometric_Fallback':
                d = np.hypot((s['lat']-lat)*111000, (s['lon']-lon)*111000*np.cos(np.radians(lat)))
                if d < 1000: is_dup = True; break
                
        if not is_dup:
            final_sites.append({
                'id': new_id_counter,
                'lat': lat,
                'lon': lon,
                'source': 'Geometric_Fallback'
            })
            new_id_counter += 1
            
    return pd.DataFrame(final_sites).set_index('id')

# ---------------------------------------------------------------------------
def segment_value(seg, meta):
    """한 구간(segment row)의 값을 VALUE 옵션에 따라 연산."""
    if VALUE == "delay":
        return float(seg[_col_seg_delay])
    if VALUE == "flow":
        return float(seg[_col_seg_flow])
        
    # --- M/G/c/c 차단확률 연산 파트 ---
    if _col_seg_len:
        L = float(seg[_col_seg_len]) * (MI2M if "mi" in _col_seg_len.lower() else 1.0)
    else:
        L = _seg_len_from_stations(seg, meta)
        
    ids = [seg[c] for c in (_col_seg_a, _col_seg_b) if c]
    ln = [meta.loc[i, _col_meta_lanes] for i in ids if meta is not None and i in meta.index and _col_meta_lanes]
    lanes = float(np.mean(ln)) if ln else 3.0
    
    Vf = VF_MPH_DEFAULT * MPH2MS
    flow = float(seg[_col_seg_flow])
    lam = flow / (300.0 if "peak" in (_col_seg_flow or "").lower() else 86400.0)
    
    return blocking_probability(L, lanes, Vf, lam)

def _seg_len_from_stations(seg, meta):
    a, b = seg[_col_seg_a], seg[_col_seg_b]
    if meta is not None and a in meta.index and b in meta.index and _col_meta_lat:
        la1, lo1 = meta.loc[a, _col_meta_lat], meta.loc[a, _col_meta_lon]
        la2, lo2 = meta.loc[b, _col_meta_lat], meta.loc[b, _col_meta_lon]
        dx = np.radians(lo2 - lo1) * R_EARTH * np.cos(np.radians((la1 + la2) / 2))
        dy = np.radians(la2 - la1) * R_EARTH
        return float(np.hypot(dx, dy))
    return 800.0

# ---------------------------------------------------------------------------
def per_junction_std(segments_csv, meta_txt=None):
    seg = pd.read_csv(segments_csv)

    # 컬럼 이름 자동 매핑 루틴
    global _col_seg_a, _col_seg_b, _col_seg_flow, _col_seg_delay, _col_seg_len
    global _col_meta_lanes, _col_meta_lat, _col_meta_lon
    _col_seg_a    = _col(seg, "station_a", "sta_a", "from", "a")
    _col_seg_b    = _col(seg, "station_b", "sta_b", "to", "b")
    _col_seg_flow = _col(seg, "flow_peak", "peak_flow", "flow", "vol", "교통량", "daily_flow")
    _col_seg_delay= _col(seg, "delay", "지연", "delay_vehhr", "veh_hr")
    _col_seg_len  = _col(seg, "len_mi", "length_mi", "len_m", "length")
    
    meta = load_meta(meta_txt) if meta_txt else None
    if meta is not None:
        _col_meta_lanes = _col(meta.reset_index(), "Lanes")
        _col_meta_lat   = _col(meta.reset_index(), "Latitude", "Lat")
        _col_meta_lon   = _col(meta.reset_index(), "Longitude", "Lon")
    else:
        _col_meta_lanes = _col_meta_lat = _col_meta_lon = None

    # 도로별 지표 특성값 선계산
    seg["__val"] = seg.apply(lambda r: segment_value(r, meta), axis=1)

    # ---------------------------------------------------------------------------
    # [보완 로직 2] 보완된 사이트 목록 기준으로 도로 구간 동적 재배정 (Re-assignment)
    # ---------------------------------------------------------------------------
    if meta is not None:
        print("[시스템] 기하학적 교차점 분석 및 사이트 보완 프로토콜 가동...")
        sites_df = discover_and_fill_sites(meta)
        print(f"[알림] 최종 확정된 분기점 수: {len(sites_df)}개 (기하학적 보완 노드 포함)")
        
        # 각 도로의 중점 좌표 계산
        seg_coords = []
        for _, r in seg.iterrows():
            a, b = r[_col_seg_a], r[_col_seg_b]
            if a in meta.index and b in meta.index:
                seg_coords.append([
                    (meta.loc[a, _col_meta_lat] + meta.loc[b, _col_meta_lat]) / 2.0,
                    (meta.loc[a, _col_meta_lon] + meta.loc[b, _col_meta_lon]) / 2.0
                ])
            else:
                seg_coords.append([34.05, -118.25]) # 결측치 발생 시 LA 중심부 fallback
        seg_coords = np.array(seg_coords)
        
        # 보완된 57개 사이트 중 가장 가까운 사이트로 셀 재할당 (유클리드/거리 기반 기본 분할형)
        site_lat_lons = sites_df[['lat', 'lon']].values
        assigned_cells = []
        for p in seg_coords:
            dists = np.hypot((site_lat_lons[:,0]-p[0])*111000, (site_lat_lons[:,1]-p[1])*111000*np.cos(np.radians(p[0])))
            assigned_cells.append(sites_df.index[np.argmin(dists)])
        
        seg["corrected_cell"] = assigned_cells
        target_cell_col = "corrected_cell"
    else:
        # 메타데이터가 없을 경우 기존 인코딩된 셀 컬럼 사용
        target_cell_col = _col(seg, "cell_weighted", "weighted_cell", "cell", "assigned_cell", "site")

    # ---------------------------------------------------------------------------
    # 최종 그룹핑 및 분기점별 불균일 통계 추출
    # ---------------------------------------------------------------------------
    rows = []
    for site, g in seg.groupby(target_cell_col):
        vals = g["__val"].dropna().values
        if len(vals) == 0: continue
        rows.append((int(site), float(np.std(vals)), float(np.mean(vals)), len(vals)))
        
    rows.sort(key=lambda t: t[1], reverse=True) # 표준편차 큰 순서대로 정렬

    allv = seg["__val"].dropna().values
    cv = float(np.std(allv) / np.mean(allv)) if np.mean(allv) else float("nan")
    return rows, cv

if __name__ == "__main__":
    # 실데이터 파일 경로 설정
    rows, cv = per_junction_std(
        "pems_d07_segments.csv", 
        meta_txt="d07_text_meta_2018_10_13.txt",
    )
    print(f"\n[결과] VALUE Mode = {VALUE}  |  네트워크 전체 최종 CV = {cv:.4f}\n")
    print(f"{'Junction(Site)':>16} {'Std(표준편차)':>14} {'Mean(평균)':>12} {'Roads(N)':>8}")
    print("-" * 55)
    for site, sd, mn, n in rows[:15]: # 상위 15개 불균일 요충지 출력
        flag = " (보완노드)" if site >= 9901 else ""
        print(f"{str(site)+flag:>16} {sd:>14.4f} {mn:>12.4f} {n:>8}")