"""
cell_erlangB_std.py
-------------------
가중치가 주어졌을 때: 멱 보로노이 셀 생성 -> 도로(본선 구간)를 셀 경계에서 '정확히' 절단
-> 각 조각의 Erlang-B(M/G/c/c) -> 셀별 표준편차.

절단은 샘플링이 아니라 닫힌형(하부 포락선)으로 정확히 계산한다.
  구간 x(t)=A+t(B-A) 위에서 멱거리 g_j(t)=|x(t)-p_j|^2 - w_j 의 t^2 계수는 모든 j에서 |B-A|^2로 동일.
  => 승자 argmin_j g_j(t) 는 직선 l_j(t)=const_j + slope_j*t 의 하부 포락선으로 결정.
     slope_j = 2 (B-A)·(A-p_j),  const_j = |A-p_j|^2 - w_j.
  두 사이트 경계: l_j=l_k  ->  t* = (const_k-const_j)/(slope_j-slope_k)  (1차식).

핵심 가정: 검지기 구간(station->station) 안에서 flow가 거의 일정 -> 모든 조각이 같은 lambda를
물려받고 길이 L만 달라진다.
"""
import numpy as np
from weighted_voronoi import blocking_probability   # 정상 부호 엔진


def assign_power(pts, sites, w):
    """점 pts(M,2)를 멱거리 |x-p|^2 - w 최소 셀에 배정."""
    d2 = ((pts[:, None, :] - sites[None, :, :]) ** 2).sum(-1) - w[None, :]
    return d2.argmin(1)


def cut_one_segment(A, B, sites, w, eps=1e-9):
    """한 구간 A->B 를 하부 포락선으로 정확 절단.
    반환: [(cell, t0, t1), ...]  (t 구간별 승자 셀)."""
    d = B - A
    diff = A[None, :] - sites                       # (K,2) = A - p_j
    slope = 2.0 * (d[0] * diff[:, 0] + d[1] * diff[:, 1])   # 기울기_j
    const = (diff ** 2).sum(1) - w                   # 상수_j
    K = len(sites)

    t = 0.0
    cur = int(np.argmin(const + slope * t))          # t=0 승자
    pieces = []
    guard = 0
    while t < 1.0 - eps:
        guard += 1
        if guard > K + 5:                            # 안전장치(수치예외)
            pieces.append((cur, t, 1.0)); break
        # cur 를 t 이후 가장 먼저 추월하는 직선 찾기
        denom = slope - slope[cur]                   # slope_j - slope_cur
        tc = np.full(K, np.inf)
        valid = denom < -eps                         # j가 cur보다 아래로 내려가려면 기울기 더 작아야
        tc[valid] = (const[cur] - const[valid]) / denom[valid]  # 교차점 t*
        tc[tc <= t + eps] = np.inf                   # 현재 이후만
        tc[cur] = np.inf
        j = int(np.argmin(tc))
        t_next = tc[j]
        if not np.isfinite(t_next) or t_next >= 1.0:
            pieces.append((cur, t, 1.0)); break
        pieces.append((cur, t, float(t_next)))
        t = float(t_next); cur = j
    return pieces


def cut_roads(sites, weights, seg_A, seg_B):
    """모든 구간을 정확 절단. 반환: 조각 dict 리스트 {seg, cell, length_m, t0, t1}."""
    sites = np.asarray(sites, float); w = np.asarray(weights, float)
    A = np.asarray(seg_A, float); B = np.asarray(seg_B, float)
    seg_len = np.hypot(*(B - A).T)
    out = []
    for i in range(len(A)):
        for (cell, t0, t1) in cut_one_segment(A[i], B[i], sites, w):
            out.append(dict(seg=i, cell=int(cell),
                            length_m=float((t1 - t0) * seg_len[i]),
                            t0=float(t0), t1=float(t1)))
    return out


def cell_erlangB_std(sites, weights, seg_A, seg_B, seg_flow, seg_lanes,
                     Vf_ms=65 * 0.44704, kj=0.15, phf=0.15, min_len_m=1.0,
                     ddof=0):
    """가중치 -> 정확 절단 -> 조각 Erlang-B -> 셀별 std.
    ddof=0 모집단 표준편차(기본), ddof=1 표본 표준편차.
    반환: (cell_std[K], pieces(각 조각에 Pc 추가))."""
    K = len(sites)
    pieces = cut_roads(sites, weights, seg_A, seg_B)
    for p in pieces:
        i = p["seg"]
        if p["length_m"] < min_len_m:
            p["Pc"] = np.nan; continue
        lam = seg_flow[i] * phf / 3600.0                   # 첨두 도착률 [veh/s]
        lanes = max(1, int(round(seg_lanes[i])))
        p["Pc"] = blocking_probability(p["length_m"], Vf_ms, lam,
                                       lanes=lanes, jam_density=kj)[0]
    cell_std = np.full(K, np.nan)
    for k in range(K):
        vals = np.array([p["Pc"] for p in pieces
                         if p["cell"] == k and np.isfinite(p["Pc"])])
        if len(vals) > ddof:                          # ddof=1이면 조각 2개 이상 필요
            cell_std[k] = vals.std(ddof=ddof)
    return cell_std, pieces


def report_cell_std(sites, weights, seg_A, seg_B, seg_flow, seg_lanes,
                    site_ids=None, sort_by_std=True, to_csv=None,
                    verbose=True, **kw):
    """셀별 Erlang-B 표준편차를 표로 정리·출력한다.
    반환: (DataFrame[cell, n_pieces, mean_ErlangB, std_ErlangB], cell_std, pieces)."""
    import pandas as pd
    cell_std, pieces = cell_erlangB_std(sites, weights, seg_A, seg_B,
                                        seg_flow, seg_lanes, **kw)
    K = len(sites)
    rows = []
    for k in range(K):
        vals = [p["Pc"] for p in pieces
                if p["cell"] == k and np.isfinite(p["Pc"])]
        rows.append({
            "cell": int(site_ids[k]) if site_ids is not None else k,
            "n_pieces": len(vals),
            "mean_ErlangB": float(np.mean(vals)) if vals else np.nan,
            "std_ErlangB": cell_std[k],
        })
    df = pd.DataFrame(rows)
    if sort_by_std:
        df = df.sort_values("std_ErlangB", ascending=False,
                            na_position="last").reset_index(drop=True)
    if to_csv:
        df.to_csv(to_csv, index=False)

    finite = cell_std[np.isfinite(cell_std)]
    if verbose:
        print(f"셀 {K}개 (조각 있는 셀 {len(finite)}개) | 총 조각 {len(pieces)}개")
        if len(finite):
            print(f"셀별 std  ->  평균 {finite.mean():.4f} | "
                  f"RMS {np.sqrt((finite**2).mean()):.4f} | "
                  f"최대 {finite.max():.4f} | 최소 {finite.min():.4f}")
        print(df.to_string(index=False,
                           formatters={"mean_ErlangB": "{:.4f}".format,
                                       "std_ErlangB": "{:.4f}".format}))
    return df, cell_std, pieces
