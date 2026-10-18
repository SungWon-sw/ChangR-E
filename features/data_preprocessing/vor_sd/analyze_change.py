"""
analyze_change.py — 두 시점(BEFORE/AFTER)의 MW 그래프-보로노이 분할이 얼마나, 어디서 바뀌었는지 분석한다.

rl_env_voronoi_mw.TrafficRLEnvMW 의 현재 인터페이스(merge_radius 로 병합된 K개 노드,
phf, rho_max, 도로 그래프 최단거리 기반 셀)를 그대로 쓰므로, train.py 가 학습한 finalW
(길이 K) 를 바로 넣을 수 있다.

BEFORE/AFTER 의 셀 경계(t 값)를 한 선분 안에서 전부 합친 '공통 sub-piece' 로 비교한다.
그러면 소유 셀이 바뀐 도로 길이가 경계 이동까지 포함해 정확히 나온다.

산출물 (--out 디렉터리)
  summary.json            핵심 수치
  region_table.csv        전체 / 바뀐 영역 / 안 바뀐 영역 / (사고 반경별) 요약
  cell_jaccard.csv        셀별 |A_i ∩ B_i| / |A_i ∪ B_i| (A=BEFORE 영역, B=AFTER 영역, 도로 길이 기준)
  cell_table.csv          셀별 유지/상실/획득 길이, Jaccard, Δa 등 상세
  transfers.csv           (이전 셀 -> 새 셀) 로 넘어간 도로 길이
  ownership_before_after.png, change_map.png, cell_jaccard.png, cell_retention.png, bp_change.png

사용 예
  python features/data_preprocessing/vor_sd/analyze_change.py \\
      --a-before finalW_before.txt --a-after finalW_after.txt \\
      --incident-lat 34.1693 --incident-lon -118.4682

  가중치 인자는 .npy / .json / 숫자가 들어 있는 텍스트(train.py 로그의 'finalW: [...]' 줄 포함),
  'zeros', 또는 'show'(rl_env_vor_show.random_search_best_a 에 박힌 SAC 결과) 를 받는다.

BP 해석 주의
  BP 는 rl_env 와 같이 (선분, 셀) 단위로 합산한 길이로 계산한 뒤 sub-piece 에 붙인다.
  BEFORE/AFTER CSV 는 교통량(vol)도 다르므로 BP 변화에는 '분할 변화'와 '교통량 변화'가
  섞여 있다. 2x2 요인분해(분할 {b,a} x 교통량 {b,a}) 로 둘을 분리해 보고한다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.data_preprocessing.vor_sd.rl_env_voronoi_mw import TrafficRLEnvMW, VF_MS
from features.data_preprocessing.vor_sd.mw_cut import (
    graph_cut_segments_fast,
    graph_cut_segments_pieces,
)
from features.data_preprocessing.vor_sd.mg_cc_batch import blocking_probability_batch

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIR = Path(__file__).resolve().parent
EPS = 1e-10


# ----------------------------------------------------------------------
# 입력: 환경 / 가중치 / 좌표
# ----------------------------------------------------------------------
def make_env(args, segments_csv):
    return TrafficRLEnvMW(
        segments_csv=str(segments_csv),
        sites_csv=str(args.sites),
        meta_txt=str(args.meta),
        rho_max=args.rho_max,
        objective=args.objective,
        phf=60.0 / args.window_min,
        merge_radius=args.merge_radius,
    )


def check_same_geometry(env_b, env_a):
    """두 환경이 같은 도로망/사이트를 쓰는지 확인한다 (다른 건 교통량뿐이어야 한다)."""
    if env_b.K != env_a.K or env_b.N != env_a.N:
        raise ValueError(f"K/N 불일치: K {env_b.K} vs {env_a.K}, N {env_b.N} vs {env_a.N}")
    for name in ("P", "Q", "site_coords", "site_node", "node_dist"):
        if not np.allclose(getattr(env_b, name), getattr(env_a, name), equal_nan=True):
            raise ValueError(f"BEFORE/AFTER 의 {name} 가 다름 — 같은 지오메트리가 아니다.")


def load_weights(spec, env):
    """a (로그 가중치, 길이 K) 를 읽는다."""
    if spec == "zeros":
        a = np.zeros(env.K)
    elif spec == "show":
        from features.data_preprocessing.vor_sd.rl_env_vor_show import random_search_best_a
        a = random_search_best_a(env)
    else:
        path = Path(spec)
        if path.suffix == ".npy":
            a = np.load(path)
        elif path.suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                obj = obj.get("finalW", obj.get("a"))
            a = np.asarray(obj, float)
        else:
            nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
                              path.read_text(encoding="utf-8"))
            a = np.array(nums, float)
    a = np.asarray(a, float).ravel()
    if len(a) != env.K:
        raise ValueError(
            f"가중치 길이 {len(a)} != env.K {env.K} ({spec}). merge_radius/사이트 파일이 "
            f"학습 때 설정과 같아야 한다 (train.py 기본: merge_radius=2500 -> K=44).")
    return a


def project_latlon(meta_txt, lat, lon):
    """env._build_road_graph 의 FF 연결자와 동일한 투영 (메타 평균 위경도 기준)."""
    meta = pd.read_csv(meta_txt, sep="\t").dropna(subset=["Latitude", "Longitude"])
    lat0, lon0 = meta.Latitude.mean(), meta.Longitude.mean()
    x = (lon - lon0) * np.cos(np.radians(lat0)) * 111320.0
    y = (lat - lat0) * 111320.0
    return np.array([x, y])


# ----------------------------------------------------------------------
# 분할 / BP
# ----------------------------------------------------------------------
def cut_state(env, a):
    """a 로 도로를 자른 결과: 가중치, (선분,셀) 유효길이 Lmat, 조각(seg, cell, t_lo, t_hi)."""
    w = env.weights(a)
    # 도달 불가 노드의 inf - inf 경고는 mw_cut 이 해당 선분을 건너뛰므로 무해하다.
    with np.errstate(invalid="ignore"):
        Lmat = graph_cut_segments_fast(
            env.P, env.Q, env.edge_u, env.edge_v, env.site_node, env.node_dist,
            w, min_len=env.min_len)
        seg, cell, t_lo, t_hi = graph_cut_segments_pieces(
            env.P, env.Q, env.edge_u, env.edge_v, env.site_node, env.node_dist, w)
    return dict(a=a - a.mean(), w=w, Lmat=Lmat, seg=seg, cell=cell, t_lo=t_lo, t_hi=t_hi)


def prob_matrix(env_traffic, Lmat):
    """교통량(lam, lanes)은 env_traffic 의 것으로 (선분, 셀) 별 차단확률을 구한다.
    min_len 미만 자투리는 Lmat 에서 0 이라 NaN 으로 남는다."""
    pm = np.full(Lmat.shape, np.nan)
    seg_i, cell_i = np.nonzero(Lmat)
    if len(seg_i) == 0:
        return pm
    L_eff = Lmat[seg_i, cell_i]
    lam_eff = env_traffic.lam[seg_i]
    if env_traffic.lam_scaling == "length":
        lam_eff = lam_eff * (L_eff / env_traffic.seg_len[seg_i])
    probs, _ = blocking_probability_batch(L_eff, VF_MS, lam_eff, env_traffic.lanes[seg_i])
    pm[seg_i, cell_i] = probs
    return pm


def common_pieces(env, st_b, st_a):
    """BEFORE/AFTER 경계를 합친 공통 sub-piece. 소유 셀은 구간 중점에서 판정한다."""
    def index(st):
        order = np.lexsort((st["t_lo"], st["seg"]))
        seg = st["seg"][order]
        starts = np.searchsorted(seg, np.arange(env.N + 1))
        return st["t_lo"][order], st["t_hi"][order], st["cell"][order], starts

    lo_b, hi_b, cell_b, st_idx_b = index(st_b)
    lo_a, hi_a, cell_a, st_idx_a = index(st_a)

    seg_o, lo_o, hi_o, cb_o, ca_o = [], [], [], [], []
    for s in range(env.N):
        b0, b1 = st_idx_b[s], st_idx_b[s + 1]
        a0, a1 = st_idx_a[s], st_idx_a[s + 1]
        if b0 == b1 or a0 == a1:        # 어느 한쪽에서 도달 불가 -> 비교 불가
            continue
        cuts = np.unique(np.r_[0.0, 1.0, lo_b[b0:b1], hi_b[b0:b1], lo_a[a0:a1], hi_a[a0:a1]])
        cuts = cuts[np.r_[True, np.diff(cuts) > EPS]]
        cuts[-1] = 1.0
        lo, hi = cuts[:-1], cuts[1:]
        mid = (lo + hi) / 2.0
        ib = np.clip(np.searchsorted(lo_b[b0:b1], mid, side="right") - 1, 0, None)
        ia = np.clip(np.searchsorted(lo_a[a0:a1], mid, side="right") - 1, 0, None)
        seg_o.append(np.full(len(lo), s))
        lo_o.append(lo)
        hi_o.append(hi)
        cb_o.append(cell_b[b0:b1][ib])
        ca_o.append(cell_a[a0:a1][ia])

    seg = np.concatenate(seg_o)
    t_lo, t_hi = np.concatenate(lo_o), np.concatenate(hi_o)
    u = env.Q[seg] - env.P[seg]
    p0 = env.P[seg] + t_lo[:, None] * u
    p1 = env.P[seg] + t_hi[:, None] * u
    return dict(seg=seg, t_lo=t_lo, t_hi=t_hi,
                before_cell=np.concatenate(cb_o), after_cell=np.concatenate(ca_o),
                p0=p0, p1=p1, length=np.linalg.norm(p1 - p0, axis=1))


def circle_inside_length(p0, p1, center, radius):
    """각 선분이 center 반경 radius 원 안에 있는 실제 길이."""
    d = p1 - p0
    q = p0 - center
    A = np.sum(d * d, axis=1)
    B = 2.0 * np.sum(q * d, axis=1)
    C = np.sum(q * q, axis=1) - radius * radius
    out = np.zeros(len(p0))
    disc = B * B - 4 * A * C
    ok = (A > 1e-12) & (disc >= 0)
    sq = np.sqrt(np.maximum(disc[ok], 0.0))
    t1 = (-B[ok] - sq) / (2 * A[ok])
    t2 = (-B[ok] + sq) / (2 * A[ok])
    frac = np.maximum(np.minimum(t2, 1.0) - np.maximum(t1, 0.0), 0.0)
    out[ok] = np.sqrt(A[ok]) * frac
    return out


# ----------------------------------------------------------------------
# 집계
# ----------------------------------------------------------------------
def wmean(x, w):
    m = np.isfinite(x) & (w > 1e-9)
    return float(np.average(x[m], weights=w[m])) if m.any() else float("nan")


def region_row(name, wt, cp, bp, valid):
    """wt: sub-piece 별 가중치(= 그 영역 안의 도로 길이)."""
    tot = float(wt.sum())
    changed = cp["changed"]
    delta = cp["delta_a"]
    row = dict(region=name, length_m=tot,
               changed_length_m=float(wt[changed].sum()),
               changed_share=float(wt[changed].sum() / tot) if tot > 0 else float("nan"))
    m = wt > 1e-9
    row["rms_delta_a"] = (float(np.sqrt(np.average(delta[m] ** 2, weights=wt[m])))
                          if m.any() else float("nan"))
    wv = np.where(valid, wt, 0.0)
    bb, ab, ba, aa = (wmean(bp[k], wv) for k in ("bb", "ab", "ba", "aa"))
    row.update(bp_before=bb, bp_after=aa, bp_total_change=aa - bb,
               bp_partition_effect=ab - bb,           # 분할만 AFTER (교통량 BEFORE)
               bp_traffic_effect=ba - bb,             # 교통량만 AFTER (분할 BEFORE)
               bp_interaction=aa - ab - ba + bb)
    return row


def cell_table(env, st_b, st_a, cp, incident):
    K = env.K
    L = cp["length"]
    cb, ca = cp["before_cell"], cp["after_cell"]
    before_len = np.bincount(cb, weights=L, minlength=K)
    after_len = np.bincount(ca, weights=L, minlength=K)
    kept = np.bincount(cb[~cp["changed"]], weights=L[~cp["changed"]], minlength=K)
    union = before_len + after_len - kept
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(union > 0, kept / union, 1.0)
    df = pd.DataFrame(dict(
        site=np.arange(K), label=env.site_labels,
        before_len_m=before_len, after_len_m=after_len, kept_len_m=kept,
        lost_len_m=before_len - kept, gained_len_m=after_len - kept,
        net_len_m=after_len - before_len, jaccard=jac,
        a_before=st_b["a"], a_after=st_a["a"], delta_a=st_a["a"] - st_b["a"],
        w_before=st_b["w"], w_after=st_a["w"]))
    if incident is not None:
        df["dist_from_incident_m"] = np.linalg.norm(env.site_coords - incident, axis=1)
    return df


# ----------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------
def distinct_cmap(n):
    import colorsys
    hues = (np.arange(n) * 0.6180339887498949) % 1.0
    return ListedColormap([colorsys.hsv_to_rgb(h, 0.65, 0.92) for h in hues])


def _frame(ax, cp, env, incident, radii, title):
    pts = np.vstack([cp["p0"], cp["p1"]])
    lo, hi = pts.min(axis=0) - 1000, pts.max(axis=0) + 1000
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("X (m)")
    ax.grid(True, linestyle="--", alpha=0.25)
    if incident is not None:
        ax.scatter(*incident, marker="*", s=420, c="black", edgecolor="white",
                   linewidth=1.6, zorder=20, label="사고")
        th = np.linspace(0, 2 * np.pi, 400)
        for r, ls in zip(radii, ["-", "--", ":", "-."]):
            ax.plot(incident[0] + r * np.cos(th), incident[1] + r * np.sin(th),
                    ls, lw=1.6, label=f"{r / 1000:g} km", zorder=15)
        ax.legend(loc="upper right", fontsize=8)


def fig_ownership(env, cp, incident, radii, path):
    lines = np.stack([cp["p0"], cp["p1"]], axis=1)
    cmap = distinct_cmap(env.K)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True, sharey=True)
    for ax, key, title in [(axes[0], "before_cell", "BEFORE — 소유 셀"),
                           (axes[1], "after_cell", "AFTER — 소유 셀")]:
        lc = LineCollection(lines, cmap=cmap, norm=Normalize(0, max(env.K - 1, 1)), linewidths=2.2)
        lc.set_array(cp[key].astype(float))
        ax.add_collection(lc)
        ax.scatter(env.site_coords[:, 0], env.site_coords[:, 1], c="red", marker="^",
                   s=60, edgecolor="black", zorder=10)
        for k, (sx, sy) in enumerate(env.site_coords):
            ax.text(sx + 150, sy + 150, env.site_labels[k], fontsize=7, zorder=11,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5))
        _frame(ax, cp, env, incident, radii, title)
    axes[0].set_ylabel("Y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_change_map(env, cp, cells, incident, radii, path):
    ch = cp["changed"]
    lines = np.stack([cp["p0"], cp["p1"]], axis=1)
    cmap = distinct_cmap(env.K)
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.add_collection(LineCollection(lines[~ch], colors="#c8c8c8", linewidths=1.0))
    lc = LineCollection(lines[ch], cmap=cmap, norm=Normalize(0, max(env.K - 1, 1)), linewidths=4.0)
    lc.set_array(cp["after_cell"][ch].astype(float))   # 새로 차지한 셀의 색
    ax.add_collection(lc)

    amax = max(float(np.abs(cells["delta_a"]).max()), 1e-9)
    sc = ax.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                    s=60 + 700 * np.abs(cells["delta_a"]) / amax, c=cells["delta_a"],
                    cmap="coolwarm", vmin=-amax, vmax=amax, edgecolor="black",
                    linewidth=0.8, zorder=10)
    top = cells.assign(act=cells["lost_len_m"] + cells["gained_len_m"]).nlargest(10, "act")
    for _, r in top.iterrows():
        sx, sy = env.site_coords[int(r["site"])]
        ax.text(sx + 200, sy + 200,
                f"#{r['label']}\n+{r['gained_len_m'] / 1000:.1f} / -{r['lost_len_m'] / 1000:.1f} km",
                fontsize=8, weight="bold", zorder=12,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.4, pad=1.5))
    share = cp["length"][ch].sum() / cp["length"].sum()
    _frame(ax, cp, env, incident, radii,
           f"소유 셀이 바뀐 도로 (색=새 소유 셀, 회색=유지)  —  {100 * share:.1f}% "
           f"({cp['length'][ch].sum() / 1000:.1f} km)")
    ax.set_ylabel("Y (m)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("사이트 Δa (점 크기 ∝ |Δa|)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_jaccard(env, cp, cells, incident, radii, path):
    """왼쪽: 유지된 도로(A∩B)를 셀의 Jaccard 색으로, 바뀐 도로는 회색. 오른쪽: 셀별 Jaccard 정렬."""
    ch = cp["changed"]
    lines = np.stack([cp["p0"], cp["p1"]], axis=1)
    jac = cells["jaccard"].values
    cmap, norm = plt.get_cmap("RdYlGn"), Normalize(0.0, 1.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax1.add_collection(LineCollection(lines[ch], colors="#d0d0d0", linewidths=1.0))
    lc = LineCollection(lines[~ch], cmap=cmap, norm=norm, linewidths=3.0)
    lc.set_array(jac[cp["before_cell"][~ch]])
    ax1.add_collection(lc)
    ax1.scatter(env.site_coords[:, 0], env.site_coords[:, 1], c=jac, cmap=cmap, norm=norm,
                s=70, edgecolor="black", linewidth=0.8, zorder=10)
    for k, (sx, sy) in enumerate(env.site_coords):
        ax1.text(sx + 200, sy + 200, f"{env.site_labels[k]}: {jac[k]:.2f}", fontsize=7, zorder=11,
                 bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.5))
    _frame(ax1, cp, env, incident, radii, "셀별 영역 유지율 J = |A∩B| / |A∪B|  (회색=바뀐 도로)")
    ax1.set_ylabel("Y (m)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, fraction=0.04,
                 pad=0.02).set_label("Jaccard")

    df = cells.sort_values("jaccard")
    y = np.arange(len(df))
    ax2.barh(y, df["jaccard"], color=cmap(norm(df["jaccard"].values)), edgecolor="black", linewidth=0.4)
    for yi, j in zip(y, df["jaccard"]):
        ax2.text(j + 0.01, yi, f"{j:.3f}", va="center", fontsize=7)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"#{l}" for l in df["label"]], fontsize=7)
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("Jaccard (1 = 변화 없음)")
    ax2.set_title("셀별 Jaccard (낮은 순)", fontsize=13, weight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_retention(cells, path, top_n=30):
    df = cells.assign(act=cells["lost_len_m"] + cells["gained_len_m"]).nlargest(top_n, "act")
    df = df.iloc[::-1]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(df) + 2.5))
    ax.barh(y, df["kept_len_m"] / 1000, color="#4c9f50", label="유지")
    ax.barh(y, -df["lost_len_m"] / 1000, color="#d9534f", label="상실")
    ax.barh(y, df["gained_len_m"] / 1000, left=df["kept_len_m"] / 1000, color="#3a7bd5", label="획득")
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.text(max(r["before_len_m"], r["after_len_m"]) / 1000 + 0.2, yi,
                f"J={r['jaccard']:.2f}  Δa={r['delta_a']:+.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"#{l}" for l in df["label"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("도로 길이 (km)")
    ax.set_title(f"셀별 도로 소유 변화 (변화량 상위 {len(df)}개 셀)", fontsize=13, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_bp(env, cp, bp, valid, regions, incident, radii, path):
    d = np.where(valid, bp["aa"] - bp["bb"], np.nan)
    lim = max(float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else 0.0, 1e-9)
    lines = np.stack([cp["p0"], cp["p1"]], axis=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw=dict(width_ratios=[1.5, 1]))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    lc = LineCollection(lines, cmap=cmap, norm=Normalize(-lim, lim), linewidths=2.5)
    lc.set_array(np.ma.masked_invalid(d))
    ax1.add_collection(lc)
    ax1.scatter(env.site_coords[:, 0], env.site_coords[:, 1], c="black", s=20, zorder=10)
    _frame(ax1, cp, env, incident, radii, "도로 조각별 BP 변화 (AFTER - BEFORE)")
    ax1.set_ylabel("Y (m)")
    fig.colorbar(lc, ax=ax1, fraction=0.04, pad=0.02).set_label("ΔBP")

    names = [r["region"] for r in regions if r["region"] in ("전체", "바뀐 영역", "안 바뀐 영역")]
    rows = [r for r in regions if r["region"] in names]
    x = np.arange(len(rows))
    w = 0.2
    for i, (key, lab, col) in enumerate([
            ("bp_partition_effect", "분할 변화", "#3a7bd5"),
            ("bp_traffic_effect", "교통량 변화", "#e8a33d"),
            ("bp_interaction", "교호작용", "#999999"),
            ("bp_total_change", "합계", "#222222")]):
        ax2.bar(x + (i - 1.5) * w, [r[key] for r in rows], w, label=lab, color=col)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['region']}\n({r['length_m'] / 1000:.0f} km)" for r in rows])
    ax2.set_ylabel("길이 가중 평균 BP 변화")
    ax2.set_title("BP 변화 요인분해 (분할 × 교통량)", fontsize=13, weight="bold")
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def parse_args(argv=None):
    od = DIR / "outputs_2008"
    p = argparse.ArgumentParser(
        description="BEFORE/AFTER MW 보로노이 분할 변화 분석",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--before-seg", type=Path, default=od / "pems_d07_segments_inc8057187_1401_before.csv")
    p.add_argument("--after-seg", type=Path, default=od / "pems_d07_segments_inc8057187_1401_after.csv")
    p.add_argument("--sites", type=Path, default=od / "pems_d07_sites.csv")
    p.add_argument("--meta", type=Path, default=DIR / "d07_text_meta_2018_10_13.txt")
    p.add_argument("--a-before", required=True, help="BEFORE 로그 가중치 (파일 | zeros | show)")
    p.add_argument("--a-after", required=True, help="AFTER 로그 가중치 (파일 | zeros | show)")
    # train.py 와 같은 env 설정이어야 K 가 맞는다
    p.add_argument("--rho-max", type=float, default=5.0)
    p.add_argument("--objective", default="within")
    p.add_argument("--window-min", type=float, default=90.0, help="phf = 60/window_min")
    p.add_argument("--merge-radius", type=float, default=2500.0)
    p.add_argument("--incident-lat", type=float, default=None)
    p.add_argument("--incident-lon", type=float, default=None)
    p.add_argument("--incident-site", default=None,
                   help="위경도 대신 병합 노드 라벨(예: 10+15) 기준으로 사고 위치 지정")
    p.add_argument("--incident-offset", default="0,-500",
                   help="--incident-site 기준 (dx,dy) [m]. 기본: 바로 아래(남쪽) 500 m")
    p.add_argument("--radii", default="1000,3000,10000", help="사고 반경 [m], 쉼표 구분")
    p.add_argument("--out", type=Path, default=None, help="기본: outputs_change/<after-seg 이름>")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = args.out or (DIR / "outputs_change" / args.after_seg.stem.replace("_after", ""))
    out.mkdir(parents=True, exist_ok=True)

    print("[1] BEFORE 환경")
    env_b = make_env(args, args.before_seg)
    print("[2] AFTER 환경")
    env_a = make_env(args, args.after_seg)
    check_same_geometry(env_b, env_a)

    a_b = load_weights(args.a_before, env_b)
    a_a = load_weights(args.a_after, env_a)

    incident, radii = None, []
    if args.incident_site is not None:
        if args.incident_site not in env_b.site_labels:
            raise ValueError(f"사이트 라벨 {args.incident_site!r} 없음. 가능: {env_b.site_labels}")
        dx, dy = (float(v) for v in args.incident_offset.split(","))
        incident = env_b.site_coords[env_b.site_labels.index(args.incident_site)] + [dx, dy]
        radii = [float(r) for r in args.radii.split(",") if r.strip()]
        print(f"사고 위치 = 사이트 #{args.incident_site} + ({dx:g}, {dy:g}) m -> "
              f"x/y = {incident[0]:.1f}, {incident[1]:.1f} m")
    elif (args.incident_lat is None) != (args.incident_lon is None):
        raise ValueError("--incident-lat 과 --incident-lon 은 같이 줘야 한다.")
    if incident is None and args.incident_lat is not None:
        incident = project_latlon(args.meta, args.incident_lat, args.incident_lon)
        radii = [float(r) for r in args.radii.split(",") if r.strip()]
        print(f"사고 위치 x/y = {incident[0]:.1f}, {incident[1]:.1f} m")

    print("[3] 분할 계산 (BEFORE/AFTER)")
    st_b, st_a = cut_state(env_b, a_b), cut_state(env_a, a_a)
    cp = common_pieces(env_b, st_b, st_a)
    cp["changed"] = cp["before_cell"] != cp["after_cell"]
    cp["delta_a"] = st_a["a"][cp["after_cell"]] - st_b["a"][cp["before_cell"]]
    print(f"공통 sub-piece {len(cp['seg']):,}개, 변경 {int(cp['changed'].sum()):,}개")

    # BP 2x2: 분할 {b,a} x 교통량 {b,a}. 키 = (분할, 교통량)
    envs = {"b": env_b, "a": env_a}
    states = {"b": st_b, "a": st_a}
    cell_key = {"b": cp["before_cell"], "a": cp["after_cell"]}
    pm = {(p, t): prob_matrix(envs[t], states[p]["Lmat"]) for p in "ba" for t in "ba"}
    bp = {p + t: pm[(p, t)][cp["seg"], cell_key[p]] for p in "ba" for t in "ba"}
    valid = np.all([np.isfinite(v) for v in bp.values()], axis=0)

    print("[4] 영역별 집계")
    L = cp["length"]
    regions = [
        region_row("전체", L, cp, bp, valid),
        region_row("바뀐 영역", np.where(cp["changed"], L, 0.0), cp, bp, valid),
        region_row("안 바뀐 영역", np.where(cp["changed"], 0.0, L), cp, bp, valid),
    ]
    for r in radii:
        inside = circle_inside_length(cp["p0"], cp["p1"], incident, r)
        regions.append(region_row(f"반경 {r / 1000:g} km", inside, cp, bp, valid))
        regions.append(region_row(f"반경 {r / 1000:g} km ∩ 바뀐",
                                  np.where(cp["changed"], inside, 0.0), cp, bp, valid))
    region_df = pd.DataFrame(regions)

    cells = cell_table(env_b, st_b, st_a, cp, incident)
    moved = (pd.DataFrame(dict(from_cell=cp["before_cell"][cp["changed"]],
                               to_cell=cp["after_cell"][cp["changed"]],
                               length_m=L[cp["changed"]]))
             .groupby(["from_cell", "to_cell"], as_index=False).sum()
             .sort_values("length_m", ascending=False))
    moved["from_label"] = [env_b.site_labels[i] for i in moved["from_cell"]]
    moved["to_label"] = [env_b.site_labels[i] for i in moved["to_cell"]]

    jac_df = pd.DataFrame(dict(
        site=cells["site"], label=cells["label"],
        A_len_m=cells["before_len_m"], B_len_m=cells["after_len_m"],
        intersection_len_m=cells["kept_len_m"],
        union_len_m=cells["before_len_m"] + cells["after_len_m"] - cells["kept_len_m"],
        jaccard=cells["jaccard"]))
    jac_df.to_csv(out / "cell_jaccard.csv", index=False, encoding="utf-8-sig")
    region_df.to_csv(out / "region_table.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(out / "cell_table.csv", index=False, encoding="utf-8-sig")
    moved.to_csv(out / "transfers.csv", index=False, encoding="utf-8-sig")

    n_valid = int(valid.sum())
    summary = dict(
        K=int(env_b.K), n_segments=int(env_b.N), n_sub_pieces=int(len(L)),
        total_length_m=float(L.sum()),
        unreachable_length_m=float(env_b.seg_len.sum() - L.sum()),   # 어느 한쪽에서 도달 불가라 비교 제외
        changed_length_m=float(L[cp["changed"]].sum()),
        changed_share=float(L[cp["changed"]].sum() / L.sum()),
        mean_cell_jaccard=float(cells["jaccard"].mean()),
        union_weighted_jaccard=float(np.average(
            cells["jaccard"], weights=cells["before_len_m"] + cells["after_len_m"]
            - cells["kept_len_m"])),
        rms_delta_a_all=float(region_df.loc[0, "rms_delta_a"]),
        bp_valid_piece_share=n_valid / len(L),
        incident_xy=None if incident is None else incident.tolist(),
        weights=dict(before=args.a_before, after=args.a_after))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    if not args.no_plots:
        print("[5] 그림 저장")
        fig_ownership(env_b, cp, incident, radii, out / "ownership_before_after.png")
        fig_change_map(env_b, cp, cells, incident, radii, out / "change_map.png")
        fig_jaccard(env_b, cp, cells, incident, radii, out / "cell_jaccard.png")
        fig_retention(cells, out / "cell_retention.png")
        fig_bp(env_b, cp, bp, valid, regions, incident, radii, out / "bp_change.png")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n" + "=" * 90)
    print("영역별 요약")
    print("=" * 90)
    print(region_df.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\n셀 변화 TOP 10 (잃거나 얻은 도로 길이 합)")
    top = cells.assign(act=cells["lost_len_m"] + cells["gained_len_m"]).nlargest(10, "act")
    print(top[["label", "lost_len_m", "gained_len_m", "jaccard", "delta_a"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print("\n가장 많이 넘어간 (이전 셀 -> 새 셀) TOP 10")
    print(moved.head(10)[["from_label", "to_label", "length_m"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    print(f"\n변경 도로 {summary['changed_length_m'] / 1000:.2f} km / "
          f"{summary['total_length_m'] / 1000:.2f} km ({100 * summary['changed_share']:.2f}%), "
          f"BP 유효 sub-piece {100 * summary['bp_valid_piece_share']:.1f}%")
    print(f"산출물: {out}")


if __name__ == "__main__":
    main()
