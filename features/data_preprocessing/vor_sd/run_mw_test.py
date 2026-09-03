# 작동하는지 테스트하는 스크립트임


"""
실제 데이터를 TrafficRLEnvMW 에 넣어보는 진단 스크립트.

사용법:
    python run_mw_test.py --segments outputs/pems_d07_segments.csv --sites outputs/pems_d07_sites.csv --meta d07_text_meta_2018_10_13.txt

확인해 주는 것
    1) 컬럼 스키마 / 좌표 단위가 맞는지
    2) a=0 (= 일반 Voronoi) 기준선
    3) 감도 진단: 가중치에 반응하는 데이터가 몇 %인가   <- 가장 중요
    4) rho_max x objective 랜덤서치로 개선 여지가 있는지
    5) 도로망을 셀 색으로 칠한 PNG
"""

import argparse
import sys

import numpy as np
import pandas as pd

REQUIRED = {
    "sites": ["x_m", "y_m"],
    "segments": ["sta_from", "vol_day_veh", "x1", "y1", "x2", "y2"],
    "meta": ["ID", "Lanes"],
}


def preflight(segments_csv, sites_csv, meta_txt):
    print("=" * 68)
    print("[1] 스키마 점검")
    print("=" * 68)
    ok = True
    sites = pd.read_csv(sites_csv)
    seg = pd.read_csv(segments_csv)
    meta = pd.read_csv(meta_txt, sep="\t")

    for name, df, cols in (("sites", sites, REQUIRED["sites"]),
                           ("segments", seg, REQUIRED["segments"]),
                           ("meta", meta, REQUIRED["meta"])):
        miss = [c for c in cols if c not in df.columns]
        mark = "OK " if not miss else "!! "
        print(f"  {mark}{name:<9} rows={len(df):<7} 필요컬럼 {cols}")
        if miss:
            ok = False
            print(f"       누락: {miss}")
            print(f"       실제: {list(df.columns)[:15]}")
    if not ok:
        sys.exit("\n[중단] 필요한 컬럼이 없습니다.")

    # 좌표 단위 점검 — 반드시 투영좌표(m)여야 한다
    x = np.r_[seg["x1"].values, seg["x2"].values]
    y = np.r_[seg["y1"].values, seg["y2"].values]
    span = max(x.max() - x.min(), y.max() - y.min())
    print(f"\n  좌표 범위 x[{x.min():.0f}, {x.max():.0f}] y[{y.min():.0f}, {y.max():.0f}]"
          f"  (span {span:,.0f})")
    if span < 1000:
        print("  !! span 이 너무 작습니다. 위경도(deg)를 넣은 건 아닌지 확인하세요.")
        print("     이 코드는 좌표가 '미터' 라고 가정합니다 (min_len=1.0m, jam_density=0.08veh/m).")
    seg_len = np.hypot(seg["x2"] - seg["x1"], seg["y2"] - seg["y1"])
    print(f"  선분 길이 [m]  min={seg_len.min():.1f}  중앙값={seg_len.median():.1f}  "
          f"max={seg_len.max():.1f}  총연장={seg_len.sum()/1000:.1f} km")
    if (seg_len < 1e-9).any():
        print(f"  !! 길이 0 인 선분 {(seg_len < 1e-9).sum()}개 — 제거가 필요합니다.")

    # 사이트 좌표가 도로와 같은 좌표계인지
    sx, sy = sites["x_m"].values, sites["y_m"].values
    inside = ((sx >= x.min()) & (sx <= x.max()) & (sy >= y.min()) & (sy <= y.max())).mean()
    print(f"  사이트가 도로 bbox 안에 있는 비율: {inside*100:.0f}%"
          + ("" if inside > 0.8 else "   !! 좌표계 불일치 의심"))

    dup = meta["ID"].duplicated().sum()
    hit = seg["sta_from"].isin(meta["ID"]).mean()
    print(f"  meta ID 중복 {dup}개 · sta_from 이 meta 에 매칭되는 비율 {hit*100:.0f}%")
    if hit < 0.9:
        print("     !! inner join 으로 선분이 많이 탈락합니다.")
    return sites, seg, meta


def diagnose(env, rho_list=(1.5, 2.0, 3.0)):
    from mw_cut import cut_segments_fast
    print("\n" + "=" * 68)
    print("[3] 감도 진단 — 가중치에 반응하는 데이터가 몇 %인가")
    print("=" * 68)

    def owners(a):
        Lm = cut_segments_fast(env.P, env.Q, env.site_coords, env.weights(a), min_len=0.0)
        return (Lm > 0).sum(1), Lm.argmax(1)

    n0, o0 = owners(np.zeros(env.K))
    print(f"  경계에 잘리는 선분 : {(n0 > 1).mean()*100:5.1f}%  ({(n0>1).sum()}/{env.N})")
    print("     -> 이 선분들만 L_eff 가 변합니다. 나머지는 P_b 가 상수입니다.")
    rng = np.random.default_rng(0)
    for rho in rho_list:
        b = np.log(rho) / 2
        chg = []
        for _ in range(15):
            a = rng.uniform(-b, b, env.K)
            a -= a.mean()
            chg.append((owners(a)[1] != o0).mean())
        print(f"  rho_max={rho:<4}: 소속 셀이 바뀌는 선분 {np.mean(chg)*100:5.1f}%")
    print("     -> 'within' 목적함수는 이 재배정에서도 신호를 받지만,")
    print("        'global' 목적함수는 잘린 선분(위 첫 줄)에서만 신호를 받습니다.")


def search(seg, sites, meta_path, rho_list=(1.0001, 1.5, 2.0, 3.0),
           objectives=("within", "global"), n_iter=100, paths=None):
    from rl_env_voronoi_mw import TrafficRLEnvMW
    print("\n" + "=" * 68)
    print(f"[4] 랜덤서치 {n_iter}회 — 개선 여지가 있는가")
    print("=" * 68)
    print(f"  {'objective':>10} {'rho_max':>8} {'J(a=0)':>12} {'best J':>12} {'개선':>8}")
    rng = np.random.default_rng(0)
    best_overall = None
    for obj in objectives:
        for rho in rho_list:
            env = TrafficRLEnvMW(*paths, rho_max=rho, objective=obj)
            J0, _, _ = env.evaluate(np.zeros(env.K))
            b, bestJ, bestA = env.a_bound, J0, np.zeros(env.K)
            for _ in range(n_iter):
                a = rng.uniform(-b, b, env.K)
                a -= a.mean()
                J, _, _ = env.evaluate(a)
                if J < bestJ:
                    bestJ, bestA = J, a
            imp = (J0 - bestJ) / max(J0, 1e-12) * 100
            print(f"  {obj:>10} {rho:>8.2f} {J0:>12.6f} {bestJ:>12.6f} {imp:>7.1f}%")
            if obj == "within" and (best_overall is None or bestJ < best_overall[0]):
                best_overall = (bestJ, bestA, env)
    return best_overall


def plot(env, a_best, path="mw_data_test.png", sub=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False})
    PAL = np.array(["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
                    "#B279A2", "#B8A02E", "#9D755D", "#7F7F7F", "#3B7EA1"])

    t = np.linspace(0, 1, sub + 1)
    A = env.P[:, None, :] + t[None, :-1, None] * (env.Q - env.P)[:, None, :]
    B = env.P[:, None, :] + t[None, 1:, None] * (env.Q - env.P)[:, None, :]
    M = 0.5 * (A + B)
    segs = np.stack([A.reshape(-1, 2), B.reshape(-1, 2)], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, a, ttl in ((axes[0], np.zeros(env.K), "a = 0  (일반 Voronoi)"),
                       (axes[1], a_best, f"최적 탐색 결과  (rho={np.exp(a_best.max()-a_best.min()):.2f})")):
        w = env.weights(a)
        d = np.linalg.norm(M.reshape(-1, 1, 2) - env.site_coords[None], axis=2) / w
        own = d.argmin(1)
        ax.add_collection(LineCollection(segs, colors=PAL[own % len(PAL)], linewidths=0.7))
        ax.scatter(env.site_coords[:, 0], env.site_coords[:, 1],
                   s=25 + 90 * (w - w.min()) / max(w.max() - w.min(), 1e-9),
                   c="black", zorder=5, marker="o", edgecolors="white", linewidths=0.8)
        ax.set_title(ttl, fontsize=11)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("도로 선분을 소속 셀 색으로 (점 크기 = 가중치)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\n  그림 저장: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    paths = (args.segments, args.sites, args.meta)

    preflight(*paths)

    from rl_env_voronoi_mw import TrafficRLEnvMW
    print("\n" + "=" * 68)
    print("[2] a = 0 기준선")
    print("=" * 68)
    env = TrafficRLEnvMW(*paths, rho_max=2.0, objective="within")
    import time
    t0 = time.time()
    J0, stds, info = env.evaluate(np.zeros(env.K))
    dt = time.time() - t0
    print(f"  evaluate() 1회 = {dt*1000:.0f} ms  ({1/dt:.0f} step/s)")
    print(f"  J(within)={info['within']:.6f}   J(global)={info['global_std']:.6f}")
    print(f"  평균 차단확률={info['mean_prob']:.4f}   조각수={info['n_pieces']}")
    print(f"  유효 셀(조각>=2) {info['n_valid_cells']}/{env.K}"
          + ("" if info["n_valid_cells"] == env.K else "   !! 조각이 거의 없는 셀이 있습니다"))
    if info.get("truncated"):
        print(f"  !! max_cuts 초과 선분 {info['truncated']}개 — cut_segments_fast(max_cuts=...) 를 키우세요")

    diagnose(env)
    best = search(None, None, None, n_iter=args.iters, paths=paths)

    if not args.no_plot and best is not None:
        plot(env, best[1])


if __name__ == "__main__":
    main()