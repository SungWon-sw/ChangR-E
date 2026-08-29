# multiplicative로 바꿈. 폴리곤을 만드는게 아니라 그냥 이차방정식 구해서 경계로 도로를 자르는 스크립트

"""
MW(곱셈가중) 보로노이로 '선분을 정확히 자르는' 벡터화 커널.

폴리곤을 전혀 만들지 않는다. 도로 선분 하나하나에 대해
    d_i(x(t)) = |x(t) - p_i| / w_i,   x(t) = P + t (Q-P),  t in [0,1]
의 argmin 이 바뀌는 t 를 이차방정식으로 정확히 구하고,
구간 중점에서 argmin 을 평가해 소유 셀을 확정한다.

  w_j^2 |x-p_i|^2 - w_i^2 |x-p_j|^2 = 0
  => alpha t^2 + beta t + gamma = 0
     alpha = |u|^2 (w_j^2 - w_i^2)
     beta  = 2 (w_j^2 (A.u) - w_i^2 (B.u))        A = P-p_i, B = P-p_j, u = Q-P
     gamma = w_j^2 |A|^2 - w_i^2 |B|^2

MW 셀이 비볼록이든, 여러 조각이든, 구멍이 있든 자동으로 처리된다.
"""

import numpy as np


def _roots_stable(alpha, beta, gamma, lin_eps=1e-12):
    """벡터화된 수치안정 이차방정식. (N,) 배열 3개 -> (N,2) 근 (없으면 nan).

    alpha ~ 0 (가중치가 거의 같을 때!) 이면 일차방정식으로 자동 강등.
    SAC 초기에는 모든 w 가 거의 같으므로 이 분기가 실제로 자주 탄다.
    """
    alpha = np.asarray(alpha, float)
    beta = np.asarray(beta, float)
    gamma = np.asarray(gamma, float)
    out = np.full(alpha.shape + (2,), np.nan)

    scale = np.maximum(np.abs(beta), np.abs(gamma))
    scale = np.where(scale > 0, scale, 1.0)
    is_lin = np.abs(alpha) <= lin_eps * scale

    # --- 일차: beta t + gamma = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lin = -gamma / beta
    ok_lin = is_lin & (np.abs(beta) > 0)
    out[..., 0] = np.where(ok_lin, t_lin, np.nan)

    # --- 이차
    quad = ~is_lin
    disc = beta * beta - 4.0 * alpha * gamma
    ok = quad & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    q = -0.5 * (beta + np.sign(np.where(beta == 0, 1.0, beta)) * sq)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(ok, q / alpha, np.nan)
        r2 = np.where(ok & (q != 0), gamma / q, np.nan)
    out[..., 0] = np.where(quad, r1, out[..., 0])
    out[..., 1] = np.where(quad, r2, np.nan)
    return out


def cut_segments(P, Q, sites, w, merge_per_cell=True, min_len=1.0):
    """모든 선분을 MW 셀 경계로 정확히 자른다.

    P, Q  : (N,2) 선분 양 끝점
    sites : (K,2)
    w     : (K,)  양수
    merge_per_cell : True 면 같은 셀이 한 선분에서 여러 조각을 가질 때 길이를 합산
                     (기존 power 코드의 poly.intersection(line).length 와 동일 의미).
                     False 면 조각별로 따로 돌려준다.
    반환: seg_idx (M,), cell_idx (M,), length (M,)
    """
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    S = np.asarray(sites, float)
    w = np.asarray(w, float)
    N, K = len(P), len(S)
    u = Q - P                                   # (N,2)
    uu = (u * u).sum(1)                         # (N,)
    seg_len = np.sqrt(uu)

    ii, jj = np.triu_indices(K, k=1)             # (M2,)
    wi2, wj2 = w[ii] ** 2, w[jj] ** 2

    A = P[:, None, :] - S[None, ii, :]           # (N,M2,2)
    B = P[:, None, :] - S[None, jj, :]
    Au = (A * u[:, None, :]).sum(-1)
    Bu = (B * u[:, None, :]).sum(-1)
    AA = (A * A).sum(-1)
    BB = (B * B).sum(-1)

    alpha = uu[:, None] * (wj2 - wi2)[None, :]
    beta = 2.0 * (wj2[None, :] * Au - wi2[None, :] * Bu)
    gamma = wj2[None, :] * AA - wi2[None, :] * BB

    roots = _roots_stable(alpha, beta, gamma)    # (N,M2,2)
    roots = roots.reshape(N, -1)
    roots = np.where((roots > 0.0) & (roots < 1.0), roots, np.nan)

    seg_out, cell_out, len_out = [], [], []
    for n in range(N):
        ts = roots[n]
        ts = ts[~np.isnan(ts)]
        cuts = np.concatenate(([0.0], np.sort(ts), [1.0]))
        cuts = cuts[np.concatenate(([True], np.diff(cuts) > 1e-12))]
        if len(cuts) < 2:
            continue
        mid = 0.5 * (cuts[:-1] + cuts[1:])                       # (m,)
        x = P[n] + mid[:, None] * u[n]                           # (m,2)
        d = np.linalg.norm(x[:, None, :] - S[None, :, :], axis=2) / w
        own = d.argmin(1)                                        # (m,)
        piece = (cuts[1:] - cuts[:-1]) * seg_len[n]

        if merge_per_cell:
            for k in np.unique(own):
                L = piece[own == k].sum()
                if L >= min_len:
                    seg_out.append(n); cell_out.append(int(k)); len_out.append(L)
        else:
            for k, L in zip(own, piece):
                if L >= min_len:
                    seg_out.append(n); cell_out.append(int(k)); len_out.append(L)

    return (np.array(seg_out, int), np.array(cell_out, int),
            np.array(len_out, float))


def owner_bruteforce(P, Q, sites, w, samples=20001):
    """검증용: 조밀 샘플링으로 셀별 길이를 근사."""
    S, w = np.asarray(sites, float), np.asarray(w, float)
    t = (np.arange(samples) + 0.5) / samples
    out = np.zeros((len(P), len(S)))
    for n in range(len(P)):
        x = P[n] + t[:, None] * (Q[n] - P[n])
        d = np.linalg.norm(x[:, None, :] - S[None, :, :], axis=2) / w
        own = d.argmin(1)
        L = np.linalg.norm(Q[n] - P[n]) / samples
        np.add.at(out[n], own, L)
    return out


# ---------------------------------------------------------------------------
# 완전 벡터화 버전 (RL 루프용). 파이썬 루프 없음.
# ---------------------------------------------------------------------------
def cut_segments_fast(P, Q, sites, w, max_cuts=64, min_len=1.0, return_info=False):
    """모든 선분 x 모든 셀의 유효길이 행렬 (N,K) 를 정확히 계산.

    1) 후보 셀 사전선별 (수학적으로 안전한 상한 사용)
         k 가 선분 위 어떤 점을 소유하면  d_k(M) <= d_min(M) + L / w_min
       (M=중점, L=선분길이). 이 부등식을 만족하는 k 만 남긴다.
    2) 남은 후보 쌍에 대해서만 이차방정식 -> 절단점
    3) 구간 중점 argmin -> bincount 로 셀별 길이 집계
    """
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    S = np.asarray(sites, float); w = np.asarray(w, float)
    N, K = len(P), len(S)
    u = Q - P
    uu = (u * u).sum(1)
    seg_len = np.sqrt(uu)

    # ---- 1) 후보 셀 선별
    M = 0.5 * (P + Q)
    dM = np.linalg.norm(M[:, None, :] - S[None, :, :], axis=2) / w      # (N,K)
    thr = dM.min(1) + seg_len / w.min()
    cand = dM <= thr[:, None]                                           # (N,K) bool
    ncand = cand.sum(1)
    C = int(ncand.max())
    cand_idx = np.argsort(~cand, axis=1, kind="stable")[:, :C]          # (N,C)
    cand_ok = np.take_along_axis(cand, cand_idx, axis=1)

    # ---- 2) 후보 쌍의 이차방정식
    ii, jj = np.triu_indices(C, k=1)
    I = cand_idx[:, ii]; J = cand_idx[:, jj]                            # (N,Mp)
    valid = cand_ok[:, ii] & cand_ok[:, jj]
    wi2 = w[I] ** 2; wj2 = w[J] ** 2

    A = P[:, None, :] - S[I]                                            # (N,Mp,2)
    B = P[:, None, :] - S[J]
    Au = (A * u[:, None, :]).sum(-1); Bu = (B * u[:, None, :]).sum(-1)
    alpha = uu[:, None] * (wj2 - wi2)
    beta = 2.0 * (wj2 * Au - wi2 * Bu)
    gamma = wj2 * (A * A).sum(-1) - wi2 * (B * B).sum(-1)

    roots = _roots_stable(alpha, beta, gamma).reshape(N, -1)
    keep = np.repeat(valid, 2, axis=1)
    roots = np.where(keep & (roots > 1e-12) & (roots < 1 - 1e-12), roots, np.nan)

    roots = np.sort(roots, axis=1)                                      # nan 은 뒤로
    nroot = (~np.isnan(roots)).sum(1)
    R = int(nroot.max())
    truncated = 0
    if R > max_cuts:                    # 방어용 상한 (정상적으론 걸리지 않는다)
        truncated = int((nroot > max_cuts).sum()); R = max_cuts
    roots = roots[:, :R]

    # ---- 3) 구간 중점에서 소유 셀 판정 (후보 셀만 비교)
    cuts = np.concatenate([np.zeros((N, 1)), np.nan_to_num(roots, nan=1.0),
                           np.ones((N, 1))], axis=1)                    # (N,R+2)
    lo, hi = cuts[:, :-1], cuts[:, 1:]
    frac = hi - lo                                                      # (N,R+1)
    mid = 0.5 * (lo + hi)
    X = P[:, None, :] + mid[:, :, None] * u[:, None, :]                 # (N,R+1,2)
    Sc = S[cand_idx]                                                    # (N,C,2)
    d = np.linalg.norm(X[:, :, None, :] - Sc[:, None, :, :], axis=3) / w[cand_idx][:, None, :]
    d = np.where(cand_ok[:, None, :], d, np.inf)                        # 패딩 슬롯 배제
    own = np.take_along_axis(cand_idx, d.argmin(2), axis=1)             # (N,R+1) -> 원래 셀 번호

    Lpiece = frac * seg_len[:, None]
    flat = (np.arange(N)[:, None] * K + own).ravel()
    Lmat = np.bincount(flat, weights=Lpiece.ravel(), minlength=N * K).reshape(N, K)
    Lmat[Lmat < min_len] = 0.0

    if return_info:
        return Lmat, dict(max_candidates=C, max_cuts_used=int(nroot.max()),
                          truncated=truncated)
    return Lmat