# M/G/c/c 차단 확률 계산하는걸 따로 빼놓음, 약간의 최적화가 적용되어 있음

"""Jain-Smith M/G/c/c 차단확률의 벡터화 버전.

weighted_voronoi.blocking_probability 와 수학적으로 동일하되,
(조각 x 셀) 수천 개를 한 번에 처리한다. 원본은 조각마다 c 번(수백~수천)
파이썬 루프를 돌기 때문에 RL 루프에서 가장 큰 병목이다.
"""

import numpy as np


def blocking_probability_batch(length_m, v_free_ms, lam, lanes,
                               jam_density=0.08, gamma=1.0, chunk=200_000):
    """(M,) 배열들을 받아 (M,) 차단확률과 (M,) 수용량 c 를 반환."""
    L = np.atleast_1d(np.asarray(length_m, float))
    lam = np.broadcast_to(np.atleast_1d(np.asarray(lam, float)), L.shape)
    lanes = np.broadcast_to(np.atleast_1d(np.asarray(lanes, float)), L.shape)

    c = np.maximum(1, np.floor(L * jam_density * lanes)).astype(np.int64)
    beta = c / 2.0
    rho = lam * (L / v_free_ms)                       # offered load
    log_rho = np.log(np.maximum(rho, 1e-300))

    out = np.empty(L.shape)
    cmax_all = int(c.max())
    # 메모리 제어를 위해 c 비슷한 것끼리 묶어 처리
    order = np.argsort(c)
    step = max(1, chunk // max(cmax_all, 1))
    for s in range(0, len(order), step):
        idx = order[s:s + step]
        cm = int(c[idx].max())
        k = np.arange(1, cm + 1, dtype=float)                       # (cm,)
        inc = (log_rho[idx][:, None] - np.log(k)[None, :]
               + ((k - 1)[None, :] / beta[idx][:, None]) ** gamma)  # -log f(k)
        logp = np.concatenate([np.zeros((len(idx), 1)),
                               np.cumsum(inc, axis=1)], axis=1)     # (m, cm+1)
        n = np.arange(cm + 1)[None, :]
        logp = np.where(n <= c[idx][:, None], logp, -np.inf)
        mx = logp.max(axis=1, keepdims=True)
        Z = np.log(np.exp(logp - mx).sum(axis=1, keepdims=True)) + mx
        log_pc = np.take_along_axis(logp, c[idx][:, None], axis=1)
        out[idx] = np.exp((log_pc - Z).ravel())
    return out, c