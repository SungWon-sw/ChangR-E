"""
weighted_voronoi.py
-------------------
가중치 보로노이 다이어그램(멱 다이어그램, power/Laguerre diagram) + Jain-Smith M/G/c/c
상태의존 차단확률(Erlang-B 일반화) 핵심 엔진.

설계 의도
  - 사이트(site)  = 교차로/신호등 = '병목 요충지'  (송도 데이터의 20개 교차로)
  - 가중치 w_i    = 각 요충지의 영향력(z좌표). RL 자리에 들어갈 자유변수.
  - 셀(cell)      = 그 요충지가 통제될 때 병목 영향권
  - 인접 간선     = 인접한 두 영향권을 잇는 '도로' (가중 들로네 = 멱 다이어그램의 쌍대)
  - 간선 차단확률 = 그 도로 구간의 M/G/c/c 막힘 확률. RL 목적함수는 이 값들의 균일도.

가중 보로노이는 '멱 거리' pow(x, p_i) = |x - p_i|^2 - w_i 로 정의한다.
  -> 셀 경계가 직선(반평면 교집합)이 되어 계산이 정확하고 빠르다.
  -> w_i 가 모두 같으면 일반(유클리드) 보로노이로 환원된다.
  -> '리프팅': p_i 를 z = |p_i|^2 - w_i 로 들어올린 점들의 하부 볼록껍질이 가중 들로네.
"""

import numpy as np
from scipy.spatial import ConvexHull


# --------------------------------------------------------------------------
# 1. 멱(가중) 보로노이 셀: 반평면 교집합 (Sutherland-Hodgman 클리핑)
# --------------------------------------------------------------------------
def _clip_halfplane(poly, a, b, c):
    """반평면 a*x + b*y <= c 로 볼록 다각형 poly(꼭짓점 리스트)를 자른다."""
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        cur_in = (a * cur[0] + b * cur[1] <= c + 1e-12)
        nxt_in = (a * nxt[0] + b * nxt[1] <= c + 1e-12)
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            d = (a * (nxt[0] - cur[0]) + b * (nxt[1] - cur[1]))
            if abs(d) > 1e-15:
                t = (c - (a * cur[0] + b * cur[1])) / d
                out.append((cur[0] + t * (nxt[0] - cur[0]),
                            cur[1] + t * (nxt[1] - cur[1])))
    return out


def power_cell(i, sites, weights, bbox):
    """사이트 i 의 멱 보로노이 셀(다각형 꼭짓점 배열)을 반환."""
    xmin, ymin, xmax, ymax = bbox
    poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    pi = sites[i]
    for j in range(len(sites)):
        if j == i:
            continue
        pj = sites[j]
        # |x-pi|^2 - wi <= |x-pj|^2 - wj  <=>  2(pj-pi)·x <= |pj|^2-wj-|pi|^2+wi
        a = 2.0 * (pj[0] - pi[0])
        b = 2.0 * (pj[1] - pi[1])
        c = (pj[0]**2 + pj[1]**2 - weights[j]) - (pi[0]**2 + pi[1]**2 - weights[i])
        poly = _clip_halfplane(poly, a, b, c)
        if len(poly) < 3:
            return np.empty((0, 2))
    return np.array(poly)


def all_power_cells(sites, weights, bbox):
    return [power_cell(i, sites, weights, bbox) for i in range(len(sites))]


# --------------------------------------------------------------------------
# 2. 인접 관계 = 가중 들로네(리프팅 -> 하부 볼록껍질). '리프팅 알고리즘'
# --------------------------------------------------------------------------
def weighted_delaunay_edges(sites, weights):
    """리프팅 z=|p|^2-w 후 하부 볼록껍질 면의 변 = 멱 다이어그램 인접 간선."""
    n = len(sites)
    lifted = np.column_stack([sites[:, 0], sites[:, 1],
                              sites[:, 0]**2 + sites[:, 1]**2 - weights])
    hull = ConvexHull(lifted)
    edges = set()
    for simplex, eq in zip(hull.simplices, hull.equations):
        if eq[2] < -1e-9:                       # 아래를 향하는 면 = 정규 삼각분할
            for a in range(3):
                for b in range(a + 1, 3):
                    e = tuple(sorted((int(simplex[a]), int(simplex[b]))))
                    edges.add(e)
    return sorted(edges)


# --------------------------------------------------------------------------
# 3. Jain-Smith M/G/c/c 상태의존 차단확률 (Erlang-B 일반화)
# --------------------------------------------------------------------------
def speed_ratio(n, beta, gamma):
    """f(n) = V_n / V_free = exp(-((n-1)/beta)^gamma).  f(1)=1."""
    if n <= 1:
        return 1.0
    return float(np.exp(-(((n - 1) / beta) ** gamma)))


def blocking_probability(length_m, v_free_ms, lam, lanes=1,
                         jam_density=0.08, beta=None, gamma=1.0):
    """
    한 도로 구간(간선)의 M/G/c/c 차단확률 P_c 를 반환.
      length_m   : 구간 길이 [m]
      v_free_ms  : 자유흐름 속도 [m/s]
      lam        : 도착률 [veh/s]
      lanes      : 차선 수
      jam_density: 차선당 정체밀도 [veh/m]  (~0.14 = 차간 7m)
    반환: (P_block, capacity_c, rho)
    """
    c = max(1, int(np.floor(length_m * jam_density * lanes)))
    if beta is None:
        beta = c / 2.0                       # 기본 보정값(데이터로 교체)
    ET1 = length_m / v_free_ms               # 자유흐름 통과시간
    rho = lam * ET1                          # offered load
    # p_n = p_0 * rho^n / (n! * prod_{k=1}^n f(k)) ; 수치안정 위해 로그합 사용
    log_terms = [0.0]                        # n=0
    acc = 0.0
    for n in range(1, c + 1):
        acc += np.log(rho + 1e-300) - np.log(n) - np.log(speed_ratio(n, beta, gamma))
        log_terms.append(acc)
    m = max(log_terms)
    probs = np.exp(np.array(log_terms) - m)
    probs /= probs.sum()
    return float(probs[-1]), c, rho


# --------------------------------------------------------------------------
# 4. 목적함수: 모든 간선 차단확률의 '균일도'(변동계수). RL 이 최소화할 대상.
# --------------------------------------------------------------------------
def edge_blocking_distribution(sites, weights, edge_attr_fn):
    """가중치 -> 인접 간선들의 차단확률 리스트와 균일도 지표를 계산."""
    edges = weighted_delaunay_edges(sites, weights)
    blocks = []
    for (i, j) in edges:
        L, vf, lam, lanes = edge_attr_fn(i, j, sites)
        Pb, _, _ = blocking_probability(L, vf, lam, lanes)
        blocks.append(Pb)
    blocks = np.array(blocks)
    cv = blocks.std() / (blocks.mean() + 1e-12)   # 변동계수: 작을수록 균일
    return edges, blocks, cv


# --------------------------------------------------------------------------
# 5. 가중치 -> 분할 -> 부하 결합:  셀이 흡수하는 수요(captured demand)
#    (가중치가 큐잉 파라미터에 실제로 영향을 주게 만드는 핵심 연결고리)
# --------------------------------------------------------------------------
def captured_demand(sites, weights, pts, vals):
    """수요점 pts(가중치 vals)를 멱거리 최근접 셀에 배정 -> 셀별 흡수 수요 합."""
    # power distance: |x-p_i|^2 - w_i
    d2 = ((pts[:, None, :] - sites[None, :, :]) ** 2).sum(-1) - weights[None, :]
    owner = d2.argmin(1)
    cap = np.zeros(len(sites))
    np.add.at(cap, owner, vals)
    return cap
