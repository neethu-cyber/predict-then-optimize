"""Daily ordering decisions for the ward cabinet.

Every evening the pharmacy decides how many packs q[i] of each medicine to place in the ward cabinet for
the next day. Left-over packs cost h[i] (return handling, expiry risk); missing packs cost p[i] (emergency
delivery). All medicines share the cabinet's space.

Plug-in rules use a forecast directly and, if the cabinet overflows, scale all orders down proportionally.
The sample average approximation (SAA) solves a stochastic linear program over demand scenarios instead:

    min  (1/S) Σ_s Σ_i ( h[i] · over[i,s] + p[i] · under[i,s] )
    s.t. over[i,s]  >= q[i] - d[i,s]           left-over packs in scenario s
         under[i,s] >= d[i,s] - q[i]           missing packs in scenario s
         Σ_i vol[i] · q[i] <= capacity          shared cabinet space
         q, over, under >= 0

Without the capacity constraint, the optimal q[i] is the p/(p+h) quantile of demand (the newsvendor
critical ratio); with it, the LP decides which medicines get the scarce space.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def scale_to_capacity(q, vol, cap):
    used = float(np.dot(vol, q))
    return q * (cap / used) if used > cap else q


def saa_order(scenarios, h, p, vol, cap):
    """scenarios: array (S, N) of demand scenarios for the N medicines. Returns the optimal order q (N,)."""
    S, N = scenarios.shape
    # variable layout: q (N) | over (N*S) | under (N*S)
    n_var = N + 2 * N * S
    c = np.concatenate([np.zeros(N), np.tile(h, S) / S, np.tile(p, S) / S])
    rows, cols, vals, b = [], [], [], []
    r = 0
    for s in range(S):
        for i in range(N):
            o, u = N + s * N + i, N + N * S + s * N + i
            rows += [r, r]; cols += [i, o]; vals += [1.0, -1.0]; b.append(scenarios[s, i]); r += 1    # q - over <= d
            rows += [r, r]; cols += [i, u]; vals += [-1.0, -1.0]; b.append(-scenarios[s, i]); r += 1  # -q - under <= -d
    rows += [r] * N; cols += list(range(N)); vals += list(vol); b.append(cap); r += 1                 # capacity
    A = coo_matrix((vals, (rows, cols)), shape=(r, n_var)).tocsr()
    res = linprog(c, A_ub=A, b_ub=np.array(b), bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:N]


def realized_cost(q, d, h, p):
    over = np.clip(q - d, 0, None)
    under = np.clip(d - q, 0, None)
    return float(np.dot(h, over) + np.dot(p, under)), float(under.sum())
