"""
Méthode C -- évolution en domaine temporel, condition de Dirichlet AdS exacte.

  psi_tt - psi_xx + V(x) psi = 0 ,   x = r_*  in [x_min, x_max]
  x_max = r_*^max (bord AdS) : psi = 0          x_min : sortant (Sommerfeld), loin.
Différences finies centrées d'ordre 2, saute-mouton, dt = h/2.
"""
import numpy as np
from scipy.integrate import solve_ivp
from model import build, horizon, peak, tortoise_gap, V_eff


def make_grid(s, L, M=1.0, Q=0.2, l=20.0, h=0.02, xmin=-150.0, xref_L=None):
    mod = build(s, M, Q, l)
    rh = max(horizon(mod))
    rp, _ = peak(mod, rh, L if xref_L is None else xref_L, rmax=0.5 * l)
    xmax = tortoise_gap(mod, rh, rp)               # r_*^max - r_*(r_peak), r_*(r_peak)=0
    N = int(round((xmax - xmin) / h))
    h = (xmax - xmin) / N
    x = xmin + h * np.arange(N + 1)
    V = np.zeros_like(x)
    f = mod['f']
    # côté horizon : dr/dx = f(r), x in [-45, 0]
    xin = x[(x <= 0) & (x >= -45.0)]
    solin = solve_ivp(lambda xx, rr: f(rr), [0.0, -45.0], [rp], t_eval=xin[::-1],
                      rtol=1e-12, atol=1e-14, method='DOP853')
    rin = solin.y[0][::-1]
    V[(x <= 0) & (x >= -45.0)] = V_eff(mod, rin, L)
    # côté AdS : du/dx = -u^2 f(1/u), x in [0, xmax)
    xout = x[(x > 0) & (x < xmax - 1e-9)]
    solout = solve_ivp(lambda xx, uu: -uu**2 * f(1.0 / uu), [0.0, xmax], [1.0 / rp],
                       t_eval=xout, rtol=1e-12, atol=1e-16, method='DOP853')
    rout = 1.0 / solout.y[0]
    V[(x > 0) & (x < xmax - 1e-9)] = V_eff(mod, rout, L)
    V[-1] = 0.0                                     # Dirichlet ; valeur inutilisée
    return x, V, xmax, rp, rh


def evolve(x, V, T, x0, sigma, xobs, dt=None):
    h = x[1] - x[0]
    dt = dt or 0.5 * h
    nsteps = int(round(T / dt))
    psi = np.exp(-(x - x0) ** 2 / (2 * sigma**2))
    psi[-1] = 0.0
    lap = np.zeros_like(psi)
    lap[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / h**2
    old = psi.copy()
    cur = psi + 0.5 * dt**2 * (lap - V * psi)
    cur[-1] = 0.0
    iobs = [int(np.argmin(np.abs(x - xo))) for xo in xobs]
    rec = np.zeros((nsteps + 1, len(xobs)))
    rec[0] = psi[iobs]
    rec[1] = cur[iobs]
    c = dt / h
    dt2 = dt**2
    for n in range(2, nsteps + 1):
        new = np.empty_like(cur)
        new[1:-1] = 2 * cur[1:-1] - old[1:-1] + dt2 * ((cur[2:] - 2 * cur[1:-1] + cur[:-2]) / h**2 - V[1:-1] * cur[1:-1])
        new[0] = cur[0] + c * (cur[1] - cur[0])
        new[-1] = 0.0
        old, cur = cur, new
        rec[n] = cur[iobs]
    t = dt * np.arange(nsteps + 1)
    return t, rec


def matrix_pencil(y, dt, K):
    """y_n ~ sum_k c_k exp(s_k n dt) ; renvoie s_k, c_k (K exponentielles)."""
    y = np.asarray(y, dtype=complex)
    N = len(y)
    Lp = N // 2
    Y = np.array([y[i:i + Lp + 1] for i in range(N - Lp)])
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    Vk = Vh[:K].conj().T
    V1, V2 = Vk[:-1], Vk[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    s = np.log(z) / dt
    # amplitudes par moindres carrés
    tt = dt * np.arange(N)
    A = np.exp(np.outer(tt, s))
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return s, c


def ringdown_fit(t, y, t1, t2, K, sub=25):
    """Ajuste y(t) sur [t1,t2] avec K exponentielles ; renvoie omega_k = i s_k (e^{-i w t})."""
    m = (t >= t1) & (t <= t2)
    tt, yy = t[m][::sub], y[m][::sub]
    dts = tt[1] - tt[0]
    s, c = matrix_pencil(yy, dts, K)
    w = 1j * s
    order = np.argsort(-np.abs(c))
    return w[order], c[order]
