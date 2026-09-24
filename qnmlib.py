import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eig

# ---------------------------------------------------------------
# Metric of the manuscript (Eqs. rtheta, Mtheta, Qtheta, f)
# ---------------------------------------------------------------
class Metric:
    def __init__(self, M=1.0, Q=0.2, l=20.0, s=0.0):
        self.M, self.Q, self.l, self.s = M, Q, l, s

    # --- theta-deformed functions and r-derivatives ---
    def rho(self, r):
        s = self.s
        if s == 0.0:
            return np.asarray(r, dtype=float) * 1.0
        return (2 * r / np.pi) * np.arctan(r / s)

    def rho1(self, r):
        s = self.s
        if s == 0.0:
            return np.ones_like(np.asarray(r, dtype=float))
        return (2 / np.pi) * np.arctan(r / s) + 2 * r * s / (np.pi * (r**2 + s**2))

    def rho2(self, r):
        s = self.s
        if s == 0.0:
            return np.zeros_like(np.asarray(r, dtype=float))
        return 4 * s**3 / (np.pi * (r**2 + s**2) ** 2)

    def Mth(self, r):
        s, M = self.s, self.M
        if s == 0.0:
            return M * np.ones_like(np.asarray(r, dtype=float))
        return (2 * M / np.pi) * (np.arctan(r / s) - r * s / (r**2 + s**2))

    def Mth1(self, r):
        s, M = self.s, self.M
        if s == 0.0:
            return np.zeros_like(np.asarray(r, dtype=float))
        return 4 * M * r**2 * s / (np.pi * (r**2 + s**2) ** 2)

    def f(self, r):
        M, Q, l = self.M, self.Q, self.l
        rho = self.rho(r)
        Mt = self.Mth(r)
        Qt = (Q / M) * Mt
        return 1 - 2 * Mt / rho + Qt**2 / rho**2 + rho**2 / l**2

    def f1(self, r):
        M, Q, l = self.M, self.Q, self.l
        rho, rho1 = self.rho(r), self.rho1(r)
        Mt, Mt1 = self.Mth(r), self.Mth1(r)
        Qt, Qt1 = (Q / M) * Mt, (Q / M) * Mt1
        return (-2 * (Mt1 * rho - Mt * rho1) / rho**2
                + 2 * Qt * Qt1 / rho**2 - 2 * Qt**2 * rho1 / rho**3
                + 2 * rho * rho1 / l**2)

    def horizon(self):
        # outer horizon: largest root of f
        rs = np.linspace(0.05, 60, 60000)
        fs = self.f(rs)
        idx = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) < 0)[0]
        r0 = rs[idx[-1]]
        return brentq(self.f, rs[idx[-1]], rs[idx[-1] + 1], xtol=1e-14)

    def V(self, r, ell):
        rho, rho1, rho2 = self.rho(r), self.rho1(r), self.rho2(r)
        f, f1 = self.f(r), self.f1(r)
        return f * (ell * (ell + 1) / rho**2 + (f1 * rho1 + f * rho2) / rho)


# ---------------------------------------------------------------
# WKB (Iyer-Will, 3rd order) using exact r_*-derivatives (D = f d/dr on Taylor series)
# ---------------------------------------------------------------
def _mp_funcs(met, ell):
    import mpmath as mp
    M, Q, l, s = mp.mpf(met.M), mp.mpf(met.Q), mp.mpf(met.l), mp.mpf(met.s)
    L = ell * (ell + 1)

    def rho(r):
        return r if s == 0 else (2 * r / mp.pi) * mp.atan(r / s)

    def rho1(r):
        return mp.mpf(1) if s == 0 else (2 / mp.pi) * mp.atan(r / s) + 2 * r * s / (mp.pi * (r**2 + s**2))

    def rho2(r):
        return mp.mpf(0) if s == 0 else 4 * s**3 / (mp.pi * (r**2 + s**2) ** 2)

    def Mth(r):
        return M if s == 0 else (2 * M / mp.pi) * (mp.atan(r / s) - r * s / (r**2 + s**2))

    def Mth1(r):
        return mp.mpf(0) if s == 0 else 4 * M * r**2 * s / (mp.pi * (r**2 + s**2) ** 2)

    def f(r):
        rh_, Mt = rho(r), Mth(r)
        Qt = (Q / M) * Mt
        return 1 - 2 * Mt / rh_ + Qt**2 / rh_**2 + rh_**2 / l**2

    def f1(r):
        rh_, r1 = rho(r), rho1(r)
        Mt, Mt1 = Mth(r), Mth1(r)
        Qt, Qt1 = (Q / M) * Mt, (Q / M) * Mt1
        return (-2 * (Mt1 * rh_ - Mt * r1) / rh_**2 + 2 * Qt * Qt1 / rh_**2
                - 2 * Qt**2 * r1 / rh_**3 + 2 * rh_ * r1 / l**2)

    def V(r):
        rh_ = rho(r)
        return f(r) * (L / rh_**2 + (f1(r) * rho1(r) + f(r) * rho2(r)) / rh_)
    return f, V


def wkb_derivs(met, ell, rp):
    """return V^{(k)}_{r_*}(rp), k=0..6"""
    import mpmath as mp
    mp.mp.dps = 60
    f, V = _mp_funcs(met, ell)
    rp = mp.findroot(lambda r: mp.diff(V, r, 1), mp.mpf(rp))
    K = 6
    Vs = mp.taylor(V, rp, K, h=mp.mpf('1e-8'))
    fs = mp.taylor(f, rp, K, h=mp.mpf('1e-8'))
    def deriv(P):
        return [ (i + 1) * P[i + 1] for i in range(len(P) - 1)]
    def mul(a, b, n):
        out = [mp.mpf(0)] * n
        for i in range(min(len(a), n)):
            for j in range(min(len(b), n - i)):
                out[i + j] += a[i] * b[j]
        return out
    P = Vs
    ders = [P[0]]
    for k in range(1, K + 1):
        dP = deriv(P)
        P = mul(fs, dP, len(dP))
        ders.append(P[0])
    return ders, rp


def wkb3(met, ell, n=0):
    import mpmath as mp
    rh = met.horizon()
    rs = np.linspace(rh * 1.0005, rh * 6, 200000)
    Vs = met.V(rs, ell)
    i = np.argmax(Vs)
    if i == 0 or i == len(rs) - 1:
        return None
    d, rp = wkb_derivs(met, ell, rs[i])
    V0, V2, V3, V4, V5, V6 = d[0], d[2], d[3], d[4], d[5], d[6]
    alpha = n + mp.mpf(1) / 2
    A = mp.sqrt(-2 * V2)
    L2 = (1 / A) * ((mp.mpf(1) / 8) * (V4 / V2) * (mp.mpf(1) / 4 + alpha**2)
                    - (mp.mpf(1) / 288) * (V3 / V2) ** 2 * (7 + 60 * alpha**2))
    L3 = (1 / (-2 * V2)) * ((mp.mpf(5) / 6912) * (V3 / V2) ** 4 * (77 + 188 * alpha**2)
                            - (mp.mpf(1) / 384) * (V3**2 * V4 / V2**3) * (51 + 100 * alpha**2)
                            + (mp.mpf(1) / 2304) * (V4 / V2) ** 2 * (67 + 68 * alpha**2)
                            + (mp.mpf(1) / 288) * (V3 * V5 / V2**2) * (19 + 28 * alpha**2)
                            - (mp.mpf(1) / 288) * (V6 / V2) * (5 + 4 * alpha**2))
    w2 = V0 + A * L2 - 1j * alpha * A * (1 + L3)
    w = mp.sqrt(w2)
    if w.real < 0:
        w = -w
    return complex(w), float(rp), float(V0)


# ---------------------------------------------------------------
# Spectral QNM solver: Dirichlet AdS boundary, ingoing at horizon
# Ingoing EF coords: (rho^2 f R')' - 2 i w rho^2 R' - i w (rho^2)' R - L R = 0
# u = r_h / r, R = u^3 phi(u), Chebyshev-Gauss collocation.
# ---------------------------------------------------------------
def spectral_qnm(met, ell, N=80, nmodes=None):
    rh = met.horizon()
    L = ell * (ell + 1)
    # Chebyshev-Gauss nodes in x in (-1,1); u=(x+1)/2 in (0,1)
    j = np.arange(N)
    x = np.cos(np.pi * (2 * j + 1) / (2 * N))
    u = (x + 1) / 2
    # Chebyshev basis T_k(x), T_k', T_k''
    k = np.arange(N)
    th = np.arccos(x)
    T = np.cos(np.outer(th, k))
    # derivatives via recurrence using U_k
    # T_k' = k U_{k-1};   T_k'' = k[(k+1)T_k - U_k]/(x^2-1)... use stable formulas
    sin_th = np.sin(th)
    with np.errstate(divide='ignore', invalid='ignore'):
        T1 = (k[None, :] * np.sin(np.outer(th, k))) / sin_th[:, None]
        # T'' = (x T' - k^2 T)/(x^2-1)
        T2 = (x[:, None] * T1 - (k[None, :] ** 2) * T) / (1 - x[:, None] ** 2)
    # d/du = 2 d/dx
    P0 = T
    P1 = 2 * T1
    P2 = 4 * T2

    r = rh / u
    rho, rho1 = met.rho(r), met.rho1(r)
    f, f1 = met.f(r), met.f1(r)
    A = rho**2 * f
    Ar = 2 * rho * rho1 * f + rho**2 * f1
    B = rho**2
    C = 2 * rho * rho1

    # R = u^3 phi
    u3 = u**3
    R0 = u3[:, None] * P0
    Ru = (3 * u**2)[:, None] * P0 + u3[:, None] * P1
    Ruu = (6 * u)[:, None] * P0 + (6 * u**2)[:, None] * P1 + u3[:, None] * P2
    # r-derivatives: d/dr = -(u^2/rh) d/du
    Rr = -(u**2 / rh)[:, None] * Ru
    Rrr = (u**2 / rh**2)[:, None] * (2 * u[:, None] * Ru + u[:, None] ** 2 * Ruu)

    # equation: A Rrr + Ar Rr - L R = i w (2 B Rr + C R)
    M0 = A[:, None] * Rrr + Ar[:, None] * Rr - L * R0
    M1 = 2 * B[:, None] * Rr + C[:, None] * R0
    # row scaling by 1/u^2
    sc = 1.0 / u**2
    M0 = sc[:, None] * M0
    M1 = sc[:, None] * M1
    ev, vec = eig(M0, M1)
    mask = np.isfinite(ev)
    ev = ev[mask]
    w = -1j * ev  # i w = ev  ->  w = -i ev
    return w, rh
