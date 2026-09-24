#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent validation of the scalar quasinormal-mode section of
"Lorentzian Matter-Space Geometry of Reissner-Nordstrom-AdS Black Holes ..."

Metric (Eqs. (rtheta), (Mtheta), (Qtheta), (f) of the manuscript), massless scalar,
psi = r_theta * Phi_rad,   d^2 psi/dr_*^2 + (omega^2 - V) psi = 0,   Eq. (pot_explicit).

Three independent tools
-----------------------
1. wkb3      : barrier-top WKB (Iyer-Will, 3rd order, exact r_*-derivatives at the peak).
              It uses ONLY Taylor coefficients of V at its maximum and assumes purely
              outgoing waves on both sides of the barrier: the AdS boundary condition
              never enters.  (The manuscript uses Konoplya's 6th order; for l=1,2 the
              3rd-order values agree with the 6th-order ones to <= 0.6 %.)
2. spectral  : genuine AdS QNMs.  Ingoing Eddington-Finkelstein coordinates, R = u^3 phi(u),
              u = r_h/r, Chebyshev-Gauss collocation -> generalized eigenproblem, linear in
              omega.  Regular at the horizon (ingoing) and normalizable (Dirichlet, psi ~ z^2)
              at the AdS boundary.  Benchmark: SAdS4, l=1, r_+=1 gives 2.79822-2.67121i
              (Horowitz-Hubeny 2000).
3. timedomain: leap-frog evolution of  d_t^2 psi = d_x^2 psi - V psi  (x = r_*), reflecting
              (normalizable) boundary at the AdS wall, absorbing (Mur) condition on the
              horizon side.  Frequencies extracted with the matrix-pencil (Prony) method.

Usage
-----
    python3 qnm_ads_validation.py benchmark      # Horowitz-Hubeny check
    python3 qnm_ads_validation.py wkb            # WKB3 vs manuscript table (l=20)
    python3 qnm_ads_validation.py spectral       # AdS QNM towers, l=20 (and l=1 with --l 1)
    python3 qnm_ads_validation.py timedomain     # prompt ringdown + late-time cavity modes
    python3 qnm_ads_validation.py all
Requires numpy, scipy, mpmath.
"""
import sys
import argparse
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp
from scipy.linalg import eig


# ---------------------------------------------------------------------------
# Metric of the manuscript
# ---------------------------------------------------------------------------
class Metric:
    def __init__(self, M=1.0, Q=0.2, l=20.0, s=0.0):
        self.M, self.Q, self.l, self.s = M, Q, l, s      # s = sqrt(theta)

    def rho(self, r):                                     # r_theta
        r = np.asarray(r, dtype=float)
        return r.copy() if self.s == 0 else (2 * r / np.pi) * np.arctan(r / self.s)

    def rho1(self, r):
        r = np.asarray(r, dtype=float)
        if self.s == 0:
            return np.ones_like(r)
        s = self.s
        return (2 / np.pi) * np.arctan(r / s) + 2 * r * s / (np.pi * (r**2 + s**2))

    def rho2(self, r):
        r = np.asarray(r, dtype=float)
        if self.s == 0:
            return np.zeros_like(r)
        s = self.s
        return 4 * s**3 / (np.pi * (r**2 + s**2) ** 2)

    def Mth(self, r):
        r = np.asarray(r, dtype=float)
        if self.s == 0:
            return self.M * np.ones_like(r)
        s = self.s
        return (2 * self.M / np.pi) * (np.arctan(r / s) - r * s / (r**2 + s**2))

    def Mth1(self, r):
        r = np.asarray(r, dtype=float)
        if self.s == 0:
            return np.zeros_like(r)
        s = self.s
        return 4 * self.M * r**2 * s / (np.pi * (r**2 + s**2) ** 2)

    def f(self, r):
        rho, Mt = self.rho(r), self.Mth(r)
        Qt = (self.Q / self.M) * Mt
        return 1 - 2 * Mt / rho + Qt**2 / rho**2 + rho**2 / self.l**2

    def f1(self, r):
        rho, rho1 = self.rho(r), self.rho1(r)
        Mt, Mt1 = self.Mth(r), self.Mth1(r)
        Qt, Qt1 = (self.Q / self.M) * Mt, (self.Q / self.M) * Mt1
        return (-2 * (Mt1 * rho - Mt * rho1) / rho**2 + 2 * Qt * Qt1 / rho**2
                - 2 * Qt**2 * rho1 / rho**3 + 2 * rho * rho1 / self.l**2)

    def horizon(self):
        rs = np.linspace(0.05, 60, 60000)
        fs = self.f(rs)
        idx = np.where(np.sign(fs[:-1]) * np.sign(fs[1:]) < 0)[0]
        return brentq(self.f, rs[idx[-1]], rs[idx[-1] + 1], xtol=1e-14)

    def V(self, r, ell):                                  # Eq. (pot_explicit)
        rho, rho1, rho2 = self.rho(r), self.rho1(r), self.rho2(r)
        f, f1 = self.f(r), self.f1(r)
        return f * (ell * (ell + 1) / rho**2 + (f1 * rho1 + f * rho2) / rho)

    def peak(self, ell):
        """radius of the exterior maximum of V (None if V is monotonic)"""
        rh = self.horizon()
        rs = np.linspace(rh * 1.0005, rh * 6, 200000)
        i = np.argmax(self.V(rs, ell))
        if i in (0, len(rs) - 1):
            return None
        res = minimize_scalar(lambda x: -self.V(x, ell), bracket=(rs[i - 1], rs[i], rs[i + 1]), tol=1e-13)
        return res.x


# ---------------------------------------------------------------------------
# 1. WKB, 3rd order (Iyer-Will) with exact tortoise-coordinate derivatives
# ---------------------------------------------------------------------------
def _mp_funcs(met, ell):
    import mpmath as mp
    M, Q, l, s = (mp.mpf(x) for x in (met.M, met.Q, met.l, met.s))
    L = ell * (ell + 1)
    rho = lambda r: r if s == 0 else (2 * r / mp.pi) * mp.atan(r / s)
    rho1 = lambda r: mp.mpf(1) if s == 0 else (2 / mp.pi) * mp.atan(r / s) + 2 * r * s / (mp.pi * (r**2 + s**2))
    rho2 = lambda r: mp.mpf(0) if s == 0 else 4 * s**3 / (mp.pi * (r**2 + s**2) ** 2)
    Mth = lambda r: M if s == 0 else (2 * M / mp.pi) * (mp.atan(r / s) - r * s / (r**2 + s**2))
    Mth1 = lambda r: mp.mpf(0) if s == 0 else 4 * M * r**2 * s / (mp.pi * (r**2 + s**2) ** 2)

    def f(r):
        rh_, Mt = rho(r), Mth(r)
        Qt = (Q / M) * Mt
        return 1 - 2 * Mt / rh_ + Qt**2 / rh_**2 + rh_**2 / l**2

    def f1(r):
        rh_, r1, Mt, Mt1 = rho(r), rho1(r), Mth(r), Mth1(r)
        Qt, Qt1 = (Q / M) * Mt, (Q / M) * Mt1
        return (-2 * (Mt1 * rh_ - Mt * r1) / rh_**2 + 2 * Qt * Qt1 / rh_**2
                - 2 * Qt**2 * r1 / rh_**3 + 2 * rh_ * r1 / l**2)

    def V(r):
        rh_ = rho(r)
        return f(r) * (L / rh_**2 + (f1(r) * rho1(r) + f(r) * rho2(r)) / rh_)
    return f, V


def wkb3(met, ell, n=0):
    """Iyer-Will 3rd-order WKB frequency (omega_R - i omega_I convention, omega_I>0 -> Im<0)."""
    import mpmath as mp
    mp.mp.dps = 60
    rp0 = met.peak(ell)
    if rp0 is None:
        return None                                       # no barrier: WKB not applicable
    f, V = _mp_funcs(met, ell)
    rp = mp.findroot(lambda r: mp.diff(V, r, 1), mp.mpf(rp0))
    K = 6
    Vs = mp.taylor(V, rp, K, h=mp.mpf('1e-8'))
    fs = mp.taylor(f, rp, K, h=mp.mpf('1e-8'))

    def mul(a, b, m):
        out = [mp.mpf(0)] * m
        for i in range(min(len(a), m)):
            for j in range(min(len(b), m - i)):
                out[i + j] += a[i] * b[j]
        return out
    P, d = Vs, [Vs[0]]
    for k in range(1, K + 1):                             # D = f d/dr  (= d/dr_*)
        dP = [(i + 1) * P[i + 1] for i in range(len(P) - 1)]
        P = mul(fs, dP, len(dP))
        d.append(P[0])
    V0, V2, V3, V4, V5, V6 = d[0], d[2], d[3], d[4], d[5], d[6]
    a = n + mp.mpf(1) / 2
    A = mp.sqrt(-2 * V2)
    L2 = (1 / A) * ((mp.mpf(1) / 8) * (V4 / V2) * (mp.mpf(1) / 4 + a**2)
                    - (mp.mpf(1) / 288) * (V3 / V2) ** 2 * (7 + 60 * a**2))
    L3 = (1 / (-2 * V2)) * ((mp.mpf(5) / 6912) * (V3 / V2) ** 4 * (77 + 188 * a**2)
                            - (mp.mpf(1) / 384) * (V3**2 * V4 / V2**3) * (51 + 100 * a**2)
                            + (mp.mpf(1) / 2304) * (V4 / V2) ** 2 * (67 + 68 * a**2)
                            + (mp.mpf(1) / 288) * (V3 * V5 / V2**2) * (19 + 28 * a**2)
                            - (mp.mpf(1) / 288) * (V6 / V2) * (5 + 4 * a**2))
    w = mp.sqrt(V0 + A * L2 - 1j * a * A * (1 + L3))
    if w.real < 0:
        w = -w
    return complex(w)


# ---------------------------------------------------------------------------
# 2. Spectral method: genuine AdS QNMs (ingoing at the horizon, Dirichlet at infinity)
# ---------------------------------------------------------------------------
def spectral_qnm(met, ell, N=140):
    """All eigenvalues omega (e^{-i omega v}) of
         (rho^2 f R')' - 2 i w rho^2 R' - i w (rho^2)' R - l(l+1) R = 0,
       R = u^3 phi(u), u = r_h/r, phi expanded in N Chebyshev polynomials, collocation at the
       N Chebyshev-Gauss nodes (both endpoints are regular-singular points and are excluded)."""
    rh = met.horizon()
    Lq = ell * (ell + 1)
    j = np.arange(N)
    x = np.cos(np.pi * (2 * j + 1) / (2 * N))
    u = (x + 1) / 2
    k = np.arange(N)
    th = np.arccos(x)
    T = np.cos(np.outer(th, k))
    T1 = (k[None, :] * np.sin(np.outer(th, k))) / np.sin(th)[:, None]
    T2 = (x[:, None] * T1 - (k[None, :] ** 2) * T) / (1 - x[:, None] ** 2)
    P0, P1, P2 = T, 2 * T1, 4 * T2                        # d/du = 2 d/dx

    r = rh / u
    rho, rho1 = met.rho(r), met.rho1(r)
    f, f1 = met.f(r), met.f1(r)
    A, Ar = rho**2 * f, 2 * rho * rho1 * f + rho**2 * f1
    B, C = rho**2, 2 * rho * rho1

    u3 = u**3
    R0 = u3[:, None] * P0
    Ru = (3 * u**2)[:, None] * P0 + u3[:, None] * P1
    Ruu = (6 * u)[:, None] * P0 + (6 * u**2)[:, None] * P1 + u3[:, None] * P2
    Rr = -(u**2 / rh)[:, None] * Ru
    Rrr = (u**2 / rh**2)[:, None] * (2 * u[:, None] * Ru + u[:, None] ** 2 * Ruu)

    M0 = A[:, None] * Rrr + Ar[:, None] * Rr - Lq * R0    # A R'' + A' R' - l(l+1) R
    M1 = 2 * B[:, None] * Rr + C[:, None] * R0            # 2 rho^2 R' + (rho^2)' R
    sc = 1.0 / u**2
    ev = eig(sc[:, None] * M0, sc[:, None] * M1, right=False)
    ev = ev[np.isfinite(ev)]
    return -1j * ev, rh                                   # M0 phi = (i w) M1 phi


def converged_modes(met, ell, N1=140, N2=180, wmax=1.2, tol=1e-6):
    """keep only eigenvalues stable under N1 -> N2 (removes spurious modes)."""
    sel = lambda w: w[(np.imag(w) < 1e-9) & (np.real(w) > 0) & (np.abs(w) < wmax)]
    w1, rh = spectral_qnm(met, ell, N1)
    w2, _ = spectral_qnm(met, ell, N2)
    w1, w2 = sel(w1), sel(w2)
    out = np.array([x for x in w2 if np.abs(w1 - x).min() < tol * max(1, abs(x))])
    return out[np.argsort(np.real(out))], rh


# ---------------------------------------------------------------------------
# 3. Time-domain evolution + matrix-pencil (Prony) extraction
# ---------------------------------------------------------------------------
def evolve(met, ell, h=0.05, zmax=420.0, T=1500.0, cfl=0.4, x0=6.0, sigma=2.0, xobs=0.0):
    """psi_tt = psi_xx - V psi on x=r_*.  Grid z_j=(j+1/2)h, x=x_inf-z (z=0 is the AdS boundary,
    where V ~ 2/z^2 and the normalizable solution is even in z).  Mur condition on the horizon side.
    x0, xobs are measured from the reference radius (peak of V, or 1.5 r_h if V has no barrier)."""
    rh = met.horizon()
    rp = met.peak(ell) or 1.5 * rh
    up = rh / rp
    rhs = lambda x, u: [-u[0] ** 2 * met.f(rh / u[0]) / rh]        # du/dx, u=r_h/r
    ev = lambda x, u: u[0]
    ev.terminal, ev.direction = True, -1
    solR = solve_ivp(rhs, [0, 1e3], [up], events=ev, dense_output=True, rtol=1e-13, atol=1e-16, method='DOP853')
    xinf = solR.t_events[0][0]                                     # x_inf - x_ref  (finite in AdS)
    solL = solve_ivp(rhs, [0, -70], [up], dense_output=True, rtol=1e-13, atol=1e-16, method='DOP853')
    z = (np.arange(int(zmax / h)) + 0.5) * h
    xr = xinf - z
    V = np.zeros_like(z)
    R, Lm = xr >= 0, (xr < 0) & (xr > -65)                         # V is < 1e-9 beyond x_ref-65
    V[R] = met.V(rh / solR.sol(xr[R])[0], ell)
    V[Lm] = met.V(rh / solL.sol(xr[Lm])[0], ell)
    dt = cfl * h
    ns = int(T / dt)
    psi = np.exp(-(xr - x0) ** 2 / (2 * sigma**2))

    def lap(p):
        o = np.empty_like(p)
        o[1:-1] = p[2:] - 2 * p[1:-1] + p[:-2]
        o[0] = p[1] - p[0]                                         # even reflection at z=0
        o[-1] = 0.0
        return o / h**2
    old, cur = psi, psi + 0.5 * dt**2 * (lap(psi) - V * psi)
    io = int(np.argmin(np.abs(xr - xobs)))
    rec = np.zeros(ns)
    c = dt / h
    for n in range(ns):
        rec[n] = cur[io]
        new = 2 * cur - old + dt**2 * (lap(cur) - V * cur)
        new[-1] = cur[-2] + ((c - 1) / (c + 1)) * (new[-2] - cur[-1])   # Mur (outgoing) at x -> -inf
        old, cur = cur, new
    return np.arange(ns) * dt + dt, rec, xinf


def matrix_pencil(y, dt, p):
    """complex frequencies (e^{-i w t}) of p damped exponentials."""
    N = len(y)
    L = N // 2
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    _, _, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh[:p].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    return 1j * np.log(z) / dt


def prompt_ringdown(tt, sig, w_guess, windows=((22, 52), (25, 55), (28, 55)), orders=(2, 3, 4), step=10):
    dt = tt[1] - tt[0]
    vals = []
    for (a, b) in windows:
        m = (tt >= a) & (tt <= b)
        for p in orders:
            w = matrix_pencil(sig[m][::step], dt * step, p)
            w = w[np.real(w) > 0]
            vals.append(w[np.argmin(np.abs(w - w_guess))])
    vals = np.array(vals)
    return vals.mean(), vals.real.std(), vals.imag.std()


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
SQ = (0.0, 0.05, 0.10, 0.15)
PAPER = {(1, 0.00): 0.3064 - 0.1000j, (1, 0.05): 0.3128 - 0.1005j, (1, 0.10): 0.3197 - 0.1011j, (1, 0.15): 0.3271 - 0.1016j,
         (2, 0.00): 0.5042 - 0.0997j, (2, 0.05): 0.5147 - 0.1002j, (2, 0.10): 0.5260 - 0.1008j, (2, 0.15): 0.5383 - 0.1013j,
         (0, 0.00): 0.1149 - 0.1011j, (0, 0.05): 0.1175 - 0.1017j, (0, 0.10): 0.1203 - 0.1022j, (0, 0.15): 0.1232 - 0.1027j}


def cmd_benchmark(a):
    met = Metric(M=1.0, Q=0.0, l=1.0, s=0.0)
    w, rh = spectral_qnm(met, 0, 60)
    w = w[(np.imag(w) < 0) & (np.real(w) > 0) & (np.abs(w) < 10)]
    w = w[np.argsort(-np.imag(w))]
    print("SAdS4  l=1  r_+=%.4f  l=0 fundamental: %.5f%+.5fi   (Horowitz-Hubeny: 2.7982-2.6712i)" % (rh, w[0].real, w[0].imag))


def cmd_wkb(a):
    print("WKB (3rd order) vs manuscript (6th order), M=1 Q=0.2 l=%g, n=0" % a.l)
    for ell in (0, 1, 2):
        for s in SQ:
            met = Metric(1.0, 0.2, a.l, s)
            w = wkb3(met, ell)
            ref = PAPER.get((ell, s)) if a.l == 20 else None
            print("  ell=%d sqrtθ=%.2f  r_peak=%s  WKB3=%s   manuscript=%s" % (
                ell, s, "%.3f" % met.peak(ell) if met.peak(ell) else "none",
                "%.4f%+.4fi" % (w.real, w.imag) if w else "n/a",
                "%.4f%+.4fi" % (ref.real, ref.imag) if ref is not None else "-"))


def cmd_spectral(a):
    print("Genuine AdS QNMs (Dirichlet), M=1 Q=0.2 l=%g  [Re w, Im w]" % a.l)
    for ell in (0, 1, 2):
        for s in SQ:
            met = Metric(1.0, 0.2, a.l, s)
            if a.l >= 5:
                ws, rh = converged_modes(met, ell, 140, 180, 1.2, 1e-6)
                ws = ws[:a.nmodes]
            else:
                ws, rh = converged_modes(met, ell, 60, 90, 12, 1e-7)
                ws = ws[np.argsort(-np.imag(ws))][:a.nmodes]
            print("  ell=%d sqrtθ=%.2f r_h=%.4f: " % (ell, s, rh) + "  ".join("%.5f%+.3ei" % (x.real, x.imag) for x in ws))


def cmd_timedomain(a):
    print("Time domain, l=20, M=1, Q=0.2 (reflecting AdS wall)")
    for ell in (1, 2):
        for s in SQ:
            met = Metric(1.0, 0.2, 20.0, s)
            tt, rec, xinf = evolve(met, ell, T=a.T)
            w, sr, si = prompt_ringdown(tt, rec, PAPER[(ell, s)])
            ref = PAPER[(ell, s)]
            print("  ell=%d sqrtθ=%.2f  t_echo=2(x_inf-x_peak)=%.1f | prompt ringdown %.4f%+.4fi (±%.4f,±%.4f) | "
                  "WKB6(manuscript) %.4f%+.4fi | diff Re %+.2f%%  Im %+.2f%%" % (
                      ell, s, 2 * xinf, w.real, w.imag, sr, si, ref.real, ref.imag,
                      100 * (w.real - ref.real) / ref.real, 100 * (w.imag - ref.imag) / ref.imag))
            if s in (0.0, 0.15):
                m = tt > 150
                y, dt = rec[m], tt[1] - tt[0]
                ww = matrix_pencil(y[::20], dt * 20, 16)
                ww = ww[(np.real(ww) > 0.2) & (np.real(ww) < 0.75) & (np.imag(ww) < 0.02)]
                ww = ww[np.argsort(np.real(ww))]
                if ell == 2:
                    print("      late-time (t>150) cavity modes: " + "  ".join("%.4f%+.4fi" % (x.real, x.imag) for x in ww[:7]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["benchmark", "wkb", "spectral", "timedomain", "all"])
    ap.add_argument("--l", type=float, default=20.0, help="AdS length (default 20, as in the manuscript)")
    ap.add_argument("--nmodes", type=int, default=5)
    ap.add_argument("--T", type=float, default=1500.0, help="time-domain final time")
    a = ap.parse_args()
    todo = ["benchmark", "wkb", "spectral", "timedomain"] if a.cmd == "all" else [a.cmd]
    for c in todo:
        globals()["cmd_" + c](a)
        print()
