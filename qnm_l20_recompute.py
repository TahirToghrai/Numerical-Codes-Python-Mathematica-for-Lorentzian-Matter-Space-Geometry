#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalcul des modes quasi-normaux scalaires (WKB 6e ordre + Pade) pour le fond
RN-AdS a deformation lorentzienne de l'article:

    f(r)  = 1 - 2 M_th(r)/r_th(r) + Q_th(r)^2/r_th(r)^2 + r_th(r)^2/l^2
    r_th  = (2r/pi) arctan(r/sqrt(theta))
    M_th  = (2M/pi) [arctan(r/sqrt(theta)) - r sqrt(theta)/(r^2+theta)],  Q_th = (Q/M) M_th
    V_eff = f [ L(L+1)/r_th^2 + (f' r_th' + f r_th'')/r_th ],   dr* = dr/f

Parametres: M=1, Q=0.2, l=20 ; sqrt(theta) in {0, 0.05, 0.10, 0.15}.

Methode: WKB a l'ordre 6 (somme Lambda_2..Lambda_6, derivees de V en r* jusqu'a
l'ordre 12) obtenu par theorie des perturbations de Rayleigh-Schrodinger autour
du sommet de la barriere (oscillateur harmonique inverse, rotation complexe).
Validation: Schwarzschild M=1, scalaire l=1 n=0 : 0.29291-0.09776i (exact 0.29294-0.09766i).
Pade: approximants P_{m/m'} (m+m'=6) de la serie en epsilon de omega^2
(Konoplya-Zhidenko-Zinhailo 2019), evalues a epsilon=1.

Usage:  python3 qnm_l20_recompute.py            -> tableau, convergence, Pade, JSON
        python3 qnm_l20_recompute.py --figure    -> + qnm_frequencies.eps / .png
Dependances: mpmath (obligatoire), matplotlib (figure).
"""
import json
import sys
import mpmath as mp

mp.mp.dps = 60
NS = 12          # ordre de Taylor (V^(12) pour le WKB 6e ordre)
M0, Q0, L_ADS = 1, mp.mpf("0.2"), 20
SQRT_THETAS = ["0.00", "0.05", "0.10", "0.15"]


# ------------------------------------------------------------------ series
def smul(a, b, N=NS):
    c = [mp.mpf(0)] * (N + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if i + j > N:
                break
            c[i + j] += ai * bj
    return c


def compose(P, s, N=NS):
    res = [mp.mpf(0)] * (N + 1)
    sp = [mp.mpf(1)] + [mp.mpf(0)] * N
    for k in range(N + 1):
        if k > 0:
            sp = smul(sp, s, N)
        for i in range(N + 1):
            res[i] += P[k] * sp[i]
    return res


def revert(a, N=NS):
    d = [mp.mpf(0)] * (N + 1)
    d[1] = 1 / a[1]
    for _ in range(N + 2):
        acc = [mp.mpf(0)] * (N + 1)
        dp = d[:]
        for j in range(2, N + 1):
            dp = smul(d, d, N) if j == 2 else smul(dp, d, N)
            for i in range(N + 1):
                acc[i] += a[j] * dp[i]
        new = [mp.mpf(0)] * (N + 1)
        new[1] = 1 / a[1]
        for i in range(N + 1):
            new[i] -= acc[i] / a[1]
        d = new
    return d


# ------------------------------------------------------------------ WKB (RSPT)
def rspt_energies(c, n, gmax=10):
    """c[k] = V^(k)(x0)/k!, variable y = x - x0. Renvoie E_j (j=0,...,gmax) avec
    omega^2 = V0 - i * sum_j E_j ."""
    a = mp.sqrt(-c[2])
    Nb = n + 3 * gmax + 12
    S = mp.zeros(Nb, Nb)
    for k in range(Nb - 1):
        v = mp.sqrt(k + 1) / mp.sqrt(2 * a)
        S[k, k + 1] = v
        S[k + 1, k] = v
    Sp = {0: mp.eye(Nb)}
    for k in range(1, gmax + 3):
        Sp[k] = Sp[k - 1] * S
    W = {m: 1j * c[m + 2] * mp.expjpi(mp.mpf(m + 2) / 4) * Sp[m + 2] for m in range(1, gmax + 1)}
    E = {0: a * (2 * n + 1)}
    psi = {0: mp.matrix([1 if i == n else 0 for i in range(Nb)])}
    for j in range(1, gmax + 1):
        E[j] = sum((W[m] * psi[j - m])[n] for m in range(1, j + 1))
        rhs = mp.zeros(Nb, 1)
        for m in range(1, j + 1):
            rhs += E[m] * psi[j - m] - W[m] * psi[j - m]
        pj = mp.zeros(Nb, 1)
        for k in range(Nb):
            if k != n:
                pj[k] = rhs[k] / (2 * a * (k - n))
        psi[j] = pj
    return E


def _w_from_w2(w2):
    w = mp.sqrt(w2)
    return -w if mp.re(w) < 0 else w


def qnm(Vr, fr, r_guess, n):
    """Renvoie dict: r0, WKB ordres 1..6, Pade (3,3),(2,4),(4,2)."""
    r0 = mp.findroot(lambda r: mp.diff(Vr, r), r_guess)
    tV = mp.taylor(lambda d: Vr(r0 + d), 0, NS)
    tg = mp.taylor(lambda d: 1 / fr(r0 + d), 0, NS - 1)
    y_of_d = [mp.mpf(0)] + [tg[j] / (j + 1) for j in range(NS)]
    Vy = compose(tV, revert(y_of_d))
    E = rspt_energies(Vy, n)
    V0 = Vy[0]
    js = (0, 2, 4, 6, 8, 10)
    out = {"r0": r0, "wkb": {}, "pade": {}}
    lam = mp.mpf(0)
    for idx, j in enumerate(js):
        lam += E[j]
        out["wkb"][idx + 1] = _w_from_w2(V0 - 1j * lam)
    coeffs = [V0] + [-1j * E[j] for j in js]          # omega^2(eps) = sum c_k eps^k
    for (Lp, Mp) in ((3, 3), (2, 4), (4, 2)):
        p, q = mp.pade(coeffs, Lp, Mp)
        num = sum(p[k] for k in range(len(p)))
        den = sum(q[k] for k in range(len(q)))
        out["pade"][f"{Lp}/{Mp}"] = _w_from_w2(num / den)
    return out


# ------------------------------------------------------------------ fond de l'article
def background(sqrt_theta, M=M0, Q=Q0, l=L_ADS):
    st = mp.mpf(sqrt_theta)
    pi, th = mp.pi, st ** 2
    if st == 0:
        rth = lambda r: r
        drth = lambda r: mp.mpf(1)
        d2rth = lambda r: mp.mpf(0)
        Mt = lambda r: mp.mpf(M)
        dMt = lambda r: mp.mpf(0)
    else:
        rth = lambda r: 2 * r / pi * mp.atan(r / st)
        drth = lambda r: 2 / pi * (mp.atan(r / st) + r * st / (r ** 2 + th))
        d2rth = lambda r: 4 * st ** 3 / (pi * (r ** 2 + th) ** 2)
        Mt = lambda r: 2 * M / pi * (mp.atan(r / st) - r * st / (r ** 2 + th))
        dMt = lambda r: 4 * M * r ** 2 * st / (pi * (r ** 2 + th) ** 2)
    Qt = lambda r: Q / M * Mt(r)
    dQt = lambda r: Q / M * dMt(r)

    def f(r):
        R = rth(r)
        return 1 - 2 * Mt(r) / R + Qt(r) ** 2 / R ** 2 + R ** 2 / l ** 2

    def df(r):
        R, dR = rth(r), drth(r)
        return (-2 * dMt(r) / R + 2 * Mt(r) * dR / R ** 2 + 2 * Qt(r) * dQt(r) / R ** 2
                - 2 * Qt(r) ** 2 * dR / R ** 3 + 2 * R * dR / l ** 2)

    def V(r, L):
        R = rth(r)
        return f(r) * (L * (L + 1) / R ** 2 + (df(r) * drth(r) + f(r) * d2rth(r)) / R)

    rh = mp.findroot(f, 1.9)
    return f, V, rh, rth, Mt


def compute(sqrt_theta, ell, n):
    f, V, rh, rth, Mt = background(sqrt_theta)
    res = qnm(lambda r: V(r, ell), f, 1.45 * rh, n)
    res["rh"] = rh
    res["rth_peak"] = rth(res["r0"])
    return res


def fmt(w, d=4):
    return f"{float(mp.re(w)):.{d}f}-{-float(mp.im(w)):.{d}f}i"


# ------------------------------------------------------------------ main
def main(make_figure=False):
    modes = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]            # (n, ell)
    data = {}
    for s in SQRT_THETAS:
        for (n, ell) in modes:
            r = compute(s, ell, n)
            data[(s, n, ell)] = r
            print(f"sqrt(theta)={s} n={n} l={ell}: r_h={float(r['rh']):.4f} r_peak={float(r['r0']):.4f} "
                  f"WKB6={fmt(r['wkb'][6])}  WKB5={fmt(r['wkb'][5])}  Pade33={fmt(r['pade']['3/3'])}",
                  flush=True)

    # ---- resume ----
    def rel(a, b):
        return float(abs(a - b) / abs(b))

    summary = {}
    for (s, n, ell), r in data.items():
        summary[f"{s}|{n}|{ell}"] = {
            "wR": float(mp.re(r["wkb"][6])), "wI": -float(mp.im(r["wkb"][6])),
            "d56": rel(r["wkb"][6], r["wkb"][5]), "d36": rel(r["wkb"][3], r["wkb"][6]),
            "pade33": rel(r["pade"]["3/3"], r["wkb"][6]),
            "pade24": rel(r["pade"]["2/4"], r["wkb"][6]),
            "pade42": rel(r["pade"]["4/2"], r["wkb"][6]),
            "rh": float(r["rh"]), "r_peak": float(r["r0"]),
        }
    json.dump(summary, open("qnm_l20_results.json", "w"), indent=1)

    if make_figure:
        make_fig()


def make_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = ["0.000", "0.025", "0.050", "0.075", "0.100", "0.125", "0.150"]
    curves = {}
    for ell in (1, 2):
        for n in (0, 1):
            xs, wr, wi = [], [], []
            for s in grid:
                w = compute(s, ell, n)["wkb"][6]
                xs.append(float(s)); wr.append(float(mp.re(w))); wi.append(-float(mp.im(w)))
            curves[(n, ell)] = (xs, wr, wi)
            print("fig", n, ell, flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    col = {1: "tab:blue", 2: "tab:red"}
    for (n, ell), (xs, wr, wi) in curves.items():
        ls = "-" if n == 0 else "--"
        lab = rf"$\ell={ell},\,n={n}$"
        ax[0].plot(xs, wr, ls, color=col[ell], marker="o", ms=3.5, label=lab)
        ax[1].plot(xs, wi, ls, color=col[ell], marker="o", ms=3.5, label=lab)
    ax[0].set_ylabel(r"$\omega_R$")
    ax[1].set_ylabel(r"$|\omega_I|$")
    for a in ax:
        a.set_xlabel(r"$\sqrt{\theta}$")
        a.grid(alpha=0.3)
        a.set_xlim(-0.003, 0.153)
    ax[0].legend(fontsize=8, ncol=2, loc="center right")
    ax[0].set_title(r"$M=1,\ Q=0.2,\ l=20$", fontsize=9)
    fig.tight_layout()
    fig.savefig("qnm_frequencies.eps")
    fig.savefig("qnm_frequencies.png", dpi=150)


if __name__ == "__main__":
    main("--figure" in sys.argv)
