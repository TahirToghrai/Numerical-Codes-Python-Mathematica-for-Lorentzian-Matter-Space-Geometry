"""
Modèle RN-AdS avec déformation lorentzienne (Manuscrit, Eqs. eq:Mtheta, eq:Qtheta,
eq:rtheta, eq:f, eq:pot_explicit).  Unités M=1.

  ds^2 = -f dt^2 + dr^2/f + rth(r)^2 dOmega^2
  f    = 1 - 2 Mth/rth + Qth^2/rth^2 + rth^2/l^2
  Mth  = (2M/pi)[atan(r/s) - r s/(r^2+s^2)],  Qth = (Q/M) Mth,  s = sqrt(theta)
  rth  = (2 r/pi) atan(r/s)
"""
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.integrate import quad

r = sp.symbols('r', positive=True)


def build(s, M=1.0, Q=0.2, l=20.0):
    """Renvoie un dict de fonctions numériques (numpy) pour (s=sqrt(theta), M, Q, l)."""
    if s == 0:
        rt = r
        Mt = sp.Float(M)
        Qt = sp.Float(Q)
    else:
        S = sp.Float(s)
        rt = 2 * r / sp.pi * sp.atan(r / S)
        Mt = 2 * M / sp.pi * (sp.atan(r / S) - r * S / (r**2 + S**2))
        Qt = Q / M * Mt
    f = 1 - 2 * Mt / rt + Qt**2 / rt**2 + rt**2 / sp.Float(l)**2
    drt = sp.diff(rt, r)
    d2rt = sp.diff(rt, r, 2)
    fp = sp.diff(f, r)
    P = rt**2 * f                      # rth^2 f
    Pp = sp.diff(P, r)
    lam = lambda e: sp.lambdify(r, e, 'numpy')
    return dict(
        f=lam(f), fp=lam(fp), fpp=lam(sp.diff(f, r, 2)),
        rt=lam(rt), drt=lam(drt), d2rt=lam(d2rt),
        P=lam(P), Pp=lam(Pp), Ppp=lam(sp.diff(P, r, 2)),
        rt_expr=rt, f_expr=f, drt_expr=drt, d2rt_expr=d2rt,
        s=s, M=M, Q=Q, l=l,
    )


def V_eff(mod, rr, L):
    """Potentiel effectif, Eq. (eq:pot_explicit), L = ell."""
    f, fp = mod['f'](rr), mod['fp'](rr)
    rt, drt, d2rt = mod['rt'](rr), mod['drt'](rr), mod['d2rt'](rr)
    return f * (L * (L + 1) / rt**2 + (fp * drt + f * d2rt) / rt)


def horizon(mod, lo=0.05, hi=None):
    """Racines positives de f (la plus grande = horizon des événements)."""
    f = mod['f']
    hi = hi or 5e3
    grid = np.geomspace(lo, hi, 60000)
    vals = f(grid)
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    roots = [brentq(f, grid[i], grid[i + 1], xtol=1e-15) for i in idx]
    return roots


def peak(mod, rh, L, rmax=None):
    """Position et valeur du maximum de V_eff à l'extérieur de l'horizon."""
    from scipy.optimize import minimize_scalar
    l = mod['l']
    rmax = rmax or 0.5 * l
    grid = np.linspace(rh * 1.0005, rmax, 20000)
    V = V_eff(mod, grid, L)
    i = np.argmax(V[: len(grid) // 2])
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: -V_eff(mod, x, L), bounds=(lo, hi), method='bounded',
                          options=dict(xatol=1e-13))
    return res.x, -res.fun


def tortoise_gap(mod, rh, r0):
    """Delta r_* = r_*^max - r_*(r0) = int_{r0}^inf dr/f  (via u = 1/r)."""
    f = mod['f']
    g = lambda u: 1.0 / (f(1.0 / u) * u**2)
    val, err = quad(g, 0.0, 1.0 / r0, epsabs=1e-13, epsrel=1e-13, limit=400)
    return val
