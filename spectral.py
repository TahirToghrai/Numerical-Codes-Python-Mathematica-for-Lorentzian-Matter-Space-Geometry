"""
Méthode B -- collocation spectrale de Chebyshev (indépendante de la méthode A).

Variable u = 1/r in [0, u_h].  Inconnue ζ(u) = e^{iω r_*} Φ, régulière à l'horizon.
Avec  ρ̂ = u·rth,  P̂ = u^4 rth^2 f :

  u P̂ ζ'' + [u P̂' - 2 P̂ + 2iω u ρ̂^2] ζ' - [2iω (ρ̂^2 - u ρ̂ ρ̂') + u L] ζ = 0

Linéaire en ω => problème généralisé  A0 ζ = ω (-A1) ζ.
  u = u_h (horizon) : la ligne de l'EDO impose la régularité.
  u = 0  (bord AdS) : ligne remplacée par Dirichlet ζ(0)=0  (mode normalisable, Delta=3).
"""
import numpy as np
import sympy as sp
import scipy.linalg as sla
from model import build, horizon

u = sp.symbols('u', positive=True)


def coeff_functions(s, M=1.0, Q=0.2, l=20.0):
    if s == 0:
        rho = sp.Integer(1)
        m = sp.Integer(1)
    else:
        S = sp.Float(s)
        rho = 1 - 2 / sp.pi * sp.atan(S * u)
        m = 1 - 2 / sp.pi * (sp.atan(S * u) + S * u / (1 + S**2 * u**2))
    Ph = rho**4 / sp.Float(l)**2 + u**2 * rho**2 - 2 * M * m * u**3 * rho + Q**2 * m**2 * u**4
    fn = lambda e: sp.lambdify(u, e, 'numpy')
    return dict(
        rho=fn(rho), drho=fn(sp.diff(rho, u)),
        Ph=fn(Ph), dPh=fn(sp.diff(Ph, u)),
    )


def cheb(N):
    """Matrice de dérivation de Chebyshev (Trefethen), nœuds x_j = cos(pi j/N)."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def spectrum(s, L, N=80, M=1.0, Q=0.2, l=20.0, rh=None):
    mod = build(s, M, Q, l)
    if rh is None:
        rh = max(horizon(mod))
    uh = 1.0 / rh
    D, x = cheb(N)
    uu = uh * (1 + x) / 2          # x=+1 -> horizon (u=uh), x=-1 -> bord (u=0)
    D1 = D * (2 / uh)
    D2 = D1 @ D1
    cf = coeff_functions(s, M, Q, l)
    rho, drho = cf['rho'](uu) * np.ones_like(uu), cf['drho'](uu) * np.ones_like(uu)
    Ph, dPh = cf['Ph'](uu), cf['dPh'](uu)
    LL = L * (L + 1)
    A0 = (np.diag(uu * Ph) @ D2 + np.diag(uu * dPh - 2 * Ph) @ D1 - np.diag(uu * LL))
    A1 = 2j * (np.diag(uu * rho**2) @ D1 - np.diag(rho**2 - uu * rho * drho))
    A0 = A0.astype(complex)
    # bord AdS : indice où u = 0  (x=-1 -> dernier nœud)
    ib = int(np.argmin(np.abs(uu)))
    A0[ib, :] = 0.0
    A0[ib, ib] = 1.0
    A1[ib, :] = 0.0
    w = sla.eig(A0, -A1, right=False)
    w = w[np.isfinite(w)]
    return w, rh


def clean(w1, w2, tol=1e-6, box=(-1.0, 3.0, -3.0, 0.5)):
    """Garde les valeurs propres stables entre deux résolutions (N et N')."""
    out = []
    for a in w1:
        if not (box[0] < a.real < box[1] and box[2] < a.imag < box[3]):
            continue
        d = np.min(np.abs(w2 - a))
        if d < tol * max(1.0, abs(a)):
            out.append(a)
    return np.array(sorted(out, key=lambda z: (-z.imag, z.real)))
