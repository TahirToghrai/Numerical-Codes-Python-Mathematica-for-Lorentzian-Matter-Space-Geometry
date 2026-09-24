"""
Méthode A -- conditions de Horowitz-Hubeny évaluées par intégration numérique.

Φ = e^{-iωr_*} ζ(r) Y e^{-iωt}   (ζ régulière à l'horizon = condition entrante)

    P ζ'' + [P' - 2iω rth^2] ζ' - [2iω rth rth' + L] ζ = 0 ,   P = rth^2 f ,  L = ell(ell+1)

  * horizon r_h : point singulier régulier ; on part de la série de Frobenius
    ζ = 1 + c1 x + c2 x^2 + ..., x = r - r_h  (entrante par construction).
  * bord AdS    : exposants 0 et 3 en u = 1/r ; la condition de normalisabilité
    (Dirichlet, Delta=3) est  ζ(r -> infini) = 0.
Les QNM sont les zéros de  F(ω) = ζ(r_max; ω).
"""
import numpy as np
from scipy.integrate import solve_ivp
import mpmath as mp
from model import build, horizon


class Shooter:
    def __init__(self, s, L, M=1.0, Q=0.2, l=20.0, eps=1e-4, rmax=1e4):
        self.mod = build(s, M, Q, l)
        self.L = L
        self.rh = max(horizon(self.mod))
        self.eps = eps
        self.rmax = rmax
        m = self.mod
        rh = self.rh
        self.p1 = float(m['Pp'](rh))
        self.p2 = float(m['Ppp'](rh)) / 2.0
        self.rth_h = float(m['rt'](rh))
        self.drt_h = float(m['drt'](rh))
        self.d2rt_h = float(m['d2rt'](rh))

    def _rhs(self, rr, y, w):
        m = self.mod
        P, Pp = m['P'](rr), m['Pp'](rr)
        rt, drt = m['rt'](rr), m['drt'](rr)
        E = Pp - 2j * w * rt**2
        G = 2j * w * rt * drt + self.L * (self.L + 1)
        z, dz = y
        return [dz, (G * z - E * dz) / P]

    def initial(self, w):
        rt, drt, d2rt = self.rth_h, self.drt_h, self.d2rt_h
        L = self.L * (self.L + 1)
        e0 = self.p1 - 2j * w * rt**2
        e1 = 2 * self.p2 - 4j * w * rt * drt
        g0 = 2j * w * rt * drt + L
        g1 = 2j * w * (drt**2 + rt * d2rt)
        c1 = g0 / e0
        c2 = (g0 * c1 + g1 - e1 * c1) / (2 * (self.p1 + e0))
        x = self.eps
        return 1 + c1 * x + c2 * x**2, c1 + 2 * c2 * x

    def F(self, w, rtol=1e-12, atol=1e-14, return_sol=False):
        w = complex(w)
        z0, dz0 = self.initial(w)
        sol = solve_ivp(self._rhs, [self.rh + self.eps, self.rmax], [z0 + 0j, dz0 + 0j],
                        method='DOP853', rtol=rtol, atol=atol, args=(w,))
        if return_sol:
            return sol
        return sol.y[0, -1]

    def root(self, w0, tol=1e-12):
        f = lambda w: self.F(w)
        w = mp.findroot(lambda z: f(complex(z)), mp.mpc(w0), solver='muller', tol=tol,
                        maxsteps=60, verify=False)
        return complex(w)
