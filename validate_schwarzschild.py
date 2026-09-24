import sympy as sp
from wkb_full import lambda2, lambda3, wkb_omega

rs, Ms = sp.symbols('rs Ms', positive=True)

def schw_scalar_derivs(ell_val, r0_guess, digits=40):
    fs = 1 - 2/rs
    Vs = fs*( ell_val*(ell_val+1)/rs**2 + 2/rs**3 )
    dVs = sp.diff(Vs, rs)
    rpeak = sp.nsolve(dVs, rs, sp.Float(r0_guess), prec=digits)
    def D(expr):
        return sp.expand(fs*sp.diff(expr, rs))
    derivs = [Vs]
    cur = Vs
    for k in range(6):
        cur = D(cur)
        derivs.append(cur)
    rpeak_f = sp.Float(rpeak, digits)
    vals = [sp.N(d.subs(rs, rpeak_f), digits) for d in derivs]
    return vals, rpeak

# Konoplya (2003/2004) Table 1 reference values (scalar, s=0), "3rd order WKB" column
reference = {
    (0,0): (0.1046, -0.1152, 2.0),   # (Re, Im, guess r0)
    (1,0): (0.2911, -0.0980, 3.5),
    (1,1): (0.2622, -0.3074, 3.5),
    (2,0): (0.4832, -0.0968, 3.0),
    (2,1): (0.4632, -0.2958, 3.0),
    (2,2): (0.4317, -0.5034, 3.0),
}

print(f"{'(l,n)':>8} | {'order1 (SW)':>20} | {'order3 (IW, mine)':>22} | {'Konoplya 3rd order (paper)':>28}")
for (ell,n), (ReRef, ImRef, guess) in reference.items():
    vals, rpeak = schw_scalar_derivs(ell, guess)
    V0,V1,V2,V3,V4,V5,V6 = vals
    om1,_,_ = wkb_omega(V0,V2,V3,V4,V5,V6, n=n, order=1, digits=40)
    om3,L2,L3 = wkb_omega(V0,V2,V3,V4,V5,V6, n=n, order=3, digits=40)
    om1 = sp.N(om1,8); om3 = sp.N(om3,8)
    print(f"({ell},{n})     | {str(om1):>20} | {str(om3):>22} | {ReRef}-{-ImRef}i")
