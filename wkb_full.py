import sympy as sp
import time

r, M, Q, l, theta, ell = sp.symbols('r M Q l theta ell', positive=True)

def build_Veff_symbolic():
    sqth = sp.sqrt(theta)
    Mth = (2*M/sp.pi)*(sp.atan(r/sqth) - r*sqth/(r**2+theta))
    Qth = (2*Q/sp.pi)*(sp.atan(r/sqth) - r*sqth/(r**2+theta))
    rth = (2*r/sp.pi)*sp.atan(r/sqth)
    f = 1 - 2*Mth/rth + Qth**2/rth**2 + rth**2/l**2
    fprime = sp.diff(f, r)
    drth = sp.diff(rth, r)
    d2rth = sp.diff(rth, r, 2)
    Veff = f*( ell*(ell+1)/rth**2 + (1/rth)*(fprime*drth + f*d2rth) )
    return f, Veff

f_sym, Veff_sym = build_Veff_symbolic()

def prec_subs(Mv, Qv, lv, thetav, ellv):
    """Substitute numeric (exact rational) parameters, keep r symbolic."""
    subs = {M: sp.Rational(Mv), Q: sp.nsimplify(Qv, rational=True),
            l: sp.Rational(lv), theta: sp.nsimplify(thetav, rational=True),
            ell: sp.Integer(ellv)}
    fN = sp.simplify(f_sym.subs(subs))
    VN = Veff_sym.subs(subs)
    return fN, VN

def find_horizon(fN, guess):
    return sp.nsolve(fN, r, guess, prec=40)

def find_peak(VN, guess):
    dV = sp.diff(VN, r)
    rpeak = sp.nsolve(dV, r, guess, prec=40)
    return rpeak

def D_operator(expr, fN):
    """d/dr_* = f(r) d/dr"""
    return sp.expand(fN*sp.diff(expr, r))

def derivatives_at_peak(VN, fN, rpeak, order=6, digits=40):
    """Return [V0, V0'' , V0''', ..., V0^(order)] wrt r_* at r=rpeak (V0' should be ~0)."""
    derivs = [VN]
    cur = VN
    for k in range(order):
        cur = D_operator(cur, fN)
        derivs.append(cur)
    # evaluate numerically
    rpeak_f = sp.Float(rpeak, digits)
    vals = [sp.N(d.subs(r, rpeak_f), digits) for d in derivs]
    return vals  # vals[0]=V0, vals[1]=V0', vals[2]=V0'', ..., vals[order]=V0^(order)

def lambda2(V2, V3, V4, nu):
    return (1/sp.sqrt(-2*V2))*( sp.Rational(1,8)*(V4/V2)*(sp.Rational(1,4)+nu**2)
             - sp.Rational(1,288)*(V3/V2)**2*(7+60*nu**2) )

def lambda3(V2, V3, V4, V5, V6, nu):
    return (1/(-2*V2))*(
        sp.Rational(5,6912)*(V3/V2)**4*(77+188*nu**2)
        - sp.Rational(1,384)*((V3**2*V4)/V2**3)*(51+100*nu**2)
        + sp.Rational(1,2304)*(V4/V2)**2*(67+68*nu**2)
        + sp.Rational(1,288)*((V3*V5)/V2**2)*(19+28*nu**2)
        + sp.Rational(1,288)*(V6/V2)*(5+4*nu**2)
    )

def wkb_omega(V0, V2, V3, V4, V5, V6, n, order=3, digits=40):
    nu = sp.Float(n,digits) + sp.Rational(1,2)
    sq = sp.sqrt(-2*V2)
    L2 = lambda2(V2,V3,V4,nu) if order>=2 else 0
    L3 = lambda3(V2,V3,V4,V5,V6,nu) if order>=3 else 0
    omega2 = V0 - sp.I*sq*(nu+L2+L3)
    omega2 = sp.N(omega2, digits)
    omega = sp.sqrt(omega2)
    # principal branch: Re>0
    omega = sp.N(omega, digits)
    if sp.re(omega) < 0:
        omega = -omega
    return omega, L2, L3

if __name__ == "__main__":
    t0 = time.time()
    # ---- Schwarzschild benchmark: Regge-Wheeler potential for scalar field, M=1 ----
    # V(r) = f(r) [ ell(ell+1)/r^2 + (1-s^2) 2M/r^3 ], f=1-2M/r, s=0 (scalar)
    rs, Ms = sp.symbols('rs Ms', positive=True)
    fs = 1 - 2*Ms/rs
    ells = sp.Integer(2)
    Vs = fs*( ells*(ells+1)/rs**2 + 2*Ms/rs**3 )
    Vs = Vs.subs(Ms, 1)
    fs = fs.subs(Ms, 1)
    dVs = sp.diff(Vs, rs)
    rpeak_s = sp.nsolve(dVs, rs, sp.Float(3.0), prec=40)
    print("Schwarzschild ell=2 peak r0 =", rpeak_s, " (expected r0=3M)")

    def D_op_s(expr):
        return sp.expand(fs*sp.diff(expr, rs))

    derivs_s = [Vs]
    cur = Vs
    for k in range(6):
        cur = D_op_s(cur)
        derivs_s.append(cur)
    rpeak_sf = sp.Float(rpeak_s, 40)
    vals_s = [sp.N(d.subs(rs, rpeak_sf), 40) for d in derivs_s]
    for i,v in enumerate(vals_s):
        print(f"  V0^({i}) = {v}")

    V0,V1,V2,V3,V4,V5,V6 = vals_s
    for order in [1,2,3]:
        om, L2, L3 = wkb_omega(V0,V2,V3,V4,V5,V6, n=0, order=order, digits=40)
        print(f"order={order}: omega = {sp.N(om,15)}   (Lambda2={sp.N(L2,6) if L2!=0 else 0}, Lambda3={sp.N(L3,6) if L3!=0 else 0})")
    print("Reference (scalar, Schwarzschild, ell=2,n=0, M=1): omega ~ 0.483644 - 0.096759 i (Berti-Cardoso-Starinets tables)")
    print("elapsed:", time.time()-t0)
