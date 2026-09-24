import sympy as sp
from wkb_full import lambda2, lambda3, wkb_omega

x, V0s, alpha = sp.symbols('x V0s alpha', positive=True)

V = V0s / sp.cosh(alpha*x)**2

derivs = [V]
cur = V
for k in range(6):
    cur = sp.diff(cur, x)
    derivs.append(cur)

vals0 = [sp.simplify(d.subs(x,0)) for d in derivs]
print("Derivatives of Poschl-Teller at x=0 (symbolic):")
for i,v in enumerate(vals0):
    print(f"  V^({i})(0) =", v)

# numeric test: pick V0=1, alpha=0.5
subs = {V0s: 1, alpha: sp.Rational(1,2)}
numvals = [sp.N(v.subs(subs), 40) for v in vals0]
V0n,V1n,V2n,V3n,V4n,V5n,V6n = numvals
print("\nNumeric (V0=1, alpha=0.5):", numvals)

alpha_n = sp.N(sp.Rational(1,2), 40)
V0_n = sp.N(1, 40)

print(f"\n{'n':>3} | {'order1':>22} | {'order2':>22} | {'order3':>22} | {'EXACT':>22}")
for n in range(4):
    om1,_,_ = wkb_omega(V0n,V2n,V3n,V4n,V5n,V6n, n=n, order=1, digits=40)
    om2,_,_ = wkb_omega(V0n,V2n,V3n,V4n,V5n,V6n, n=n, order=2, digits=40)
    om3,_,_ = wkb_omega(V0n,V2n,V3n,V4n,V5n,V6n, n=n, order=3, digits=40)
    nu = n + sp.Rational(1,2)
    exact = sp.sqrt(V0_n - alpha_n**2/4) - sp.I*alpha_n*nu
    print(f"{n:>3} | {str(sp.N(om1,10)):>22} | {str(sp.N(om2,10)):>22} | {str(sp.N(om3,10)):>22} | {str(sp.N(exact,10)):>22}")
