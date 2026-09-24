import numpy as np, json, time, sys
from td import *

X0S=[-6.0,0.0,6.0]; SIGS=[1.5,2.0,3.0]; A1=[14,18,22]; A2=[40,46]; XO=3.0

def run(s, L):
    x,V,xmax,rp,rh = make_grid(s,L,h=0.02)
    res={}
    for x0 in X0S:
        for sig in SIGS:
            t,rec = evolve(x,V,T=100.0,x0=x0,sigma=sig,xobs=[XO])
            tb=abs(x0); techo=2*xmax-x0-XO
            for a1 in A1:
                for a2 in A2:
                    t1=tb+a1; t2=min(tb+a2,techo-6)
                    if t2-t1<18: continue
                    w,c=ringdown_fit(t,rec[:,0],t1,t2,K=2)
                    wsel = w[np.argmax(w.real)]

                    # >>> AJOUT ICI : voir les DEUX racines avant sélection <
                    if L==0 and s==0.0 and abs(wsel.real) <= TOL:
                        print(f"  [DEBUG] config x0={x0} sig={sig} a1={a1} a2={a2}: w (les 2 racines) = {w!r}, c = {c!r}")
                    # <<< fin de l'ajout >>>

                    res[(x0,sig,a1,a2)] = wsel
    return res, xmax
TOL = 1e-3  # bien en dessous de l'echelle typique de Re(w)~0.1-0.5

allres={}
t0=time.time()
for L in [1,2,0]:
    for s in [0.0,0.05,0.10,0.15]:
        allres[(L,s)],xmax = run(s,L)
         # >>> AJOUT ICI, juste après avoir rempli allres[(L,s)], avant le filtrage <
        for k, v in allres[(L,s)].items():
            if abs(v.real) <= TOL:
                print(f"  [DEBUG] ell={L} s={s:4.2f}: clé fautive {k} -> valeur brute = {v!r}")
        # <<< fin de l'ajout >>
        vals = np.array(list(allres[(L,s)].values()))
        good = np.abs(vals.real) > TOL
        r = vals[good]
        n_bad = int((~good).sum())
        tag = f"  [{n_bad} degenerate fit(s) dropped]" if n_bad else ""
        print(f"ell={L} s={s:4.2f}: n={len(r)}/{len(vals)}  Re={r.real.mean():.4f}+-{r.real.std():.4f}  Im={r.imag.mean():.4f}+-{r.imag.std():.4f}   ({time.time()-t0:.0f}s){tag}",flush=True)

print("\nDécalage relatif theta=0 -> theta (configuration par configuration)")
out={}
for L in [1,2,0]:
    base=allres[(L,0.0)]
    for s in [0.05,0.10,0.15]:
        keys=[k for k in base if k in allres[(L,s)]
              and abs(base[k].real) > TOL
              and abs(allres[(L,s)][k].real) > TOL]
        n_all=len([k for k in base if k in allres[(L,s)]])
        dR=np.array([allres[(L,s)][k].real/base[k].real-1 for k in keys])
        dI=np.array([allres[(L,s)][k].imag/base[k].imag-1 for k in keys])
        out[(L,s)]=(dR.mean(),dR.std(),dI.mean(),dI.std(),len(keys))
        print(f"  ell={L} s={s:4.2f}: d(wR)/wR = {100*dR.mean():+.2f}% +- {100*dR.std():.2f}%   d|wI|/|wI| = {100*dI.mean():+.2f}% +- {100*dI.std():.2f}%   (n={len(keys)}/{n_all})",flush=True)
json.dump({f"{k[0]}_{k[1]}":[complex(v).real for v in allres[k].values()]+[complex(v).imag for v in allres[k].values()] for k in allres}, open('td_protocol.json','w'))
json.dump({f"{k[0]}_{k[1]}":v for k,v in out.items()}, open('td_shifts.json','w'))
