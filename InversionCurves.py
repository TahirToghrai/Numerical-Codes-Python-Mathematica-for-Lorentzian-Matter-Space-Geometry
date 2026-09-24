import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

pi = np.pi

# ============================================================
# Fonctions de l'article (inchangées)
# ============================================================
def r_theta(rh, sqrt_theta):
    if sqrt_theta == 0:
        return rh
    return (2*rh/pi) * np.arctan(rh / sqrt_theta)

def Q_theta(rh, sqrt_theta, Q_val):
    if sqrt_theta == 0:
        return Q_val
    x = rh / sqrt_theta
    return (2*Q_val/pi) * (np.arctan(x) - x/(1+x**2))

def chi_theta(rh, sqrt_theta, Q_val):
    if sqrt_theta == 0:
        return 0.0
    x = rh / sqrt_theta
    dQdr = (4*Q_val*rh**2*sqrt_theta) / (pi*(rh**2 + sqrt_theta**2)**2)
    dRdr = (2/pi)*np.arctan(x) + (2*rh*sqrt_theta)/(pi*(rh**2 + sqrt_theta**2))
    return dQdr / dRdr

def g_rho(rho, Qth, chi):
    return -1/(8*pi*rho**2) + Qth**2/(8*pi*rho**4) - Qth*chi/(4*pi*rho**3)

def g_prime(rho, Qth, chi):
    return 1/(4*pi*rho**3) - Qth**2/(2*pi*rho**5) + 3*Qth*chi/(4*pi*rho**4)

def T_inv_formula(rho, Qth, chi):
    return -rho**2 * g_prime(rho, Qth, chi)

def P_inv_formula(rho, Qth, chi):
    T = T_inv_formula(rho, Qth, chi)
    return T/(2*rho) + g_rho(rho, Qth, chi)

# ============================================================
# Calcul de la courbe d'inversion complète
# ============================================================
def get_inversion_curve(Q_val, sqrt_theta, rh_max=3.0, n=10000):
    rh = np.linspace(0.001, rh_max, n)
    T_list, P_list, rho_list = [], [], []
    for r in rh:
        rho = r_theta(r, sqrt_theta)
        Qth = Q_theta(r, sqrt_theta, Q_val)
        chi = chi_theta(r, sqrt_theta, Q_val)
        T = T_inv_formula(rho, Qth, chi)
        P = P_inv_formula(rho, Qth, chi)
        T_list.append(T)
        P_list.append(P)
        rho_list.append(rho)
    return np.array(T_list), np.array(P_list), np.array(rho_list), rh

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(figsize=(7,6))

Q_val = 1.5
colors_theta = {0.0: 'black', 0.05: 'blue', 0.10: 'red', 0.15: 'green'}
sqrt_theta_vals = [0.0, 0.05, 0.10, 0.15]
linestyles = {0.0: '-', 0.05: '--', 0.10: '-.', 0.15: ':'}

P_black = None
T_black = None
legend_items = []

for sqrt_th in sqrt_theta_vals:
    T_inv, P_inv, rho_inv, rh_inv = get_inversion_curve(Q_val, sqrt_th, rh_max=3.0)
    mask = T_inv > 0
    if np.any(mask):
        T_pos = T_inv[mask]
        P_pos = P_inv[mask]
        idx_sort = np.argsort(P_pos)
        P_pos = P_pos[idx_sort]
        T_pos = T_pos[idx_sort]
        mask_display = (P_pos > -0.5) & (P_pos < 2.0) & (T_pos < 1.0)
        if np.any(mask_display):
            # Stockage de la courbe noire (sqrt_theta = 0) pour les zones
            if sqrt_th == 0.0:
                P_black = P_pos[mask_display].copy()
                T_black = T_pos[mask_display].copy()
            
            line, = ax.plot(
                P_pos[mask_display],
                T_pos[mask_display],
                linestyle=linestyles[sqrt_th],
                color=colors_theta[sqrt_th],
                linewidth=2,
                zorder=3,
                label=fr'$\sqrt{{\theta}}={sqrt_th}$'
            )
            legend_items.append(line)

# ============================================================
# ZONES COOLING / HEATING (fill_betweenx, analogue à la figure de référence)
# ============================================================
if P_black is not None:
    xmin, xmax = -0.5, 1.5
    
    # COOLING : à droite de la courbe d'inversion (mu_JT > 0)
    ax.fill_betweenx(
        T_black, P_black, xmax,
        alpha=0.08, color='red', zorder=0
    )
    
    # HEATING : à gauche de la courbe d'inversion (mu_JT < 0)
    ax.fill_betweenx(
        T_black, xmin, P_black,
        alpha=0.08, color='blue', zorder=0
    )
    
    # Labels sur la figure
    ax.text(0.9, 0.25, 'Cooling', fontsize=12, fontweight='bold', color='darkred', zorder=4)
    ax.text(0.1, 0.75, 'Heating', fontsize=12, fontweight='bold', color='darkblue', zorder=4)

# ============================================================
# Légende avec mu_JT
# ============================================================
cool_patch = Patch(facecolor='red',   edgecolor='none', alpha=0.25, label=r'$\mu_{JT} > 0$')
heat_patch = Patch(facecolor='blue',  edgecolor='none', alpha=0.25, label=r'$\mu_{JT} < 0$')

legend_items.extend([cool_patch, heat_patch])

ax.legend(handles=legend_items, loc='upper left', fontsize=9, framealpha=0.9)

ax.set_xlabel('$P$', fontsize=14)
ax.set_ylabel('$T$', rotation=0, fontsize=14, labelpad=15)
ax.set_title(r'$Q = 0.2$', fontsize=14)

ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig('/home/tahir/Downloads/inversion_Q15_analogue_ref.pdf', dpi=1200)
plt.show()
