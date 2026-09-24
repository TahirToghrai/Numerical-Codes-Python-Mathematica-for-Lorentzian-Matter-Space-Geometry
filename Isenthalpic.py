import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

pi = np.pi

# ============================================================
# Fonctions (inchangées)
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

def isenthalpic_curve_Q0(H, Q_val, r_min=0.01, r_max=5.0, n=5000):
    r = np.linspace(r_min, r_max, n)
    P_vals, T_vals, r_vals = [], [], []
    for rh in r:
        P = (3/(4*pi*rh**3)) * (H - rh/2 - Q_val**2/(2*rh))
        T = 3*H/(2*pi*rh**2) - 1/(2*pi*rh) - Q_val**2/(pi*rh**3)
        if T > 0 and P > -1.0 and P < 10.0:
            P_vals.append(P)
            T_vals.append(T)
            r_vals.append(rh)
    return np.array(P_vals), np.array(T_vals), np.array(r_vals)

# ============================================================
# FIGURE
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6.2))

Q_val = 1.5

# Isenthalpiques
H_values = [2.0, 3.0, 5.0, 8.0, 12.0, 15.0]
H_labels = [0.3, 0.4, 0.5, 0.7, 1.0, 1.5]
colors = ['black', 'blue', 'red', 'green', 'purple', 'orange']

legend_handles = []

for i, (H, Hlab) in enumerate(zip(H_values, H_labels)):
    P_iso, T_iso, _ = isenthalpic_curve_Q0(H, Q_val, r_max=3.0)
    if len(P_iso) > 10:
        line, = ax.plot(P_iso, T_iso, '--', color=colors[i],
                       linewidth=1.5, label=fr'$H={Hlab}$')
        legend_handles.append(line)

# Courbe d'inversion
H_env_range = np.linspace(1.5, 20.0, 300)
env_P, env_T = [], []
for H in H_env_range:
    P_iso, T_iso, _ = isenthalpic_curve_Q0(H, Q_val, r_max=3.0)
    if len(T_iso) > 5:
        idx_max = np.argmax(T_iso)
        env_P.append(P_iso[idx_max])
        env_T.append(T_iso[idx_max])

env_P = np.array(env_P)
env_T = np.array(env_T)
idx_sort = np.argsort(env_P)

ax.plot(env_P[idx_sort], env_T[idx_sort], 'k-', linewidth=3, zorder=3)

# Zones colorées
ax.fill_betweenx(env_T[idx_sort], env_P[idx_sort], 2.5,
                alpha=0.08, color='red', zorder=0)
ax.fill_betweenx(env_T[idx_sort], -0.5, env_P[idx_sort],
                alpha=0.08, color='blue', zorder=0)

# Textes Heating / Cooling
ax.text(1.8, 1.2, 'Cooling', fontsize=12, fontweight='bold',
        color='darkred', zorder=4)
ax.text(-0.4, 1.2, 'Heating', fontsize=12, fontweight='bold',
        color='darkblue', zorder=4)

# ============================================================
# LÉGENDE : 5 colonnes × 2 lignes
# ============================================================
cool_patch = Patch(facecolor='red',  edgecolor='none', alpha=0.25,
                   label=r'$\mu_{JT} > 0$')
heat_patch = Patch(facecolor='blue', edgecolor='none', alpha=0.25,
                   label=r'$\mu_{JT} < 0$')
inv_line = Line2D([0], [0], color='black', lw=3, ls='-',
                  label=r'Inversion curve ($\sqrt{\theta}=0.1$)')
dummy = Line2D([0], [0], linestyle='None', marker='None',
               color='none', label='')

legend_handles.extend([cool_patch, heat_patch, inv_line, dummy])

# --- Placement remonté (bbox y = -0.06 au lieu de -0.10) ---
ax.legend(handles=legend_handles,
          loc='upper center',
          bbox_to_anchor=(0.5, -0.06),   # ← remonté : moins d'espace sous l'axe
          ncol=5,
          fontsize=9,
          framealpha=0.9,
          handlelength=2.5,
          columnspacing=1.0,
          handletextpad=0.4)

# Axes
ax.set_xlabel('$P$', fontsize=14)
ax.xaxis.set_label_coords(1.02, 0.03)
ax.set_ylabel('$T$', rotation=0, fontsize=14, labelpad=15)
ax.set_title(r'$Q = 0.2$', fontsize=14)

ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, 2.0)

# --- bottom réduit car la légende remonte ---
plt.subplots_adjust(bottom=0.14, left=0.10, right=0.95, top=0.92)

plt.savefig('/home/tahir/Downloads/JT_Q02_legend_5x2_raised.pdf', dpi=1200)
plt.show()
