# Numerical Codes (Python & Mathematica) for *Lorentzian Matter-Space Geometry of Reissner–Nordström–AdS Black Holes*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

Python and Mathematica codes used to produce every numerical result, table, and figure in:

> T. Toghrai, A. Daassou, N. Mansour, R. Benbrik, A. El Boukili, M. B. Sedra,
> **"Lorentzian Matter-Space Geometry of Reissner–Nordström–AdS Black Holes: Thermodynamic Stability, PV Criticality, Joule–Thomson Expansion, and Quasinormal Modes,"**
> submitted to *European Physical Journal Plus* (2026).

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Paper results → code mapping](#paper-results--code-mapping)
- [Citation](#citation)
- [License](#license)
- [Authors & contact](#authors--contact)

## Overview

The metric is built from a double Lorentzian smearing of the RN-AdS mass and
charge sources, via the deformed areal coordinate $r_\theta(r)$ selected by the
Complete Geometric Consistency (CGC) principle. These codes numerically and
symbolically:

- verify the effective energy-momentum tensor and the weak/null/strong energy
  conditions across the near-horizon and far-field regions;
- check the first law $d\mathcal{H}_{\rm eff}=T_{\rm th}\,dS+V\,dP+\Phi_\theta\,dQ$
  and the Smarr formula exactly;
- compute the isobaric heat capacity and Gibbs free energy for thermodynamic
  stability analysis;
- solve the PV-criticality equation of state (free-$T$, Kubiznák–Mann
  prescription), the critical point, critical exponents, and the Maxwell
  equal-area construction;
- compute the Joule–Thomson coefficient, inversion curve, and the
  $T_i^{\min}/T_c$ ratio across $\sqrt{\theta}\in[0,0.15]$;
- extract quasinormal-mode frequencies via sixth-order WKB (with Padé
  resummation) and cross-check them against an independent time-domain
  evolution;
- compare the Lorentzian profile against the standard Gaussian smearing.

## Repository structure

```
.
├── python/
│   ├── thermodynamics/        # Heat capacity, Gibbs free energy, phase structure
│   ├── pv_criticality/        # Equation of state, critical point, exponents, Maxwell construction
│   ├── joule_thomson/         # JT coefficient, inversion curve, T_i^min/T_c ratio
│   ├── quasinormal_modes/     # 6th-order WKB + Padé, time-domain cross-check
│   └── utils/                 # Deformed metric r_θ(r), shared constants, plotting helpers
├── mathematica/
│   ├── FirstLaw.nb             # Symbolic verification of the first law & Smarr formula
│   ├── EnergyConditions.nb     # Effective EMT and energy-condition sweep
│   └── GaussianComparison.nb   # Gaussian vs. Lorentzian smearing comparison
├── data/                       # Cached numerical outputs (CSV) behind the paper's tables/figures
├── figures/                    # Scripts/notebooks regenerating each paper figure
├── requirements.txt
├── LICENSE.md
├── .gitignore
└── README.md
```

> **Note:** this layout mirrors the paper's section structure and is meant as
> a starting scaffold — rename the files/folders above to match your actual
> scripts before pushing.

## Requirements

**Python ≥ 3.9**
```
numpy
scipy
mpmath
matplotlib
pandas
```
```bash
pip install -r requirements.txt
```

**Mathematica ≥ 12.0** (built-in kernel functions only; no paid add-on packages required).

## How to run

```bash
git clone https://github.com/TahirToghrai/Numerical-Codes-Python-Mathematica-for-Lorentzian-Matter-Space-Geometry.git
cd Numerical-Codes-Python-Mathematica-for-Lorentzian-Matter-Space-Geometry
pip install -r requirements.txt
python python/thermodynamics/heat_capacity.py
```
Each script/notebook writes its output to `data/` and/or `figures/`.

## Paper results → code mapping

| Paper result | Section | Script / notebook |
|---|---|---|
| Deformed metric $r_\theta(r)$, CGC principle | Sec. 2 | `python/utils/metric.py` |
| Effective $T^{\mu\nu}$ & energy conditions | Sec. 2.5, App. B | `mathematica/EnergyConditions.nb` |
| First law & Smarr formula (exact check) | Sec. 3, App. A | `mathematica/FirstLaw.nb` |
| Isobaric heat capacity, Gibbs free energy | Sec. 3.7–3.8 | `python/thermodynamics/` |
| Critical point, exponents, Maxwell construction | Sec. 4 | `python/pv_criticality/` |
| JT coefficient, inversion curve, $T_i^{\min}/T_c\approx0.500$ | Sec. 5 | `python/joule_thomson/` |
| 6th-order WKB + Padé QNM frequencies | Sec. 6.2–6.3 | `python/quasinormal_modes/wkb6.py` |
| Time-domain cross-check | Sec. 6.4 | `python/quasinormal_modes/time_domain.py` |
| Gaussian vs. Lorentzian comparison | App. C | `mathematica/GaussianComparison.nb` |

## Citation

Please cite both the paper and this code repository:

```bibtex
@article{Toghrai2026Lorentzian,
  author  = {Toghrai, T. and Daassou, A. and Mansour, N. and Benbrik, R. and El Boukili, A. and Sedra, M. B.},
  title   = {Lorentzian Matter-Space Geometry of Reissner--Nordstr{\"o}m--AdS Black Holes:
             Thermodynamic Stability, {PV} Criticality, {Joule--Thomson} Expansion,
             and Quasinormal Modes},
  journal = {European Physical Journal Plus},
  year    = {2026},
  note    = {in press}
}

@software{Toghrai2026Code,
  author    = {Toghrai, T. and Daassou, A. and Mansour, N. and Benbrik, R. and El Boukili, A. and Sedra, M. B.},
  title     = {Numerical Codes (Python \& Mathematica) for
               "Lorentzian Matter-Space Geometry of Reissner--Nordstr{\"o}m--AdS Black Holes"},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```
Fill in the volume/pages/DOI/arXiv ID once assigned.

## License

Released under the [MIT License](LICENSE.md).

## Authors & contact

| Author | Affiliation | ORCID |
|---|---|---|
| T. Toghrai (corresponding) | Univ. Moulay Ismail (MRA/STI) & Cadi Ayyad Univ. (LP2EA) | 0000-0001-7142-0158 |
| A. Daassou | Cadi Ayyad University (LP2EA) | 0000-0001-9439-5047 |
| N. Mansour | Univ. Moulay Ismail (MRA/STI) | 0000-0002-9993-8714 |
| R. Benbrik | Cadi Ayyad University (LP2EA) | 0000-0002-5159-0325 |
| A. El Boukili | Univ. Moulay Ismail (MRA/STI) | 0000-0002-3277-9640 |
| M. B. Sedra | Univ. Ibn Tofail (LPMS) | 0000-0001-5160-7564 |

Corresponding author: t.toghrai@edu.umi.ac.ma
