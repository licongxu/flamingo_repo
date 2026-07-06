"""Final figures + metrics + manifest for the universal tSZ model.

Usage: python -m autoresearch.make_final [best.npy] [nonlinear:0/1]
Produces (in autoresearch/figures/):
  fig_spectra.pdf/.png   : 5 spectra (full + q>50,20,10,5) data vs theory + residual panel
  fig_residual_cv.pdf/.png: residuals vs the cosmic-variance band
  final_metrics.json     : accuracy summary + best-fit parameters
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

bestfile = sys.argv[1] if len(sys.argv) > 1 else 'fit_cat_linear.npy'
if len(sys.argv) > 2 and sys.argv[2] == '1':
    os.environ['CATMODEL_NONLINEAR'] = '1'

from flamingo import paths
import autoresearch.cat_model as CM

FIGDIR = paths.DATA.parent / 'autoresearch' / 'figures'
FIGDIR.mkdir(exist_ok=True)
ELLFIT = 60.0

mp = np.load(paths.DATA / 'nb09_tsz_map_ps.npz')
ellb = mp['ellb']
dmap = {'full': mp['dl_map'], 50: mp['dl_q50'], 20: mp['dl_q20'], 10: mp['dl_q10'], 5: mp['dl_q5']}
fsky = {'full': 0.9993, 50: 0.9993, 20: 0.9817, 10: 0.9261, 5: 0.8405}
SELS = CM.SELS

m = CM.CatModel()
Pl = m._Pl(ellb)
x = np.load(paths.DATA.parent / 'autoresearch' / bestfile)
a2h = 10 ** x[0]
ab, sp = [], []
for b in range(CM.NQ):
    la, c500, gamma, beta = x[1 + 4 * b:5 + 4 * b]
    ab.append(10 ** la)
    sp.append(dict(c500=float(c500), gamma=float(gamma), alpha=1.0510, beta=float(beta)))
pred = m.predict(ellb, np.array(ab), a2h, sp, Pl=Pl)

colors = {'full': '0.2', 50: '#0072B2', 20: '#E69F00', 10: '#009E73', 5: '#D55E00'}
labels = {'full': 'full sky', 50: r'$q>50$', 20: r'$q>20$', 10: r'$q>10$', 5: r'$q>5$'}

# ---- Figure 1: spectra + residuals ----
fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.0, 7.6), height_ratios=[2.3, 1.3], sharex=True)
for s in SELS:
    c = colors[s]
    ax.loglog(ellb, dmap[s], 'o', ms=3.0, color=c, alpha=0.55, mec='none')
    ax.loglog(ellb, pred[s], '-', lw=1.8, color=c, label=labels[s])
    r = pred[s] / dmap[s] - 1
    axr.semilogx(ellb[ellb >= ELLFIT], 100 * r[ellb >= ELLFIT], '-o', lw=1.0, ms=2.6, color=c)
ax.set_ylabel(r'$D_\ell=\ell(\ell+1)C_\ell^{yy}/2\pi$')
ax.set_ylim(2e-15, 2.4e-12); ax.set_xlim(12, 6000)
ax.legend(fontsize=9, ncol=2, frameon=False, loc='lower right')
ax.set_title('Universal catalogue-anchored tSZ model vs FLAMINGO', fontsize=11)
axr.axhspan(-1, 1, color='green', alpha=0.15, label=r'$\pm1\%$')
axr.axhline(0, color='k', lw=0.6)
axr.set_ylim(-8, 8)
axr.set_xlabel(r'multipole $\ell$'); axr.set_ylabel(r'theory/map $-1$ [\%]')
axr.legend(fontsize=8, frameon=False, loc='upper right')
fig.tight_layout()
fig.savefig(FIGDIR / 'fig_spectra.pdf'); fig.savefig(FIGDIR / 'fig_spectra.png', dpi=140)

# ---- Figure 2: residual vs cosmic variance ----
fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
elf = ellb[ellb >= ELLFIT]
for s in SELS:
    r = (pred[s] / dmap[s] - 1)[ellb >= ELLFIT]
    ax2.semilogx(elf, 100 * r, '-o', lw=0.9, ms=2.6, color=colors[s], label=labels[s])
cv = 100 * np.sqrt(2.0 / ((2 * elf + 1) * 30.0 * np.mean(list(fsky.values()))))
ax2.fill_between(elf, -cv, cv, color='grey', alpha=0.25, label=r'$\pm1\sigma$ Gaussian CV')
ax2.fill_between(elf, -2 * cv, 2 * cv, color='grey', alpha=0.12)
ax2.axhline(0, color='k', lw=0.5)
ax2.set_xlabel(r'multipole $\ell$'); ax2.set_ylabel(r'theory/map $-1$ [\%]')
ax2.set_ylim(-8, 8); ax2.set_xlim(elf.min(), 6000)
ax2.legend(fontsize=8, ncol=2, frameon=False)
ax2.set_title('Residuals lie at the single-realization sample-variance floor', fontsize=10)
fig2.tight_layout()
fig2.savefig(FIGDIR / 'fig_residual_cv.pdf'); fig2.savefig(FIGDIR / 'fig_residual_cv.png', dpi=140)

# ---- metrics + manifest ----
metrics = {'bestfile': bestfile, 'nonlinear': os.environ.get('CATMODEL_NONLINEAR', '0') == '1',
           'a2h': float(a2h), 'a_qbin': [float(a) for a in ab], 'shapes': sp,
           'qbin_edges_q': CM.QBIN_EDGES, 'R_mask_theta500': 5.0, 'ellfit_min': ELLFIT}
for cut in (60, 150, 250):
    fm = ellb >= cut
    allr = np.concatenate([np.abs(pred[s][fm] / dmap[s][fm] - 1) for s in SELS])
    metrics[f'ell>={cut}'] = dict(median_pct=float(np.median(allr) * 100),
                                  p90_pct=float(np.percentile(allr, 90) * 100),
                                  max_pct=float(allr.max() * 100),
                                  frac_below_1pct=float(np.mean(allr < 0.01)),
                                  frac_below_2pct=float(np.mean(allr < 0.02)))
per = {}
for s in SELS:
    fm = ellb >= 150
    rel = np.abs(pred[s][fm] / dmap[s][fm] - 1)
    per[str(s)] = dict(median_pct=float(np.median(rel) * 100), max_pct=float(rel.max() * 100))
metrics['per_selection_ell>=150'] = per
with open(FIGDIR.parent / 'final_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('wrote figures to', FIGDIR)
print(json.dumps({k: metrics[k] for k in ('ell>=60', 'ell>=150', 'ell>=250')}, indent=2))
