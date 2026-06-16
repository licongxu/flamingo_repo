"""Plot catalogue-model fit vs map: spectra + residuals, for VLM inspection."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import autoresearch.fit_cat as F

x = np.load(sys.argv[1] if len(sys.argv) > 1 else 'autoresearch/fit_cat_best.npy')
pred = F.predict(x)
ellb, dmap = F.ellb, F.dmap
SELS = F.SELS
colors = {'full': '0.25', 50: 'C0', 20: 'C1', 10: 'C2', 5: 'C3'}
labels = {'full': 'full sky', 50: 'q>50', 20: 'q>20', 10: 'q>10', 5: 'q>5'}

fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 9.5), height_ratios=[2.2, 1.5], sharex=True)
for s in SELS:
    c = colors[s]
    ax.loglog(ellb, dmap[s], 'o', ms=2.6, color=c, alpha=0.45)
    ax.loglog(ellb, pred[s], '-', lw=1.7, color=c, label=labels[s])
    r = pred[s] / dmap[s] - 1
    axr.semilogx(ellb, r, '-', lw=1.0, color=c, alpha=0.5)
    axr.semilogx(ellb[ellb >= F.ELLFIT], r[ellb >= F.ELLFIT], 'o', ms=2.6, color=c)
ax.set_ylabel(r'$D_\ell=\ell(\ell+1)C_\ell^{yy}/2\pi$')
ax.set_ylim(2e-15, 3e-12); ax.set_xlim(12, 6000)
ax.legend(fontsize=9, ncol=2)
ax.set_title('Catalogue-anchored universal model vs FLAMINGO (points=map, lines=theory)')
axr.axhspan(-0.01, 0.01, color='green', alpha=0.18)
axr.axhline(0, color='k', lw=0.6)
for y in (-0.05, 0.05, -0.02, 0.02):
    axr.axhline(y, color='grey', ls=':', lw=0.6)
axr.set_ylim(-0.10, 0.10)
axr.set_xlabel(r'$\ell$'); axr.set_ylabel('theory/map - 1')
axr.axvline(F.ELLFIT, color='purple', ls='--', lw=0.8)
fig.tight_layout()
out = 'autoresearch/fit_cat_plot.png'
fig.savefig(out, dpi=130)
print('saved', out)
# also where are the worst points
for s in SELS:
    r = np.abs(pred[s] / dmap[s] - 1)[ellb >= F.ELLFIT]
    el = ellb[ellb >= F.ELLFIT]
    bad = el[r > 0.01]
    print(f'{str(s):>5}: {len(bad)} pts >1%, at ell=' + ','.join(f'{b:.0f}' for b in bad[:12]))
