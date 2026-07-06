"""Plot data vs model for all 5 selections + residual panel. For VLM inspection."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import autoresearch.fit01 as F
import autoresearch.model as M

bestfile = sys.argv[1] if len(sys.argv) > 1 else 'autoresearch/fit01_best.npy'
x = np.load(bestfile)
pred = F.predict(x)
ellb, dmap = F.ellb, F.dmap
SELS = ['full', 50, 20, 10, 5]
colors = {'full': '0.3', 50: 'C0', 20: 'C1', 10: 'C2', 5: 'C3'}
labels = {'full': 'full sky', 50: 'q>50', 20: 'q>20', 10: 'q>10', 5: 'q>5'}

fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.5, 9), height_ratios=[2.2, 1.4], sharex=True)
for s in SELS:
    c = colors[s]
    ax.loglog(ellb, dmap[s], 'o', ms=2.8, color=c, alpha=0.5)
    ax.loglog(ellb, pred[s], '-', lw=1.7, color=c, label=labels[s])
    axr.semilogx(ellb, pred[s] / dmap[s] - 1, '-', lw=1.4, color=c)
    axr.semilogx(ellb[ellb >= F.ELLFIT], (pred[s] / dmap[s] - 1)[ellb >= F.ELLFIT],
                 'o', ms=2.5, color=c)
ax.set_ylabel(r'$D_\ell=\ell(\ell+1)C_\ell^{yy}/2\pi$')
ax.set_ylim(3e-15, 3e-12); ax.set_xlim(12, 6000)
ax.legend(fontsize=9, ncol=2); ax.set_title('Universal masked-tSZ model vs FLAMINGO map (points=map, lines=model)')
axr.axhspan(-0.01, 0.01, color='green', alpha=0.15)
axr.axhline(0, color='k', lw=0.6)
for y in (-0.05, 0.05):
    axr.axhline(y, color='grey', ls=':', lw=0.7)
axr.set_ylim(-0.20, 0.20)
axr.set_xlabel(r'$\ell$'); axr.set_ylabel('model/map - 1')
axr.axvline(F.ELLFIT, color='purple', ls='--', lw=0.8)
fig.tight_layout()
out = bestfile.replace('.npy', '_plot.png')
fig.savefig(out, dpi=130)
print('saved', out)
