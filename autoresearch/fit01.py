"""Iter 3: joint fit of the universal model to all 5 spectra.

Model per selection q:  D(q,ell) = a1h*C1h(q) + a2h*C2h(q) + a_diff*Diff(ell)
  C1h(q), C2h(q): completeness-weighted 1h/2h (profile + sigma_lnY dependent)
  Diff(ell): masking-ROBUST template (same for every q incl. full) = full-sky 2h shape,
             to absorb the diffuse/large-scale tSZ that masking small discs leaves intact.

Outer optimizer (Nelder-Mead) over [log a1h, log a2h, log a_diff, beta, c500, gamma, sigma_lnY].
Objective: log-space, sample-variance-weighted residual over ELLFIT..6000 for all 5 spectra.
Reports per-q median and 90th-pct |model/data-1|, and fraction of points within 1%.
"""
import numpy as np
from scipy.optimize import minimize
import autoresearch.model as M

ELLFIT = 60.0
mod = M.Model()
ellb, dmap = M.load_map()
SELS = ['full', 50, 20, 10, 5]

fitmask = ellb >= ELLFIT
elf = ellb[fitmask]
# sample-variance weight: modes per bin ~ (2l+1)*dl*fsky ; use full-sky fsky~1, dl=30
modes = (2 * elf + 1) * 30.0
wll = np.sqrt(modes)
wll = wll / wll.mean()


def interp(d):
    return np.exp(np.interp(np.log(ellb), np.log(mod.ell_np), np.log(np.clip(d, 1e-30, None))))


def predict(params):
    la1, la2, lad, beta, c500, gamma, slny = params
    pp = dict(P0=M.A10['P0'], c500=c500, gamma=gamma, alpha=M.A10['alpha'], beta=beta)
    base = mod.base_spectra(pp, slny)
    diff_template = interp(base['full'][1])    # full-sky 2h shape, masking-robust
    out = {}
    for s in SELS:
        c1 = interp(base[s][0]); c2 = interp(base[s][1])
        out[s] = 10**la1 * c1 + 10**la2 * c2 + 10**lad * diff_template
    return out


def objective(params):
    la1, la2, lad, beta, c500, gamma, slny = params
    if not (4.0 < beta < 7.5 and 0.5 < c500 < 3.0 and 0.05 < gamma < 0.8 and 0.05 < slny < 0.6):
        return 1e6
    pred = predict(params)
    r = 0.0
    for s in SELS:
        d = dmap[s][fitmask]; p = pred[s][fitmask]
        r += np.sum(wll * (np.log(p) - np.log(d)) ** 2)
    return r


def report(params, tag=''):
    pred = predict(params)
    print(f'--- {tag} params: la1={params[0]:.3f} la2={params[1]:.3f} lad={params[2]:.3f} '
          f'beta={params[3]:.3f} c500={params[4]:.3f} gamma={params[5]:.3f} slny={params[6]:.3f}')
    allres = []
    for s in SELS:
        d = dmap[s][fitmask]; p = pred[s][fitmask]
        rel = np.abs(p / d - 1)
        allres.append(rel)
        print(f'  {str(s):>5}: median={np.median(rel)*100:5.2f}%  90pct={np.percentile(rel,90)*100:6.2f}%  '
              f'max={rel.max()*100:6.2f}%  frac<1%={np.mean(rel<0.01)*100:4.0f}%')
    allres = np.concatenate(allres)
    print(f'  ALL  : median={np.median(allres)*100:5.2f}%  90pct={np.percentile(allres,90)*100:6.2f}%  '
          f'frac<1%={np.mean(allres<0.01)*100:4.0f}%')
    return np.median(allres)


if __name__ == '__main__':
    import sys
    x0 = np.array([np.log10(1.0), np.log10(1.0), np.log10(0.3),
                   M.A10['beta'], M.A10['c500'], M.A10['gamma'], 0.173])
    report(x0, 'initial'); sys.stdout.flush()
    best = None
    for trial in range(4):
        rng = np.random.default_rng(trial)
        start = x0 + rng.normal(0, [0.3, 0.3, 0.5, 0.4, 0.2, 0.05, 0.05])
        res = minimize(objective, start, method='Nelder-Mead',
                       options=dict(maxiter=1500, xatol=1e-3, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
        print(f'trial {trial}: fun={res.fun:.4f}  nit={res.nit}'); sys.stdout.flush()
    print('\n=== BEST ===')
    report(best.x, 'best')
    np.save(M.paths.DATA.parent / 'autoresearch' / 'fit01_best.npy', best.x)
