"""Fit the catalogue-anchored model with per-qbin shapes. Full-shape, log-space, all 5 q."""
import sys
import numpy as np
from scipy.optimize import minimize
from flamingo import paths
import autoresearch.cat_model as CM

ELLFIT = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
NITER = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
NTRIAL = int(sys.argv[3]) if len(sys.argv) > 3 else 5

mp = np.load(paths.DATA / 'nb09_tsz_map_ps.npz')
ellb = mp['ellb']
dmap = {'full': mp['dl_map'], 50: mp['dl_q50'], 20: mp['dl_q20'], 10: mp['dl_q10'], 5: mp['dl_q5']}
SELS = CM.SELS
NQ = CM.NQ
m = CM.CatModel()
Pl = m._Pl(ellb)

fm = ellb >= ELLFIT
elf = ellb[fm]
wll = np.sqrt((2 * elf + 1) * 30.0); wll /= wll.mean()


# param layout: [log a2h] + per qbin [log a_b, c500, gamma, beta]
def unpack(p):
    a2h = 10 ** p[0]
    ab, sp = [], []
    for b in range(NQ):
        la, c500, gamma, beta = p[1 + 4 * b: 5 + 4 * b]
        ab.append(10 ** la)
        sp.append(dict(c500=c500, gamma=gamma, alpha=1.0510, beta=beta))
    return np.array(ab), a2h, sp


def predict(p):
    ab, a2h, sp = unpack(p)
    return m.predict(ellb, ab, a2h, sp, Pl=Pl)


def objective(p):
    if not (-0.6 < p[0] < 0.7):       # a2h in [0.25, 5]; keep away from 0 (no-floor minimum)
        return 1e6
    for b in range(NQ):
        la, c500, gamma, beta = p[1 + 4 * b: 5 + 4 * b]
        if not (0.3 < c500 < 5.0 and 0.02 < gamma < 1.3 and 3.2 < beta < 10.0 and -0.4 < la < 0.4):
            return 1e6
    pr = predict(p)
    r = 0.0
    for s in SELS:
        r += np.sum(wll * (np.log(pr[s][fm]) - np.log(dmap[s][fm])) ** 2)
    return r


def report(p, tag=''):
    pr = predict(p)
    ab, a2h, sp = unpack(p)
    print(f'--- {tag}: a2h={a2h:.3f}  ab=' + ' '.join(f'{a:.3f}' for a in ab))
    for b in range(NQ):
        print(f'    qbin{b}: c500={sp[b]["c500"]:.3f} gamma={sp[b]["gamma"]:.3f} beta={sp[b]["beta"]:.3f}')
    allr = []
    for s in SELS:
        rel = np.abs(pr[s][fm] / dmap[s][fm] - 1)
        allr.append(rel)
        print(f'  {str(s):>5}: med={np.median(rel)*100:5.2f}%  90pct={np.percentile(rel,90)*100:6.2f}%  '
              f'max={rel.max()*100:6.2f}%  <1%={np.mean(rel<0.01)*100:3.0f}%')
    allr = np.concatenate(allr)
    print(f'  ALL  : med={np.median(allr)*100:5.2f}%  90pct={np.percentile(allr,90)*100:6.2f}%  '
          f'max={allr.max()*100:6.2f}%  <1%={np.mean(allr<0.01)*100:3.0f}%  <2%={np.mean(allr<0.02)*100:3.0f}%')
    return objective(p)


def clip_bounds(p):
    p = p.copy()
    for b in range(NQ):
        p[1 + 4 * b] = np.clip(p[1 + 4 * b], -0.35, 0.35)
        p[2 + 4 * b] = np.clip(p[2 + 4 * b], 0.35, 4.5)
        p[3 + 4 * b] = np.clip(p[3 + 4 * b], 0.05, 1.2)
        p[4 + 4 * b] = np.clip(p[4 + 4 * b], 3.4, 9.5)
    return p


if __name__ == '__main__':
    import os
    outfile = paths.DATA.parent / 'autoresearch' / os.environ.get('OUTFILE', 'fit_cat_best.npy')
    warm = os.environ.get('WARMSTART')
    if warm:
        x0 = np.load(paths.DATA.parent / 'autoresearch' / warm)
    else:
        la = np.log10(0.86)
        x0 = np.array([np.log10(1.20),
                       la, 1.099, 0.219, 5.613,
                       la, 1.687, 0.204, 4.846,
                       la, 1.530, 0.020, 5.324,
                       la, 1.409, 0.393, 5.077])
    report(x0, 'warm-start'); sys.stdout.flush()
    best = None
    for trial in range(NTRIAL):
        rng = np.random.default_rng(trial)
        scale = np.array([0.2] + [0.1, 0.3, 0.08, 0.5] * NQ)
        start = clip_bounds((x0 if best is None else best.x)
                            + (0 if trial == 0 else rng.normal(0, scale)))
        res = minimize(objective, start, method='Powell',
                       options=dict(maxiter=NITER, xtol=1e-5, ftol=1e-8))
        if best is None or res.fun < best.fun:
            best = res
        print(f'trial {trial}: fun={res.fun:.5f} nit={res.nit}'); sys.stdout.flush()
        np.save(outfile, best.x)
        report(best.x, f'after trial {trial}'); sys.stdout.flush()
    print('\n=== BEST ===')
    report(best.x, 'best')
    np.save(outfile, best.x)
