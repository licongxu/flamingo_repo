"""Catalogue-anchored 1-halo (Poisson) tSZ power spectrum, with EXACT masking.

C_1h(q_cut, ell) = (1/4pi) sum_{a: q_a <= q_cut} [ Y_ang,a * Ghat_{b(a)}(ell*theta500,a) ]^2

- Y_ang = Y_5R500c / dA^2  (dA = r_comoving/(1+z)), integrated Compton-y [sr].
- Ghat_b: stacked empirical harmonic shape for the cluster's q-bin (nb11_profiles.npz),
  Ghat_b(0)=1, argument u = ell*theta500 [rad].
- masking: drop clusters with q_from_Y_5R500c > q_cut (the field used to build the map mask).

Implemented by histogramming sum(Y_ang^2) in theta500 bins per q-bin (per selection),
then C(ell) = (1/4pi) sum_b sum_thetabin W_b(theta) * Ghat_b(ell*theta)^2.
"""
import numpy as np
from flamingo import paths
from flamingo.catalogue import load_catalogue

QBIN_EDGES = [5.0, 10.0, 20.0]     # -> bins 0:(q<5) 1:(5-10) 2:(10-20) 3:(>20)
NQ = 4
Q_CUTS = (50, 20, 10, 5)


def _load():
    cat = load_catalogue(paths.HYDRO_CATALOGUE)
    z = cat['z'].values
    dA = cat['r_comoving_Mpc'].values / (1 + z)
    y_ang = cat['Y_5R500c_Mpc2'].values / dA ** 2
    th500 = cat['theta500_arcmin'].values / (180.0 / np.pi * 60.0)   # rad
    q = cat['q_from_Y_5R500c'].values
    qb = np.clip(np.digitize(q, QBIN_EDGES), 0, NQ - 1)
    d = np.load(paths.DATA / 'nb11_profiles.npz')
    Ghat = {b: (d['kk'], d[f'Ghat_{b}']) for b in range(NQ)}
    return z, y_ang, th500, q, qb, Ghat


class Cat1h:
    def __init__(self, n_theta=240):
        self.z, self.y_ang, self.th500, self.q, self.qb, self.Ghat = _load()
        finite = np.isfinite(self.y_ang) & np.isfinite(self.th500)
        self.finite = finite
        tmin, tmax = np.nanmin(self.th500[finite]), np.nanmax(self.th500[finite])
        self.tedges = np.logspace(np.log10(tmin * 0.999), np.log10(tmax * 1.001), n_theta + 1)
        self.tmid = np.sqrt(self.tedges[:-1] * self.tedges[1:])

    def weights(self, q_cut):
        """W[b, theta_bin] = sum of Y_ang^2 over surviving clusters in q-bin b."""
        surv = self.finite & (~(self.q > q_cut) if q_cut is not None else np.ones_like(self.q, bool))
        W = np.zeros((NQ, len(self.tmid)))
        for b in range(NQ):
            sel = surv & (self.qb == b)
            W[b], _ = np.histogram(self.th500[sel], bins=self.tedges,
                                   weights=self.y_ang[sel] ** 2)
        return W

    def cl_1h(self, ell, q_cut):
        ell = np.atleast_1d(ell)
        W = self.weights(q_cut)
        cl = np.zeros(len(ell))
        for b in range(NQ):
            kk, G = self.Ghat[b]
            # G2[ell, thetabin] = Ghat_b(ell*tmid)^2
            u = ell[:, None] * self.tmid[None, :]
            G2 = np.interp(u, kk, G, left=G[0], right=0.0) ** 2
            cl += G2 @ W[b]
        return cl / (4 * np.pi)


if __name__ == '__main__':
    c = Cat1h()
    mp = np.load(paths.DATA / 'nb09_tsz_map_ps.npz')
    ellb = mp['ellb']
    dmap = {'full': mp['dl_map'], 50: mp['dl_q50'], 20: mp['dl_q20'],
            10: mp['dl_q10'], 5: mp['dl_q5']}
    pref = ellb * (ellb + 1) / (2 * np.pi)
    print(f'{"ell":>6} ' + ' '.join(f'{s:>8}' for s in ['full', 50, 20, 10, 5]) + '   (1h_cat/map)')
    cl = {s: pref * c.cl_1h(ellb, (None if s == 'full' else s)) for s in ['full', 50, 20, 10, 5]}
    for L in [50, 150, 500, 1000, 2000, 3000, 5000]:
        k = np.argmin(np.abs(ellb - L))
        print(f'{ellb[k]:6.0f} ' + ' '.join(f'{cl[s][k]/dmap[s][k]:8.3f}' for s in ['full', 50, 20, 10, 5]))
