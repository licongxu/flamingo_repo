"""Catalogue-anchored universal masked-tSZ model.

C(q,ell) = a1h * C_1h(q,ell)  +  a2h * C_2h(q,ell)

C_1h: exact Poisson sum over SURVIVING catalogue clusters with a parametric, exactly
      Y-normalised harmonic shape ghat(u; shape), u=ell*theta500:
        C_1h = (1/4pi) sum_{q_a<=q_cut} [ Y_ang,a * ghat(ell*theta500,a) ]^2
      ghat(u) = Hankel_0 of the projected GNFW shape, normalised ghat(0)=1, so
      y_ell -> Y_ang as ell->0 (integrated Compton-y is the catalogue Y_5R500c).

C_2h: halo-model clustering with the per-z-shell bracket evaluated as a catalogue sum
      over SURVIVING clusters (exact masking of the bias-weighted Y), times P(k=ell/chi,z).

Masking is EXACT for both terms (drop catalogue clusters with q_from_Y_5R500c>q_cut).
The shape and the two amplitudes are the only fitted quantities (selection is the data).
Reads hmfast/cosmocnc only; never modifies them.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10')

import numpy as np
import jax.numpy as jnp
from mcfit import Hankel

from flamingo import paths
from flamingo.catalogue import load_catalogue, D3A_COSMOLOGY
from flamingo.profiles import gnfw, projected_shape
from hmfast.halos import HaloModel

ARCMIN_PER_RAD = 180.0 / np.pi * 60.0
QBIN_EDGES = [5.0, 10.0, 20.0]
NQ = 4
Q_CUTS = (50, 20, 10, 5)
SELS = ['full', 50, 20, 10, 5]


def ghat_of_shape(c500, gamma, alpha, beta, u_grid):
    """Normalised harmonic-space shape ghat(u), ghat(0)=1, for a GNFW profile.

    g(x) = projected GNFW shape (LOS integral, peak 1). ghat(u) = 2pi int x g(x) J0(u x) dx,
    normalised by its u->0 value (= 2pi int x g(x) dx). Returned interpolated onto u_grid.
    """
    x = np.logspace(-3.5, np.log10(8.0), 2048)
    pfunc = lambda r: gnfw(r, P0=1.0, c500=c500, gamma=gamma, alpha=alpha, beta=beta)
    g = np.asarray(projected_shape(jnp.asarray(x), p_func=pfunc, s_max=8.0, n_s=3000))
    # mcfit Hankel(nu=0): A(x) -> H(u) = int_0^inf A(x) J0(u x) x dx. Pass A=g.
    u, H = Hankel(x, lowring=True, nu=0)(g, extrap=True)
    ghat = H / np.interp(u.min(), u, H)        # normalise ghat(u->0)=1 (the 2pi cancels)
    return np.interp(u_grid, u, ghat, left=1.0, right=0.0)


class CatModel:
    def __init__(self, lmax=6000, nz_shell=40):
        cat = load_catalogue(paths.HYDRO_CATALOGUE)
        z = cat['z'].values
        dA = cat['r_comoving_Mpc'].values / (1 + z)
        self.chi = cat['r_comoving_Mpc'].values
        self.z = z
        self.y_ang = cat['Y_5R500c_Mpc2'].values / dA ** 2
        self.th500 = cat['theta500_arcmin'].values / ARCMIN_PER_RAD
        self.q = cat['q_from_Y_5R500c'].values
        self.qb = np.clip(np.digitize(self.q, QBIN_EDGES), 0, 3)
        self.finite = np.isfinite(self.y_ang) & np.isfinite(self.th500) & np.isfinite(self.chi)

        # bias per cluster from hmfast (interpolated on a grid), using M_200m for bias.
        hm = HaloModel(cosmology=D3A_COSMOLOGY)
        mg = jnp.logspace(12.5, 15.6, 40); zg = jnp.geomspace(0.01, 3.0, 40)
        bgrid = np.asarray(hm.halo_bias.halo_bias(hm, mg, zg))   # (Nm,Nz)
        from scipy.interpolate import RegularGridInterpolator
        bint = RegularGridInterpolator((np.log10(np.asarray(mg)), np.asarray(zg)), bgrid,
                                       bounds_error=False, fill_value=None)
        M200m = np.clip(cat['M_200m_Msun'].values, 10**12.6, 10**15.5)
        self.bias = bint(np.stack([np.log10(M200m), np.clip(z, 0.011, 2.99)], -1))

        # z-shells for the 2-halo
        self.zsh_edges = np.linspace(0.0, max(3.0, z.max() + 1e-3), nz_shell + 1)
        self.zsh = 0.5 * (self.zsh_edges[:-1] + self.zsh_edges[1:])
        self.dVdz = np.asarray(D3A_COSMOLOGY.comoving_volume_element(jnp.asarray(self.zsh)))  # Mpc3/sr
        self.dz = self.zsh_edges[1] - self.zsh_edges[0]
        # P(k,z) table
        kk = np.asarray(D3A_COSMOLOGY.pk(self.zsh[0])[0])
        self.k_arr = kk
        linpk = os.environ.get('CATMODEL_NONLINEAR', '0') != '1'
        self.Pk = np.array([np.asarray(D3A_COSMOLOGY.pk(zz, linear=linpk)[1]) for zz in self.zsh])
        self.chi_sh = (1 + self.zsh) * np.asarray(D3A_COSMOLOGY.angular_diameter_distance(
            jnp.asarray(self.zsh)))
        # theta500 histogram support
        f = self.finite
        self.tedges = np.logspace(np.log10(self.th500[f].min() * 0.999),
                                  np.log10(self.th500[f].max() * 1.001), 240)
        self.tmid = np.sqrt(self.tedges[:-1] * self.tedges[1:])
        self.zidx = np.clip(np.digitize(z, self.zsh_edges) - 1, 0, nz_shell - 1)

        self.u_grid = np.logspace(-4, np.log10(6000.0 * self.th500[f].max() * 1.5), 4000)
        self.lmax = lmax
        self._precompute_hists()

    def _surv(self, q_cut):
        if q_cut is None:
            return self.finite
        return self.finite & ~(self.q > q_cut)

    def _precompute_hists(self):
        """Per-selection, per-qbin histograms in theta500 (1h) and (z-shell,theta500) (2h).

        Depend only on the selection (which clusters survive) and the cluster q-bin, not on
        the profile shape, so per-qbin shapes can be varied cheaply during fitting.
        """
        self.H1 = {}   # H1[s][b, theta]       = sum Yang^2     over surviving qbin-b clusters
        self.H2 = {}   # H2[s][b, iz, theta]   = sum bias*Yang  over surviving qbin-b clusters
        nz, nt = len(self.zsh), len(self.tmid)
        for s in SELS:
            q = None if s == 'full' else s
            surv = self._surv(q)
            H1 = np.zeros((NQ, nt)); H2 = np.zeros((NQ, nz, nt))
            for b in range(NQ):
                sel = surv & (self.qb == b)
                H1[b], _ = np.histogram(self.th500[sel], bins=self.tedges,
                                        weights=self.y_ang[sel] ** 2)
                H2[b], _, _ = np.histogram2d(self.zidx[sel], self.th500[sel],
                                             bins=[np.arange(nz + 1) - 0.5, self.tedges],
                                             weights=(self.bias * self.y_ang)[sel])
            self.H1[s] = H1; self.H2[s] = H2
        self.dVcom = self.dVdz * 4 * np.pi * self.dz                       # (nz,)

    def set_shapes(self, shape_params):
        """shape_params: list of NQ dicts(c500,gamma,alpha,beta), one per q-bin."""
        self.ghat = np.array([ghat_of_shape(sp['c500'], sp['gamma'], sp['alpha'], sp['beta'],
                                            self.u_grid) for sp in shape_params])   # (NQ, ngrid)

    def _gmat(self, ell):
        u = ell[:, None] * self.tmid[None, :]                              # (Nell, Ntheta)
        return np.stack([np.interp(u, self.u_grid, self.ghat[b], left=1.0, right=0.0)
                         for b in range(NQ)])                              # (NQ, Nell, Ntheta)

    def cl_1h(self, ell, s, gmat=None, ab=None):
        G = self._gmat(ell) if gmat is None else gmat
        ab = np.ones(NQ) if ab is None else ab
        cl = sum(ab[b] ** 2 * ((G[b] ** 2) @ self.H1[s][b]) for b in range(NQ))
        return cl / (4 * np.pi)

    def _Pl(self, ell):
        Pl = np.empty((len(self.zsh), len(ell)))
        for iz in range(len(self.zsh)):
            Pl[iz] = np.interp((ell + 0.5) / self.chi_sh[iz], self.k_arr, self.Pk[iz])
        return Pl

    def cl_2h(self, ell, s, gmat=None, Pl=None, ab=None):
        G = self._gmat(ell) if gmat is None else gmat          # (NQ, Nell, Ntheta)
        ab = np.ones(NQ) if ab is None else ab
        Bmat = sum(ab[b] * (G[b] @ self.H2[s][b].T) for b in range(NQ))  # (Nell, nz)
        Bdens = Bmat / self.dVcom[None, :]
        if Pl is None:
            Pl = self._Pl(ell)
        integ = (self.dVdz * self.dz)[None, :] * Pl.T * Bdens ** 2   # (Nell, nz)
        return integ.sum(axis=1)

    def predict(self, ell, ab, a2h, shape_params, Pl=None):
        """ab: per-qbin Y-amplitude (len NQ); a2h: global 2-halo amplitude."""
        self.set_shapes(shape_params)
        G = self._gmat(ell)
        if Pl is None:
            Pl = self._Pl(ell)
        pref = ell * (ell + 1) / (2 * np.pi)
        ab = np.asarray(ab)
        out = {}
        for s in SELS:
            c1 = self.cl_1h(ell, s, gmat=G, ab=ab)
            c2 = self.cl_2h(ell, s, gmat=G, Pl=Pl, ab=ab)
            out[s] = pref * (c1 + a2h * c2)
        return out
