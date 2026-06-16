"""Universal masked-tSZ power spectrum model (autoresearch).

Components, all on a shared (ell, M, z) grid:
  C_1h(q) : cluster 1-halo, down-weighted by the n=2 completeness fraction cf2(M,z;q)
  C_2h(q) : 2-halo bracket, down-weighted by the n=1 completeness fraction cf1(M,z;q)

The SELECTION (snr field) is held fixed to the catalogue B=1 Arnaud definition (nb06);
it is the survey selection function and is independent of the pressure profile we fit
for the power spectrum. The completeness fractions depend only on (snr, sigma_lnY, q).

Free model knobs (set by the caller / fitter):
  pressure-profile GNFW params -> reshape y_ell(M,z)
  a1h  : 1-halo amplitude
  a2h  : 2-halo (clustering) amplitude  [freed to absorb nonlinear/diffuse clustering]
  sigma_lnY : intrinsic Y-M scatter entering the completeness

This module only READS hmfast; it never modifies it.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10')

import numpy as np
import jax.numpy as jnp

from flamingo import paths
from flamingo.catalogue import D3A_COSMOLOGY
from hmfast.halos import HaloModel, convert_m_delta, MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.utils import Const
from hmfast.tracers.tsz_completeness import (
    compute_theta500_arcmin, load_sigma_y0_curve, sigma_y0_from_theta,
    conditional_An_undetected)

A10 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
B_SEL = 1.0
Q_CUTS = (50, 20, 10, 5)


def load_map():
    mp = np.load(paths.DATA / 'nb09_tsz_map_ps.npz')
    return mp['ellb'], {'full': mp['dl_map'], 50: mp['dl_q50'], 20: mp['dl_q20'],
                        10: mp['dl_q10'], 5: mp['dl_q5']}


class Model:
    def __init__(self, lmax=6000, nl=40, nm=60, nz=60):
        self.hm = HaloModel(cosmology=D3A_COSMOLOGY)
        self.ell = jnp.logspace(1.0, np.log10(lmax), nl)
        self.m = jnp.logspace(11.0, 15.5, nm)
        self.z = jnp.geomspace(0.001, 3.0, nz)
        self.ell_np = np.asarray(self.ell)
        self.pref = self.ell_np * (self.ell_np + 1) / (2 * np.pi)
        self._build_selection()

    def _build_selection(self):
        hm, m, z = self.hm, self.m, self.z
        mdef500 = MassDefinition(500, 'critical')
        c_old = hm.concentration.c_delta(hm, m, z)
        m500c = convert_m_delta(hm.cosmology, m, z, hm.mass_definition, mdef500, c_old=c_old)
        r500c = mdef500.r_delta(hm.cosmology, m500c, z)
        h = hm.cosmology.H0 / 100.0
        E_z = jnp.atleast_1d(hm.cosmology.hubble_parameter(z))[None, :] / hm.cosmology.H0
        P_500c = (1.65 * (h / 0.7) ** 2 * E_z ** (8.0 / 3.0)
                  * ((m500c * h / B_SEL) / (0.7 * 3.0e14)) ** (2.0 / 3.0 + 0.12) * (0.7 / h) ** 1.5)
        SIGMA_T_CM2, MEC2_EV, I_SHAPE = 6.6524587e-25, 510998.95, 0.470502095
        y0 = (2.0 * (SIGMA_T_CM2 / MEC2_EV) * A10['P0'] * P_500c
              * (r500c * Const._Mpc_over_m_ * 100.0) * I_SHAPE)
        theta500 = compute_theta500_arcmin(hm, m, z, B_SEL)
        coeff, _ = load_sigma_y0_curve()
        self.snr = y0 / sigma_y0_from_theta(theta500, coeff)

    def cf(self, q, n, sigma_lnY):
        if q is None:
            return jnp.ones_like(self.snr)
        norm = np.exp(0.5 * (n * sigma_lnY) ** 2)
        return conditional_An_undetected(self.snr, sigma_lnY, float(q), n_power=n) / norm

    def base_spectra(self, profile_params, sigma_lnY):
        """Return raw (un-amplitude-scaled) C1h(q) and C2h(q) in D_ell, per selection."""
        tracer = tSZTracer(profile=GNFWPressureProfile(**profile_params, B=B_SEL))
        hm, ell, m, z = self.hm, self.ell, self.m, self.z
        out = {}
        for key, q in [('full', None)] + [(qc, qc) for qc in Q_CUTS]:
            w1 = self.cf(q, 2, sigma_lnY)
            w2 = self.cf(q, 1, sigma_lnY)
            if q is None:
                c1 = np.asarray(hm.cl_1h(tracer, tracer, l=ell, m=m, z=z))
                c2 = np.asarray(hm.cl_2h(tracer, tracer, l=ell, m=m, z=z))
            else:
                c1 = np.asarray(hm.cl_1h_masked(tracer, tracer, l=ell, m=m, z=z, mask_mz=w1))
                c2 = np.asarray(hm.cl_2h_masked(tracer, tracer, l=ell, m=m, z=z, mask_mz=w2))
            out[key] = (self.pref * c1, self.pref * c2)
        return out


if __name__ == '__main__':
    M = Model()
    ellb, dmap = load_map()
    base = M.base_spectra(A10, 0.173)

    def interp(d):
        return np.exp(np.interp(np.log(ellb), np.log(M.ell_np), np.log(np.clip(d, 1e-30, None))))

    print('2-halo MASKING RESPONSE C2h(q)/C2h(full) vs map low-ell masked/full:')
    c2full = interp(base['full'][1])
    c1full = interp(base['full'][0])
    for L in [30, 60, 120, 250, 500]:
        k = np.argmin(np.abs(ellb - L))
        row2 = [interp(base[q][1])[k] / c2full[k] for q in Q_CUTS]
        rowmap = [dmap[q][k] / dmap['full'][k] for q in Q_CUTS]
        print(f'  ell~{ellb[k]:5.0f}  2h-resp ' + ' '.join(f'{x:.3f}' for x in row2)
              + '   map-resp ' + ' '.join(f'{x:.3f}' for x in rowmap))
    print('\n1-halo MASKING RESPONSE C1h(q)/C1h(full):')
    for L in [30, 120, 500, 2000]:
        k = np.argmin(np.abs(ellb - L))
        row1 = [interp(base[q][0])[k] / c1full[k] for q in Q_CUTS]
        print(f'  ell~{ellb[k]:5.0f}  ' + ' '.join(f'{x:.3f}' for x in row1))
