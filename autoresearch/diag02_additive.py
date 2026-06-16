"""Iter 1: decisive test of the additive decomposition

  map(q, ell) = LS(ell) [masking-robust large-scale/diffuse]  +  C1h_undet(q, ell)

C1h_undet(q) = GNFW 1-halo down-weighted by the n=2 completeness fraction (nb06).
If the residual  R(q) = map(q) - C1h_undet(q)  is q-independent, the universal theory is
LS(ell) + C1h_undet(q), and LS is the same curve for every selection (incl. full sky, cf=1).
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

mp = np.load(paths.DATA / 'nb09_tsz_map_ps.npz')
ellb = mp['ellb']
dmap = {50: mp['dl_q50'], 20: mp['dl_q20'], 10: mp['dl_q10'], 5: mp['dl_q5']}
full = mp['dl_map']

A10 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
B = 1.0
tracer = tSZTracer(profile=GNFWPressureProfile(**A10, B=B))
hm = HaloModel(cosmology=D3A_COSMOLOGY)
SIGMA_LNY = 0.173

ell_th = jnp.logspace(1.0, np.log10(6000.0), 40)
m = jnp.logspace(11.0, 15.5, 60)
z = jnp.geomspace(0.001, 3.0, 60)

# --- SNR field from the same Arnaud GNFW (nb06) ---
mdef500 = MassDefinition(500, 'critical')
c_old = hm.concentration.c_delta(hm, m, z)
m500c = convert_m_delta(hm.cosmology, m, z, hm.mass_definition, mdef500, c_old=c_old)
r500c = mdef500.r_delta(hm.cosmology, m500c, z)
h = hm.cosmology.H0 / 100.0
E_z = jnp.atleast_1d(hm.cosmology.hubble_parameter(z))[None, :] / hm.cosmology.H0
P_500c = (1.65 * (h / 0.7) ** 2 * E_z ** (8.0 / 3.0)
          * ((m500c * h / B) / (0.7 * 3.0e14)) ** (2.0 / 3.0 + 0.12) * (0.7 / h) ** 1.5)
SIGMA_T_CM2, MEC2_EV, I_SHAPE = 6.6524587e-25, 510998.95, 0.470502095
y0 = (2.0 * (SIGMA_T_CM2 / MEC2_EV) * A10['P0'] * P_500c
      * (r500c * Const._Mpc_over_m_ * 100.0) * I_SHAPE)
theta500 = compute_theta500_arcmin(hm, m, z, B)
coeff, _ = load_sigma_y0_curve()
snr = y0 / sigma_y0_from_theta(theta500, coeff)
norm2 = np.exp(0.5 * (2 * SIGMA_LNY) ** 2)
norm1 = np.exp(0.5 * (1 * SIGMA_LNY) ** 2)

ell_np = np.asarray(ell_th)
pref = ell_np * (ell_np + 1) / (2 * np.pi)
cl1h_full = np.asarray(hm.cl_1h(tracer, tracer, l=ell_th, m=m, z=z))
cl2h_full = np.asarray(hm.cl_2h(tracer, tracer, l=ell_th, m=m, z=z))
dl_1h_full = pref * cl1h_full
dl_2h_full = pref * cl2h_full


def i(d):
    return np.exp(np.interp(np.log(ellb), np.log(ell_np), np.log(np.clip(d, 1e-30, None))))


d1h_full = i(dl_1h_full)
d2h_full = i(dl_2h_full)

c1h_undet = {}
for q in (50, 20, 10, 5):
    w1h = conditional_An_undetected(snr, SIGMA_LNY, float(q), n_power=2) / norm2
    cl1h_m = np.asarray(hm.cl_1h_masked(tracer, tracer, l=ell_th, m=m, z=z, mask_mz=w1h))
    c1h_undet[q] = i(pref * cl1h_m)

print("TEST: residual map(q) - C1h_undet(q), normalized by map_full (should be q-independent)")
print(f'{"ell":>6}  ' + '  '.join(f'q{q:<5}' for q in (50, 20, 10, 5)) + f'   {"1h_full/full":>11} {"2h_full/full":>11}')
for L in [30, 50, 100, 200, 500, 1000, 2000, 4000]:
    k = np.argmin(np.abs(ellb - L))
    res = [(dmap[q][k] - c1h_undet[q][k]) / full[k] for q in (50, 20, 10, 5)]
    print(f'{ellb[k]:6.0f}  ' + '  '.join(f'{r:6.3f}' for r in res)
          + f'   {d1h_full[k]/full[k]:11.3f} {d2h_full[k]/full[k]:11.3f}')

np.savez(paths.DATA.parent / 'autoresearch' / 'diag02_cache.npz',
         ellb=ellb, full=full, **{f'map_q{q}': dmap[q] for q in (50, 20, 10, 5)},
         d1h_full=d1h_full, d2h_full=d2h_full,
         **{f'c1h_undet_q{q}': c1h_undet[q] for q in (50, 20, 10, 5)})
print('\nsaved diag02_cache.npz')
