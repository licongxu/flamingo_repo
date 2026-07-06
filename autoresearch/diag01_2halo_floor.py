"""Iter 0 diagnostic: is the masked map's low-ell floor set by the full-sky 2-halo term?

Builds the full-sky 1h/2h split with hmfast (A10, B=1, D3A cosmology, Tinker08) and
compares the 2-halo curve to the masked map datapoints.
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.10')

import numpy as np
import jax.numpy as jnp

from flamingo import paths
from flamingo.catalogue import D3A_COSMOLOGY
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

mp = np.load(paths.DATA / 'nb09_tsz_map_ps.npz')
ellb = mp['ellb']
dl = {'full': mp['dl_map'], 'q50': mp['dl_q50'], 'q20': mp['dl_q20'],
      'q10': mp['dl_q10'], 'q5': mp['dl_q5']}

A10 = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)
tracer = tSZTracer(profile=GNFWPressureProfile(**A10, B=1.0))
hm = HaloModel(cosmology=D3A_COSMOLOGY)

ell_th = jnp.logspace(1.0, np.log10(6000.0), 50)
m = jnp.logspace(11.0, 15.5, 60)
z = jnp.geomspace(0.001, 3.0, 60)
cl1h = np.asarray(hm.cl_1h(tracer, tracer, l=ell_th, m=m, z=z))
cl2h = np.asarray(hm.cl_2h(tracer, tracer, l=ell_th, m=m, z=z))
ell_th = np.asarray(ell_th)
pref = ell_th * (ell_th + 1) / (2 * np.pi)
dl_1h, dl_2h = pref * cl1h, pref * cl2h


def i(ell_t, d):
    return np.exp(np.interp(np.log(ellb), np.log(ell_t), np.log(d)))


d1 = i(ell_th, dl_1h)
d2 = i(ell_th, dl_2h)
print(f'{"ell":>6} {"2h/full":>8} {"q5/full":>8} {"q10/full":>8} {"q20/full":>8} '
      f'{"2h/q5":>7} {"2h/q10":>7} {"2h/q20":>7}')
for L in [30, 50, 100, 200, 500, 1000, 2000]:
    k = np.argmin(np.abs(ellb - L))
    print(f'{ellb[k]:6.0f} {d2[k]/dl["full"][k]:8.3f} '
          f'{dl["q5"][k]/dl["full"][k]:8.3f} {dl["q10"][k]/dl["full"][k]:8.3f} '
          f'{dl["q20"][k]/dl["full"][k]:8.3f} '
          f'{d2[k]/dl["q5"][k]:7.3f} {d2[k]/dl["q10"][k]:7.3f} {d2[k]/dl["q20"][k]:7.3f}')

# full-sky theory total vs map
print('\nfull-sky theory (1h+2h)/map:')
for L in [30, 100, 500, 1500, 3000, 5000]:
    k = np.argmin(np.abs(ellb - L))
    print(f'  ell~{ellb[k]:6.0f}  (1h+2h)/map={(d1[k]+d2[k])/dl["full"][k]:.3f}  '
          f'1h frac={d1[k]/(d1[k]+d2[k]):.2f}')
