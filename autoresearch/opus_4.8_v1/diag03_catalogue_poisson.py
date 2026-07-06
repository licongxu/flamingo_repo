"""Iter 2: exact catalogue Poisson (white) 1-halo floor, with masking.

Low-ell limit: y_ell -> Y_ang, so C_1h -> (1/4pi) sum_a Y_ang,a^2 (white in C_l).
Masking q>q_cut removes the DETECTED clusters from the sum exactly (no profile model).
The catalogue q is 'q_from_Y_5R500c' (same field used to build the map mask).
Compare the catalogue white level (full and masked) to the map's low-ell D_ell.
"""
import numpy as np
from flamingo import paths
from flamingo.catalogue import load_catalogue

cat = load_catalogue(paths.HYDRO_CATALOGUE)
z = cat['z'].values
dA = cat['r_comoving_Mpc'].values / (1 + z)
y_ang = cat['Y_5R500c_Mpc2'].values / dA ** 2          # integrated y [sr]
q = cat['q_from_Y_5R500c'].values

mp = np.load(paths.DATA / 'nb09_tsz_map_ps.npz')
ellb = mp['ellb']
dmap = {'full': mp['dl_map'], 50: mp['dl_q50'], 20: mp['dl_q20'],
        10: mp['dl_q10'], 5: mp['dl_q5']}

print(f'catalogue clusters: {len(cat)},  q range [{np.nanmin(q):.2f}, {np.nanmax(q):.0f}]')


def Cwhite(surv_mask):
    return np.sum(y_ang[surv_mask] ** 2) / (4 * np.pi)


C_full = Cwhite(np.ones(len(cat), bool))
print(f'\nfull-sky white C = {C_full:.3e}')
print(f'{"sel":>6} {"survivors":>10} {"Cwhite":>11} {"Cw/Cfull":>9}  '
      f'D_l@ell30  D_l@ell60  D_l@ell100   map D_l@30 / white@30')
for q_cut, key in [(np.inf, 'full'), (50, 50), (20, 20), (10, 10), (5, 5)]:
    surv = ~(q > q_cut)        # undetected survive (NaN q -> survive)
    Cw = Cwhite(surv)
    row = []
    for L in (30, 60, 100):
        dl_white = L * (L + 1) / (2 * np.pi) * Cw
        row.append(dl_white)
    k30 = np.argmin(np.abs(ellb - 30))
    dl_white_30 = 30 * 31 / (2 * np.pi) * Cw
    print(f'{str(key):>6} {surv.sum():10d} {Cw:11.3e} {Cw/C_full:9.3f}  '
          f'{row[0]:9.2e}  {row[1]:9.2e}  {row[2]:9.2e}   {dmap[key][k30]/dl_white_30:7.2f}')

# Direct: how much of the map's low-ell power is the catalogue Poisson floor?
print('\nmap D_l vs catalogue-white D_l at low ell (white = l(l+1)/2pi * C_white):')
for q_cut, key in [(np.inf, 'full'), (20, 20), (5, 5)]:
    surv = ~(q > q_cut)
    Cw = Cwhite(surv)
    print(f'  --- {key} ---')
    for L in (30, 60, 100, 200):
        k = np.argmin(np.abs(ellb - L))
        dl_white = ellb[k] * (ellb[k] + 1) / (2 * np.pi) * Cw
        print(f'    ell~{ellb[k]:5.0f}  map={dmap[key][k]:.3e}  white={dl_white:.3e}  white/map={dl_white/dmap[key][k]:.3f}')
