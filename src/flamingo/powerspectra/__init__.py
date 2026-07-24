"""Power-spectrum estimation.

Three backends; only the light healpy one is imported eagerly:

- :mod:`flamingo.powerspectra.anafast`  full-sky Cl via healpy (imported here).
- :mod:`flamingo.powerspectra.namaster` mask-decoupled pseudo-Cl; import
  explicitly, requires pymaster (``pip install -e ".[powerspectra]"``).
- :mod:`flamingo.powerspectra.flat`     flat-sky FFT Cl of a stamp; import
  explicitly, requires pixell (``pip install -e ".[stamps]"``).
"""
from .anafast import bin_cl, full_sky_cl

__all__ = ["full_sky_cl", "bin_cl"]
