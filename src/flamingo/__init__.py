"""General-purpose tools for FLAMINGO lightcone maps, catalogues, and profiles.

Subpackages (import explicitly to keep package import cheap):

- ``flamingo.maps``         HEALPix I/O and sampling (NumPy/healpy); flat-sky
                            stamps in ``maps.stamps`` (optional, needs pixell).
- ``flamingo.masking``      binary cluster discs masks (NumPy/healpy).
- ``flamingo.profiles``     GNFW pressure model + projection (JAX/GPU) and
                            map-measured radial stacks (NumPy/healpy).
- ``flamingo.catalogue``    SOAP catalogue loading and rotation-frame geometry.
- ``flamingo.powerspectra`` full-sky Cl via healpy; mask-decoupled pseudo-Cl
                            via NaMaster and flat-sky FFT Cl via pixell in
                            explicit submodules (optional dependencies).
- ``flamingo.theory``       halo-model C_ell^yy via hmfast (loads emulators).

The numeric model kernels (``flamingo.profiles.gnfw`` and
``flamingo.profiles.projection``) are written in JAX and enable float64; the
HEALPix-bound code stays in NumPy because healpy pixel queries are CPU-only.
"""
from __future__ import annotations

from . import paths

__version__ = "0.1.0"
__all__ = ["paths", "__version__"]
