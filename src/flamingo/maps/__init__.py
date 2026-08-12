"""HEALPix map I/O and sampling.

Flat-sky stamps live in :mod:`flamingo.maps.stamps`; import that module
explicitly, it requires pixell (``pip install -e ".[stamps]"``).
"""
from .io import nside_of, read_map, write_map
from .sampling import sample_at, sample_neighbour_max

__all__ = ["read_map", "write_map", "nside_of", "sample_at", "sample_neighbour_max"]
