"""Halo-model theory spectra via hmfast.

Importing this subpackage loads the hmfast emulators (slow the first time).
"""
from .clyy import cl_yy, dl_of_cl

__all__ = ["cl_yy", "dl_of_cl"]
