"""Project-wide JAX configuration.

Importing this module enables 64-bit precision, which the pressure-profile and
projection kernels rely on (the FLAMINGO halo model and y-map dynamic range
span many orders of magnitude). Import it before any ``jax.numpy`` use::

    from flamingo import jaxconfig  # noqa: F401  (enables x64)

The call is idempotent, so importing it from several modules is safe.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
