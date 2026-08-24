"""Pressure profiles: GNFW model (JAX) and map-measured stacks (NumPy)."""
from .stacking import default_n_jobs, normalized_profile, stack_normalized

__all__ = [
    "gnfw",
    "A10_PARAMS",
    "projected_shape",
    "y500_normalized_projected",
    "normalized_profile",
    "stack_normalized",
    "default_n_jobs",
]


def __getattr__(name: str):
    # Keep JAX off the stacking import path so fork workers are safe.
    if name in {"A10_PARAMS", "gnfw"}:
        from .gnfw import A10_PARAMS, gnfw

        globals().update(A10_PARAMS=A10_PARAMS, gnfw=gnfw)
        return globals()[name]
    if name in {"projected_shape", "y500_normalized_projected"}:
        from .projection import projected_shape, y500_normalized_projected

        globals().update(
            projected_shape=projected_shape,
            y500_normalized_projected=y500_normalized_projected,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
