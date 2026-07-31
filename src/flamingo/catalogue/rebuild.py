"""Identity-safe FLAMINGO catalogue rebuild metadata and primitives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Family = Literal["l1", "l2"]

L1_VARIANTS = (
    "L1_m9",
    "fgas+2sigma",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
    "Jet",
    "Jet_fgas-4sigma",
)

GEOMETRY_COLUMNS = (
    "snap",
    "soap_index",
    "z",
    "x_Mpc",
    "y_Mpc",
    "z_Mpc",
    "r_comoving_Mpc",
    "shell_idx",
    "theta_nat_rad",
    "phi_nat_rad",
    "lon_nat_deg",
    "lat_nat_deg",
    "theta_rot_rad",
    "phi_rot_rad",
    "lon_rot_deg",
    "lat_rot_deg",
    "x_rot_Mpc",
    "y_rot_Mpc",
    "z_rot_Mpc",
)

SOAP_COLUMNS = (
    "M_500c_Msun",
    "M_200c_Msun",
    "M_200m_Msun",
    "R_500c_Mpc",
    "R_200c_Mpc",
    "R_200m_Mpc",
)

L1_Y_COLUMNS = (
    "Y_500c_Mpc2",
    "Y_500c_noAGN_Mpc2",
    "Y_5R500c_Mpc2",
    "Y_5R500c_noAGN_Mpc2",
)

APERTURE_COLUMNS = (
    "theta_500_arcmin",
    "Y_500cyl_arcmin2",
    "sigma_Y500_arcmin2",
    "npix_in_aperture",
    "q_from_aperture",
)


@dataclass(frozen=True)
class CatalogueTarget:
    """One simulation variant/lightcone and its four canonical CSV products."""

    family: Family
    run: str
    variant: str
    lightcone: int
    catalogue_dir: Path
    map_path: Path

    @property
    def key(self) -> str:
        if self.family == "l1":
            return f"L1_m9/{self.variant}"
        return f"L2p8_m9/lightcone{self.lightcone}"

    @property
    def catalogue_stem(self) -> str:
        return self.variant if self.family == "l1" else "L2p8_m9"

    @property
    def canonical_csvs(self) -> tuple[Path, Path, Path, Path]:
        stem = self.catalogue_stem
        prefix = "halo_catalogue_M500c_"
        suffix = f"_{stem}_yang26rot"
        names = (
            f"{prefix}1e13_zlt3{suffix}.csv",
            f"{prefix}5e13_zlt3{suffix}_qfrommz.csv",
            f"{prefix}5e13_zlt3{suffix}_qfrommz_alpha_fixed_1p12.csv",
            f"{prefix}5e13_zlt3{suffix}_qfrommap.csv",
        )
        return tuple(self.catalogue_dir / name for name in names)  # type: ignore[return-value]


def base_columns(family: Family) -> tuple[str, ...]:
    """Canonical base-catalogue schema for a simulation family."""
    if family == "l1":
        return (*GEOMETRY_COLUMNS, *SOAP_COLUMNS, *L1_Y_COLUMNS)
    if family == "l2":
        return (*GEOMETRY_COLUMNS, *SOAP_COLUMNS)
    raise ValueError(f"unknown catalogue family: {family}")


def q_columns(family: Family) -> tuple[str, ...]:
    """Canonical q-from-mass/redshift schema."""
    return (*base_columns(family), "q_from_mz")


def qmap_columns(family: Family) -> tuple[str, ...]:
    """Canonical q-from-map schema."""
    return (*base_columns(family), *APERTURE_COLUMNS)


def catalogue_targets(root: Path) -> tuple[CatalogueTarget, ...]:
    """Return the complete canonical L1 feedback and L2 lightcone matrix."""
    root = Path(root)
    targets = [
        CatalogueTarget(
            family="l1",
            run="L1_m9",
            variant=variant,
            lightcone=0,
            catalogue_dir=root / "L1_m9/catalogues",
            map_path=(
                root
                / "L1_m9/maps"
                / f"y_unlensed_{variant}_lc0_nside4096.fits"
            ),
        )
        for variant in L1_VARIANTS
    ]
    targets.extend(
        CatalogueTarget(
            family="l2",
            run="L2p8_m9",
            variant="L2p8_m9",
            lightcone=lightcone,
            catalogue_dir=(
                root / "L2p8_m9" / f"lightcone{lightcone}" / "catalogues"
            ),
            map_path=(
                root
                / "L2p8_m9"
                / f"lightcone{lightcone}"
                / "healpix_map"
                / f"y_unlensed_L2p8_m9_lc{lightcone}.fits"
            ),
        )
        for lightcone in range(8)
    )
    return tuple(targets)
