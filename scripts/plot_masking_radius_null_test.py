"""Figures and summary table for the masking-radius null test.

Reads the per-point ``.npz`` files written by
:mod:`masking_radius_null_test` (and, where present, the random-position
controls from :mod:`masking_radius_random_control`) and answers one question:
**how much does the measured tSZ power spectrum depend on the choice to mask
each detected cluster out to ``4 theta_500``?**

Three products:

``masking_radius_null_test``
    Residual power ``D_ell(R) / D_ell(unmasked)`` against masking radius
    ``R = r / theta_500``, one panel per simulation and detection threshold,
    one curve per multipole bin. The production radius is marked. A curve that
    has flattened by ``R = 4`` means the choice is safe at that multipole.

``masking_radius_convergence``
    The same curves normalised to the widest radius run, ``D_ell(R) /
    D_ell(8 theta_500) - 1``, i.e. the fractional power still left to remove at
    radius ``R``. This is the direct systematic-error budget for the choice.

``masking_radius_random_control``
    The real sweep against the random-position control at fixed ``q``. The
    control masks the same number of discs with the same radii at random sky
    positions, so it isolates how much of the decrement is the estimator
    responding to lost area rather than to removed cluster signal.

Run::

    python scripts/plot_masking_radius_null_test.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data_paper" / "masking_radius_null_test"
FIGURES = REPO / "figures"

VARIANTS = ["L1_m9", "L2p8_m9"]
Q_CUTS = [20.0, 10.0, 5.0, 1.0]
R_MULTS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

#: Production masking radius, in units of theta_500.
R_PRODUCTION = 4.0

#: Log-bin indices shown as individual curves (of the 12 bins, centres
#: ~100, 150, 224, 334, 497, 741, 1106, 1650, 2461, 3671, 5476, 8168).
ELL_SHOW = [0, 3, 6, 9, 11]

#: Radii at which the residual systematic is tabulated against R = 4.
R_TABLE = [5.0, 6.0, 8.0]


def radius_tag(r_mult: float) -> str:
    return f"r{r_mult:g}".replace(".", "p")


def load_sweep(variant: str, cut: float, *, random: bool = False) -> dict:
    """Stack one ``(variant, q_cut)`` sweep over masking radius.

    Returns ``r`` (radii actually on disk), ``dl_12`` of shape
    ``(n_radii, 12)``, ``ell_log``, ``f_sky`` and the unmasked reference.
    """
    kind = "random_" if random else ""
    radii, dl, fsky = [], [], []
    for r_mult in R_MULTS:
        path = DATA / f"{variant}_{kind}qgt{cut:g}_{radius_tag(r_mult)}.npz"
        if not path.exists():
            continue
        point = np.load(path)
        radii.append(r_mult)
        dl.append(point["dl_12"])
        fsky.append(float(point["f_sky_raw"]))

    unmasked = np.load(DATA / f"{variant}_unmasked.npz")
    return dict(
        r=np.array(radii),
        dl_12=np.array(dl) if dl else np.zeros((0, 12)),
        f_sky=np.array(fsky),
        ell_log=unmasked["ell_log"],
        dl_unmasked=unmasked["dl_12"],
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def _panel_grid(ylabel: str, title: str):
    fig, axes = plt.subplots(
        len(VARIANTS),
        len(Q_CUTS),
        figsize=(4.0 * len(Q_CUTS), 3.4 * len(VARIANTS)),
        sharex=True,
        sharey="row",
        gridspec_kw={"hspace": 0.12, "wspace": 0.06},
    )
    axes = np.atleast_2d(axes)
    fig.suptitle(title, y=0.97)
    for j in range(len(Q_CUTS)):
        axes[-1, j].set_xlabel(r"masking radius $R$  [$\theta_{500}$]")
    for i in range(len(VARIANTS)):
        axes[i, 0].set_ylabel(ylabel)
    return fig, axes


def figure_residual(sweeps: dict) -> None:
    """``D_ell(R) / D_ell(unmasked)`` -- the fraction of power that survives."""
    fig, axes = _panel_grid(
        r"$D_\ell(R)\,/\,D_\ell^{\rm unmasked}$",
        "Masking-radius null test: surviving tSZ power",
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ELL_SHOW)))

    for i, variant in enumerate(VARIANTS):
        for j, cut in enumerate(Q_CUTS):
            ax = axes[i, j]
            s = sweeps[variant, cut]
            for c, k in zip(colors, ELL_SHOW) if len(s["r"]) else ():
                ratio = s["dl_12"][:, k] / s["dl_unmasked"][k]
                ax.plot(
                    np.concatenate([[0.0], s["r"]]),
                    np.concatenate([[1.0], ratio]),
                    "o-",
                    ms=3.5,
                    lw=1.4,
                    color=c,
                    label=rf"$\ell\simeq{s['ell_log'][k]:.0f}$",
                )
            ax.axvline(R_PRODUCTION, color="0.4", ls="--", lw=1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlim(0.0, max(R_MULTS) + 0.3)
            ax.text(
                0.97, 0.95, rf"{variant}, $q>{cut:g}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left", frameon=False)
    axes[0, 0].annotate(
        r"paper: $4\theta_{500}$",
        xy=(R_PRODUCTION, 0.08), xytext=(R_PRODUCTION + 0.4, 0.08),
        fontsize=7, color="0.4",
    )
    _save(fig, FIGURES / "masking_radius_null_test")


def figure_convergence(sweeps: dict) -> None:
    """``D_ell(R)/D_ell(R_max) - 1`` -- power still left to remove at radius R."""
    fig, axes = _panel_grid(
        r"$D_\ell(R)\,/\,D_\ell(8\theta_{500}) - 1$",
        r"Masking-radius null test: convergence toward the widest mask",
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ELL_SHOW)))

    for i, variant in enumerate(VARIANTS):
        for j, cut in enumerate(Q_CUTS):
            ax = axes[i, j]
            s = sweeps[variant, cut]
            widest = list(s["r"]).index(max(s["r"])) if len(s["r"]) else None
            for c, k in zip(colors, ELL_SHOW) if len(s["r"]) else ():
                ax.plot(
                    s["r"], s["dl_12"][:, k] / s["dl_12"][widest, k] - 1.0,
                    "o-", ms=3.5, lw=1.4, color=c,
                    label=rf"$\ell\simeq{s['ell_log'][k]:.0f}$",
                )
            ax.axvline(R_PRODUCTION, color="0.4", ls="--", lw=1.0)
            ax.axhline(0.0, color="0.7", lw=0.8)
            ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlim(0.0, max(R_MULTS) + 0.3)
            ax.text(
                0.97, 0.95, rf"{variant}, $q>{cut:g}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower left", frameon=False)
    _save(fig, FIGURES / "masking_radius_convergence")


def figure_random_control(sweeps: dict, controls: dict) -> None:
    """Real masking against same-radius discs scattered at random positions."""
    cuts = sorted({cut for _, cut in controls}, reverse=True)
    if not cuts:
        print("no random controls on disk; skipping control figure", flush=True)
        return

    fig, axes = plt.subplots(
        1, len(cuts), figsize=(4.6 * len(cuts), 3.6), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )
    axes = np.atleast_1d(axes)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(ELL_SHOW)))

    for ax, cut in zip(axes, cuts):
        variant = next(v for v, c in controls if c == cut)
        s, ctrl = sweeps[variant, cut], controls[variant, cut]
        for c, k in zip(colors, ELL_SHOW):
            ax.plot(
                s["r"], s["dl_12"][:, k] / s["dl_unmasked"][k],
                "o-", ms=3.5, lw=1.4, color=c,
                label=rf"$\ell\simeq{s['ell_log'][k]:.0f}$",
            )
            ax.plot(
                ctrl["r"], ctrl["dl_12"][:, k] / ctrl["dl_unmasked"][k],
                "s--", ms=3.0, lw=1.1, color=c, alpha=0.65,
            )
        ax.axvline(R_PRODUCTION, color="0.4", ls="--", lw=1.0)
        ax.set_xlabel(r"masking radius $R$  [$\theta_{500}$]")
        ax.set_xlim(0.0, max(R_MULTS) + 0.3)
        ax.text(
            0.97, 0.95, rf"{variant}, $q>{cut:g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
        )
    axes[0].set_ylabel(r"$D_\ell(R)\,/\,D_\ell^{\rm unmasked}$")
    axes[0].legend(fontsize=7, loc="lower left", frameon=False, title="solid: clusters")
    axes[-1].plot([], [], "s--", color="0.4", label="random positions")
    axes[-1].legend(fontsize=7, loc="lower left", frameon=False)
    fig.suptitle("Masking-radius null test: random-position control", y=1.0)
    _save(fig, FIGURES / "masking_radius_random_control")


def write_table(sweeps: dict) -> None:
    """Residual systematic of the production radius, per multipole and cut."""
    lines = [
        "# Masking-radius null test: fractional change in D_ell when the mask",
        "# is widened beyond the production radius R = 4 theta_500.",
        "#   delta(R) = D_ell(R) / D_ell(4 theta_500) - 1",
        "# A small |delta| means the production choice has already converged.",
        "#",
        f"# {'variant':9s} {'q>':>4s} {'ell':>6s} "
        + " ".join(f"{'d(R=' + f'{r:g}' + ')':>10s}" for r in R_TABLE)
        + f" {'fsky(R=4)':>10s}",
    ]
    for variant in VARIANTS:
        for cut in Q_CUTS:
            s = sweeps[variant, cut]
            if R_PRODUCTION not in list(s["r"]):
                continue
            i4 = list(s["r"]).index(R_PRODUCTION)
            for k in range(len(s["ell_log"])):
                deltas = []
                for r in R_TABLE:
                    if r not in list(s["r"]):
                        deltas.append(np.nan)
                        continue
                    ir = list(s["r"]).index(r)
                    deltas.append(s["dl_12"][ir, k] / s["dl_12"][i4, k] - 1.0)
                lines.append(
                    f"  {variant:9s} {cut:>4g} {s['ell_log'][k]:>6.0f} "
                    + " ".join(f"{d:>10.4f}" for d in deltas)
                    + f" {s['f_sky'][i4]:>10.4f}"
                )
    out = DATA / "masking_radius_null_test_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(REPO)}", flush=True)
    print("\n".join(lines))


def main() -> None:
    sweeps = {
        (v, cut): load_sweep(v, cut) for v in VARIANTS for cut in Q_CUTS
    }
    missing = [k for k, s in sweeps.items() if len(s["r"]) < len(R_MULTS)]
    if missing:
        print(f"warning: incomplete sweeps {missing}", flush=True)

    controls = {}
    for v in VARIANTS:
        for cut in Q_CUTS:
            c = load_sweep(v, cut, random=True)
            if len(c["r"]):
                controls[v, cut] = c

    figure_residual(sweeps)
    figure_convergence(sweeps)
    figure_random_control(sweeps, controls)
    write_table(sweeps)


if __name__ == "__main__":
    main()
