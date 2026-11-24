#!/usr/bin/env python3
"""Figure 3: per-pixel metacal shear response, JAX-GalSim autodiff vs GalSim finite differences.

One noiseless bulge + disk galaxy (Gaussian PSF) goes through metacal (deconvolve, shear,
reconvolve with the PSF dilated by 1 + 2|g|; see `metacal_common.py`). The response
images dI/dg1 and dI/dg2 at g = 0 are computed:

* by autodiff of the JAX-GalSim metacal image (`jax.jacfwd`);
* by central finite differences of the GalSim metacal image, (I(h) - I(-h)) / 2h, for
  each step in FD_STEPS.

Rows: g1, g2. Columns: autodiff and the finite differences (one color scale), then
for each step the difference from autodiff relative to the largest autodiff response.

Also prints the agreement for the noisy galaxy (flux S/N = 20, with the noise
correction), quoted in the text.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import Normalize
from metacal_common import (
    METACAL_STEP,
    STAMP,
    autodiff_response,
    central_fd,
    fd_vs_ad,
    setup,
)
from paper_style import RES_CMAP, TEXT_WIDTH, pixel_axes, set_style

# ---------------------------------------------------------------- parameters
FD_STEPS = (1e-6, METACAL_STEP)  # small step, and the step used by ngmix / metadetect
RESPONSE_CMAP = "PuOr_r"  # diverging: the response changes sign across the galaxy
PIXEL_TICKS = (-15, 0, 15)  # pixels from the stamp center
TITLE_SIZE = 7  # points
# ----------------------------------------------------------------------------


def step_label(h):
    return rf"$h = 10^{{{np.log10(h):.0f}}}$"


def responses(noise):
    _, f_gs, f_jgs, im0 = setup(noise)
    ad = autodiff_response(f_jgs)
    fd = {h: np.array([central_fd(f_gs, h, i) for i in range(2)]) for h in FD_STEPS}
    return ad, fd, np.abs(im0).max()


def main(out: Path = typer.Option(..., help="Output PNG path.")):
    set_style()

    # numbers for the text: noisy galaxy with the noise correction
    ad_n, fd_n, _ = responses(noise=True)
    for h in FD_STEPS:
        e = fd_vs_ad(ad_n, fd_n[h])
        print(
            f"noisy, {h:.0e}: max|FD - AD| / max|AD| = {e[0]:.1e} (g1), {e[1]:.1e} (g2)"
        )

    ad, fd, peak = responses(noise=False)
    for h in FD_STEPS:
        e = fd_vs_ad(ad, fd[h])
        print(
            f"noiseless, {h:.0e}: max|FD - AD| / max|AD| = {e[0]:.1e} (g1), {e[1]:.1e} (g2)"
        )

    half = STAMP / 2
    extent = (-half, half, -half, half)
    n_resp = 1 + len(FD_STEPS)

    fig, axes = plt.subplots(
        2, n_resp + len(FD_STEPS), figsize=(TEXT_WIDTH, 2.55), constrained_layout=True
    )
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)

    # responses: one color scale for both rows and all methods
    v = np.abs(ad).max() / peak
    norm = Normalize(vmin=-v, vmax=v)
    for i in range(2):
        for ax, r in zip(axes[i, :n_resp], [ad[i]] + [fd[h][i] for h in FD_STEPS]):
            imr = ax.imshow(
                r / peak, origin="lower", extent=extent, cmap=RESPONSE_CMAP, norm=norm
            )
    cb = fig.colorbar(
        imr, ax=axes[:, :n_resp].ravel().tolist(), shrink=0.8, pad=0.015, aspect=30
    )
    cb.set_label(r"$\partial I / \partial g$ / peak")

    # residuals: one color scale per step (both rows), power of ten in the colorbar label
    for j, h in enumerate(FD_STEPS):
        res = np.array([(fd[h][i] - ad[i]) / np.abs(ad[i]).max() for i in range(2)])
        exp = int(np.floor(np.log10(np.abs(res).max())))
        res = res / 10.0**exp
        vr = np.abs(res).max()
        col = axes[:, n_resp + j]
        for ax, r in zip(col, res):
            imd = ax.imshow(
                r, origin="lower", extent=extent, cmap=RES_CMAP, norm=Normalize(-vr, vr)
            )
        cbd = fig.colorbar(imd, ax=col.tolist(), shrink=0.8, pad=0.015, aspect=30)
        cbd.ax.set_title(rf"$\times 10^{{{exp}}}$", fontsize=6, pad=3, loc="left")

    for i in range(2):
        row = axes[i]
        pixel_axes(row, PIXEL_TICKS)
        if i == 0:
            for ax in row:
                ax.tick_params(labelbottom=False)
        row[0].text(
            -0.4,
            0.5,
            rf"$g_{i + 1}$",
            transform=row[0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )

    axes[0, 0].set_title("JAX-GalSim\nautodiff", fontsize=TITLE_SIZE)
    for j, h in enumerate(FD_STEPS):
        axes[0, 1 + j].set_title(f"GalSim FD\n{step_label(h)}", fontsize=TITLE_SIZE)
        axes[0, n_resp + j].set_title(
            f"FD $-$ autodiff\n{step_label(h)}", fontsize=TITLE_SIZE
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
