#!/usr/bin/env python3
"""Appendix figure: agreement of GalSim finite differences with JAX-GalSim autodiff
against the finite-difference step, for the metacal shear response of Figure 3.

Same noiseless galaxy and metacal steps as Figure 3. `--noise` instead uses the noisy
galaxy (flux S/N = 20) with the noise correction (`metacal_common.setup(noise=True)`);
it is not used in the paper (the noise does not change the autodiff point and muddies
it). For each step h, the largest per-pixel
difference between the central-difference response, (I(h) - I(-h)) / 2h, and the
autodiff response, relative to the largest autodiff response; the worse of g1 and g2.

What sets the curve (noiseless; from the checks in the paper notes):

* small h (below about 3e-8): round-off of the images, divided by h; the minimum is
  about 2e-7 at h = 3e-8;
* larger h: central differences at g = 0 converge as O(h), about 3.7 h: both I(h) and
  I(-h) use the target PSF dilated by 1 + 2h (the k-space interpolant of the
  `InterpolatedImage` also has a jump in its second derivative at g = 0, but that
  term is about 1000 times smaller);
* around h = 1e-3: a small step above the line, from the hard cutoff of `Deconvolve`
  at maxk, which moves with the shear, so k-grid points switch on and off between
  I(h) and I(-h). No grid point switches for h <= 3e-4; the first ones switch at
  h = 5e-4 (g1) and 1e-3 (g2), exactly where the curves leave the line, and moving the
  cutoff to high k removes the step (checked in the paper notes). With noise (`--noise`) this jump is much larger (0.25-0.6 at
  h >= 1e-3), since the deconvolved high-k noise is not damped.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from metacal_common import METACAL_STEP, autodiff_response, central_fd, fd_vs_ad, setup
from paper_style import COLUMN_WIDTH, set_style

# ---------------------------------------------------------------- parameters
STEPS = np.logspace(-9, -1, 17)
# ----------------------------------------------------------------------------


def main(
    out: Path = typer.Option(..., help="Output PNG path."),
    noise: bool = typer.Option(
        False, help="Noisy galaxy (flux S/N = 20) with the noise correction."
    ),
):
    set_style()

    _, f_gs, f_jgs, _ = setup(noise=noise)
    ad = autodiff_response(f_jgs)

    err = np.array(
        [fd_vs_ad(ad, [central_fd(f_gs, h, i) for i in range(2)]) for h in STEPS]
    )  # (len(STEPS), 2)
    for h, e in zip(STEPS, err):
        print(f"h = {h:.0e}: g1 {e[0]:.1e}  g2 {e[1]:.1e}")
    worst = err.max(axis=1)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4), constrained_layout=True)
    ax.loglog(STEPS, worst, "-", marker="o", ms=2.5, lw=1, color="C0")
    ax.axvline(METACAL_STEP, color="0.5", ls=":", lw=0.8)
    ax.text(
        METACAL_STEP * 0.8,
        0.03,
        "metacal step",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.4",
    )
    ax.set_xlabel("finite-difference step $h$")
    ax.set_ylabel(r"max$|$FD $-$ autodiff$|$ / max$|$autodiff$|$")
    ax.tick_params(which="both", direction="in", top=True, right=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
