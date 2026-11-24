#!/usr/bin/env python3
"""Validation number for Sec 4.3: GalSim versus JAX-GalSim over many full scenes.

Same setup as Figure 2 (`src/fig2_scene.py`: 250-pixel scenes, round Moffat PSF
FWHM 0.8 arcsec, bins of the 250-pixel run with outliers), for samples
0, ..., N_SCENES - 1 of the timing run with seed 45 (Figure 2 is sample 0).

For each scene:
* GalSim "default" (GalSim picks stamp and FFT size per galaxy), which also checks that
  every JAX-GalSim bin is at least as large as GalSim's stamp (Check 1);
* GalSim "matched" (each galaxy drawn with its JAX-GalSim bin's stamp and FFT size);
* JAX-GalSim (compiled once, reused for all scenes).

Prints max|JAX-GalSim - GalSim| / peak (peak of the GalSim default scene) for both
residuals, per scene summary and the maximum over all scenes.

Usage (from the repository root, CPU is enough):
    uv run python paper/validate_scenes.py --catsim-fpath output/OneDegSq.fits \
        --cache-dir output/cache
"""

import os

# before importing jax; the scene itself (random positions) depends on it
os.environ["JAX_ENABLE_X64"] = "True"

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import typer
from jax import device_put, jit, random

from draw_scene_functions import (
    draw_all_bins_jgs,
    draw_all_galsim_stratified,
    draw_galsim,
    draw_jgs_vmap_scatter_stamps,
    get_good_sizes_galsim,
    get_one_full_sample,
    prepare_catalog,
    prepare_per_bin_samples,
)
from fig2_scene import (
    BUFFER,
    FFT_SIZE_BINS,
    IMAGE_SLEN,
    MAX_MAG,
    MAX_N_GALS_BINS,
    MAX_N_GALS_GLOBAL,
    MAX_N_ITERS,
    N_SAMPLES_RUN,
    SEED,
    STAMP_SLEN_BINS,
    psf_galsim,
    psf_jgs,
)

# ---------------------------------------------------------------- parameters
N_SCENES = 100
# ----------------------------------------------------------------------------


def main(
    catsim_fpath: Path = typer.Option(..., help="CatSim OneDegSq.fits file."),
    cache_dir: Path = typer.Option(..., help="Folder to cache GalSim good sizes."),
    n_scenes: int = typer.Option(N_SCENES, help="Number of scenes."),
    out: Path = typer.Option(None, help="Optional .npz with per-scene results."),
):
    assert jax.config.jax_enable_x64
    psf_gs = psf_galsim()
    device = jax.devices()[0]
    print(f"device: {device}")

    # catalog, good sizes and cuts, as in fig2_scene.load_scene
    cat = prepare_catalog(str(catsim_fpath), min_hlr=0.0, max_mag=MAX_MAG)
    good_sizes, good_fft_sizes = get_good_sizes_galsim(
        cat=cat, psf=psf_gs, max_mag=MAX_MAG, min_hlr=0.0, out_path=cache_dir
    )
    cat["good_size"] = good_sizes
    cat["good_fft_size"] = good_fft_sizes
    cat = cat[cat["good_size"] + BUFFER <= IMAGE_SLEN]
    for s, f in zip(STAMP_SLEN_BINS, FFT_SIZE_BINS):
        assert np.all(cat[cat["good_size"] <= s]["good_fft_size"] <= f)

    # JAX-GalSim drawing function, compiled once
    draw_fncs = [
        jit(partial(draw_jgs_vmap_scatter_stamps, ilen=IMAGE_SLEN, slen=s, fft_size=f, max_n_gals=m))
        for s, f, m in zip(STAMP_SLEN_BINS, FFT_SIZE_BINS, MAX_N_GALS_BINS)
    ]
    all_draw_fnc = jit(
        partial(
            draw_all_bins_jgs,
            ilen=IMAGE_SLEN,
            draw_fncs=draw_fncs,
            device=device,
            n_bins=len(STAMP_SLEN_BINS),
        )
    )

    rkeys = random.split(random.key(SEED), N_SAMPLES_RUN)
    rows = []
    for ii in range(n_scenes):
        _, k2 = random.split(rkeys[ii])
        sample, n_gals, gsizes = get_one_full_sample(
            k2, cat=cat, ilen=IMAGE_SLEN, max_n_gals=MAX_N_GALS_GLOBAL
        )
        gs_default = draw_galsim(
            sample,
            n_gals,
            ilen=IMAGE_SLEN,
            psf=psf_gs,
            check_stamp_sizes=True,
            max_slen=STAMP_SLEN_BINS[-1],
            good_sizes=gsizes,
            buffer=BUFFER,
            size_bins=STAMP_SLEN_BINS,
        )
        gs_matched = draw_all_galsim_stratified(
            sample,
            ilen=IMAGE_SLEN,
            psf=psf_gs,
            gsizes=gsizes,
            stamp_slen_bins=STAMP_SLEN_BINS,
            fft_size_bins=FFT_SIZE_BINS,
            buffer=BUFFER,
        )
        samples_per_bin, n_iters_per_bin = prepare_per_bin_samples(
            sample,
            gsizes,
            stamp_slen_bins=STAMP_SLEN_BINS,
            max_n_gals_bins=MAX_N_GALS_BINS,
            max_n_iters=MAX_N_ITERS,
            buffer=BUFFER,
        )
        inputs = device_put(
            (samples_per_bin, jnp.asarray(n_iters_per_bin), psf_jgs()), device=device
        )
        jgs_scene = np.asarray(jax.block_until_ready(all_draw_fnc(*inputs)))

        peak = gs_default.max()
        r_def = np.abs(jgs_scene - gs_default).max() / peak
        r_mat = np.abs(jgs_scene - gs_matched).max() / peak
        dflux_def = (jgs_scene - gs_default).sum() / gs_default.sum()
        dflux_mat = (jgs_scene - gs_matched).sum() / gs_default.sum()
        rows.append((n_gals, r_def, r_mat, dflux_def, dflux_mat))
        print(
            f"scene {ii:3d}  n_gals={n_gals:4d}  default {r_def:.2e} ({dflux_def:+.1e})"
            f"  matched {r_mat:.2e} ({dflux_mat:+.1e})",
            flush=True,
        )

    a = np.array(rows)
    print(f"\n{n_scenes} scenes, {int(a[:, 0].sum())} galaxies (Check 1 passed for all)")
    for name, j in (("default", 1), ("matched", 2)):
        print(
            f"{name:8s} max|res|/peak: max {a[:, j].max():.2e}, median {np.median(a[:, j]):.2e};"
            f"  max|dflux/flux| {np.abs(a[:, j + 2]).max():.1e}"
        )
    if out is not None:
        np.savez(out, n_gals=a[:, 0], res_default=a[:, 1], res_matched=a[:, 2],
                 dflux_default=a[:, 3], dflux_matched=a[:, 4])
        print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
