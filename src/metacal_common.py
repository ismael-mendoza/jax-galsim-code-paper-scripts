"""Shared setup for the metacal shear-response figures (Figure 3 and Figure A3).

Metacalibration (metacal) as in JAX-GalSim's `tests/jax/test_metacal_jax.py`, written
out step by step in both GalSim and JAX-GalSim:

1. the observed galaxy image and the PSF image become `InterpolatedImage`s (lanczos15);
2. the galaxy is deconvolved by the PSF, sheared by (g1, g2), and reconvolved with the
   target PSF, drawn with a fixed FFT size and `method="no_pixel"`. The target is a
   Gaussian fit to the PSF dilated by 1 + 2|g| (as in metacal); the PSF here is a round
   Gaussian, so the fit is the PSF itself;
3. noise correction (noisy case only): a second, independent noise image is rotated by
   90 degrees, goes through step 2 with the same shear, is rotated back, and is added.
   A 90-degree rotation flips the sign of (g1, g2), so this cancels, on average, the
   shear response of the sheared noise in the galaxy image.

The per-pixel shear response is the derivative of the metacal image with respect to
(g1, g2) at g = 0: by autodiff (`jax.jacfwd`) in JAX-GalSim, and by central finite
differences of GalSim's metacal image.

|g| has a kink at g = 0, so the image is not differentiable there through the dilation.
Central differences use the same dilation for +h and -h, so they converge to the
symmetric derivative, in which the dilation term cancels: the response with the
dilation held at its g = 0 value. Autodiff gets the same derivative by holding the
dilation fixed (`jax.lax.stop_gradient`).

Everything is float64: `drawImage` returns float32 by default in both codes, and
JAX-GalSim's `InterpolatedImage` keeps float32 inputs as float32.
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"

import galsim
import jax
import jax.numpy as jnp
import jax_galsim as jgs
import numpy as np

from draw_scene_functions import get_bd_galsim

# ---------------------------------------------------------------- parameters
PIXEL_SCALE = 0.2  # arcsec / pixel
STAMP = 33  # pixels, as in test_metacal_jax.py
FFT = 128  # fixed FFT size for the metacal images, as in test_metacal_jax.py
INTERPOLANT = "lanczos15"  # x_interpolant of the InterpolatedImages (ngmix default)

# galaxy: bulge (Spergel, nu = -0.6) + disk (exponential), as in Figures 2 and A1
GALAXY = {
    "flux_d": 7e4,  # electrons
    "flux_b": 3e4,  # electrons
    "hlr_d": 0.6,  # arcsec
    "hlr_b": 0.25,  # arcsec
    "q_d": 0.6,
    "q_b": 0.8,
    "beta": 30.0,  # degrees
}

PSF_FWHM = 0.9  # arcsec, round Gaussian, as in test_metacal_jax.py
# target PSF: Gaussian(fwhm=PSF_FWHM) dilated by 1 + 2|g| (see `_dilation`)

SNR = 20.0  # flux S/N, sqrt(sum I^2) / sigma for the noiseless image
SEED = 42

# finite differences
METACAL_STEP = 0.01  # step used by ngmix / metadetect
# ----------------------------------------------------------------------------


def make_images(noise=True):
    """Observed galaxy image, PSF image, and the noise-correction image (None if noiseless).

    All drawn by GalSim in float64; both codes get the same arrays.
    """
    psf = galsim.Gaussian(fwhm=PSF_FWHM)
    gal = get_bd_galsim(**GALAXY, psf=psf)
    kw = {"nx": STAMP, "ny": STAMP, "scale": PIXEL_SCALE, "dtype": np.float64}
    im = gal.drawImage(**kw).array
    psf_im = psf.drawImage(**kw).array
    if not noise:
        return im, psf_im, None

    sigma = np.sqrt((im**2).sum()) / SNR
    rng = np.random.default_rng(SEED)
    noisy = im + rng.normal(0.0, sigma, im.shape)
    nse = rng.normal(0.0, sigma, im.shape)
    return noisy, psf_im, nse


def _dilation_gs(g1, g2):
    return 1.0 + 2.0 * np.hypot(g1, g2)


def _dilation_jgs(g):
    """1 + 2|g| held fixed under differentiation (see module docstring)."""
    return jax.lax.stop_gradient(1.0 + 2.0 * jnp.sqrt(g[0] ** 2 + g[1] ** 2))


def _gs_render(img, psf_im, g1, g2):
    ii = galsim.InterpolatedImage(
        galsim.ImageD(img), scale=PIXEL_SCALE, x_interpolant=INTERPOLANT
    )
    ip = galsim.InterpolatedImage(
        galsim.ImageD(psf_im), scale=PIXEL_SCALE, x_interpolant=INTERPOLANT
    )
    prof = galsim.Convolve(
        galsim.Convolve(ii, galsim.Deconvolve(ip)).shear(g1=g1, g2=g2),
        galsim.Gaussian(fwhm=PSF_FWHM).dilate(_dilation_gs(g1, g2)),
        gsparams=galsim.GSParams(minimum_fft_size=FFT, maximum_fft_size=FFT),
    )
    return prof.drawImage(
        nx=STAMP, ny=STAMP, scale=PIXEL_SCALE, method="no_pixel", dtype=np.float64
    ).array


def _jgs_render(img, psf_im, g):
    ii = jgs.InterpolatedImage(
        jgs.ImageD(img), scale=PIXEL_SCALE, x_interpolant=INTERPOLANT
    )
    ip = jgs.InterpolatedImage(
        jgs.ImageD(psf_im), scale=PIXEL_SCALE, x_interpolant=INTERPOLANT
    )
    prof = jgs.Convolve(
        jgs.Convolve(ii, jgs.Deconvolve(ip)).shear(g1=g[0], g2=g[1]),
        jgs.Gaussian(fwhm=PSF_FWHM).dilate(_dilation_jgs(g)),
        gsparams=jgs.GSParams(minimum_fft_size=FFT, maximum_fft_size=FFT),
    )
    return prof.drawImage(
        nx=STAMP, ny=STAMP, scale=PIXEL_SCALE, method="no_pixel", dtype=jnp.float64
    ).array


def metacal_galsim(img, psf_im, nse, g1, g2):
    """GalSim metacal image, with the noise correction if `nse` is not None."""
    out = _gs_render(img, psf_im, g1, g2)
    if nse is None:
        return out
    return out + np.rot90(_gs_render(np.rot90(nse, 1), psf_im, g1, g2), 3)


def make_metacal_jgs(img, psf_im, nse):
    """JAX-GalSim metacal image as a function of g = (g1, g2), as `metacal_galsim`."""
    img, psf_im = jnp.asarray(img), jnp.asarray(psf_im)
    nse = None if nse is None else jnp.asarray(nse)

    def f(g):
        out = _jgs_render(img, psf_im, g)
        if nse is None:
            return out
        return out + jnp.rot90(_jgs_render(jnp.rot90(nse, 1), psf_im, g), 3)

    return f


def autodiff_response(f_jgs):
    """dI/dg1 and dI/dg2 at g = 0 by autodiff, shape (2, STAMP, STAMP)."""
    jac = jax.jit(jax.jacfwd(f_jgs))(jnp.zeros(2))  # (STAMP, STAMP, 2)
    return np.moveaxis(np.asarray(jac), -1, 0)


def central_fd(f_gs, h, i):
    """Central finite difference of the GalSim metacal image along g_i (i = 0, 1) at g = 0."""
    e = np.zeros(2)
    e[i] = h
    return (f_gs(*e) - f_gs(*(-e))) / (2 * h)


def fd_vs_ad(ad, fd):
    """max|FD - AD| / max|AD| for each component, shape (2,)."""
    return np.array(
        [np.abs(fd[i] - ad[i]).max() / np.abs(ad[i]).max() for i in range(2)]
    )


def setup(noise=True):
    """Images, GalSim metacal function, JAX-GalSim metacal function, and checks at g = 0.

    noise=True: flux S/N = SNR plus the noise correction; noise=False: noiseless galaxy.
    """
    img, psf_im, nse = make_images(noise)

    def f_gs(g1, g2):
        return metacal_galsim(img, psf_im, nse, g1, g2)

    f_jgs = make_metacal_jgs(img, psf_im, nse)

    # same metacal image at g = 0 in both codes
    im_gs = f_gs(0.0, 0.0)
    im_jgs = np.asarray(jax.jit(f_jgs)(jnp.zeros(2)))
    diff = np.abs(im_jgs - im_gs).max() / np.abs(im_gs).max()
    print(f"metacal image at g = 0: max|JAX-GalSim - GalSim| / peak = {diff:.1e}")
    assert diff < 1e-13, "GalSim and JAX-GalSim metacal images differ at g = 0"
    return img, f_gs, f_jgs, im_gs
