# jax-galsim-code-paper-scripts
Scripts for reproducing figures and results of the first jax galsim code paper release.

## Setup

JAX-GalSim is installed (editable) from a checkout next to this repo, `../JAX-GalSim`.

```bash
uv sync                # CPU
uv sync --extra cuda   # GPU (cluster)
```

The CatSim catalog is expected at `output/OneDegSq.fits` (locally and on the cluster).

## Layout

- `src/`: shared modules (`draw_scene_functions.py`, `fig2_scene.py`), installed into
  the environment, so any script can import them.
- `bin/`: timing, profiling and CPU-versus-GPU runs (GPU node).
- `sbatch/gpu-master.sbatch`: makes all GPU results in `output/`.
- `paper/`: figures and tables; `make -C paper` writes them to the paper repository.
- `output/`: results (`timing_results/`, `gpu_results/`, `profiles/`, `cache/`).
