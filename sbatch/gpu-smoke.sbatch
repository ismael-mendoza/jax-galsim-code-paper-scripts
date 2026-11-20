#!/bin/bash
#SBATCH -J gpu-smoke
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu-a100_1g.5gb
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --mem-per-cpu=8g

# Quick check before `sbatch/gpu-master.sbatch`, on a small MIG slice of an A100
# (1/7 of the GPU, 5 GB), which is usually idle. It checks the environment, that JAX
# runs on the GPU, and runs each step of gpu-master.sbatch once with tiny settings
# (250-pixel scenes only, since 1000-pixel scenes may not fit in 5 GB). Results go to
# output/smoke/, not to the folders the paper reads. Timings here are meaningless.
#
# `sbatch sbatch/gpu-smoke.sbatch`, or `bash sbatch/gpu-smoke.sbatch` inside an
# interactive GPU shell.

module purge
module load cuda cudnn
unset LD_LIBRARY_PATH

set -eo pipefail

cd /home/ismael/scratch/jax-galsim-code-paper-scripts

SMOKE_DIR="output/smoke"
CATSIM="output/OneDegSq.fits"
mkdir -p ${SMOKE_DIR}

RUN="uv run --extra cuda python3"

# ---------------------------------------------------------------- 0. environment
echo "=== environment: $(date)"
test -f ${CATSIM} || { echo "missing ${CATSIM}"; exit 1; }
test -d ../JAX-GalSim || { echo "missing ../JAX-GalSim"; exit 1; }
nvidia-smi

# JAX_PLATFORMS=cuda makes JAX fail here instead of falling back to the CPU
JAX_PLATFORMS=cuda ${RUN} -c "
import jax, jax_galsim
print('jax', jax.__version__, 'jax_galsim', jax_galsim.__version__, jax.devices())
assert jax.default_backend() == 'gpu', jax.devices()
print('sum on GPU:', jax.numpy.ones(1000).sum())
"

# ---------------------------------------------------------------- 1. timing
echo "=== timing: $(date)"
${RUN} bin/timing_tests_general.py --stamp-slen-bins-str '63,83,103,250' --max-n-gals-bins-str '65,20,5,3' --fft-size-bins-str "128,128,128,256" --image-slen 250 --max-n-gals-global 150 --out-dir ${SMOKE_DIR}/timing --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --include-outliers --scan-or-vmap 'scatter-vmap' --cpu-or-gpu 'gpu' --catsim-fpath ${CATSIM} --n-samples 2

# ---------------------------------------------------------------- 2. roofline profile
echo "=== profile: $(date)"
${RUN} bin/profile_scene_gpu.py --stamp-slen-bins-str "63,83,103,250" --max-n-gals-bins-str "65,20,5,3" --fft-size-bins-str "128,128,128,256" --image-slen 250 --max-n-gals-global 150 --out-dir ${SMOKE_DIR}/profiles --max-n-iters 5 --seed 42 --scan-or-vmap 'scatter-vmap' --catsim-fpath ${CATSIM} --n-repeats 1

# ---------------------------------------------------------------- 3. CPU versus GPU scene
echo "=== CPU versus GPU scene: $(date)"
${RUN} bin/scene_cpu_gpu.py --out ${SMOKE_DIR}/scene_cpu_gpu.npz --catsim-fpath ${CATSIM} --cache-dir ${SMOKE_DIR}/cache

echo "=== smoke test passed: $(date)"
