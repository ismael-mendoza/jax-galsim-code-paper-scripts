#!/bin/bash
#SBATCH -J gpu-master
#SBATCH -t 05:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem-per-cpu=8g
#SBATCH --mail-user=ismael@umd.edu
#SBATCH --mail-type=BEGIN,END

# All GPU results for the JAX-GalSim paper, in one job (`sbatch sbatch/gpu-master.sbatch`):
#   1. full-scene timing runs (Table 2): 250 and 1000 pixels, with and without outliers
#   2. roofline profiles: one scene drawn 3 times inside one trace (no compilation or
#      transfer in the trace), 250 and 1000 pixels
#   3. the Figure 2 scene drawn by JAX-GalSim on the CPU and on the GPU (appendix figure)
# The job stops at the first failing command; later steps do not run.
# Expected run time: about 1 hour with N_SAMPLES=1000 (most of it GalSim in the timing
# runs, about 0.06 s per 250-pixel scene and 0.9 s per 1000-pixel scene).

module purge
module load cuda cudnn
unset LD_LIBRARY_PATH

# stop at the first failing command (set after the environment setup, since `module`
# can return nonzero harmlessly)
set -eo pipefail

# run from the repository root; `uv run` uses this repo's .venv, with JAX-GalSim
# installed (editable) from ../JAX-GalSim
cd /home/ismael/scratch/jax-galsim-code-paper-scripts

TIMING_DIR="output/timing_results"  # Table 2 input (paper/table2_timing.py)
PROFILE_DIR="output/profiles"
GPU_DIR="output/gpu_results"  # Figure A2 input (paper/figureA2_cpu_gpu.py)
CACHE_DIR="output/cache"  # GalSim good sizes for the Figure 2 scene
CATSIM="output/OneDegSq.fits"
N_SAMPLES=1000  # scenes per timing run
mkdir -p ${TIMING_DIR} ${PROFILE_DIR} ${GPU_DIR} ${CACHE_DIR}

# the cuda extra installs the GPU build of jax
RUN="uv run --extra cuda python3"

nvidia-smi
${RUN} -c "import jax; print(jax.__version__, jax.devices())"

# ---------------------------------------------------------------- 1. timing (Table 2)
echo "=== timing runs: $(date)"

${RUN} bin/timing_tests_general.py --stamp-slen-bins-str '63,83,103,250' --max-n-gals-bins-str '65,20,5,3' --fft-size-bins-str "128,128,128,256" --image-slen 250 --max-n-gals-global 150 --out-dir ${TIMING_DIR} --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --include-outliers --scan-or-vmap 'scatter-vmap' --cpu-or-gpu 'gpu' --catsim-fpath ${CATSIM} --n-samples ${N_SAMPLES}

${RUN} bin/timing_tests_general.py --stamp-slen-bins-str '63,83,103,250,1000' --max-n-gals-bins-str '1050,300,40,20,3' --fft-size-bins-str "128,128,128,256,1024" --image-slen 1000 --max-n-gals-global 2000 --out-dir ${TIMING_DIR} --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --include-outliers --scan-or-vmap 'scatter-vmap' --cpu-or-gpu 'gpu' --catsim-fpath ${CATSIM} --n-samples ${N_SAMPLES}

${RUN} bin/timing_tests_general.py --out-dir ${TIMING_DIR} --stamp-slen-bins-str '63,83,121' --max-n-gals-bins-str '65,20,10' --fft-size-bins-str "128,128,128" --image-slen 250 --max-n-gals-global 150 --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --scan-or-vmap 'scatter-vmap' --cpu-or-gpu 'gpu' --catsim-fpath ${CATSIM} --n-samples ${N_SAMPLES}

${RUN} bin/timing_tests_general.py --out-dir ${TIMING_DIR} --stamp-slen-bins-str '63,83,123' --max-n-gals-bins-str '1050,300,50' --fft-size-bins-str "128,128,128" --image-slen 1000 --max-n-gals-global 2000 --max-n-iters 5 --psf-type "vary-moffat" --seed 45 --scan-or-vmap 'scatter-vmap' --cpu-or-gpu 'gpu' --catsim-fpath ${CATSIM} --n-samples ${N_SAMPLES}

# ---------------------------------------------------------------- 2. roofline profiles
# same settings as gpu-roofline-scene-profile.sbatch, plus 3 draws per trace
echo "=== roofline profiles: $(date)"

${RUN} bin/profile_scene_gpu.py --stamp-slen-bins-str "63,83,103,250" --max-n-gals-bins-str "65,20,5,3" --fft-size-bins-str "128,128,128,256" --image-slen 250 --max-n-gals-global 150 --out-dir ${PROFILE_DIR} --max-n-iters 5 --seed 42 --scan-or-vmap 'scatter-vmap' --catsim-fpath ${CATSIM} --n-repeats 3

${RUN} bin/profile_scene_gpu.py --stamp-slen-bins-str "63,83,103,250,1000" --max-n-gals-bins-str "1050,300,40,20,3" --fft-size-bins-str "128,128,128,256,1024" --image-slen 1000 --max-n-gals-global 2000 --out-dir ${PROFILE_DIR} --max-n-iters 5 --seed 42 --scan-or-vmap 'scatter-vmap' --catsim-fpath ${CATSIM} --n-repeats 3

# ---------------------------------------------------------------- 3. CPU versus GPU scene
# Figure 2 scene; copy output/gpu_results/ back and run `make -C paper figureA2`
# on any machine
echo "=== CPU versus GPU scene: $(date)"

${RUN} bin/scene_cpu_gpu.py --out ${GPU_DIR}/scene_cpu_gpu.npz --catsim-fpath ${CATSIM} --cache-dir ${CACHE_DIR}

echo "=== done: $(date)"
