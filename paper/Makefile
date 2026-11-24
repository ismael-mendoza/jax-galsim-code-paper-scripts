# Make all figures and tables for the JAX-GalSim paper (run from the repository root).
#
#   make -C paper            # all figures and tables
#   make -C paper figure1    # one figure
#   make -C paper table1     # versions in Table 1 (the table itself is by hand)
#   make -C paper FIG_DIR=/some/other/folder
#
# Figures are only remade when their script (or a script they use) changes;
# `make -B` forces a full rebuild.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO := $(abspath $(HERE)/..)
SRC := $(REPO)/src
OUTPUT := $(REPO)/output

FIG_DIR ?= $(HOME)/Sync/manuscripts/jax-galsim-paper/figures
TAB_DIR ?= $(abspath $(FIG_DIR)/../tables)
CATSIM ?= $(OUTPUT)/OneDegSq.fits
CACHE_DIR ?= $(OUTPUT)/cache
PYTHON ?= uv run --project $(REPO) python

# Results made on a GPU node by sbatch/gpu-master.sbatch (copy output/ back from the cluster)
TIMING_RESULTS ?= $(OUTPUT)/timing_results
GPU_RESULTS ?= $(OUTPUT)/gpu_results
SCENE_CPU_GPU := $(GPU_RESULTS)/scene_cpu_gpu.npz

FIG1 := $(FIG_DIR)/figure1_fixed_sizes.png
FIG2 := $(FIG_DIR)/figure2_full_scene.png
FIG3 := $(FIG_DIR)/figure3_metacal_response.png
FIGA1 := $(FIG_DIR)/figureA1_moffat_table.png
FIGA2 := $(FIG_DIR)/figureA2_cpu_gpu.png
FIGA3 := $(FIG_DIR)/figureA3_metacal_steps.png
TAB1_VERSIONS := $(TAB_DIR)/table1_versions.txt
TAB2 := $(TAB_DIR)/table2_timing.txt

.PHONY: all figure1 figure2 figure3 figureA1 figureA2 figureA3 table1 table2 clean FORCE

all: figure1 figure2 figure3 figureA1 figureA2 figureA3 table1 table2

figure1: $(FIG1)
figure2: $(FIG2)
figure3: $(FIG3)
figureA1: $(FIGA1)
figureA2: $(FIGA2)
figureA3: $(FIGA3)
table1: $(TAB1_VERSIONS)
table2: $(TAB2)

# Always rerun: the versions change without any file here changing.
$(TAB1_VERSIONS): FORCE
	cd $(HERE) && $(PYTHON) table1_versions.py --out $@

$(FIG1): $(HERE)/figure1_fixed_sizes.py $(HERE)/paper_style.py
	cd $(HERE) && $(PYTHON) figure1_fixed_sizes.py --out $@

$(FIG2): $(HERE)/figure2_full_scene.py $(HERE)/paper_style.py $(SRC)/fig2_scene.py $(SRC)/draw_scene_functions.py
	cd $(HERE) && $(PYTHON) figure2_full_scene.py --out $@ --catsim-fpath $(CATSIM) --cache-dir $(CACHE_DIR)

$(FIG3): $(HERE)/figure3_metacal_response.py $(HERE)/metacal_common.py $(HERE)/paper_style.py $(SRC)/draw_scene_functions.py
	cd $(HERE) && $(PYTHON) figure3_metacal_response.py --out $@

$(FIGA1): $(HERE)/figureA1_moffat_table.py $(HERE)/paper_style.py $(SRC)/draw_scene_functions.py
	cd $(HERE) && $(PYTHON) figureA1_moffat_table.py --out $@

$(FIGA2): $(HERE)/figureA2_cpu_gpu.py $(HERE)/paper_style.py $(SCENE_CPU_GPU)
	cd $(HERE) && $(PYTHON) figureA2_cpu_gpu.py --npz $(SCENE_CPU_GPU) --out $@

$(FIGA3): $(HERE)/figureA3_metacal_steps.py $(HERE)/metacal_common.py $(HERE)/paper_style.py $(SRC)/draw_scene_functions.py
	cd $(HERE) && $(PYTHON) figureA3_metacal_steps.py --out $@

$(TAB2): $(HERE)/table2_timing.py $(wildcard $(TIMING_RESULTS)/*/time_array_results.npz)
	cd $(HERE) && $(PYTHON) table2_timing.py --results-dir $(TIMING_RESULTS) --out $@

clean:
	rm -f $(FIG1) $(FIG2) $(FIG3) $(FIGA1) $(FIGA2) $(FIGA3) $(TAB1_VERSIONS) $(TAB2)

FORCE:
