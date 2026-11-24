"""Table 2: full-scene timing, GalSim (1 CPU) versus JAX-GalSim (1 GPU).

Reads the `time_array_results.npz` files written by `bin/timing_tests_general.py`
(arrays `t_galsim`, `t_jgalsim`, `t_transfer`, one entry per scene, in seconds) and
writes a LaTeX table (to `\\input` in main.tex) with the mean and median time per
image for each of the four runs: 250 and 1000 pixel scenes, with and without outliers.

Runs are found by folder name, `{image_slen}-{n_samples}-...-{outliers|no-outliers}`,
inside `--results-dir` (default `output/timing_results`). A run with no folder gets an
empty row.

Notes on the timing script:
- JAX-GalSim compilation is timed separately (`t_compile`: first call minus the first
  timed draw) and given in its own column; it is not in the per-image times.
- The first transfer of each run includes device warm-up (about 0.4 s), which raises
  the mean transfer time; the median is not affected.
"""

from pathlib import Path

import numpy as np
import typer

# ---------------------------------------------------------------- parameters
RESULTS_DIR = Path(__file__).resolve().parents[1] / "output" / "timing_results"
IMAGE_SIZES = (250, 1000)  # pixels
OUTLIER_LABELS = {
    "outliers": "Yes",
    "no-outliers": "No",
}  # folder suffix -> table entry
FMT = "{:.4f}"  # seconds
# speedup = median GalSim time / median JAX-GalSim time (transfer not included)
# ----------------------------------------------------------------------------


def find_run(results_dir: Path, image_slen: int, suffix: str) -> Path | None:
    matches = [
        p
        for p in results_dir.glob(f"{image_slen}-*-{suffix}/time_array_results.npz")
        # "-outliers" also matches "-no-outliers"
        if suffix == "no-outliers" or not p.parent.name.endswith("-no-outliers")
    ]
    if len(matches) > 1:
        raise ValueError(f"More than one run for {image_slen}, {suffix}: {matches}")
    return matches[0] if matches else None


def summarize(fpath: Path) -> dict:
    d = np.load(fpath)
    out = {"n": len(d["t_galsim"])}
    for key in ("t_galsim", "t_jgalsim", "t_transfer"):
        assert len(d[key]) == out["n"]
        out[key] = (np.mean(d[key]), np.median(d[key]))
    out["speedup"] = out["t_galsim"][1] / out["t_jgalsim"][1]
    out["t_compile"] = float(d["t_compile"])
    return out


def table_row(image_slen: int, outliers: str, s: dict | None) -> str:
    cells = [f"${image_slen} \\times {image_slen}$", outliers]
    if s is None:
        cells += [""] * 8
    else:
        for key in ("t_galsim", "t_jgalsim", "t_transfer"):
            cells += [FMT.format(v) for v in s[key]]
        cells.append(f"{s['speedup']:.1f}")
        cells.append(f"{s['t_compile']:.0f}")
    return " & ".join(cells) + " \\\\"


def make_table(rows: list[str], n_samples: list[int]) -> str:
    n_str = ", ".join(sorted({str(n) for n in n_samples}))
    lines = [
        "% Table 2: full-scene timing. Written by jax-galsim-code-paper-scripts paper/table2_timing.py;",
        "% do not edit by hand. Use in main.tex with \\input{tables/table2_timing.txt}.",
        "\\begin{table*}",
        "\\centering",
        "\\footnotesize",
        "\\caption{Timing comparison for drawing full-scene images}",
        "\\label{tab:timing}",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{GalSim (s)} & \\multicolumn{2}{c}{JAX-GalSim (s)}"
        " & \\multicolumn{2}{c}{Transfer (s)} & & \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        "Image (pixels) & Outliers & Mean & Median & Mean & Median & Mean & Median & Speedup & Compile (s) \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\smallskip",
        "\\parbox{\\textwidth}{\\emph{Note.} Tming results comparing drawing full galaxy scenes using GalSim running on a single CPU core vs. JAX-GalSim on one A100 GPU. JAX-GalSim vectorizes the rendering of individual galaxy stamps and pasting them into the full scene (independently for each size bin). GalSim performs both of these operations sequentially."
        " The times (in seconds) are aggregated over "
        + n_str
        + " scenes. The `JAX-GalSim' column does not include compilation or transfer time."
        " `Transfer' is the time to bin the galaxy parameters and move them to the GPU. "
        "`Speedup' is the ratio of the median GalSim and JAX-GalSim times, without transfer or compilation times. `Compile' is the one-time JAX-GalSim compilation time of each run (all size bins), not included in any other column. For more details see Section \\FIXME{YY}.}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def main(
    out: Path = typer.Option(..., help="Output .txt path (LaTeX)."),
    results_dir: Path = typer.Option(RESULTS_DIR, help="Folder with the timing runs."),
):
    rows, n_samples = [], []
    for image_slen in IMAGE_SIZES:
        for suffix, label in OUTLIER_LABELS.items():
            fpath = find_run(results_dir, image_slen, suffix)
            s = summarize(fpath) if fpath is not None else None
            print(
                f"{image_slen} {suffix:12s} {fpath.parent.name if fpath else 'MISSING'}"
            )
            if s is not None:
                n_samples.append(s["n"])
                print(
                    f"   n={s['n']}  galsim {s['t_galsim'][0]:.4f}/{s['t_galsim'][1]:.4f}"
                    f"  jgs {s['t_jgalsim'][0]:.4f}/{s['t_jgalsim'][1]:.4f}"
                    f"  transfer {s['t_transfer'][0]:.4f}/{s['t_transfer'][1]:.4f}"
                    f"  speedup {s['speedup']:.1f}"
                )
            rows.append(table_row(image_slen, label, s))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(make_table(rows, n_samples))
    print(f"Saved {out}")


if __name__ == "__main__":
    typer.run(main)
