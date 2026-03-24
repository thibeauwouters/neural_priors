"""
Compare lambda_tilde posteriors for BNS and NSBH hypotheses across EOS/population
choices, for the three GW events.

Output structure:
    figures/compare_tilde/
        GW170817/bns.pdf
        GW170817/nsbh.pdf
        GW190425/bns.pdf
        ...

CONFIGURATION
─────────────
Edit the two sections marked *** CONFIGURE HERE *** below:

1. CONFIGS  — which (population, eos) NF runs to overlay, with colour/style.
              Adding or removing an entry automatically updates every plot.
2. Global knobs: events, x-range, KDE bandwidth, figure size, font sizes.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from dataclasses import dataclass
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_posterior_data, EOS_SAMPLES_NAMES_DICT, POPULATION_NAMES_DICT

# ===========================================================================
# *** CONFIGURE HERE (1/2): which NF runs to show  ***
# ===========================================================================
# Each entry produces one curve per plot.  Colors/styles are fully explicit so
# you can swap them without touching the plotting code.
# "label" supports LaTeX.

@dataclass
class RunConfig:
    population: str
    eos:        str
    color:      str
    linestyle:  str
    linewidth:  float = 2.0
    label:      str  = ""

    def __post_init__(self):
        if not self.label:
            pop = POPULATION_NAMES_DICT.get(self.population, self.population)
            eos = EOS_SAMPLES_NAMES_DICT.get(self.eos, self.eos)
            self.label = rf"{pop}, {eos}"


CONFIGS: list[RunConfig] = [
    RunConfig("gaussian",        "radio",        "#0472b0", "solid"),
    RunConfig("gaussian",        "radio_chiEFT", "#de8f05", "solid"),
    RunConfig("gaussian",        "radio_NICER",  "#ca7abc", "solid"),
    RunConfig("double_gaussian", "radio_chiEFT", "#de8f05", "dashed"),
]

# ===========================================================================
# *** CONFIGURE HERE (2/2): global settings ***
# ===========================================================================

GW_EVENTS       = ["GW170817", "GW190425", "GW230529"]
SOURCE_TYPES    = ["bns", "nsbh"]     # one output file each
BASE_DIR        = "../final_results"
OUTPUT_BASE_DIR = "./figures/compare_tilde"

N_GRID = 1000
KDE_BW = None   # None → scipy Scott's rule

# X-axis limits: set to None for adaptive (percentile-based) limits derived
# from the data, or provide explicit floats to pin one or both ends.
# Per-(event, source) overrides can be added to XLIM_OVERRIDES below.
XLIM_MIN: float | None = 0.0      # None → adaptive lower bound
XLIM_MAX: float | None = None     # None → adaptive upper bound

# Percentile used for the adaptive upper bound (e.g. 99.5 clips a long tail)
XLIM_UPPER_PERCENTILE = 99.5

# Optional per-plot overrides: keys are (event, source) tuples.
# Each value is a (xmin, xmax) tuple; use None in either position to stay adaptive.
# Example:
#   XLIM_OVERRIDES = {("GW230529", "nsbh"): (0.0, 500.0)}
XLIM_OVERRIDES: dict[tuple[str, str], tuple[float | None, float | None]] = {
    ("GW230529", "nsbh"): (None, 300.0),
}

# Default-run style
DEFAULT_COLOR     = "dimgray"
DEFAULT_LINESTYLE = "dashed"
DEFAULT_LINEWIDTH = 1.8

# Figure / font sizes
FIG_SIZE  = (7, 5)
FS_TICKS  = 18
FS_LABELS = 20
FS_LEGEND = 14
FS_TITLE  = 20

SOURCE_TYPE_LABEL = {"bns": "BNS", "nsbh": "NSBH"}

# ===========================================================================
# Matplotlib style  (no need to touch)
# ===========================================================================

plt.rcParams.update({
    "axes.grid":       False,
    "text.usetex":     True,
    "font.family":     "serif",
    "font.serif":      ["Computer Modern Serif"],
    "ytick.color":     "black",
    "xtick.color":     "black",
    "axes.labelcolor": "black",
    "axes.edgecolor":  "black",
    "xtick.labelsize": FS_TICKS,
    "ytick.labelsize": FS_TICKS,
    "axes.labelsize":  FS_LABELS,
    "legend.fontsize": FS_LEGEND,
})

# ===========================================================================
# I/O helpers
# ===========================================================================

def _path_nf(event: str, source: str, cfg: RunConfig) -> str:
    return os.path.join(BASE_DIR, event, source, cfg.population, cfg.eos, "samples.npz")


def _path_default(event: str, source: str) -> str:
    return os.path.join(BASE_DIR, event, source, "default", "samples.npz")


def _load_lambda_tilde(path: str) -> np.ndarray | None:
    posterior = load_posterior_data(path, fast_mode=True)
    if posterior is None:
        return None
    lt = posterior.get("lambda_tilde")
    return np.asarray(lt) if lt is not None else None


def _kde(samples: np.ndarray, x: np.ndarray) -> np.ndarray:
    s = samples[np.isfinite(samples)]
    if len(s) < 2:
        return np.zeros_like(x)
    return gaussian_kde(s, bw_method=KDE_BW)(x)

# ===========================================================================
# Per-event/source plot
# ===========================================================================

def _compute_xlim(
    all_samples: list[np.ndarray],
    event: str,
    source: str,
) -> tuple[float, float]:
    """Return (xmin, xmax) for the plot, respecting overrides and adaptive logic."""
    override = XLIM_OVERRIDES.get((event, source), (None, None))
    ov_min, ov_max = override

    # Gather finite values from all datasets for adaptive bounds
    combined = np.concatenate([s[np.isfinite(s)] for s in all_samples if len(s) > 0])

    xmin = ov_min if ov_min is not None else (
        XLIM_MIN if XLIM_MIN is not None else float(np.percentile(combined, 0))
    )
    xmax = ov_max if ov_max is not None else (
        XLIM_MAX if XLIM_MAX is not None else float(np.percentile(combined, XLIM_UPPER_PERCENTILE))
    )
    return xmin, xmax


def make_plot(event: str, source: str, out_path: str) -> None:
    """Create and save one compare_tilde figure for (event, source)."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    handles = []
    all_samples: list[np.ndarray] = []

    # --- collect all samples first (needed for adaptive x limits) ---
    default_samples = _load_lambda_tilde(_path_default(event, source))
    if default_samples is not None:
        all_samples.append(default_samples)

    nf_samples: list[np.ndarray | None] = []
    for cfg in CONFIGS:
        s = _load_lambda_tilde(_path_nf(event, source, cfg))
        nf_samples.append(s)
        if s is not None:
            all_samples.append(s)

    xmin, xmax = _compute_xlim(all_samples, event, source)
    x_grid = np.linspace(xmin, xmax, N_GRID)

    # --- default (agnostic) run ---
    if default_samples is not None:
        ax.plot(x_grid, _kde(default_samples, x_grid),
                color=DEFAULT_COLOR, linestyle=DEFAULT_LINESTYLE,
                linewidth=DEFAULT_LINEWIDTH, zorder=2)
        handles.append(mlines.Line2D(
            [], [], color=DEFAULT_COLOR, linestyle=DEFAULT_LINESTYLE,
            linewidth=DEFAULT_LINEWIDTH,
            label=rf"{SOURCE_TYPE_LABEL[source]} (default prior)"))
    else:
        print(f"[WARN] default not found: {_path_default(event, source)}")

    # --- NF runs ---
    for cfg, samples in zip(CONFIGS, nf_samples):
        if samples is not None:
            ax.plot(x_grid, _kde(samples, x_grid),
                    color=cfg.color, linestyle=cfg.linestyle,
                    linewidth=cfg.linewidth, zorder=3)
            handles.append(mlines.Line2D(
                [], [], color=cfg.color, linestyle=cfg.linestyle,
                linewidth=cfg.linewidth, label=cfg.label))
        else:
            print(f"[WARN] not found: {_path_nf(event, source, cfg)}")

    # Axes
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\tilde{\Lambda}$")
    ax.set_ylabel(r"$p(\tilde{\Lambda})$")
    ax.set_title(rf"{event} — {SOURCE_TYPE_LABEL[source]}", fontsize=FS_TITLE, pad=10)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Legend inside axes (upper right by default; easy to move)
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=FS_LEGEND)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    for event in GW_EVENTS:
        for source in SOURCE_TYPES:
            out = os.path.join(OUTPUT_BASE_DIR, event, f"{source}.pdf")
            make_plot(event, source, out)
