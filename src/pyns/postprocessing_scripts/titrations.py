import argparse
import os
import shutil
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from pyns.postprocessing_utils import compute_recruitment_curves

def _load_npy_dict(path: str) -> Dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            return arr.item()
        except Exception:
            pass
    if isinstance(arr, dict):
        return arr
    raise ValueError(f"Unsupported results format in {path}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _mirror_out_dir(results_root: str, out_parent: Optional[str] = None) -> str:
    results_root = os.path.abspath(results_root.rstrip(os.sep))
    parent, name = (out_parent if out_parent else os.path.dirname(results_root)), os.path.basename(results_root)
    out_root = os.path.join(parent, f"{name}_postprocessed")
    _ensure_dir(out_root)
    return out_root


def _plot_recruitment_all(curves: Dict[str, Any], groups: List[str], out_dir: str) -> None:
    plt.figure(figsize=(7, 4))
    plotted = False
    for group in groups:
        if "DR" in group or "DL" in group or "afferent" in group.lower():
            color = "#2f6b69"
        elif "VR" in group or "VL" in group or "efferent" in group.lower():
            color = "#84009b"
        else:
            # a random color
            color = None
        if group not in curves:
            continue
        data = curves[group]
        stim = data["stim_factors"]
        perc = data["recruitment_percentage"]
        plt.plot(stim, perc, lw=2, marker="o", label=group, color=color)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Stimulation factor")
    plt.ylabel("Recruited axons [%]")
    plt.title("Recruitment curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "titr_recruitment_all_groups.png"), dpi=200)
    plt.close()


def _plot_init_sites_rows(curves: Dict[str, Any], groups: List[str], out_dir: str) -> None:
    # Collect x positions (init sites) per group and plot as scatter rows
    xs = []
    ys = []
    labels = []
    colors = []
    y_pos = 0  # Track actual y-position for groups with data
    for group in groups:
        if "DR" in group or "DL" in group or "afferent" in group.lower():
            color = "#2f6b69"
        elif "VR" in group or "VL" in group or "efferent" in group.lower():
            color = "#84009b"
        else:
            # a random color
            color = "#%06x" % (hash(group) & 0xFFFFFF)
        if group not in curves:
            continue
        ap_sites = [v for v in curves[group].get("ap_init_sites", []) if v is not None]
        if len(ap_sites) == 0:
            continue
        xs.extend(ap_sites)
        ys.extend([y_pos] * len(ap_sites))
        labels.append(group)
        colors.extend([color] * len(ap_sites))
        y_pos += 1  # Increment only when data is plotted
    if len(xs) == 0:
        return
    plt.figure(figsize=(8, 1.5 + 0.6 * max(1, len(labels))))
    plt.scatter(xs, ys, s=18, alpha=0.8, c=colors)
    plt.yticks(range(len(labels)), labels)
    plt.xlim(0, 1)
    plt.xlabel("AP initiation site along axons length")
    plt.title("AP initiation sites by group (at the threshold of each axon)")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "titr_ap_init_sites_all_groups.png"), dpi=200)
    plt.close()


def process_titration_run(run_dir: str, out_run_dir: str, groups: List[str]) -> None:
    res_path = os.path.join(run_dir, "axons_titration_results.npy")
    cfg_src = os.path.join(run_dir, "titrations_config.yaml")
    if not os.path.isfile(res_path):
        return

    _ensure_dir(out_run_dir)

    # Copy config if present
    if os.path.isfile(cfg_src):
        shutil.copy2(cfg_src, os.path.join(out_run_dir, os.path.basename(cfg_src)))

    results = _load_npy_dict(res_path)
    curves = compute_recruitment_curves(results, groups, separator="_")
    _plot_recruitment_all(curves, groups, out_run_dir)
    _plot_init_sites_rows(curves, groups, out_run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess titration results: generate recruitment figures and copy config."
    )
    parser.add_argument(
        "results_dir",
        help="Path to the results root directory containing timestamped run folders.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["S1_DR", "S2_DR", "S1_VR", "S2_VR"],
        help="Group names (default: S1_DR S2_DR S1_VR S2_VR)",
    )
    parser.add_argument(
        "--separator",
        default="_",
        help="Group name separator used to split keywords (default: '_'). Use empty string to disable.",
    )
    parser.add_argument(
        "--stim-factor-step",
        type=float,
        default=None,
        help="Step size for stimulation factors; mutually exclusive with --n-stim-factors.",
    )
    parser.add_argument(
        "--n-stim-factors",
        type=int,
        default=None,
        help="Number of stimulation factors to linearly space; mutually exclusive with --stim-factor-step.",
    )
    parser.add_argument(
        "--postprocessed-dir",
        default=None,
        help=(
            "Parent directory in which to create the '<results_dir>_postprocessed' folder. "
            "Defaults to the parent of results_dir."
        ),
    )

    args = parser.parse_args()
    results_root = os.path.abspath(args.results_dir)
    out_root = _mirror_out_dir(results_root, args.postprocessed_dir)

    # Parse groups and separator
    groups = args.groups if args.groups else []
    separator = args.separator if (args.separator is not None and args.separator != "") else None

    # Validate mutual exclusivity
    if args.stim_factor_step is not None and args.n_stim_factors is not None:
        raise SystemExit("Error: --stim-factor-step and --n-stim-factors are mutually exclusive.")

    # Walk all subdirectories and process those containing titration outputs
    for root, dirs, files in os.walk(results_root):
        if "axons_titration_results.npy" in files:
            # Compute relative path from results_root to this folder
            rel = os.path.relpath(root, results_root)
            out_run_dir = os.path.join(out_root, rel)
            # Pass CLI-configured parameters
            # print the current directory being processed
            print(f"Processing results in: {root}")
            res_path = os.path.join(root, "axons_titration_results.npy")
            cfg_src = os.path.join(root, "titrations_config.yaml")
            _ensure_dir(out_run_dir)
            if os.path.isfile(cfg_src):
                shutil.copy2(cfg_src, os.path.join(out_run_dir, os.path.basename(cfg_src)))

            results = _load_npy_dict(res_path)
            curves = compute_recruitment_curves(
                results,
                groups,
                separator=separator,
                stim_factor_step=args.stim_factor_step,
                n_stim_factors=args.n_stim_factors,
            )
            _plot_recruitment_all(curves, groups, out_run_dir)
            _plot_init_sites_rows(curves, groups, out_run_dir)


if __name__ == "__main__":
    main()
