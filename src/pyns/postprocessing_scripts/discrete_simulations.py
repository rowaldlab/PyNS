import argparse
import os
import shutil
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from pyns.postprocessing_utils import compute_air_eir_curves

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


def _group_color(name: str) -> str:
    n = name.lower()
    if "afferents" in n:
        return "#2f6b69"
    if "efferents" in n:
        return "#84009b"
    # fallback random but stable by name
    import random
    random.seed(hash(name) & 0xFFFFFFFF)
    return "#%06x" % random.randint(0, 0xFFFFFF)


def _plot_recruitment_all(curves: Dict[str, Any], out_dir: str) -> None:
    # Plot all groups' recruitment curves on one figure
    any_plotted = False
    plt.figure(figsize=(8, 5))
    for key, data in curves.items():
        if "recruitment_percentage" not in data:
            continue
        stim = data["stim_factors"]
        perc = data["recruitment_percentage"]
        plt.plot(stim, perc, lw=2, marker="o", label=key, color=_group_color(key))
        any_plotted = True
    if not any_plotted:
        plt.close()
        return
    plt.xlabel("Stimulation factor")
    plt.ylabel("Recruited axons [%]")
    plt.title("Recruitment curves")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "discrete-sim_recruitment_all_groups.png"), dpi=200)
    plt.close()


def _plot_air_eir_per_group(key: str, data: Dict[str, Any], out_dir: str) -> None:
    # Plot AIR vs EIR for a single efferent group with specified colors
    if not all(k in data for k in ("n_eir", "n_air")):
        return
    stim = data["stim_factors"]
    plt.figure(figsize=(7, 4))
    # Colors as requested: AIR -> #2f6b69, EIR -> #84009b
    plt.plot(stim, data["n_air"], marker="^", lw=2, label="AIR (reflex)", color="#2f6b69")
    plt.plot(stim, data["n_eir"], marker="s", lw=2, label="EIR (direct)", color="#84009b")
    plt.xlabel("Stimulation factor")
    plt.ylabel("Responses")
    plt.title(f"AIR vs EIR - {key}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"discrete-sim_air_eir_{key}.png"), dpi=200)
    plt.close()


def process_discrete_run(run_dir: str, out_run_dir: str, groups: List[str]) -> None:
    res_path = os.path.join(run_dir, "axons_discrete_simulations_results.npy")
    cfg_src = os.path.join(run_dir, "discrete_simulations_config.yaml")
    if not os.path.isfile(res_path):
        return

    _ensure_dir(out_run_dir)

    # Copy config if present
    if os.path.isfile(cfg_src):
        shutil.copy2(cfg_src, os.path.join(out_run_dir, os.path.basename(cfg_src)))

    results = _load_npy_dict(res_path)
    curves = compute_air_eir_curves(results, groups, separator="_")

    _plot_recruitment_all(curves, out_run_dir)
    # Make AIR/EIR plots per efferent group
    for key, data in curves.items():
        if "efferents" in key.lower():
            _plot_air_eir_per_group(key, data, out_run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postprocess discrete simulation results: generate AIR/EIR and recruitment figures and copy config."
    )
    parser.add_argument(
        "results_dir",
        help="Path to the results root directory containing timestamped run folders.",
    )
    parser.add_argument(
        "--postprocessed-dir",
        default=None,
        help=(
            "Parent directory in which to create the '<results_dir>_postprocessed' folder. "
            "Defaults to the parent of results_dir."
        ),
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

    args = parser.parse_args()
    results_root = os.path.abspath(args.results_dir)
    out_root = _mirror_out_dir(results_root, args.postprocessed_dir)
    groups = args.groups if args.groups else []
    separator = args.separator if (args.separator is not None and args.separator != "") else None

    # Walk all subdirectories and process those containing discrete outputs
    for root, dirs, files in os.walk(results_root):
        if "axons_discrete_simulations_results.npy" in files:
            rel = os.path.relpath(root, results_root)
            out_run_dir = os.path.join(out_root, rel)
            # Inline processing to inject CLI-configured parameters
            print(f"Processing results in: {root}")
            res_path = os.path.join(root, "axons_discrete_simulations_results.npy")
            cfg_src = os.path.join(root, "discrete_simulations_config.yaml")
            _ensure_dir(out_run_dir)
            if os.path.isfile(cfg_src):
                shutil.copy2(cfg_src, os.path.join(out_run_dir, os.path.basename(cfg_src)))
            # load config to get passive_end_nodes parameter
            with open(cfg_src, "r") as f:
                import yaml
                cfg = yaml.safe_load(f)
            passive_end_nodes = cfg.get("passive_end_nodes", False)

            results = _load_npy_dict(res_path)
            curves = compute_air_eir_curves(results, groups, separator=separator, passive_end_nodes=passive_end_nodes)

            _plot_recruitment_all(curves, out_run_dir)
            for key, data in curves.items():
                if "efferents" in key.lower():
                    _plot_air_eir_per_group(key, data, out_run_dir)


if __name__ == "__main__":
    main()
