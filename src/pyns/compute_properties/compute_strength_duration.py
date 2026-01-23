'''
###########################################
# File: pyns/compute_strength_duration.py
# Project: pyns
# Author: Abdallah Alashqar (abdallah.j.alashqar@fau.de)
# -----
# PI: Andreas Rowald, PhD (andreas.rowald@fau.de)
# Associate Professor for Digital Health
# Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
https://www.pdh.med.fau.de/
############################################
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
from ..utils import *
import time
import argparse
import os
import copy
try:
    from mpi4py import MPI
except Exception:
    MPI = None
    print(
        "Warning: MPI (mpi4py) is not installed; parallel processing is disabled. Using rank=0, size=1.",
        flush=True,
    )

from ..titration_utils import titrate_axon
from ..utils import DummyComm

# get project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute strength-duration curve of an axon model under extracellular stimulation"
    )
    parser.add_argument(
        "--field_path",
        type=str,
        required=False,
        default=os.path.join(project_path, "test_dataset", "lumbar-tSCS_cathode_T11-T12_anode_navel-sides_units_V_m_cropped.h5"),
        help="Path to the electric field file (.npy or .h5)",
    )
    parser.add_argument(
        "--axons_path",
        type=str,
        required=False,
        default=os.path.join(project_path, "test_dataset", "RightSoleusAxons_diams_from_Schalow1992_cropped.npy"),
        help="Path to the axon trajectories file (.npy)",
    )
    parser.add_argument(
        "--axon_name",
        type=str,
        required=False,
        default=None,
        help="Name of the axon to simulate (if None, use the first axon in the file)",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        required=False,
        default="alashqar",
        help="Model variant to use: 'Alashqar', 'Gaines', or 'MRG'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=os.path.join(os.getcwd(), "strength_duration_results"),
        help="Directory to save the results",
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        required=False,
        default=None,
        help="Suffix to add to the results filename",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        required=False,
        default=None,
        help="Axon diameter in um to overwrite the one in the axon trajectory file",
    )
    parser.add_argument(
        "--sensory_only",
        action="store_true",
        help="If set, simulate only a sensory type of the axon",
    )
    parser.add_argument(
        "--motor_only",
        action="store_true",
        help="If set, simulate only a motor type of the axon",
    )
    parser.add_argument(
        "--time_step",
        type=float,
        required=False,
        default=0.005,
        help="Time step for the simulation in ms",
    )
    parser.add_argument(
        "--first_pulse_width",
        type=float,
        required=False,
        default=0.01,
        help="First pulse width to simulate in ms",
    )
    parser.add_argument(
        "--last_pulse_width",
        type=float,
        required=False,
        default=2.5,
        help="Last pulse width to simulate in ms",
    )
    parser.add_argument(
        "--num_pulse_widths",
        type=int,
        required=False,
        default=8,
        help="Number of pulse widths to simulate",
    )

    comm = MPI.COMM_WORLD if MPI else DummyComm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parser.parse_args()
    if rank == 0:
        parsed_arguments_dict = vars(args)
        print(f"Parsed arguments:", flush=True)
        for key, value in parsed_arguments_dict.items():
            print(f"\t{key}: {value}", flush=True)
        print("----------------------------------------------------", flush=True)
    comm.Barrier()
    field_path = args.field_path
    axons_path = args.axons_path
    axon_name = args.axon_name
    output_dir = args.output_dir
    result_suffix = args.result_suffix
    diameter = args.diameter
    sensory_only = args.sensory_only
    motor_only = args.motor_only
    model_variant = args.model_variant
    dt = args.time_step
    init_hoc_path = os.path.join(project_path, "init_diff_v.hoc")

    # motor_only cannot be True if model_variant is mrg
    if sensory_only and model_variant.lower() == "mrg":
        raise ValueError("sensory_only cannot be True if model_variant is 'MRG' ('MRG' has only a motor model)")

    if model_variant.lower() == "alashqar":
        tuned_flag = True
    elif model_variant.lower() == "gaines":
        tuned_flag = False
    elif model_variant.lower() == "mrg":
        tuned_flag = False
    else:
        raise ValueError(
            f"Invalid model_variant argument: {model_variant}. Accepted arguments: 'Alashqar', 'Gaines', 'MRG'"
        )

    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if field_path.endswith(".npy"):
        field_dict = np.load(field_path, allow_pickle=True)[()]
    elif field_path.endswith(".h5"):
        with h5py.File(field_path, "r") as f:
            field_dict = {key: f[key][()] for key in f.keys()}
    else:
        raise ValueError("field_path must be a .npy or .h5 file!")
    field_dict["x"] *= 1e6  # m to um
    field_dict["y"] *= 1e6  # m to um
    field_dict["z"] *= 1e6  # m to um
    # define the range used to filter axons with a safety margin of 100 um on each side
    x_range = [field_dict["x"].min() + 100, field_dict["x"].max() - 100]
    y_range = [field_dict["y"].min() + 100, field_dict["y"].max() - 100]
    z_range = [field_dict["z"].min() + 100, field_dict["z"].max() - 100]

    axons_dict = np.load(axons_path, allow_pickle=True)[()]
    if axon_name is not None:
        axons_dict = {key: axons_dict[key] for key in axons_dict.keys() if axon_name in key}
    else:
        # use only the first axon
        axons_dict = {list(axons_dict.keys())[0]: axons_dict[list(axons_dict.keys())[0]]}
    axon_dicts = filter_axon_trajectories(
        axons_dict,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        min_axon_length=40.0 * 1e3,
        rank=rank,
    )

    model_types = ["sensory", "motor"]
    if model_variant.lower() == "mrg":
        model_types = ["MRG"]
    elif sensory_only:
        model_types = ["sensory"]
    elif motor_only:
        model_types = ["motor"]

    results_dict = {model_type: {} for model_type in model_types}
    pulse_widths = np.linspace(
        args.first_pulse_width, args.last_pulse_width, args.num_pulse_widths
    )
    for model_type in model_types:
        thresholds = np.zeros((len(pulse_widths)))
        if rank == 0:
            print(f"Model type: {model_type}")
        if diameter is not None:
            axon_info["diam"] = diameter
        else:
            diameter = axon_info["diam"]

        if rank == 0:
            print(f"Distributing pulse width simulations among {size} processes...", flush=True)
        local_pulse_widths = np.array_split(pulse_widths, size)[rank]
        local_thresholds = np.zeros((len(local_pulse_widths)))
        t1 = time.time()
        for i, pw in enumerate(local_pulse_widths):
            axon_info = copy.deepcopy(axon_dicts[0])
            if rank == 0:
                print(f"\t[Rank 0]: Titrating threshold for pulse width: {pw} ms", flush=True)
            sim_dur = 1.0 + pw + 3.0  # ms
            stim_t, stim_pulse = create_single_pulse_waveform(
                amplitude=1.0, start_at=1.0, end_at=1.0 + pw, time_step=dt, stim_dur=sim_dur
            )
            local_thresholds[i] = titrate_axon(
                axon_info,
                model_type=model_type,
                field_dict=field_dict,
                stim_pulse=stim_pulse,
                dt=dt,
                sim_dur=sim_dur,
                tuned_flag=tuned_flag,
                record_v=False,
                passive_end_nodes=True,
                verbose=False,
            )
            if rank == 0:
                print(f"\t\t[Rank 0]: Found threshold: {local_thresholds[i]}", flush=True)
        t2 = time.time()
        if rank == 0:
            print(f"\t[Rank 0]: Time taken for local pulse widths: {t2 - t1}", flush=True)
        
        gathered_thresholds = comm.gather(local_thresholds, root=0)
        gathered_pulse_widths = comm.gather(local_pulse_widths, root=0)
        if rank == 0:
            print(f"Finished simulations for all pulse widths for model type: {model_type}", flush=True)
            # combine gathered thresholds
            all_thresholds = np.concatenate(gathered_thresholds)
            pulse_widths = np.concatenate(gathered_pulse_widths)
            # sort pulse widths and thresholds accordingly
            sorted_indices = np.argsort(pulse_widths)
            pulse_widths = pulse_widths[sorted_indices]
            all_thresholds = all_thresholds[sorted_indices]
            results_dict[model_type]["pulse_widths"] = pulse_widths
            results_dict[model_type]["thresholds"] = all_thresholds
    
    # save results
    if rank == 0:
        if result_suffix is None:
            result_suffix = f"{model_variant.lower()}"
        result_path = os.path.join(output_dir, f"strength-duration_{result_suffix}_diam{diameter}.npy")
        np.save(result_path, results_dict)
        print(f"Strength-duration results saved to: {result_path}", flush=True)