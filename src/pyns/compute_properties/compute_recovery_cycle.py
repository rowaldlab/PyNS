'''
###########################################
# File: pyns/compute_recovery_cycle.py
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

from ..titration_utils import titrate_axon, titrate_axon_with_cond_pulse
from ..utils import DummyComm

# get project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute recovery cycle of an axon model under extracellular stimulation"
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
        default=os.path.join(os.getcwd(), "rec_cycle_results"),
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
        "--pulse_width",
        type=float,
        required=False,
        default=0.1,
        help="Pulse width for the stimulation pulse in ms",
    )
    parser.add_argument(
        "--sim_dur",
        type=float,
        required=False,
        default=5.0,
        help="Total duration of the simulation in ms",
    )
    parser.add_argument(
        "--first_isi",
        type=float,
        required=False,
        default=2.0,
        help="First inter-stimulus interval to simulate in ms",
    )
    parser.add_argument(
        "--last_isi",
        type=float,
        required=False,
        default=100.0,
        help="Last inter-stimulus interval to simulate in ms",
    )
    parser.add_argument(
        "--num_intervals",
        type=int,
        required=False,
        default=25,
        help="Number of inter-stimulus intervals to simulate",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        required=False,
        default="log",
        help="Method to sample inter-stimulus intervals: 'linear' using numpy.linspace or 'log' using numpy.geomspace",
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
    pulse_width = args.pulse_width
    sim_dur = args.sim_dur
    first_isi = args.first_isi
    last_isi = args.last_isi
    num_intervals = args.num_intervals
    sampling_method = args.sampling_method
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

    if sampling_method == "log":
        intervals = np.geomspace(first_isi, last_isi, num_intervals)
    elif sampling_method == "linear":
        intervals = np.linspace(first_isi, last_isi, num_intervals)
    else:
        raise ValueError("sampling_method must be 'linear' or 'log'")
    results_dict = {model_type: {} for model_type in model_types}

    for model_type in model_types:
        thresholds = np.zeros((len(intervals)))
        if rank == 0:
            print(f"Model type: {model_type}", flush=True)

        axon_info = copy.deepcopy(axon_dicts[0])
        if diameter is not None:
            axon_info["diam"] = diameter
        else:
            diameter = axon_info["diam"]
        
        stim_t, stim_pulse = create_single_pulse_waveform(
            amplitude=1.0, start_at=1.0, end_at=1.0 + pulse_width, time_step=dt
        )

        if rank == 0:
            cond_pulse_threshold = titrate_axon(
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
            print(f"\t Threshold for one pulse: {cond_pulse_threshold}", flush=True)
        else:
            cond_pulse_threshold = None

        cond_pulse_threshold = comm.bcast(cond_pulse_threshold, root=0)

        if rank == 0:
            print(f"\t Distributing intervals to ranks, with size: {size}. Number of intervals per rank: {len(intervals)//size}", flush=True)
        local_intervals = np.array_split(intervals, size)[rank]
        local_thresholds = np.zeros(len(local_intervals))

        time_st = time.time()
        for int_i, inter_stim_interval in enumerate(local_intervals):
            if rank == 0:
                print(f"\t[Rank 0]: Interval: {inter_stim_interval} ({int_i+1}/{len(local_intervals)})", flush=True)
            local_thresholds[int_i] = titrate_axon_with_cond_pulse(
                axon_info,
                model_type,
                field_dict,
                pulse_width,
                inter_stim_interval=inter_stim_interval,
                cond_pulse_threshold=cond_pulse_threshold,
                tuned_flag=tuned_flag,
                dt=dt,
                passive_end_nodes=True,
                verbose=False,
            )
        t_end = time.time()
        if rank == 0:
            print(f"\t[Rank 0]: Time taken for all intervals: {t_end - time_st}", flush=True)

        gathered_thresholds = comm.gather(local_thresholds, root=0)
        if rank == 0:
            print(f"Finished simulations for all intervals for model type: {model_type}", flush=True)
            thresholds = np.concatenate(gathered_thresholds)
            results_dict[model_type] = {
                "pulse_width": pulse_width,
                "intervals": intervals,
                "thresholds": thresholds,
                "axon_name": axon_name,
                "diam": axon_info["diam"],
                "cond_pulse_threshold": cond_pulse_threshold,
            }

    if rank == 0:
        print("===========", flush=True)
        if result_suffix is None:
            result_suffix = f"{model_variant.lower()}"
        result_path = os.path.join(output_dir, f"rec-cycle_{result_suffix}_diam{diameter}.npy")
        np.save(result_path, results_dict, allow_pickle=True)