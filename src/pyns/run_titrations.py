'''
###########################################
# File: pyns/run_titrations.py
# Project: pyns
# Author: Abdallah Alashqar (abdallah.j.alashqar@fau.de)
# -----
# PI: Andreas Rowald, PhD (andreas.rowald@fau.de)
# Associate Professor for Digital Health
# Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
https://www.pdh.med.fau.de/
############################################
'''

import os
import sys
import copy
import numpy as np
import h5py
try:
    from mpi4py import MPI
except Exception:
    MPI = None
    print(
        "Warning: MPI (mpi4py) is not installed; parallel processing is disabled. Using rank=0, size=1.",
        flush=True,
    )

import time
import datetime
from .utils import (
    pulse_file_to_pulse,
    filter_axon_trajectories,
    create_single_pulse_waveform,
    create_cont_stim_waveform,
    save_results,
    DummyComm,
)
from .axon_models import *
import yaml
from .sim_utils import simulate_axon, discretize_and_interpolate_v
from .arguments_parsers import parse_titrations_arguments

if __name__ == "__main__":
    comm = MPI.COMM_WORLD if MPI else DummyComm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = parse_titrations_arguments()
    config_dict = config.to_dict()

    if config.model_variant.lower() == "alashqar":
        model_type = None # determined from axon name based on afferents_kws_any and efferents_kws_any
        tuned_flag = True
    elif config.model_variant.lower() == "gaines":
        model_type = None # determined from axon name based on afferents_kws_any and efferents_kws_any
        tuned_flag = False
    elif config.model_variant.lower() == "mrg":
        model_type = "MRG"
        tuned_flag = False
    else:
        if rank == 0:
            raise ValueError(
                f"Invalid model_variant argument: {config.model_variant}. Accepted arguments: 'Alashqar', 'Gaines', 'MRG'"
            )
        sys.exit()

    # results_dir_sim is date and time stamped
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.results_dir_suffix:
        folder_name = f"{datetime_str}_{config.results_dir_suffix}"
    else:
        folder_name = f"{datetime_str}"
    results_dir_sim = os.path.join(config.results_dir, folder_name)
    if rank == 0:
        results_dir_sim_n = 0
        while os.path.isdir(results_dir_sim):
            results_dir_sim_n += 1
            results_dir_sim = f"{results_dir_sim}_{results_dir_sim_n}"
        os.makedirs(results_dir_sim)
        # save parsed arguments in a txt file
        settings_file_path = os.path.join(results_dir_sim, "titrations_config.yaml")
        with open(settings_file_path, 'w') as file:
            yaml.dump(config_dict, file)

    if rank == 0:
        print("******************************", flush=True)
        print(
            f"Starting simulations for EM field in {os.path.basename(config.field_path)}...",
            flush=True,
        )

    # Loading field dict:
    if rank == 0:
        t_start_all = time.perf_counter()
        print(f"\tLoading field...", flush=True)
    if config.field_path.endswith(".npy") and os.path.isfile(config.field_path):
        field_dict = np.load(config.field_path, allow_pickle=True)[()]
    elif config.field_path.endswith(".h5") and os.path.isfile(config.field_path):
        with h5py.File(config.field_path, "r") as f:
            field_dict = {key: f[key][()] for key in f.keys()}
    else:
        if rank == 0:
            if not os.path.isfile(config.field_path):
                error_msg = f"field_path {config.field_path} does not exist!"
            else:
                error_msg = f"field_path {config.field_path} must be a .npy or .h5 file!"
            raise ValueError(error_msg)
        sys.exit()

    field_dict["x"] *= 1e6  # m to um
    field_dict["y"] *= 1e6  # m to um
    field_dict["z"] *= 1e6  # m to um

    if rank == 0:
        print(f"\tLoading axon coordinates...", flush=True)
        # define the range used to filter axons with a safety margin of 1000 um on each side
        x_range = [field_dict["x"].min() + 1000, field_dict["x"].max() - 1000]
        y_range = [field_dict["y"].min() + 1000, field_dict["y"].max() - 1000]
        z_range = [field_dict["z"].min() + 1000, field_dict["z"].max() - 1000]
        if os.path.isdir(config.axons_path):
            subgroup_filenames = os.listdir(config.axons_path)
            # print(f"\tShortlested axon names: {short_list}")
            axons_file_paths = [
                os.path.join(config.axons_path, fname) for fname in subgroup_filenames
            ]
            axons_dict = {}
            for axons_path in axons_file_paths:
                # load subgroup and update axons_dict
                subgroup_dict = np.load(axons_path, allow_pickle=True)[()]
                axons_dict.update(subgroup_dict)
        elif os.path.isfile(config.axons_path):
            axons_dict = np.load(config.axons_path, allow_pickle=True)[()]
        else:
            raise ValueError(f"Invalid axons path: {config.axons_path}")
        print(
            f"\tNumber of axons before filtering: {len(axons_dict)}",
            flush=True,
        )

        # fiber_names_chosen = [k for k in axons_dict.keys() if "VR" in k and "L3" in k][0]
        # axons_dict = {k: v for k, v in axons_dict.items() if k == fiber_names_chosen}

        axon_dicts = filter_axon_trajectories(
            copy.deepcopy(axons_dict),
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            min_axon_length=config.min_fiber_length * 1e3,
            axons_kws_any=config.axons_kws_any,
            rank=rank,
        )
        
        if config.debug:
            axon_dicts = np.random.choice(axon_dicts, 5)
            # axon_dicts = axon_dicts[100:105]
            print(
                f"Axons names to simulate: {[axd['axon_name'] for axd in axon_dicts]}"
            )
        print(f"\tNumber of axons after filtering: {len(axon_dicts)}", flush=True)
        np.random.shuffle(axon_dicts)
    else:
        axon_dicts = None
        # params_dicts = None
    axon_dicts = comm.bcast(axon_dicts, root=0)

    if config.pulse_path is not None:
        if rank == 0:
            print(f"\tLoading pulse from {config.pulse_path}...", flush=True)
        stim_t, stim_pulse = pulse_file_to_pulse(config.pulse_path, stim_dur=config.sim_dur, time_step=config.time_step)
    else:
        if config.pulse_shape not in ["biphasic", "monophasic"]:
            if rank == 0:
                raise ValueError(f"Invalid pulse_shape argument: {config.pulse_shape}. Accepted arguments: 'biphasic', 'monophasic'")
            sys.exit()
        biphasic_flag = False
        if config.pulse_shape == "biphasic":
            biphasic_flag = True
        if config.cont_stim_waveform:
            stim_t, stim_pulse = create_cont_stim_waveform(
                silence_period=config.pulse_silence_period,
                burst_freq=config.cont_stim_freq,
                burst_width=config.pulse_width,
                freq=config.cont_stim_carrier_freq,
                time_step=config.time_step,
                total_stim_dur=config.sim_dur,
                biphasic=biphasic_flag,
                amplitude=config.pulse_amplitude,
            )
        else:
            stim_t, stim_pulse = create_single_pulse_waveform(
                amplitude=config.pulse_amplitude,
                time_step=config.time_step,
                biphasic=biphasic_flag,
                start_at=config.pulse_silence_period,
                end_at=config.pulse_silence_period + config.pulse_width,
                stim_dur=config.sim_dur,
            )
    if rank == 0:
        # create a plot of the pulse and save in the results directory
        fig = plt.figure(figsize=(20, 5))
        plt.plot(stim_t, stim_pulse)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title("Stim. Pulse")
        plt.savefig(os.path.join(results_dir_sim, "stim_pulse.png"))
        plt.close()
        print(
            f"\tA total of {len(axon_dicts)} axons will be split on {size} processors",
            flush=True,
        )

    axons_sub_list = np.array_split(axon_dicts, size)[rank]
    n_axons_per_process = len(axons_sub_list)
    n_all_axons = len(axon_dicts)
    axons_results = []
    # discretize once
    if rank == 0:
        print(f"\tdiscretizing axons...", flush=True)

    efferent_axons_sub_list = []
    other_axons_sub_list = axons_sub_list
    if config.efferent_kws_any is not None:
        efferent_axons_sub_list = [
            axon_d for axon_d in axons_sub_list if any(kw in axon_d["axon_name"] for kw in config.efferent_kws_any)
        ]
        other_axons_sub_list = [
            axon_d for axon_d in axons_sub_list if not any(kw in axon_d["axon_name"] for kw in config.efferent_kws_any)
        ]
    if rank == 0:
        print(f"\t\tNumber of efferent axons in this process: {len(efferent_axons_sub_list)}", flush=True)
        print(f"\t\tNumber of other axons in this process: {len(other_axons_sub_list)}", flush=True)
    
    if len(efferent_axons_sub_list) == 0:
        efferents_discretized = []
    else:
        efferents_discretized = discretize_and_interpolate_v(
            efferent_axons_sub_list,
            field_dict,
            model_type=model_type,
            tuned_flag=tuned_flag,
            motoneuron=True,
            paramfit_method=config.paramfit_method,
        )
    if len(other_axons_sub_list) == 0:
        other_discretized = []
    else:
        other_discretized = discretize_and_interpolate_v(
            other_axons_sub_list,
            field_dict,
            model_type=model_type,
            tuned_flag=tuned_flag,
            motoneuron=False,
            paramfit_method=config.paramfit_method,
        )
    for axon_obj_dict in efferents_discretized:
        axon_obj_dict["stim_factor"] = config.initial_stim_factor
        axon_obj_dict["high_thresh"] = 0
        axon_obj_dict["last_no_spk"] = 0
        if config.no_motoneuron:
            axon_obj_dict["MN"] = False
        else:
            axon_obj_dict["MN"] = True
    for axon_obj_dict in other_discretized:
        axon_obj_dict["stim_factor"] = config.initial_stim_factor
        axon_obj_dict["high_thresh"] = 0
        axon_obj_dict["last_no_spk"] = 0
        axon_obj_dict["MN"] = False
    axons_sub_list = efferents_discretized + other_discretized
    if rank == 0:
        print(f"\tFinished discretizing axons!", flush=True)

    field_dict = None
    record_v_default = config.record_v
    while len(axons_sub_list) > 0:
        if rank == 0:
            t_start = time.perf_counter()

        for axon_i, axon_obj_dict in enumerate(axons_sub_list):
            plot_axon_vm_flag = False
            record_v = record_v_default
            if config.axons_to_plot is not None and len(config.axons_to_plot) > 0:
                if axon_obj_dict["name"] in config.axons_to_plot:
                    print(f"\tRank {rank}: Plotting Vm for axon {axon_obj_dict['name']}", flush=True)
                    plot_axon_vm_flag = True
                    record_v = True
            result, axon_obj_dict = simulate_axon(
                axon_obj_dict=axon_obj_dict, 
                stim_pulse=stim_pulse,
                stim_factor=axon_obj_dict["stim_factor"],
                passive_end_nodes=config.passive_end_nodes,
                prepassive_nodes_as_endnodes=config.prepassive_nodes_as_endnodes,
                debug=config.debug,
                record_v=record_v,
                exclude_end_node=config.exclude_end_node,
                time_step=config.time_step,
                sim_dur=config.sim_dur,
                motoneuron=axon_obj_dict["MN"],
                input_spikes=None,
                fiber_traj_groups=None,
                recorded_v_nodes=config.recorded_v_nodes,
                recorded_ap_times_nodes=config.recorded_ap_times_nodes,
                disable_extracellular_stim=False,
                titration=True,
                titration_conv_perc=config.titration_conv_perc,
                initial_reduction_factor=config.initial_reduction_factor,
                initial_increment_factor=config.initial_increment_factor,
                max_stim_factor=config.max_stim_factor,
                n_req_spikes=config.number_of_req_spikes,
                min_stim_factor=config.min_stim_factor,
                output_dir=results_dir_sim,
                plot_axon_vm=plot_axon_vm_flag,
            )
            axons_sub_list[axon_i] = axon_obj_dict
            if result is not None:
                result = {k: {kk: vv for kk, vv in v.items() if not kk=="AP_times" and not kk=="MN"} for k, v in result.items()}
                axons_results.append(result)

        # keep only axons that did not spike
        axons_sub_list = [
            axon_obj_dict
            for axon_obj_dict in axons_sub_list
            if axon_obj_dict["name"]
            not in [list(axon_res.keys())[0] for axon_res in axons_results]
        ]

        if rank == 0:
            prog = (len(axons_results) / n_axons_per_process) * 100.0
            t_end = time.perf_counter()
            print(f"\t\tRank 0:")
            print(
                f"\t\t Finished one simulation loop in: {t_end - t_start}",
                flush=True,
            )
            print(
                f"\t\t Number of axons results so far: {len(axons_results)}",
                flush=True,
            )
            print(f"\t\t Progress: {np.round(prog, 2)}%", flush=True)

    # release memory by setting axon_dicts to None
    axon_dicts = None
    if rank == 0:
        t_end_all = time.perf_counter()
        print(
            f"\t\tFinished all simulations in {t_end_all - t_start_all} seconds! Gathering results from all processes...",
            flush=True,
        )
        t_gather_start = time.perf_counter()
    axon_results_gathered = comm.gather(axons_results, root=0)
    if rank == 0:
        axon_results_all = {
            k: v
            for sublist in axon_results_gathered
            for axon_res in sublist
            for k, v in axon_res.items()
        }
        t_gather_end = time.perf_counter()
        print(
            f"\t\tGathered all results in {t_gather_end - t_gather_start} seconds!",
            flush=True,
        )
        if config.debug:
            # print all axon results
            print("----------------------------------------------------", flush=True)
            print("Axon results:")
            for axon_name, axon_res in axon_results_all.items():
                if "spike" in axon_res:
                    print(f"\t{axon_name}: {axon_res['spike']}")
                else:
                    print(f"\t{axon_name}: no spike")
            print("----------------------------------------------------", flush=True)

        # dump axon results
        print(f"\tDumping axon results...", flush=True)

        output_npy_path = os.path.join(results_dir_sim, "axons_titration_results.npy")
        save_results(axon_results_all, output_npy_path)
        print("----------------------------------------------------", flush=True)
    else:
        pass

    MPI.COMM_WORLD.Barrier()

    sys.exit()
