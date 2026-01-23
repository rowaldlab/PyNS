'''
###########################################
# File: pyns/run_discrete_simulations.py
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
import yaml

import time
import datetime
from .utils import (
    create_cont_stim_waveform,
    pulse_file_to_pulse,
    filter_axon_trajectories,
    create_single_pulse_waveform,
    axon_dicts_to_afferent_efferent_groups,
    axon_names_to_traj_groups,
    save_results,
    DummyComm,
)
from .sim_utils import simulate_axons, discretize_and_interpolate_v
import matplotlib.pyplot as plt
from .arguments_parsers import parse_discrete_simulations_arguments

if __name__ == "__main__":
    comm = MPI.COMM_WORLD if MPI else DummyComm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    config = parse_discrete_simulations_arguments()
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
    
    # afferents_only, efferents_only, other_axons_only are mutually exclusive
    if sum([config.afferents_only, config.efferents_only, config.other_axons_only]) > 1:
        raise ValueError("Only one of afferents_only, efferents_only, other_axons_only can be True!")
    
    stim_amplitudes = None
    afferent_results_all = None
    if config.efferents_only and config.enable_synaptic_transmission and config.afferents_results_path is None:
        raise ValueError("If efferents_only and enable_synaptic_transmission is True, afferents_results_path must be provided!")
    elif config.efferents_only:
        if not os.path.isfile(config.afferents_results_path) and not os.path.isdir(config.afferents_results_path):
            raise ValueError(f"Invalid afferents_results_path: {config.afferents_results_path}")
        if rank == 0:
            # if path is a directory look for folder containing same field_path and matching settings
            if os.path.isdir(config.afferents_results_path):
                for subdir in os.listdir(config.afferents_results_path):
                    subdir_config_path = os.path.join(config.afferents_results_path, subdir, "discrete_simulations_config.yaml")
                    if not os.path.exists(subdir_config_path):
                        continue
                    # load config file
                    with open(subdir_config_path, 'r') as file:
                        config = yaml.safe_load(file)
                    # make sure all relevant settings match
                    if all([
                        os.path.basename(config.field_path) == os.path.basename(config["field_path"]),
                        config.passive_end_nodes == config["passive_end_nodes"],
                        config.pulse_silence_period == config["pulse_silence_period"],
                        config.pulse_amplitude == config["pulse_amplitude"],
                        config.pulse_shape == config["pulse_shape"],
                        config.pulse_width == config["pulse_width"],
                        config.cont_stim_waveform == config["cont_stim_waveform"],
                        config.cont_stim_freq == config["cont_stim_freq"],
                        config.cont_stim_carrier_freq == config["cont_stim_carrier_freq"],
                        config.model_variant == config["model_variant"],
                        config.paramfit_method == config["paramfit_method"],
                    ]):
                        afferents_results_path = os.path.join(config.afferents_results_path, subdir, "axons_discrete_simulations_results.npy")
                        break
            if not os.path.isfile(afferents_results_path):
                afferent_results_all = None
            else:
                print(f"\tLoading afferent results from {afferents_results_path}", flush=True)
                afferent_results_all = np.load(afferents_results_path, allow_pickle=True)[()]
                afferent_results_all = {k:v for k, v in afferent_results_all.items() if any([kw in k for kw in ["DR", "DL"]])}
                if len(afferent_results_all) == 0:
                    afferent_results_all = None
        else:
            afferent_results_all = None

        afferent_results_all = comm.bcast(afferent_results_all, root=0)
        if afferent_results_all is None and rank == 0:
            print(f"\t!!! [WARNING] No matching afferent results found in {afferents_results_path} !!!", flush=True)
        elif rank == 0:
            print(f"\tLoaded afferent results from {afferents_results_path}", flush=True)

    stim_amplitudes = config.stim_amplitudes
    if stim_amplitudes is not None and rank == 0:
        print(f"\tUsing stim_amplitudes from arguments: {stim_amplitudes}", flush=True)

    if afferent_results_all is not None:
        afferent_keys = list(afferent_results_all.keys())
        stim_factors_afferents = list(afferent_results_all[afferent_keys[0]]["results"].keys())
        if stim_amplitudes is not None:
            if rank == 0:
                print(f"\tFiltering stim_amplitudes based on afferent results...", flush=True)
            stim_amplitudes = [stamp for stamp in stim_amplitudes if stamp in stim_factors_afferents]
        else:
            if rank == 0:
                print(f"\tUsing stim_amplitudes based on afferent results...", flush=True)
            stim_amplitudes = stim_factors_afferents

    if stim_amplitudes is None:
        if rank == 0:
            print(f"\tUsing initial_stim_factor: {config.initial_stim_factor}, stim_factor_step: {config.stim_factor_step}, max_stim_factor: {config.max_stim_factor}", flush=True)
        stim_amplitudes = np.arange(config.initial_stim_factor, config.max_stim_factor, config.stim_factor_step)
    
    # results_dir_sim is date and time stamped
    sub_dirname = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if config.results_dir_suffix:
        folder_name = f"{sub_dirname}_{config.results_dir_suffix}"
    else:
        folder_name = f"{sub_dirname}"
    results_dir_sim = os.path.join(config.results_dir, folder_name)
    if rank == 0:
        results_dir_sim_n = 0
        while os.path.isdir(results_dir_sim):
            results_dir_sim_n += 1
            results_dir_sim = f"{results_dir_sim}_{results_dir_sim_n}"
        made_dir = False
        while not made_dir:
            results_dir_sim_n += 1
            results_dir_sim = f"{results_dir_sim}_{results_dir_sim_n}"
            if not os.path.isdir(results_dir_sim):
                try:
                    os.makedirs(results_dir_sim)
                    made_dir = True
                except:
                    made_dir = False
        print(f"Results will be saved in {results_dir_sim}", flush=True)
        # save parsed arguments in a txt file
        settings_file_path = os.path.join(results_dir_sim, "discrete_simulations_config.yaml")
        # write parsed arguments to yaml file
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
            all_axon_names = os.listdir(config.axons_path)
            # print(f"\tShortlested axon names: {short_list}")
            axon_file_paths = [
                os.path.join(config.axons_path, fname) for fname in all_axon_names
            ]
            axons_dict = {}
            for axon_path in axon_file_paths:
                axon_name = os.path.basename(axon_path).replace(".npy", "")
                axon_points = np.load(axon_path)
                axons_dict[axon_name] = axon_points
        elif os.path.isfile(config.axons_path):
            axons_dict = np.load(config.axons_path, allow_pickle=True)[()]
        else:
            raise ValueError(f"Invalid axons path: {config.axons_path}")
        print(
            f"\tNumber of axons before filtering: {len(axons_dict)}",
            flush=True,
        )

        axon_dicts = filter_axon_trajectories(
            copy.deepcopy(axons_dict),
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            min_axon_length=config.min_fiber_length * 1e3,
            axons_kws_any=config.axons_kws_any,
            rank=rank,
        )

        print(f"\tNumber of axons after filtering: {len(axon_dicts)}", flush=True)
        np.random.shuffle(axon_dicts)
    else:
        axon_dicts = None
        # params_dicts = None
    axon_dicts = comm.bcast(axon_dicts, root=0)
    sim_dur_afferent = config.sim_dur_afferent
    if sim_dur_afferent is None:
        sim_dur_afferent = config.sim_dur
    sim_dur_efferent = config.sim_dur_efferent
    if sim_dur_efferent is None:
        sim_dur_efferent = config.sim_dur
    sim_dur_other = config.sim_dur
    if sim_dur_other is None:
        sim_dur_other = config.sim_dur
    sim_dur = np.max([sim_dur_afferent, sim_dur_efferent, sim_dur_other])

    if config.pulse_path is not None:
        if rank == 0:
            print(f"\tLoading pulse from {config.pulse_path}...", flush=True)
        sim_dur = np.max([sim_dur_afferent, sim_dur_efferent, sim_dur_other])
        stim_t, stim_pulse = pulse_file_to_pulse(config.pulse_path, stim_dur=sim_dur, time_step=config.time_step)
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
                total_stim_dur=sim_dur,
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
                stim_dur=sim_dur,
            )
    if rank == 0:
        print(
            f"\tA total of {len(axon_dicts)} axons will be split on {size} processors",
            flush=True,
        )
        # create a plot of the pulse and save in the results directory
        fig = plt.figure(figsize=(20, 5))
        plt.plot(stim_t, stim_pulse)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title("Stim. Pulse")
        plt.savefig(os.path.join(results_dir_sim, "stim_pulse.png"))
        plt.close()
        # zoom in from 0 to 5
        if sim_dur > 5.0:
            fig = plt.figure(figsize=(20, 5))
            plt.plot(stim_t, stim_pulse)
            plt.xlim(0, 5)
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (a.u.)")
            plt.title("Stim. Pulse (0-5 ms)")
            plt.savefig(os.path.join(results_dir_sim, "stim_pulse_zoom.png"))
            plt.close()
        print(
            f"\tA total of {len(axon_dicts)} axons will be split on {size} processors",
            flush=True,
        )

    # get afferent axons, efferent axons, and others
    afferent_axons, efferent_axons, other_axons = axon_dicts_to_afferent_efferent_groups(
        axon_dicts,
        afferent_kws_all=config.syn_afferent_kws_all,
        efferent_kws_all=config.syn_efferent_kws_all,
        afferent_kws_any=config.afferent_kws_any,
        efferent_kws_any=config.efferent_kws_any,
        root_kws_any=config.root_kws_any,
        )
    axon_names = [k["axon_name"] for k in axon_dicts]
    fiber_traj_groups = axon_names_to_traj_groups(
        axon_names,
        afferent_kws_all=config.syn_afferent_kws_all,
        efferent_kws_all=config.syn_efferent_kws_all,
        root_kws_any=config.root_kws_any,
        )

    if config.debug:
        # Choose only up to 5 axons from each group for debugging
        afferent_axons = afferent_axons[:5]
        efferent_axons = efferent_axons[:5]
        other_axons = other_axons[:5]

    afferent_axons_sub_list = np.array_split(afferent_axons, size)[rank]
    efferent_axons_sub_list = np.array_split(efferent_axons, size)[rank]
    other_axons_sub_list = np.array_split(other_axons, size)[rank]
    # print length of each list
    if rank == 0:
        print(f"\tNumber of all afferent axons: {len(afferent_axons)}", flush=True)
        print(f"\tNumber of all efferent axons: {len(efferent_axons)}", flush=True)
        print(f"\tNumber of all other axons: {len(other_axons)}", flush=True)
        print(f"\tDiscretizing and interpolating potential for afferent, efferent, and other axons...", flush=True)
    # discretize and interpolate
    t_discr_start = time.perf_counter()
    if not any([
        all([config.efferents_only, config.enable_synaptic_transmission, afferent_results_all is not None]),
        config.other_axons_only
        ]):
        afferents_discretized = discretize_and_interpolate_v(
            afferent_axons_sub_list,
            field_dict,
            model_type=model_type,
            tuned_flag=tuned_flag,
            paramfit_method=config.paramfit_method,
            )
    if not any([config.afferents_only, config.other_axons_only]):
        efferents_discretized = discretize_and_interpolate_v(
            efferent_axons_sub_list,
            field_dict,
            model_type=model_type,
            tuned_flag=tuned_flag,
            motoneuron=True,
            paramfit_method=config.paramfit_method,
            )
    if not any([config.afferents_only, config.efferents_only]) and not config.skip_other_axons:
        others_discretized = discretize_and_interpolate_v(
            other_axons_sub_list,
            field_dict,
            model_type=model_type,
            tuned_flag=tuned_flag,
            paramfit_method=config.paramfit_method,
            )
    t_discr_end = time.perf_counter()
    if rank == 0:
        print(f"\tDiscretization and interpolation took {t_discr_end - t_discr_start} seconds!", flush=True)

    field_dict = None

    axon_results_all = {}
    # first simulate afferent axons
    if len(afferent_axons) > 0 and not any([all([config.efferents_only,  config.enable_synaptic_transmission, afferent_results_all is not None]), config.other_axons_only]):
        if rank == 0:
            print(f"\tRANK 0: Simulating afferent axons...", flush=True)
        t1 = time.perf_counter()
        afferent_results = simulate_axons(
            axons_sub_list=afferents_discretized,
            stim_t=stim_t,
            stim_pulse=stim_pulse,
            stim_factor_step=config.stim_factor_step,
            initial_stim_factor=config.initial_stim_factor,
            max_stim_factor=config.max_stim_factor,
            passive_end_nodes=config.passive_end_nodes,
            prepassive_nodes_as_endnodes=config.prepassive_nodes_as_endnodes,
            exclude_end_node=False,
            debug=config.debug,
            time_step=config.time_step,
            sim_dur=sim_dur_afferent,
            record_v=config.record_v,
            recorded_v_nodes=config.recorded_v_nodes,
            recorded_ap_times_nodes=config.recorded_ap_times_nodes,
            save_only_processed_responses=config.save_only_processed_responses,
            stim_amplitudes=stim_amplitudes,
            output_dir=results_dir_sim,
            plot_axons=config.axons_to_plot,
        )
        t2 = time.perf_counter()
        if rank == 0:
            print(f"\tRANK 0: Finished simulating afferent axons in {t2-t1} seconds!", flush=True)
            # gather results
        afferent_results_gathered = comm.gather(afferent_results, root=0)
        if rank == 0:
            print(f"\tGathered results from all processors!", flush=True)
        comm.Barrier()
        # results is a dictionary, keys are axon names, values are dictionaries with results
        if rank == 0:
            # gather all results in one dict
            afferent_results_all = {
                ax_name: ax_sim_res for ax_sims_res in afferent_results_gathered for ax_name, ax_sim_res in ax_sims_res.items()
            }
            axon_results_all.update(afferent_results_all)
        else:
            afferent_results_all = None

        # if afferents_only is True, save results and exit
        if config.afferents_only:
            if rank == 0:
                t_end_all = time.perf_counter()
                print(
                    f"\t\tFinished all simulations in {t_end_all - t_start_all} seconds!",
                    flush=True,
                )
                print(f"\tDumping axon results of afferents and exiting...", flush=True)
                output_npy_path = os.path.join(results_dir_sim, "axons_discrete_simulations_results.npy")
                save_results(axon_results_all, output_npy_path)
                print("----------------------------------------------------", flush=True)
            sys.exit()

        afferent_results_all = comm.bcast(afferent_results_all, root=0)

        comm.Barrier()
        if rank == 0:
            print(f"\tBroadcasted results to all processors!", flush=True)
            print(f"\tNumber of afferent results: {len(afferent_results_all)}", flush=True)
    else:
        # update axon_results_all with afferent_results_all if they were loaded from file
        if rank == 0 and afferent_results_all is not None:
            axon_results_all.update(afferent_results_all)

    # then simulate efferent axons
    if rank == 0:
        print(f"\tRANK 0: Simulating efferent axons...", flush=True)
    efferents_motoneuron = True
    if config.no_motoneuron:
        if config.enable_synaptic_transmission:
            # raise an error since synaptic transmission requires motoneuron model
            if rank == 0:
                raise ValueError("enable_synaptic_transmission cannot be True if no_motoneuron is True!")
            sys.exit()
        efferents_motoneuron = False
    if len(efferent_axons) > 0 and not any([config.afferents_only, config.other_axons_only]):
        t1 = time.perf_counter()
        if config.enable_synaptic_transmission:
            projecting_axons_results = copy.deepcopy(afferent_results_all)
        else:
            projecting_axons_results = None
        efferent_results = simulate_axons(
            axons_sub_list=efferents_discretized,
            stim_t=stim_t,
            stim_pulse=stim_pulse,
            stim_factor_step=config.stim_factor_step,
            initial_stim_factor=config.initial_stim_factor,
            max_stim_factor=config.max_stim_factor,
            passive_end_nodes=config.passive_end_nodes,
            prepassive_nodes_as_endnodes=config.prepassive_nodes_as_endnodes,
            exclude_end_node=False,
            debug=config.debug,
            time_step=config.time_step,
            sim_dur=config.sim_dur_efferent,
            record_v=config.record_v,
            recorded_v_nodes=config.recorded_v_nodes, # recorded_v_nodes=[0, -1]
            recorded_ap_times_nodes=config.recorded_ap_times_nodes,
            projecting_axons_results=projecting_axons_results,
            motoneuron=efferents_motoneuron,
            fiber_traj_groups=fiber_traj_groups,
            disable_extracellular_stim=config.disable_extracellular_efferent,
            save_only_processed_responses=config.save_only_processed_responses,
            syn_weight=config.syn_weight,
            stim_amplitudes=stim_amplitudes,
            proj_freq=config.proj_freq,
            output_dir=results_dir_sim,
            plot_axons=config.axons_to_plot,
        )
        t2 = time.perf_counter()
        if rank == 0:
            print(f"\tRANK 0: Finished simulating efferent axons in {t2-t1} seconds!", flush=True)
        efferent_results_gathered = comm.gather(efferent_results, root=0)
        if rank == 0:
            print(f"\tGathered efferent results from all processors!", flush=True)
        comm.Barrier()
        if rank == 0:
            efferent_results_all = {
                ax_name: ax_sim_res for ax_sims_res in efferent_results_gathered for ax_name, ax_sim_res in ax_sims_res.items()
            }
            axon_results_all.update(efferent_results_all)

        if config.efferents_only:
            if rank == 0:
                t_end_all = time.perf_counter()
                print(
                    f"\t\tFinished all simulations in {t_end_all - t_start_all} seconds!",
                    flush=True,
                )
                print(f"\tDumping axon results of efferents and exiting...", flush=True)
                output_npy_path = os.path.join(results_dir_sim, "axons_discrete_simulations_results.npy")
                save_results(axon_results_all, output_npy_path)
                print("----------------------------------------------------", flush=True)
            sys.exit()

        comm.Barrier()

    # and now other axons
    if rank == 0:
        print(f"\tRANK 0: Simulating other axons...", flush=True)
    if len(other_axons) > 0 and not any([config.afferents_only, config.efferents_only]) and not config.skip_other_axons:
        t1 = time.perf_counter()
        other_results = simulate_axons(
            axons_sub_list=others_discretized,
            stim_t=stim_t,
            stim_pulse=stim_pulse,
            stim_factor_step=config.stim_factor_step,
            initial_stim_factor=config.initial_stim_factor,
            max_stim_factor=config.max_stim_factor,
            passive_end_nodes=config.passive_end_nodes,
            prepassive_nodes_as_endnodes=config.prepassive_nodes_as_endnodes,
            exclude_end_node=False,
            debug=config.debug,
            time_step=config.time_step,
            sim_dur=config.sim_dur_other,
            record_v=config.record_v,
            recorded_v_nodes=config.recorded_v_nodes,
            recorded_ap_times_nodes=config.recorded_ap_times_nodes,
            save_only_processed_responses=config.save_only_processed_responses,
            stim_amplitudes=stim_amplitudes,
            output_dir=results_dir_sim,
            plot_axons=config.axons_to_plot,
        )
        t2 = time.perf_counter()
        if rank == 0:
            print(f"\tRANK 0: Finished simulating other axons in {t2-t1} seconds!", flush=True)
        other_results_gathered = comm.gather(other_results, root=0)
        if rank == 0:
            print(f"\tGathered other axons results from all processors!", flush=True)
        comm.Barrier()
        if rank == 0:
            other_results_all = {
                ax_name: ax_sim_res for ax_sims_res in other_results_gathered for ax_name, ax_sim_res in ax_sims_res.items()
            }
            axon_results_all.update(other_results_all)

        if config.other_axons_only:
            if rank == 0:
                t_end_all = time.perf_counter()
                print(
                    f"\t\tFinished all simulations in {t_end_all - t_start_all} seconds!",
                    flush=True,
                )
                print(f"\tDumping axon results of other axons and exiting...", flush=True)
                output_npy_path = os.path.join(results_dir_sim, "axons_discrete_simulations_results.npy")
                save_results(axon_results_all, output_npy_path)
                print("----------------------------------------------------", flush=True)
            sys.exit()

        comm.Barrier()

    if rank == 0:
        t_end_all = time.perf_counter()
        print(
            f"\t\tFinished all simulations in {t_end_all - t_start_all} seconds!",
            flush=True,
        )
    if rank == 0:
        # dump axon results
        print(f"\tDumping axon results...", flush=True)
        output_npy_path = os.path.join(results_dir_sim, "axons_discrete_simulations_results.npy")
        save_results(axon_results_all, output_npy_path)
        print("----------------------------------------------------", flush=True)
    else:
        pass

    MPI.COMM_WORLD.Barrier()

    sys.exit()
