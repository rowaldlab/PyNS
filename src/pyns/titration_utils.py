'''
###########################################
# File: pyns/titration_utils.py
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
# from sim_hoc_utils import *
from .utils import *
from . import sim_utils
from importlib import reload
reload(sim_utils)
from .sim_utils import *
import time

def titrate_axon_with_cond_pulse(
    axon_info,
    model_type,
    field_dict,
    pulse_width,
    inter_stim_interval,
    cond_pulse_threshold,
    tuned_flag=False,
    dt=0.005,
    verbose=False,
    record_v=False,
    titration_conv_perc=1.0,
    initial_increment_factor=1.5,
    max_stim_factor=3000.0,
    initial_reduction_factor=0.5,
    paramfit_method="continuous",
    passive_end_nodes=False,
    exclude_end_node_flag=False,
    prepassive_nodes_as_endnodes=False,
    init_hoc_path=None,
):
    """ Titrate the threshold for a given pulse width and inter-stimulus interval """
    if init_hoc_path is None:
        from .config import get_package_data_dir
        import os
        pkg_dir = get_package_data_dir()
        init_hoc_path = os.path.join(pkg_dir, "init_diff_v.hoc")
    
    if verbose:
        print(f"\t\t Searching threshold for interval: {inter_stim_interval}", flush=True)

    t_start = time.time()
    axon_obj_dict = discretize_and_interpolate_v_fiber(
        axon_info,
        model_type,
        tuned_flag,
        field_dict,
        paramfit_method=paramfit_method,
    )
    axon_obj_dict["stim_factor"] = cond_pulse_threshold
    axon_obj_dict["high_thresh"] = 0
    axon_obj_dict["last_no_spk"] = 0
    sim_dur = 2*pulse_width + inter_stim_interval + 1.0 + 3.0
    while True:
        stim_t, stim_pulse = create_multiple_pulses_waveform(
            start_at=[1.0, 1.0 + pulse_width + inter_stim_interval],
            end_at=[1.0 + pulse_width, 1.0 + 2*pulse_width + inter_stim_interval],
            amplitude=[cond_pulse_threshold, axon_obj_dict["stim_factor"]],
            time_step=dt,
            stim_dur=sim_dur,
        )
        axon_obj = MyelinatedAxon(discretized_dict=axon_obj_dict)
        axon_obj.initialize_neuron(
            passive_end_nodes=passive_end_nodes,
        )
        axon_obj.setup_recorders(
            record_v=record_v,
            recorded_v_nodes=None,
            recorded_ap_times_nodes=None,
            )
        axon_obj.assign_v_ext()
        if axon_obj_dict["stim_factor"] < 0.1:
            break
        t_one_sim_start = time.time()
        axon_res = axon_obj.run_simulation(
            stim_factor=1.0,
            stim_pulse=stim_pulse,
            dt=dt,
            tstop=sim_dur,
            output_path=None,
            return_only_spiking=True,
            exclude_end_node=exclude_end_node_flag,
            prepassive_nodes_as_endnodes=prepassive_nodes_as_endnodes,
            delete_hoc_objects=True,
            init_hoc_path=init_hoc_path,
            min_n_spikes_per_node=2,
        )
        t_one_sim_end = time.time()
        if verbose:
            print(
                f"\t\t\t Time taken for one sim: {t_one_sim_end - t_one_sim_start}",
                flush=True,
            )
        titration_result = update_axon_titration(
            axon_obj_dict,
            axon_res,
            verbose=verbose,
            titration_conv_perc=titration_conv_perc,
            initial_reduction_factor=initial_reduction_factor,
            initial_increment_factor=initial_increment_factor,
            max_stim_factor=max_stim_factor,
        )
        if titration_result is not None:
            if verbose:
                print(f" Finished titration for fiber {axon_obj_dict['name']}", flush=True)
            break
    t_end = time.time()
    if verbose:
        print(
            f"\t\t Titration for fiber {axon_obj_dict['name']} took: {t_end - t_start} seconds",
            flush=True,
        )
    if "ErrorMessage" in titration_result[axon_obj_dict['name']].keys():
        return -1
    else:
        axon_obj_dict["spike"] = axon_res["spike"]
    return axon_obj_dict["stim_factor"]

def titrate_axon(
    axon_info,
    model_type,
    field_dict,
    stim_pulse,
    dt,
    sim_dur,
    tuned_flag=False,
    initial_stim_factor=30.0,
    record_v=False,
    titration_conv_perc=1.0,
    initial_increment_factor=1.5,
    max_stim_factor=3000.0,
    initial_reduction_factor=0.5,
    verbose=False,
    passive_end_nodes=False,
    exclude_end_node_flag=False,
    prepassive_nodes_as_endnodes=False,
    paramfit_method="continuous",
    return_sim_res=False,
    min_stim_factor=0.001,
    mod_params_node=None,
    mod_params_mysa=None,
    mod_params_flut=None,
    mod_params_stin=None,
    v_init=None,
    motoneuron=False,
    init_hoc_path=None,
):
    if init_hoc_path is None:
        from .config import get_package_data_dir
        import os
        pkg_dir = get_package_data_dir()
        init_hoc_path = os.path.join(pkg_dir, "init_diff_v.hoc")
    
    t_start = time.time()
    axon_obj_dict = discretize_and_interpolate_v_fiber(
        axon_info,
        model_type,
        tuned_flag,
        field_dict,
        motoneuron=motoneuron,
        paramfit_method=paramfit_method,
    )
    axon_obj_dict["stim_factor"] = initial_stim_factor
    axon_obj_dict["high_thresh"] = 0
    axon_obj_dict["last_no_spk"] = 0
    while True:
        axon_obj = MyelinatedAxon(discretized_dict=axon_obj_dict)
        axon_obj.initialize_neuron(
            passive_end_nodes=passive_end_nodes,
            mod_params_node=mod_params_node,
            mod_params_mysa=mod_params_mysa,
            mod_params_flut=mod_params_flut,
            mod_params_stin=mod_params_stin,
            v_init=v_init,
            end_connected_to_mn=motoneuron,
        )
        axon_obj.setup_recorders(
            record_v=record_v,
            recorded_v_nodes=None,
            recorded_ap_times_nodes=None,
            )
        axon_obj.assign_v_ext()
        t_one_sim_start = time.time()
        if motoneuron:
            mn = Motoneuron(
                name=f"MN_{axon_obj.name}",
                init_seg_diam=axon_obj.axonD*(3.5/6.4),
                initseg_length=mn_initseg_length,
                soma_length=mn_soma_length,
                soma_coord=axon_obj_dict["mn_soma_coord"],
                initseg_coord=axon_obj_dict["mn_initseg_coord"],
                ) # dima ratio from Cullheim and Kellerth, 1978
            # extSegPot_mn = interpolate_3d(field_dict, np.copy(axon_obj.segments_midpoints)[-1, :])[0]
            mn.setup_recorders(record_v=record_v)
            mn.assign_v_ext(
                v_ext_soma=axon_obj_dict["mn_soma_v"],
                v_ext_initseg=axon_obj_dict["mn_initseg_v"])
            # connect the axon to the motoneuron
            mn.initSegment.connect(axon_obj.sections_list[-1], 1, 1)
        axon_res = axon_obj.run_simulation(
            stim_factor=axon_obj_dict["stim_factor"],
            stim_pulse=stim_pulse,
            dt=dt,
            tstop=sim_dur,
            output_path=None,
            return_only_spiking=True,
            prepassive_nodes_as_endnodes=prepassive_nodes_as_endnodes,
            delete_hoc_objects=True,
            init_hoc_path=init_hoc_path,
            exclude_end_node=exclude_end_node_flag,
        )
        t_one_sim_end = time.time()
        if verbose:
            print(
                f"\t\t\t Time taken for one sim: {t_one_sim_end - t_one_sim_start}",
                flush=True,
            )
        titration_result = update_axon_titration(
            axon_obj_dict,
            axon_res,
            verbose=verbose,
            titration_conv_perc=titration_conv_perc,
            initial_reduction_factor=initial_reduction_factor,
            initial_increment_factor=initial_increment_factor,
            max_stim_factor=max_stim_factor,
            min_stim_factor=min_stim_factor,
        )
        if titration_result is not None:
            if verbose:
                print(f" Finished titration for fiber {axon_obj_dict['name']}", flush=True)
            break
    t_end = time.time()
    if verbose:
        print(
            f"\t\t Titration for fiber {axon_obj_dict['name']} took: {t_end - t_start} seconds",
            flush=True,
        )
    if titration_result is None or "ErrorMessage" in titration_result[axon_obj_dict['name']].keys():
        return -1
    else:
        axon_obj_dict["spike"] = axon_res["spike"]
    if return_sim_res:
        return axon_obj_dict["stim_factor"], axon_res
    return axon_obj_dict["stim_factor"]