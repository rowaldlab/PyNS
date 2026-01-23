'''
###########################################
# File: pyns/sim_utils.py
# Project: pyns
# Author: Abdallah Alashqar (abdallah.j.alashqar@fau.de)
# -----
# PI: Andreas Rowald, PhD (andreas.rowald@fau.de)
# Associate Professor for Digital Health
# Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
https://www.pdh.med.fau.de/
############################################
'''

from doctest import debug
import numpy as np
import re
try:
    from mpi4py import MPI
except Exception:
    MPI = None

import traceback
from .axon_models import MyelinatedAxon, Motoneuron, UnmyelinatedAxon
import time
from .utils import *
from .utils import DummyComm
from neuron import h
from .sim_analysis_utils import classify_responses, get_ap_init_nodes, get_ap_times_at_mn
import copy
import os

rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

mn_initseg_length = 1000 # 1000 from Moraud et al. 2016
mn_soma_length = 53.04 # 53.04 from Cullheim et al. 1987
node_length = 1 # node length in um

# roots from C1-C8, T1-T12, L1-L5, S1-S5
ROOTS_ORDERED = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
    "S1", "S2", "S3", "S4", "S5",
    ]

def update_axon_titration(
    axon_obj_dict,
    axon_res,
    verbose=False,
    titration_conv_perc=1.0,
    initial_reduction_factor=0.5,
    initial_increment_factor=1.5,
    max_stim_factor=300.0,
    min_stim_factor=0.01,
):
    if len(axon_res) > 0 and "spike" in axon_res:
        if verbose:
            print(
                f"\t\t Axon {axon_obj_dict['name']} spiked with stim factor: {axon_obj_dict['stim_factor']}",
                flush=True,
            )
        # axon spiked
        if axon_obj_dict["high_thresh"] == 0:
            stim_factor_step_percent = 100.0
        else:
            stim_factor_step = (
                axon_obj_dict["high_thresh"] - axon_obj_dict["stim_factor"]
            )
            try:
                stim_factor_step_percent = (
                    stim_factor_step / axon_obj_dict["stim_factor"]
                ) * 100
            except ZeroDivisionError:
                print(f"\t\t\t!!! [ERROR] Axon {axon_obj_dict['name']} stim factor is zero, cannot compute step percent", flush=True)
                axon_res = {
                    "ErrorMessage": "Stim factor is zero, cannot compute step percent"
                }
                return {axon_obj_dict["name"]: axon_res}
        if stim_factor_step_percent < titration_conv_perc:
            if verbose:
                print(
                    f"\t\t\t Stim factor step: {stim_factor_step_percent} < {titration_conv_perc}%",
                    flush=True,
                )
            return {axon_obj_dict["name"]: axon_res}
        else:
            if verbose:
                print(
                    f"\t\t\t Stim factor step: {stim_factor_step_percent} > {titration_conv_perc}%",
                    flush=True,
                )
                print(
                    f"\t\t\t Decreasing stim factor step to: {axon_obj_dict['stim_factor'] * initial_reduction_factor}",
                    flush=True,
                )
            # decrease stim factor
            axon_obj_dict["high_thresh"] = axon_obj_dict["stim_factor"]
            if axon_obj_dict["stim_factor"] <= min_stim_factor:
                axon_res = {
                    "ErrorMessage": f"Stim factor went below min limit of {min_stim_factor}"
                }
                return {axon_obj_dict["name"]: axon_res}
            if axon_obj_dict["last_no_spk"] == 0:
                axon_obj_dict["stim_factor"] = (
                    axon_obj_dict["stim_factor"] * initial_reduction_factor
                )
            else:
                axon_obj_dict["stim_factor"] = (
                    axon_obj_dict["last_no_spk"] + axon_obj_dict["stim_factor"]
                ) / 2.0
    else:
        if axon_obj_dict["stim_factor"] >= max_stim_factor:
            axon_res = {
                "ErrorMessage": f"Stim factor exceeded the max limit of {max_stim_factor}"
            }
            return {axon_obj_dict["name"]: axon_res}
        axon_obj_dict["last_no_spk"] = axon_obj_dict["stim_factor"]
        if verbose:
            print(
                f"\t\t Axon {axon_obj_dict['name']} did not spike with stim factor: {axon_obj_dict['stim_factor']}",
                flush=True,
            )
        if not axon_obj_dict["high_thresh"] == 0:
            # increase stim factor
            if verbose:
                print(
                    f"\t\t\t Increasing stim factor to: {(axon_obj_dict['high_thresh'] + axon_obj_dict['stim_factor']) / 2}",
                    flush=True,
                )
            axon_obj_dict["stim_factor"] = (
                axon_obj_dict["high_thresh"] + axon_obj_dict["stim_factor"]
            ) / 2
        else:
            if verbose:
                print(
                    f"\t\t\t Increasing stim factor to: {axon_obj_dict['stim_factor'] * initial_increment_factor}",
                    flush=True,
                )
            axon_obj_dict["stim_factor"] = (
                axon_obj_dict["stim_factor"] * initial_increment_factor
            )
    return None

def plot_membrane_potential_from_res(axon_res, axon_name, fig_path, highlight_color="r"):
    # make_dir for fig path if it does not exist
    if not os.path.exists(os.path.dirname(fig_path)):
        os.makedirs(os.path.dirname(fig_path))
    # plot MN membrane potential
    t_vector = axon_res['time_vector']
    node_is = [int(k.split("_")[2]) for k in axon_res['membrane_potential'].keys()]
    nnodes = len(node_is)
    plt.figure(figsize=[20, int(nnodes * 0.1)])
    plt.title(axon_name)
    node_count = 0
    keys_list = list(axon_res['membrane_potential'].keys())
    for k in keys_list[::-1]:
        v = axon_res['membrane_potential'][k]
        if node_count >= 55:
            break
        ni = int(k.split("_")[2])
        k = f"AP_times_node_{ni}"
        if not k in axon_res['AP_times'].keys():
            if ni in [0, 1]:
                k = f"AP_times_first_node"
            elif ni in [nnodes-1, nnodes-2]:
                k = f"AP_times_last_node"
            else:
                print(f"CURRENT KEYS:", flush=True)
                for k in axon_res['AP_times'].keys():
                    print(f"\t{k}", flush=True)
                raise ValueError(f"Node {ni} does not have AP times recorded.") 
        ap_times = np.array(axon_res['AP_times'][k])
        # color = "k"
        plt.plot(
            t_vector,
            10 * ni + np.array(v),
            color="#58595b",
            linewidth=0.5,
        )
        node_count += 1
        for ap_time in ap_times:
            time_mask = (t_vector >= ap_time - 0.1) & (t_vector <= ap_time + 0.45)
            time_ap = t_vector[time_mask]
            plt.plot(
                time_ap,
                10 * ni + np.array(v)[time_mask],
                color=highlight_color,
                linewidth=0.5,
            )
    # plot mn init segment
    if "MN" in axon_res.keys():
        mn_result = axon_res["MN"]["results"]
        plt.plot(
            mn_result["t"],
            10 * (nnodes + 1) + np.array(mn_result['v_initseg']),
            color="#a7a9ac",
            linewidth=0.5,
        )
        # plot mn soma
        plt.plot(
            mn_result["t"],
            10 * (nnodes + 2) + np.array(mn_result['v_soma']),
            color="#231f20",
            linewidth=0.5,
        )
    plt.savefig(fig_path)
    # save as eps as well
    plt.savefig(fig_path.replace(".png", ".eps"), format="eps")
    plt.close()
    # create one plot only for mn
    if "MN" in axon_res.keys():
        plt.figure(figsize=[20, 5])
        plt.title(axon_name)
        mn_result = axon_res["MN"]["results"]
        plt.plot(
            mn_result["t"],
            np.array(mn_result['v_initseg']),
            color="b",
            linewidth=2,
            alpha=0.8,
            label="MN init seg",
        )
        plt.plot(
            mn_result["t"],
            np.array(mn_result['v_soma']),
            color="g",
            linewidth=2,
            alpha=0.8,
            label="MN soma",
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane potential (mV)")
        plt.legend()
        plt.savefig(fig_path.replace(".png", "_mnOnly.png"))
        # save as eps as well
        plt.savefig(fig_path.replace(".png", "_mnOnly.eps"), format="eps")
        plt.close()

def discretize_and_interpolate_v_fiber(
        axon_info,
        model_type,
        tuned_flag,
        field_dict,
        paramfit_method="continuous",
        motoneuron=False,
        unmyelinated_model="sundt",
        afferent_kws_any=["sensory", "_Aalpha", "_DR", "_DL", "dorsal", "afferent"],
        efferent_kws_any=["motor", "_alpha", "_VR", "_VL", "ventral", "efferent"],
        ):
    try:
        inner_diam=None
        if "inner_diam" in axon_info:
            inner_diam = axon_info["inner_diam"]
        if axon_info["diam"] > 1:
            axon_obj = MyelinatedAxon(
                axon_name=axon_info["axon_name"],
                axon_coords=np.copy(axon_info["points"]),
                fiber_diameter=axon_info["diam"],
                axon_inner_diameter=inner_diam,
                model_type=model_type,
                tuned_model=tuned_flag,
                params_fit_method=paramfit_method,
                afferent_kws_any=afferent_kws_any,
                efferent_kws_any=efferent_kws_any,
            )
        else:
            axon_obj = UnmyelinatedAxon(
                axon_name=axon_info["axon_name"],
                axon_coords=np.copy(axon_info["points"]),
                model=unmyelinated_model,
            )
        # interpolate voltages
        axon_obj.interpolate_v_on_sections(field_dict)
        axon_obj_dict = axon_obj.to_dict()

        if motoneuron:
            direction = np.copy(axon_obj.segments_midpoints)[-1, :] - np.copy(axon_obj.segments_midpoints)[-2, :]
            direction = direction/np.linalg.norm(direction)
            initseg_l = mn_initseg_length
            soma_l = mn_soma_length
            initseg_coord = np.copy(axon_obj.segments_midpoints)[-1, :] + direction*((initseg_l/2) + node_length/2)
            soma_coord = initseg_coord + direction*(initseg_l/2 + soma_l/2)
            mn_coords = np.concatenate((initseg_coord, soma_coord))
            extSegPot_mn = interpolate_3d(field_dict, mn_coords)
            axon_obj_dict["mn_initseg_v"] = extSegPot_mn[0]
            axon_obj_dict["mn_soma_v"] = extSegPot_mn[1]
            axon_obj_dict["mn_initseg_coord"] = initseg_coord
            axon_obj_dict["mn_soma_coord"] = soma_coord

    except Exception as e:
        print(
            f"!!! [ERROR] !!! Could not create axon object for: {axon_info['axon_name']}",
            flush=True,
        )
        print(f"\t!!! ERROR MESSAGE: {str(e)}", flush=True)
        print(f"\t!!!   Traceback: {traceback.format_exc()}", flush=True)
        print(f"\t!!! FOLLOWING ARE AXON PARAMS: ", flush=True)
        print(f"\t\t!!! AXON DIAM: {axon_info['diam']}", flush=True)
        print(
            f"\t\t!!! AXON LENGTH: {get_arcline_length(axon_info['points'])}",
            flush=True,
        )
        axon_obj_dict = None
    return axon_obj_dict


def discretize_and_interpolate_v(
        fibers_list,
        field_dict,
        model_type="Gaines",
        tuned_flag=False,
        paramfit_method="continuous",
        motoneuron=False,
        unmyelinated_model="sundt",
        afferent_kws_any=["sensory", "_Aalpha", "_DR", "_DL", "dorsal", "afferent"],
        efferent_kws_any=["motor", "_alpha", "_VR", "_VL", "ventral", "efferent"],
        ):
    axons_discretized_list = [
        discretize_and_interpolate_v_fiber(
            axon_info=axon_info,
            model_type=model_type,
            tuned_flag=tuned_flag,
            field_dict=field_dict,
            paramfit_method=paramfit_method,
            motoneuron=motoneuron,
            unmyelinated_model=unmyelinated_model,
            afferent_kws_any=afferent_kws_any,
            efferent_kws_any=efferent_kws_any,
        ) for axon_info in fibers_list
    ]
    # exclude None
    axons_discretized_list = [axon for axon in axons_discretized_list if axon is not None]
    return axons_discretized_list

def simulate_axon(
        axon_obj_dict,
        stim_pulse,
        stim_factor=1.0,
        passive_end_nodes=False,
        prepassive_nodes_as_endnodes=False,
        debug=False,
        record_v=False,
        exclude_end_node=False,
        time_step=0.005,
        sim_dur=4.0,
        motoneuron=False,
        input_spikes=None,
        fiber_traj_groups=None,
        recorded_v_nodes=[-1],
        recorded_ap_times_nodes=[-1],
        disable_extracellular_stim=False,
        titration=False,
        titration_conv_perc=1.0,
        initial_reduction_factor=0.5,
        initial_increment_factor=1.5,
        max_stim_factor=300.0,
        syn_weight=0.0056994,
        n_req_spikes=1,
        proj_freq=50.0,
        output_dir="./outputs",
        plot_axon_vm=False,
        min_stim_factor=0.01,
        init_hoc_path=None,
        ):
    return_only_spiking = False
    if titration and not plot_axon_vm:
        return_only_spiking = True
    
    # Set default init_hoc_path if not provided
    if init_hoc_path is None:
        from .config import get_package_data_dir
        pkg_dir = get_package_data_dir()
        init_hoc_path = os.path.join(pkg_dir, "init_diff_v.hoc")
    try:
        if axon_obj_dict["fiberD"] > 1:
            axon_obj = MyelinatedAxon(
                discretized_dict=axon_obj_dict,
                )
        else:
            axon_obj = UnmyelinatedAxon(discretized_dict=axon_obj_dict)
        try:
            if debug:
                print(
                    f"\t\t\tSimulating axon: {axon_obj.name} with stim factor: {stim_factor}",
                    flush=True,
                )
            if axon_obj_dict["fiberD"] > 1:
                axon_obj.initialize_neuron(
                    passive_end_nodes=passive_end_nodes,
                    end_connected_to_mn=motoneuron,
                )
                axon_obj.setup_recorders(
                    record_v=record_v,
                    recorded_v_nodes=recorded_v_nodes,
                    recorded_ap_times_nodes=recorded_ap_times_nodes,
                    dt=time_step
                    )
            else:
                axon_obj.initialize_neuron()
                axon_obj.setup_recorders(
                    record_v=record_v,
                    recorded_v_secs=recorded_v_nodes,
                    recorded_ap_times_secs=recorded_ap_times_nodes,
                    dt=time_step
                    )
            axon_obj.assign_v_ext()

            if motoneuron:
                if debug:
                    print(f"\t\t\tCreating motoneuron for axon: {axon_obj.name}", flush=True)
                mn = Motoneuron(
                    name=f"MN_{axon_obj.name}",
                    init_seg_diam=axon_obj.fiberD*(3.5/6.4),
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
                if input_spikes is not None:
                    if fiber_traj_groups is None:
                        raise ValueError("Fiber trajectory groups are not provided!")
                    fiber_segment = axon_obj.name.split("_")[0]
                    if "_VR_" in axon_obj.name:
                        pos = "DR"
                        pos_efferent = "VR"
                    elif "_VL_" in axon_obj.name:
                        pos = "DL"
                        pos_efferent = "VL"
                    traj_name = get_traj_name(axon_obj.name)
                    afferent_axon_names = copy.deepcopy(fiber_traj_groups[traj_name][fiber_segment][pos])
                    # add afferents of segments below and above
                    root_index = ROOTS_ORDERED.index(fiber_segment)
                    if root_index > 0:
                        prev_root = ROOTS_ORDERED[root_index - 1]
                        if prev_root in fiber_traj_groups[traj_name]:
                            afferent_axon_names += copy.deepcopy(fiber_traj_groups[traj_name][prev_root][pos])
                    if root_index < len(ROOTS_ORDERED) - 1:
                        next_root = ROOTS_ORDERED[root_index + 1]
                        if next_root in fiber_traj_groups[traj_name]:
                            afferent_axon_names += copy.deepcopy(fiber_traj_groups[traj_name][next_root][pos])
                    efferent_fiber_names = copy.deepcopy(fiber_traj_groups[traj_name][fiber_segment][pos_efferent])
                    efferent_fiber_names_sampled = efferent_fiber_names
                    if proj_freq != 100.0:
                        # get fiber diams of efferent and sort them
                        efferent_fiber_diams = [float(re.search(r"(\d+\.?\d*)um", fname).group(1)) for fname in efferent_fiber_names]
                        diam_sort_inds = np.argsort(efferent_fiber_diams)
                        efferent_fiber_names_sorted = [efferent_fiber_names[i] for i in diam_sort_inds]
                        n_efferent_fibers_to_sample = int(np.round(len(efferent_fiber_names_sorted)*(proj_freq/100.0)))
                        # uniform sampling of efferent fibers
                        if n_efferent_fibers_to_sample < 1:
                            n_efferent_fibers_to_sample = 1
                        sampling_inds = np.linspace(0, len(efferent_fiber_names_sorted)-1, n_efferent_fibers_to_sample).astype(int)
                        efferent_fiber_names_sampled = [efferent_fiber_names_sorted[i] for i in sampling_inds]
                    # if current motor fiber is not in the sampled list, skip adding inputs
                    if axon_obj.name in efferent_fiber_names_sampled:
                        # get number of fibers from same traj_name and pos but all segments
                        projecting_axons = {axon_name: {
                            'AP_init_sites': axon_res["results"][stim_factor]['AP_init_sites'],
                            'segment_midpoints' : axon_res["segment_midpoints"],
                        } for axon_name, axon_res in input_spikes.items() if axon_name in afferent_axon_names}
                        axon_input_spikes = get_ap_times_at_mn(projecting_axons, mn.soma_coord)
                        if debug:
                            print(f"\t\t\t\tAfferent fibers: {len(afferent_axon_names)}")
                            print(f"\t\t\t\tFiber input spikes: {axon_input_spikes}")
                        mn.set_Ia_afferent_inputs(axon_input_spikes, syn_weight=syn_weight)
            if disable_extracellular_stim:
                stim_factor = 0.0
            if axon_obj_dict["fiberD"] > 1:
                axon_res = axon_obj.run_simulation(
                    stim_factor=stim_factor,
                    stim_pulse=stim_pulse,
                    dt=time_step,
                    tstop=sim_dur,
                    return_only_spiking=return_only_spiking,
                    exclude_end_node=exclude_end_node,
                    prepassive_nodes_as_endnodes=prepassive_nodes_as_endnodes,
                    delete_hoc_objects=True,
                    init_hoc_path=init_hoc_path,
                    min_n_spikes_per_node=n_req_spikes,
                )
            else:
                axon_res = axon_obj.run_simulation(
                    stim_factor=stim_factor,
                    stim_pulse=stim_pulse,
                    dt=time_step,
                    tstop=sim_dur,
                    return_only_spiking=return_only_spiking,
                    delete_hoc_objects=True,
                    min_n_spikes_per_sec=n_req_spikes,
                )
                # return {"ss": {"sss":"ss"}}, axon_obj_dict
        except Exception as e:
            print(
                f"!!! [ERROR] !!! Could not simulate axon: {axon_obj.name}",
                flush=True,
            )
            print(f"\t!!! ERROR MESSAGE: {str(e)}", flush=True)
            print(f"\t!!!   Traceback: {traceback.format_exc()}", flush=True)
            print(f"\t!!! FOLLOWING ARE AXON PARAMS: ", flush=True)
            print(f"\t\t!!! AXON DIAM: {axon_obj.fiberD}", flush=True)
            print(f"\t\t!!! AXON LENGTH: {axon_obj.total_length}", flush=True)
            axon_res = {"ErrorMessage": str(e)}
            if titration:
                 return {axon_obj.name: axon_res}, axon_obj_dict
            return {axon_obj.name: axon_res}
    except Exception as e:
        print(
            f"!!! [ERROR] !!! Could not create axon object for: {axon_obj_dict['name']}",
            flush=True,
        )
        print(f"\t!!! ERROR MESSAGE: {str(e)}", flush=True)
        print(f"\t!!!   Traceback: {traceback.format_exc()}", flush=True)
        print(f"\t!!! FOLLOWING ARE AXON PARAMS: ", flush=True)
        print(f"\t\t!!! AXON DIAM: {axon_obj_dict['fiberD']}", flush=True)
        axon_res = {"ErrorMessage": str(e)}
        if titration:
            return {axon_obj_dict['name']: axon_res}, axon_obj_dict
        return {axon_obj.name: axon_res}
    if motoneuron:
        axon_res["MN"] = {"Ia_afferent_inputs": mn.Ia_afferent_inputs, "results": mn.get_recorders_npy()}
        axon_res["ConnectedMN"] = True
        h.delete_section(sec=mn.soma)
        h.delete_section(sec=mn.initSegment)
        # plot_membrane_potential
        if plot_axon_vm:
            fig_dir = os.path.join(output_dir, "membrane_plots", axon_obj.name)
            if not os.path.exists(fig_dir):
                try:
                    os.makedirs(fig_dir)
                except:
                    pass
            fig_path = os.path.join(output_dir, "membrane_plots", axon_obj.name, f"MN_v_stimF{stim_factor}.png")
            plot_membrane_potential_from_res(
                axon_res,
                axon_name=axon_obj.name,
                fig_path=fig_path,
                highlight_color="#7f2b8f",
            )
        mn = None
    else:
        if plot_axon_vm:
            fig_dir = os.path.join(output_dir, "membrane_plots", axon_obj.name)
            if not os.path.exists(fig_dir):
                try:
                    os.makedirs(fig_dir)
                except:
                    pass
            fig_path = os.path.join(output_dir, "membrane_plots", axon_obj.name, f"Fiber_v_stimF{stim_factor}.png")
            plot_membrane_potential_from_res(
                axon_res,
                axon_name=axon_obj.name,
                fig_path=fig_path,
                highlight_color="#316b69",
            )
    if titration:
        titration_result = update_axon_titration(
            axon_obj_dict,
            axon_res,
            verbose=debug,
            titration_conv_perc=titration_conv_perc,
            initial_reduction_factor=initial_reduction_factor,
            initial_increment_factor=initial_increment_factor,
            max_stim_factor=max_stim_factor,
            min_stim_factor=min_stim_factor,
        )
        return titration_result, axon_obj_dict
    return {axon_obj.name: axon_res}

def simulate_axons(
    axons_sub_list,
    stim_t,
    stim_pulse,
    stim_factor_step=1.0,
    initial_stim_factor=1.0,
    max_stim_factor=300.0,
    passive_end_nodes=False,
    prepassive_nodes_as_endnodes=False,
    debug=False,
    record_v=False,
    exclude_end_node=False,
    time_step=0.005,
    sim_dur=4.0,
    recorded_v_nodes=None,
    recorded_ap_times_nodes=None,
    projecting_axons_results=None,
    motoneuron=False,
    fiber_traj_groups=None,
    disable_extracellular_stim=False,
    save_only_processed_responses=False,
    syn_weight=0.0056994,
    stim_amplitudes=None,
    proj_freq=50.0,
    output_dir="./outputs",
    plot_axons=[],
    ):

    comm = MPI.COMM_WORLD if MPI else DummyComm()
    rank = comm.Get_rank()
    
    if stim_amplitudes is None:
        stim_amplitudes = np.arange(initial_stim_factor, max_stim_factor, stim_factor_step)
    axons_results = {
        axon_info["name"]: {
            "segment_types": axon_info["segments_types"] if "segments_types" in axon_info else None,
            "segment_midpoints": axon_info["segments_midpoints"],
            "diameter": axon_info["fiberD"],
            "nnodes": axon_info["axonnodes"] if "axonnodes" in axon_info else None,
            "results": {np.round(k, 2): {} for k in stim_amplitudes},
            "connected_to_mn": motoneuron,
        }
        for axon_info in axons_sub_list
    }
    t_last_nonzero_in_pulse = stim_t[np.argwhere(stim_pulse != 0)[-1][0]]
    for stim_amp_i, stim_amp in enumerate(stim_amplitudes):
        stim_amp = np.round(stim_amp, 2)
        if rank == 0:
            t_start = time.perf_counter()
            print(f"\t\tStarting simulation loop for stim factor {stim_amp} ({stim_amp_i+1}/{len(stim_amplitudes)})...", flush=True)
        
        record_v_default = record_v
        n_spiking_axons = 0
        for axon_obj_dict in axons_sub_list:
            plot_axon_vm_flag = False
            record_v = record_v_default
            if plot_axons is not None and len(plot_axons) > 0:
                if axon_obj_dict["name"] in plot_axons:
                    plot_axon_vm_flag = True
                    record_v = True
            axon_res = simulate_axon(
                axon_obj_dict=axon_obj_dict,
                stim_pulse=stim_pulse,
                stim_factor=stim_amp,
                passive_end_nodes=passive_end_nodes,
                prepassive_nodes_as_endnodes=prepassive_nodes_as_endnodes,
                debug=debug,
                record_v=record_v,
                exclude_end_node=exclude_end_node,
                time_step=time_step,
                sim_dur=sim_dur,
                recorded_v_nodes=recorded_v_nodes,
                recorded_ap_times_nodes=recorded_ap_times_nodes,
                motoneuron=motoneuron,
                input_spikes=projecting_axons_results,
                fiber_traj_groups=fiber_traj_groups,
                disable_extracellular_stim=disable_extracellular_stim,
                syn_weight=syn_weight,
                proj_freq=proj_freq,
                output_dir=output_dir,
                plot_axon_vm=plot_axon_vm_flag,
            )
            if not "spikes_list" in list(axon_res.values())[0]:
                if "spike" in list(axon_res.values())[0]:
                    n_spiking_axons += 1
            else:
                if len(list(axon_res.values())[0]['spikes_list']) > 0:
                    n_spiking_axons += 1
            if save_only_processed_responses:
                if motoneuron:
                    # For efferents, save only classified responses
                    try:
                        # axons_results[list(axon_res.keys())[0]]["results"][stim_amp]["MN"] = list(axon_res.values())[0]["MN"]
                        axons_results[list(axon_res.keys())[0]]["results"][stim_amp]["responses_classified"] = classify_responses(list(axon_res.values())[0], last_t_in_pulse=t_last_nonzero_in_pulse, sim_dur=sim_dur)
                    except Exception as e:
                        print(
                            f"!!! [ERROR] !!! Could not classify responses for axon: {list(axon_res.keys())[0]}",
                            flush=True,
                        )
                        print(f"\t!!! ERROR MESSAGE: {str(e)}", flush=True)
                        print(f"\t!!!   Traceback: {traceback.format_exc()}", flush=True)
                else:
                    # For afferent fibers, only save AP times at last node and AP init sites
                    axons_results[list(axon_res.keys())[0]]["results"][stim_amp] = {
                        "AP_times": {"AP_times_last_node": list(axon_res.values())[0]['AP_times']['AP_times_last_node']},
                        "AP_init_sites": get_ap_init_nodes(list(axon_res.values())[0], reference="last", default_cond_v=67.8),
                    }
            else:
                axons_results[list(axon_res.keys())[0]]["results"][stim_amp] = {
                    "AP_times": list(axon_res.values())[0]['AP_times'],
                }
                if record_v:
                    axons_results[list(axon_res.keys())[0]]["results"][stim_amp]["t"] = list(axon_res.values())[0]['time_vector']
                    axons_results[list(axon_res.keys())[0]]["results"][stim_amp]["membrane_potential"] = list(axon_res.values())[0]['membrane_potential']
                if motoneuron:
                    axons_results[list(axon_res.keys())[0]]["results"][stim_amp]["MN"] = list(axon_res.values())[0]["MN"]
                    axons_results[list(axon_res.keys())[0]]["results"][stim_amp]["responses_classified"] = classify_responses(list(axon_res.values())[0], last_t_in_pulse=t_last_nonzero_in_pulse, sim_dur=sim_dur)
                else:
                    axons_results[list(axon_res.keys())[0]]["results"][stim_amp]["AP_init_sites"] = get_ap_init_nodes(list(axon_res.values())[0], reference="last", default_cond_v=67.8)

        # axons_with_spikes = [k for k, v in axons_results.items() if len(v["results"][stim_amp]['APs']) > 0]
        if rank == 0:
            t_end = time.perf_counter()
            print(f"\t\tRank 0:")
            print(
                f"\t\t Finished one loop iteration in: {t_end - t_start} seconds!",
                flush=True,
            )
            print(
                f"\t\t Number of axons results so far: {len(axons_results)}",
                flush=True,
            )
            print(
                f"\t\t Number of axons with spikes: {n_spiking_axons}",
                flush=True,
            )
    return axons_results