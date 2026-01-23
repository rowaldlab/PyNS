'''
###########################################
# File: pyns/utils.py
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
import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator
from scipy import signal
import re
import matplotlib.pyplot as plt


class DummyComm:
    """Dummy MPI communicator for fallback when MPI is not available."""

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, data, root=0):
        return data

    def gather(self, data, root=0):
        return [data]

    def Barrier(self):
        return None


def get_traj_name(axon_name):
    segment = axon_name.split("_")[0]
    axon_name_splits = axon_name.split("_")
    # find the index containint 'traj'
    if not "traj" in axon_name_splits:
        return ""
    traj_index = axon_name_splits.index("traj")
    # find the index containing 'fiber', if no fiber is found, find the index containing 'diam', else use the end of the list
    if "fiber" not in axon_name_splits:
        if "diam" in axon_name_splits:
            fiber_index = axon_name_splits.index("diam")
        else:
            fiber_index = len(axon_name_splits)
    else:
        fiber_index = axon_name_splits.index("fiber")
    traj_name = "_".join(axon_name_splits[traj_index+1:fiber_index])
    return traj_name

def axon_names_to_traj_groups(axon_names, afferent_kws_all=None, efferent_kws_all=None, root_kws_any=["rlet", "anstm"]):
    # parameter checks: axon_names should be a list of strings, all others should be lists or None
    if not isinstance(axon_names, list) or not all(isinstance(k, str) for k in axon_names):
        raise ValueError("axon_names should be a list of strings")
    if afferent_kws_all is not None and not isinstance(afferent_kws_all, list):
        raise ValueError("afferent_kws_all should be a list or None")
    if efferent_kws_all is not None and not isinstance(efferent_kws_all, list):
        raise ValueError("efferent_kws_all should be a list or None")
    if root_kws_any is not None and not isinstance(root_kws_any, list):
        raise ValueError("root_kws_any should be a list or None")
    
    traj_names = list(set([get_traj_name(k) for k in axon_names if any(kw in k for kw in root_kws_any)]))
    # for each traj name, get the dorsal and ventral root axons
    axon_groups = {}
    for traj_name in traj_names:
        axon_groups[traj_name] = {}
        # get unique segments
        segments = list(set([k.split("_")[0] for k in axon_names if any(kw in k for kw in root_kws_any) and f"traj_{traj_name}" in k]))
        for seg in segments:
            axon_groups[traj_name][seg] = {}
            for side in ["R", "L"]:
                dorsal_pos = f"D{side}"
                ventral_pos = f"V{side}"
                if afferent_kws_all is not None:
                    d_axons_in_seg = [k for k in axon_names if seg==k.split("_")[0] and f"_{dorsal_pos}_" in k and f"traj_{traj_name}" in k and all(kw in k for kw in afferent_kws_all)]
                else:
                    d_axons_in_seg = [k for k in axon_names if seg==k.split("_")[0] and f"_{dorsal_pos}_" in k and f"traj_{traj_name}" in k]
                if efferent_kws_all is not None:
                    v_axons_in_seg = [k for k in axon_names if seg==k.split("_")[0] and f"_{ventral_pos}_" in k and f"traj_{traj_name}" in k and all(kw in k for kw in efferent_kws_all)]
                else:
                    v_axons_in_seg = [k for k in axon_names if seg==k.split("_")[0] and f"_{ventral_pos}_" in k and f"traj_{traj_name}" in k]
                axon_groups[traj_name][seg][dorsal_pos] = d_axons_in_seg
                axon_groups[traj_name][seg][ventral_pos] = v_axons_in_seg
    return axon_groups

def axon_dicts_to_afferent_efferent_groups(
        axon_dicts,
        afferent_kws_all=None,
        efferent_kws_all=None,
        afferent_kws_any=["Sensory", "sensory", "_Aalpha", "_DR", "_DL"],
        efferent_kws_any=["Motor", "motor", "_alpha", "_VR", "_VL"],
        root_kws_any=["rlet", "anstm"]):
    
    # parameter checks: axon_dicts should be a list of dicts, all others should be lists or None
    if not isinstance(axon_dicts, list) or not all(isinstance(k, dict) for k in axon_dicts):
        raise ValueError("axon_dicts should be a list of dicts")
    if afferent_kws_all is not None and not isinstance(afferent_kws_all, list):
        raise ValueError("afferent_kws_all should be a list or None")
    if efferent_kws_all is not None and not isinstance(efferent_kws_all, list):
        raise ValueError("efferent_kws_all should be a list or None")
    if afferent_kws_any is not None and not isinstance(afferent_kws_any, list):
        raise ValueError("afferent_kws_any should be a list or None")
    if efferent_kws_any is not None and not isinstance(efferent_kws_any, list):
        raise ValueError("efferent_kws_any should be a list or None")
    if root_kws_any is not None and not isinstance(root_kws_any, list):
        raise ValueError("root_kws_any should be a list or None")
    
    # resolve condtions one by one
    if root_kws_any is not None and len(root_kws_any) > 0:
        root_condition = [any([kw in axon_dict["axon_name"] for kw in root_kws_any]) for axon_dict in axon_dicts]
    else:
        root_condition = [True for _ in axon_dicts]
    if afferent_kws_any is not None and len(afferent_kws_any) > 0:
        afferent_condition = [any([kw in axon_dict["axon_name"] for kw in afferent_kws_any]) for axon_dict in axon_dicts]
    else:
        afferent_condition = [True for _ in axon_dicts]
    if efferent_kws_any is not None and len(efferent_kws_any) > 0:
        efferent_condition = [any([kw in axon_dict["axon_name"] for kw in efferent_kws_any]) for axon_dict in axon_dicts]
    else:
        efferent_condition = [True for _ in axon_dicts]
    if afferent_kws_all is not None and len(afferent_kws_all) > 0:
        afferent_kws_all_condition = [all([kw in axon_dict["axon_name"] for kw in afferent_kws_all]) for axon_dict in axon_dicts]
    else:
        afferent_kws_all_condition = [True for _ in axon_dicts]
    if efferent_kws_all is not None and len(efferent_kws_all) > 0:
        efferent_kws_all_condition = [all([kw in axon_dict["axon_name"] for kw in efferent_kws_all]) for axon_dict in axon_dicts]
    else:
        efferent_kws_all_condition = [True for _ in axon_dicts]

    afferent_indices = np.argwhere((np.array(root_condition, dtype=bool) & np.array(afferent_condition, dtype=bool) & np.array(afferent_kws_all_condition, dtype=bool)))[:,0]
    efferent_indices = np.argwhere((np.array(root_condition, dtype=bool) & np.array(efferent_condition, dtype=bool) & np.array(efferent_kws_all_condition, dtype=bool)))[:,0]
    # if afferent_indices and efferent_indices overlap, removel all and add to other
    overlap_indices = np.intersect1d(afferent_indices, efferent_indices)
    afferent_indices = np.setdiff1d(afferent_indices, overlap_indices)
    efferent_indices = np.setdiff1d(efferent_indices, overlap_indices)
    other_indices = np.setdiff1d(np.arange(len(axon_dicts)), np.concatenate((afferent_indices, efferent_indices)))
    
    afferent_axons = [axon_dicts[i] for i in afferent_indices]
    efferent_axons = [axon_dicts[i] for i in efferent_indices]
    other_axons = [axon_dicts[i] for i in other_indices]

    return afferent_axons, efferent_axons, other_axons

def pulse_file_to_pulse(pulse_path, stim_dur=5, time_step=0.025, start_at=0):
    x, y = np.genfromtxt(pulse_path)
    pulse_x = np.arange(0, stim_dur, time_step)
    pulse_y = np.zeros((len(pulse_x)))
    for orig_i, y_val in enumerate(y[:-1]):
        start_index = np.argmin(np.abs(x[orig_i] + start_at - pulse_x))
        start_val = y_val
        end_index = np.argmin(np.abs(x[orig_i + 1] + start_at - pulse_x))
        end_val = y[orig_i + 1]
        n_samples = end_index - start_index
        pulse_y[start_index:end_index] = np.linspace(start_val, end_val, n_samples)
    return pulse_x, pulse_y


def create_cont_stim_waveform(
    silence_period=1,
    burst_freq=30,
    carrier_freq=10000,
    burst_width=1,
    time_step=0.005,
    total_stim_dur=250,
    amplitude=1.0,
    biphasic=False,
):
    """Create a continuous stimulation pulse train with given parameters"""
    # convert freq to be in ms
    burst_freq = burst_freq * 1e-3
    carrier_freq = carrier_freq * 1e-3

    # first generate the 10 kHz pulse
    time_vector_one_pulse = np.arange(0, burst_width, time_step)
    sq_signal = signal.square(2 * np.pi * carrier_freq * time_vector_one_pulse)
    # if biphasic:
    #     sq_signal = -signal.square(2 * amplitude * np.pi * freq * time_vector)
    # else:
    #     sq_signal = signal.square(2 * amplitude * np.pi * (freq/2.) * time_vector)
    if not biphasic:
        # convert -ve values to 0
        sq_signal = np.maximum(sq_signal, 0)

    # print(f"Frequency: {carrier_freq*1e3} kHz, Burst freq: {1/burst_freq} Hz, Burst width: {burst_width} ms, Time res: {time_step} ms, Total stim dur: {total_stim_dur} ms, Amplitude: {amplitude}, Biphasic: {biphasic}")
    if carrier_freq == 1.0*1e-3 and biphasic:
        # print("Warning: freq is 1 Hz and biphasic is True, setting biphasic to False")
        # split into two halves of +ve and -ve amplitude
        half_index = len(sq_signal) // 2
        sq_signal[:half_index] = amplitude
        sq_signal[half_index:] = -amplitude

    # get silece period and concatenate it to the pulse
    burst_period = 1 / burst_freq
    post_silence_period = burst_period - burst_width
    post_silence_vector = np.arange(0, post_silence_period, time_step)
    post_silence = np.zeros((len(post_silence_vector)))
    pulse = np.concatenate((sq_signal, post_silence))

    # repeat the pulse to fill in 1 second with 30 Hz
    n_repeats = np.ceil(total_stim_dur / burst_period)
    if n_repeats < 1 and total_stim_dur > burst_width:
        n_repeats = 1
    pulse = np.tile(pulse, int(n_repeats))
    # time_vector = np.arange(0, len(pulse) * time_step, time_step)

    # prepend silence period
    silence_vector = np.zeros((int(silence_period / time_step)))
    pulse = np.concatenate((silence_vector, pulse))
    # time_vector = np.arange(0, len(pulse) * time_step, time_step)
    pulse = pulse[:int(total_stim_dur / time_step)]
    # time_vector = time_vector[:int(total_stim_dur / time_step)]
    time_vector = np.arange(0, total_stim_dur, time_step)
    if len(pulse) < len(time_vector):
        # pad with zeros
        pulse = np.concatenate((pulse, np.zeros((len(time_vector) - len(pulse)))))
    elif len(pulse) > len(time_vector):
        # truncate
        pulse = pulse[:len(time_vector)]
    
    # multiply by amplitude
    pulse = pulse * amplitude

    return time_vector, pulse

def create_single_pulse_waveform(
    stim_dur=5,
    time_step=0.025,
    start_at=1,
    end_at=3,
    amplitude=1.0,
    biphasic=False,
):
    pulse_x = np.arange(0, stim_dur, time_step)
    pulse_y = np.zeros((len(pulse_x)))
    start_index = np.argmin(np.abs(start_at - pulse_x))
    end_index = np.argmin(np.abs(end_at - pulse_x))
    if biphasic:
        end_index1 = start_index + (end_index - start_index) // 2
        pulse_y[start_index:end_index1] = amplitude
        pulse_y[end_index1:end_index] = -amplitude
    else:
        pulse_y[start_index:end_index] = amplitude
    return pulse_x, pulse_y


def create_multiple_pulses_waveform(
    stim_dur=5,
    time_step=0.025,
    start_at=[1, 4],
    end_at=[2, 5],
    amplitude=[0.25, 0.5],
    biphasic=False,
):
    """Create multiple pulses with different start and end times"""
    pulse_x = np.arange(0, stim_dur, time_step)
    pulse_y = np.zeros((len(pulse_x)))
    for start, end, amp in zip(start_at, end_at, amplitude):
        start_index = np.argmin(np.abs(start - pulse_x))
        end_index = np.argmin(np.abs(end - pulse_x))
        if biphasic:
            end_index1 = start_index + (end_index - start_index) // 2
            pulse_y[start_index:end_index1] = amp
            pulse_y[end_index1:end_index] = -amp
        else:
            pulse_y[start_index:end_index] = amp
    return pulse_x, pulse_y

def interpolate_3d(field_dict, interpolation_points):
    interp = RegularGridInterpolator(
        (field_dict["x"], field_dict["y"], field_dict["z"]), field_dict["field_values"]
    )
    return interp(interpolation_points)

def get_arcline_length(line_points, return_length_per_point=False):
    """Calculate the length of an arc line defined by a list of points"""
    line_lengths = np.sqrt(
        np.sum(
            np.square(np.diff(line_points, n=1, axis=0)),
            axis=1,
        )
    )
    line_lengths = np.insert(line_lengths, 0, 0.0)
    if return_length_per_point:
        return np.sum(line_lengths), np.cumsum(line_lengths)
    return np.sum(line_lengths)

def filter_axon_trajectories(
    axons_dict, x_range, y_range, z_range, min_axon_length=5, axons_kws_any=None, rank=0, default_diam=16.0
):
    # axon_points are expected to be in mm and ranges are in um

    axon_dicts = []
    lengths = []
    removed_axons_names_length = []
    removed_axons_names_range = []
    for axon_name, org_axon_points in axons_dict.items():
        if axons_kws_any is not None:
            if not any([kw in axon_name for kw in axons_kws_any]):
                continue
        # axon_points = np.load(axon_path)
        # print(np.min(axon_points), np.max(axon_points))
        axon_points = org_axon_points * 1e3  # mm to um
        indices_to_keep = np.argwhere(
            (axon_points[:, 0] > x_range[0])
            & (axon_points[:, 0] < x_range[1])
            & (axon_points[:, 1] > y_range[0])
            & (axon_points[:, 1] < y_range[1])
            & (axon_points[:, 2] > z_range[0])
            & (axon_points[:, 2] < z_range[1])
        )[:, 0]
        if len(indices_to_keep) == len(axon_points):
            axon_points = axon_points[indices_to_keep]
            total_len = np.sum(np.linalg.norm(np.diff(axon_points, axis=0), axis=1))
            if total_len >= min_axon_length:
                # axon_file_name = os.path.basename(axon_path)
                diam_string = [
                    sub_string
                    for s_i, sub_string in enumerate(axon_name.split("_"))
                    if "um" in sub_string and axon_name.split("_")[s_i-1] == "diam"
                ]
                innerdiam_string = [
                    sub_string
                    for s_i, sub_string in enumerate(axon_name.split("_"))
                    if "um" in sub_string and axon_name.split("_")[s_i-1] == "axondiam"
                ]
                if len(diam_string) == 0:
                    diam = default_diam
                else:
                    diam = float(re.findall(r"[-+]?(?:\d*\.*\d+)", diam_string[0])[0])
                if len(innerdiam_string) == 0:
                    inner_diam = None
                else:
                    inner_diam = float(re.findall(r"[-+]?(?:\d*\.*\d+)", innerdiam_string[0])[0])
                # print("diam_string: ", diam_string)
                # diam = float(re.findall(r"[-+]?(?:\d*\.*\d+)", diam_string)[0])
                axon_name = axon_name.replace(".npy", "")
                axon_dicts.append(
                    {
                        "points": axon_points,  # mm to um
                        "diam": diam,  # um
                        "inner_diam": inner_diam,
                        "axon_name": axon_name,
                        "length": total_len,
                    }
                )
                lengths.append(total_len)
            else:
                removed_axons_names_length.append(axon_name)
        elif len(indices_to_keep) > 0:
            axon_traj_splits = []
            traj_inds = []
            last_ind = -1
            # print(f"\t\t Traj: {axon_name} original length: {len(axon_points)}")
            for ind in np.arange(axon_points.shape[0]):
                if (
                    axon_points[ind, 0] > x_range[0]
                    and (axon_points[ind, 0] < x_range[1])
                    and (axon_points[ind, 1] > y_range[0])
                    and (axon_points[ind, 1] < y_range[1])
                    and (axon_points[ind, 2] > z_range[0])
                    and (axon_points[ind, 2] < z_range[1])
                ):
                    if last_ind == ind - 1 or last_ind == -1:
                        traj_inds.append(ind)
                        last_ind = ind
                    else:
                        axon_traj_splits.append(traj_inds.copy())
                        traj_inds = []
                        last_ind = -1
                    if ind == len(axon_points) - 1 and len(traj_inds) > 0:
                        axon_traj_splits.append(traj_inds.copy())
                        traj_inds = []
                        last_ind = -1
                else:
                    if len(traj_inds) > 0:
                        axon_traj_splits.append(traj_inds.copy())
                        traj_inds = []
                        last_ind = -1
            # print(f"\t\t\t Got {len(axon_traj_splits)} traj splits, with lengths {[len(traj) for traj in axon_traj_splits]}")
            axon_points_inds = axon_traj_splits[
                np.argmax(np.array([len(traj) for traj in axon_traj_splits]))
            ]
            axon_points = axon_points[np.array(axon_points_inds), :]
            # print(f"\t\t\t LENGTH: {len(axon_points)}")
            total_len = np.sum(np.linalg.norm(np.diff(axon_points, axis=0), axis=1))
            if total_len >= min_axon_length:
                # axon_file_name = os.path.basename(axon_path)
                diam_string = [
                    sub_string
                    for s_i, sub_string in enumerate(axon_name.split("_"))
                    if "um" in sub_string and axon_name.split("_")[s_i-1] == "diam"
                ]
                innerdiam_string = [
                    sub_string
                    for s_i, sub_string in enumerate(axon_name.split("_"))
                    if "um" in sub_string and axon_name.split("_")[s_i-1] == "axondiam"
                ]
                if len(diam_string) == 0:
                    diam = default_diam
                else:
                    diam = float(re.findall(r"[-+]?(?:\d*\.*\d+)", diam_string[0])[0])
                if len(innerdiam_string) == 0:
                    inner_diam = None
                else:
                    inner_diam = float(re.findall(r"[-+]?(?:\d*\.*\d+)", innerdiam_string[0])[0])
                # print("diam_string: ", diam_string)
                # diam = float(re.findall(r"[-+]?(?:\d*\.*\d+)", diam_string)[0])
                axon_name = axon_name.replace(".npy", "")
                axon_dicts.append(
                    {
                        "points": axon_points,  # mm to um
                        "diam": diam,  # um
                        "inner_diam": inner_diam,
                        "axon_name": axon_name,
                        "length": total_len,
                    }
                )
                lengths.append(total_len)
            else:
                removed_axons_names_length.append(axon_name)
        else:
            removed_axons_names_range.append(axon_name)
    if lengths and rank == 0:
        print(f"\t\t Filtered axons minimum length: {np.min(lengths)}", flush=True)
        print(f"\t\t Filtered axons maximum length: {np.max(lengths)}", flush=True)
    if rank == 0:
        print(
            f"\t\t List of axons excluded due to being out of field: {removed_axons_names_range}",
            flush=True,
        )
        print(
            f"\t\t List of axons excluded due to minimum length criterion: {removed_axons_names_length}",
            flush=True,
        )
    return axon_dicts

def save_results(results_to_save, output_npy_path):
    try:
        np.save(output_npy_path, results_to_save, allow_pickle=True)
    except Exception as e:
        print(f"\t !!! Saving in a npy failed with this error: {e} !!!")
        print(f"\t   Saving with pickle...")
        if os.path.isfile(output_npy_path):
            try:
                os.remove(output_npy_path)
            except:
                pass
        pkl_path = output_npy_path.replace(".npy", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
