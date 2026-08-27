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
import h5py
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

    def Split(self, color=0, key=0):
        return self

    def Split_type(self, split_type, key=0):
        return self

    def send(self, data, dest=0, tag=0):
        return None

    def recv(self, source=0, tag=0):
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

def create_capacitive_stim_waveform(
        silence_period=1,
        total_stim_dur=5,
        amplitude=1.0,
        time_step=0.005,
        frequency=0.0,
        pulse_width=1.0,
        tau=0.5,                 # R_load * C_block [ms] -- sets droop AND tail
        i_compliance=np.inf,     # V_rail / R_load, same units as amplitude
):
    """Create a capacitive stimulation pulse train with given parameters.

    One RC governs everything, so there is no independent droop rate.

    i_compliance = inf        -> regulated throughout, flat top (mid-cost long-pulse TENS)
    i_compliance <= amplitude -> never regulates, exponential droop from t=0
                                 (low-cost short-pulse TENS: pass i_compliance=amplitude)
    in between                -> flat until the cap eats the headroom, then
                                 droops (a current source that clips mid-pulse)
    """
    period = 1000.0 / frequency if frequency > 0 else total_stim_dur

    n_period_steps = int(period / time_step)
    single_pulse = np.zeros((n_period_steps))

    # active phase: flat while regulating, exponential once compliance is gone
    n_pulse_steps = min(int(pulse_width / time_step), n_period_steps)
    t_active = np.arange(n_pulse_steps) * time_step
    peak = min(amplitude, i_compliance)
    t_knee = tau * (i_compliance / peak - 1.0) if peak > 0 else np.inf
    active_seg = peak * np.exp(-np.maximum(0.0, t_active - t_knee) / tau)
    single_pulse[:n_pulse_steps] = active_seg

    # recovery tail: peak set by charge conservation, NOT by amplitude
    if n_pulse_steps < n_period_steps:
        q_pulse = active_seg.sum() * time_step
        t_tail = np.arange(n_period_steps - n_pulse_steps) * time_step
        single_pulse[n_pulse_steps:] = -(q_pulse / tau) * np.exp(-t_tail / tau)

    active_dur = total_stim_dur - silence_period
    n_repeats = int(np.ceil(active_dur / period)) if period > 0 else 1
    repeated_pulse = np.tile(single_pulse, n_repeats)
    repeated_pulse = repeated_pulse[:int(active_dur / time_step)]
    silence_vector = np.zeros((int(silence_period / time_step)))
    pulse = np.concatenate((silence_vector, repeated_pulse))
    pulse = pulse[:int(total_stim_dur / time_step)]
    time_vector = np.arange(0, total_stim_dur, time_step)

    return time_vector, pulse

def create_cont_stim_waveform(
    silence_period=1,
    burst_freq=0,
    carrier_freq=0,
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
    if not biphasic:
        # convert -ve values to 0
        sq_signal = np.maximum(sq_signal, 0)

    # print(f"Frequency: {carrier_freq*1e3} kHz, Burst freq: {1/burst_freq} Hz, Burst width: {burst_width} ms, Time res: {time_step} ms, Total stim dur: {total_stim_dur} ms, Amplitude: {amplitude}, Biphasic: {biphasic}")
    # if there is no carrier frequency, then the pulse is just a square wave of amplitude and duration burst_width
    if carrier_freq == 0 and biphasic:
        # print("Warning: freq is 1 Hz and biphasic is True, setting biphasic to False")
        # split into two halves of +ve and -ve amplitude
        half_index = len(sq_signal) // 2
        sq_signal[:half_index] = amplitude
        sq_signal[half_index:] = -amplitude

    # get silece period and concatenate it to the pulse
    # if burst_freq == 0, then the burst period is equal to the total stim duration: only one burst will be generated
    if burst_freq == 0:
        burst_period = total_stim_dur
    else:
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

def interpolate_3d(field_dicts, interpolation_points):
    interp_list = []
    for field_dict in field_dicts:
        interp = RegularGridInterpolator(
            (field_dict["x"], field_dict["y"], field_dict["z"]), field_dict["field_values"]
        )
        interp_list.append(interp(interpolation_points))
    return interp_list

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
        if axons_kws_any:
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
        if len(indices_to_keep) > 0:
            axon_points = axon_points[indices_to_keep]
            total_len = np.sum(np.linalg.norm(np.diff(axon_points, axis=0), axis=1))
            if total_len >= min_axon_length:
                # axon_file_name = os.path.basename(axon_path)
                diam_string = [
                    sub_string
                    for sub_string in axon_name.split("_")
                    if "um" in sub_string
                ][0]
                # print("diam_string: ", diam_string)
                diam = float(re.findall(r"[-+]?(?:\d*\.*\d+)", diam_string)[0])
                axon_name = axon_name.replace(".npy", "")
                axon_dicts.append(
                    {
                        "points": axon_points,  # mm to um
                        "diam": diam,  # um
                        "axon_name": axon_name,
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


def _dict_to_hdf5_group(group, d):
    """Recursively write a (possibly nested) dict of arrays/scalars into an h5py group.

    HDF5 group/dataset names must be strings, but dict keys here can be floats, tuples, etc.
    (e.g. stim factors). To preserve the original key type, entries are stored under an
    index-based name and the real key is pickled into a "__key__" attribute.
    """
    for idx, (key, val) in enumerate(d.items()):
        name = f"item_{idx}"
        key_blob = np.frombuffer(pickle.dumps(key, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8)
        if isinstance(val, dict):
            subgroup = group.create_group(name)
            subgroup.attrs["__key__"] = key_blob
            _dict_to_hdf5_group(subgroup, val)
            continue
        if val is None:
            dset = group.create_dataset(name, data=np.zeros(1, dtype=np.uint8))
            dset.attrs["__key__"] = key_blob
            dset.attrs["__none__"] = True
            continue
        try:
            arr = np.asarray(val)
            if arr.dtype != object:
                dset = group.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
                dset.attrs["__key__"] = key_blob
                continue
        except Exception:
            pass
        # fallback for arbitrary/irregular python objects that can't be stored as a plain array
        blob = np.frombuffer(pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8)
        dset = group.create_dataset(name, data=blob)
        dset.attrs["__key__"] = key_blob
        dset.attrs["__pickled__"] = True


def _hdf5_group_to_dict(group):
    """Recursively read an h5py group back into a dict, restoring original key types."""
    d = {}
    for name, item in group.items():
        if "__key__" in item.attrs:
            key = pickle.loads(np.asarray(item.attrs["__key__"]).tobytes())
        else:
            key = name  # fallback for entries written without key metadata
        if isinstance(item, h5py.Group):
            d[key] = _hdf5_group_to_dict(item)
        elif item.attrs.get("__none__", False):
            d[key] = None
        elif item.attrs.get("__pickled__", False):
            d[key] = pickle.loads(item[()].tobytes())
        else:
            d[key] = item[()]
    return d


def save_results_hdf5(results_dict, filepath, group_name=None, mode="w"):
    """Dump a (possibly nested) results dict to an HDF5 file, optionally under a named group.

    Used for temporary per-process/per-node result dumps instead of `.npy`/pickle, since HDF5
    supports compression and appending additional groups to an existing file (mode="a").
    """
    with h5py.File(filepath, mode) as f:
        target = f.create_group(group_name) if group_name else f
        _dict_to_hdf5_group(target, results_dict)


def load_results_hdf5(filepath, group_name=None):
    """Load a results dict previously written with `save_results_hdf5`."""
    with h5py.File(filepath, "r") as f:
        target = f[group_name] if group_name else f
        return _hdf5_group_to_dict(target)


def prune_afferent_results_for_synaptic_transmission(afferent_results_all):
    """Strip afferent results down to only what synaptic transmission needs (segment_midpoints
    and, per stim amplitude, AP_init_sites), instead of the full per-axon results (AP_times,
    recorded membrane potentials, etc.). Used to shrink the copy broadcast to every rank, since
    that full dict would otherwise be replicated in every single MPI process's memory.
    """
    return {
        axon_name: {
            "segment_midpoints": axon_res["segment_midpoints"],
            "results": {
                stim_key: {"AP_init_sites": stim_res["AP_init_sites"]}
                for stim_key, stim_res in axon_res["results"].items()
            },
        }
        for axon_name, axon_res in afferent_results_all.items()
    }


def merge_distributed_results(local_results, comm, node_comm, local_rank, global_rank, results_dir_sim, tag, mpi_module=None, broadcast_to_all=False, prune_fn=None):
    """Combine each rank's results dict while minimizing shared filesystem traffic.

    Instead of every rank dumping/reloading one temp file each (O(size), or O(size^2) when every
    rank also has to reload every other rank's file), this:
      1. Combines all ranks on a node in-memory via `node_comm.gather` (no disk I/O).
      2. Has only node leaders write to the shared filesystem, under results_dir_sim (one file per
         node instead of per rank).
      3. Throttles those writes with a token relay so leaders touch the shared filesystem one at
         a time instead of all at once.
    Only rank 0 reads the (few) per-node files back and merges them; the merged dict is broadcast
    to all ranks only if `broadcast_to_all` is True. The temp files are removed afterward.
    `mpi_module` should be the imported `mpi4py.MPI` module (or None when MPI is unavailable).
    `prune_fn`, if provided, is applied (on rank 0 only) to shrink the dict actually sent over the
    broadcast, so every rank isn't forced to hold a full-size copy in memory; rank 0 still returns
    the unpruned `merged_results` for its own use (e.g. saving to disk).
    """
    node_results_list = node_comm.gather(local_results, root=0)
    node_results = {}
    if local_rank == 0:
        for res in node_results_list:
            node_results.update(res)

    is_leader = (local_rank == 0)
    color = 0 if is_leader else (mpi_module.UNDEFINED if mpi_module is not None else None)
    leader_comm = comm.Split(color, global_rank)

    shared_tmp_dir = os.path.join(results_dir_sim, f"tmp_{tag}_results")
    if is_leader:
        os.makedirs(shared_tmp_dir, exist_ok=True)
        leader_rank = leader_comm.Get_rank()
        n_leaders = leader_comm.Get_size()
        shared_file_path = os.path.join(shared_tmp_dir, f"results_{tag}_node{leader_rank}.h5")

        # token relay: only one node leader touches the shared filesystem at a time
        if leader_rank != 0:
            leader_comm.recv(source=leader_rank - 1, tag=42)

        save_results_hdf5(node_results, shared_file_path)

        if leader_rank != n_leaders - 1:
            leader_comm.send(True, dest=leader_rank + 1, tag=42)

    comm.Barrier()

    merged_results = None
    if global_rank == 0:
        merged_results = {}
        for fname in sorted(os.listdir(shared_tmp_dir)):
            if fname.startswith(f"results_{tag}_node") and fname.endswith(".h5"):
                merged_results.update(load_results_hdf5(os.path.join(shared_tmp_dir, fname)))
    if broadcast_to_all:
        # only rank 0 needs the pruned dict computed; other ranks pass None and receive it via bcast
        to_broadcast = prune_fn(merged_results) if (prune_fn is not None and global_rank == 0) else merged_results
        received = comm.bcast(to_broadcast, root=0)
        if global_rank != 0:
            merged_results = received

    comm.Barrier()
    if global_rank == 0:
        for fname in os.listdir(shared_tmp_dir):
            if fname.startswith(f"results_{tag}_node") and fname.endswith(".h5"):
                os.remove(os.path.join(shared_tmp_dir, fname))
        try:
            os.rmdir(shared_tmp_dir)
        except OSError:
            pass

    return merged_results
