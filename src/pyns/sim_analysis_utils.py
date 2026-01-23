'''
###########################################
# File: pyns/sim_analysis_utils.py
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

def classify_responses(axon_result, last_t_in_pulse=3.0, sim_dur=5.0, fig_path=None):

    mn = axon_result["MN"]
    # get ap_times at initseg
    ap_soma = mn['results']['ap_soma_times']
    ap_initseg = mn['results']['ap_initseg_times']
    n_synapse_activ = 0
    # synapse is activated only if there is an AP at the initial segment happening after soma AP
    for soma_ap_time in ap_soma:
        # get the earliest AP at initseg
        if len(ap_initseg) > 0:
            closest_ap_time = ap_initseg[np.argmin(np.abs(np.array(ap_initseg) - soma_ap_time))]
            if closest_ap_time > soma_ap_time:
                n_synapse_activ += 1
    n_responses = len(axon_result['AP_times']["AP_times_first_node"])
    start_indices = [0] * n_responses
    n_spks_per_node = [len(axon_result['AP_times'][k]) for k in axon_result['AP_times'].keys()]
    # nodes_with_more_than_one = [ni for ni, n_spks in enumerate(n_spks_per_node) if n_spks > 1]
    nodes_keys_all = list(axon_result['AP_times'].keys())
    spiking_node_keys = [k for k in axon_result['AP_times'].keys() if "AP_times_node_" in k and len(axon_result['AP_times'][k]) > 0]
    if n_responses < np.max(n_spks_per_node):
        # get the nodes that have their last spike happening at the end
        last_spike_time_per_node = {k: np.max(axon_result['AP_times'][k]) for k in axon_result['AP_times'].keys() if k in spiking_node_keys}
        nodes_spiking_at_end = [k for k, v in last_spike_time_per_node.items() if np.abs(sim_dur - v) < 2]
        # for each node, check that the next node does not have a spike occuring at or after the last spike time
        last_checked_spike_time = -1
        last_checked_node_index = -1
        for k_i, k in enumerate(nodes_spiking_at_end):
            node_index = int(k.replace("AP_times_node_", ""))
            next_k = f"AP_times_node_{node_index-1}"
            if not next_k in nodes_keys_all:
                # try last
                next_k = "AP_times_first_node"
                if not next_k in nodes_keys_all:
                    continue
            if not any(axon_result['AP_times'][next_k] >= last_spike_time_per_node[k]):
                if node_index == 0:
                    continue
                if (k_i == 0) or not (node_index == last_checked_node_index+1 and last_spike_time_per_node[k] >= last_checked_spike_time):
                    # append response and start index
                    n_responses += 1
                    start_indices.append(node_index)
                last_checked_spike_time = last_spike_time_per_node[k]
                last_checked_node_index = node_index

    earliest_ap_initseg = sim_dur + 1
    if n_synapse_activ > 0:
        earliest_ap_initseg = np.min(ap_initseg)

    n_direct_activ = n_responses
    direct_activ_init_nodes = []
    n_reflex_activ= 0
    spike_times_per_train = [[] for i in range(n_responses)]
    spike_nodesinds_per_train = [[] for i in range(n_responses)]
    
    reflex_responses_time_at_0 = []
    direct_responses_time_at_0 = []
    reflex_earliest_spike_time = []
    direct_earliest_spike_time = []
    responses_cond_vel = []
    if n_responses > 0:
        for r_i in range(n_responses):
            start_ind = start_indices[r_i]
            nodes_keys = nodes_keys_all.copy()
            if start_ind == 0:
                start_index = nodes_keys.index("AP_times_first_node")
            else:
                start_index = nodes_keys.index(f"AP_times_node_{start_ind}")
            if not nodes_keys[-1] == "AP_times_last_node":
                end_index = nodes_keys.index("AP_times_last_node")
            else:
                end_index = len(nodes_keys) - 1
            nodes_keys = nodes_keys[start_index:end_index+1]
            # make sure that all node keys in between follow a numerical order
            nodes_inds = [int(key.replace("AP_times_node_", "")) for key in nodes_keys[1:-1]]
            if not np.all(np.diff(nodes_inds) == 1):
                raise ValueError("Node keys are not in numerical order")

            if start_ind == 0:
                resp_ind = r_i
                while len(axon_result['AP_times']["AP_times_first_node"])-1 < resp_ind:
                    resp_ind -= 1
                spike_times_per_train[r_i].append(axon_result['AP_times']["AP_times_first_node"][resp_ind])
                spike_nodesinds_per_train[r_i].append(0)
            else:
                spike_times_per_train[r_i].append(axon_result['AP_times'][nodes_keys[0]][-1])
                spike_nodesinds_per_train[r_i].append(start_ind)
        for ni, key in enumerate(nodes_keys_all):
            # skip node 0 and last node
            if ni == 0 or key == "AP_times_first_node":
                continue
            if key == "AP_times_last_node":
                current_node_index = len(nodes_keys_all) - 1
            else:
                current_node_index = int(key.replace("AP_times_node_", ""))
            if len(axon_result['AP_times'][key]) > 0:
                spike_times_current_node = np.copy(axon_result['AP_times'][key])
                # print(f"\t\t Node {ni} has {len(spike_times_current_node)} spikes")
                spike_times_last_detectable = [spike_times[-1] for spike_times in spike_times_per_train]
                responses_time_diffs = np.zeros(n_responses)
                for spk_i, spk_time in enumerate(spike_times_current_node):
                    response_inds = np.arange(len(spike_times_per_train))
                    response_inds = [i for i in response_inds if start_indices[i] <= current_node_index]
                    spike_times_last_detectable_filtered = [spike_times_last_detectable[i] for i in response_inds]
                    closest_index = response_inds[np.argmin(np.abs(np.array(spike_times_last_detectable_filtered) - spk_time))]
                    # find the closest spike time to the last spike time
                    # closest_index = np.argmin(np.abs(np.array(spike_times_last_detectable) - spk_time))
                    # check if another spike was already assigned to this response
                    if responses_time_diffs[closest_index] > 0:
                        # if yes, check if the current spike time is closer to the last spike time
                        if np.abs(spk_time - spike_times_last_detectable[closest_index]) < responses_time_diffs[closest_index]:
                            # if yes, update the spike time
                            spike_times_per_train[closest_index][-1] = spk_time
                            spike_nodesinds_per_train[closest_index][-1] = ni
                            responses_time_diffs[closest_index] = np.abs(spk_time - spike_times_last_detectable[closest_index])
                        else:
                            # if no, skip this spike time
                            continue
                    else:
                        spike_times_per_train[closest_index].append(spk_time)
                        spike_nodesinds_per_train[closest_index].append(ni)
                        responses_time_diffs[closest_index] = np.abs(spk_time - spike_times_last_detectable[closest_index])

        if fig_path is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # for each spike train check where is the earliest spike time
        n_invalid_responses = 0
        for r_i in range(n_responses):
            
            if len(spike_times_per_train[r_i]) < 2:
                n_invalid_responses += 1
                continue
            propagate_time_per_node = np.mean(np.abs(np.diff(spike_times_per_train[r_i])))
            cond_vel = 0
            if propagate_time_per_node > 0:
                cond_vel = (axon_result["axon_nodetonode_dist"] * 1e-6) / (propagate_time_per_node * 1e-3)
            responses_cond_vel.append(cond_vel)

            earliest_spike_index = np.argmin(spike_times_per_train[r_i])
            time_at_node_0 = ((spike_nodesinds_per_train[r_i][earliest_spike_index]) * propagate_time_per_node) + spike_times_per_train[r_i][earliest_spike_index]

            if fig_path is not None:
                ax.plot(spike_nodesinds_per_train[r_i], spike_times_per_train[r_i], "o", label=f"Response {r_i}")
                # earliest spike index with 'x'
                ax.plot(spike_nodesinds_per_train[r_i][earliest_spike_index], spike_times_per_train[r_i][earliest_spike_index], "x", color="red")

            earliest_spike_time = spike_times_per_train[r_i][earliest_spike_index]

            if all([earliest_spike_index == len(spike_times_per_train[r_i]) - 1,
                    n_synapse_activ > 0,
                    earliest_spike_time > earliest_ap_initseg,
                    n_reflex_activ < n_synapse_activ]):
                n_reflex_activ += 1
                reflex_responses_time_at_0.append(time_at_node_0)
                reflex_earliest_spike_time.append(earliest_spike_time)
            elif all([earliest_spike_index in [len(spike_times_per_train[r_i]) - 2, len(spike_times_per_train[r_i]) - 3],
                      np.min(spike_times_per_train[r_i]) > last_t_in_pulse,
                      n_synapse_activ > 0,
                      earliest_spike_time > earliest_ap_initseg,
                      n_reflex_activ < n_synapse_activ]):
                n_reflex_activ += 1
                reflex_responses_time_at_0.append(time_at_node_0)
                reflex_earliest_spike_time.append(earliest_spike_time)
            else:
                try:
                    direct_activ_init_nodes.append({
                        "node": nodes_keys_all[spike_nodesinds_per_train[r_i][earliest_spike_index]].replace("AP_times_", ""),
                        "time": spike_times_per_train[r_i][earliest_spike_index],
                        "time_at_first_periph_node": spike_times_per_train[r_i][0],
                    })
                    direct_earliest_spike_time.append(spike_times_per_train[r_i][earliest_spike_index])
                except:
                    print(f"Error in response {r_i} with spike_nodesinds_per_train of length {len(spike_nodesinds_per_train[r_i])} and earliest spike index {earliest_spike_index}")

                direct_responses_time_at_0.append(time_at_node_0)
        if fig_path is not None:
            ax.set_xlabel("Node index")
            ax.set_ylabel("Spike time (ms)")
            ax.set_title("Spike times per response")
            ax.legend()
            fig.savefig(fig_path)
            plt.close(fig)
        
        n_responses -= n_invalid_responses
        n_direct_activ = n_responses - n_reflex_activ
        
    return {
        "n_responses": n_responses,
        "n_synapse_activ": n_synapse_activ,
        "n_direct_activ": n_direct_activ,
        "n_reflex_activ": n_reflex_activ,
        "direct_activ_init_nodes": direct_activ_init_nodes,
        "direct_responses_time_at_node0": direct_responses_time_at_0,
        "reflex_responses_time_at_node0": reflex_responses_time_at_0,
        "direct_earliest_spike_time": direct_earliest_spike_time,
        "reflex_earliest_spike_time": reflex_earliest_spike_time,
        "responses_cond_vel": responses_cond_vel,
    }

def get_ap_init_nodes(axon_result, reference="last", default_cond_v=67.8):
    if reference == "last":
        n_responses = len(axon_result['AP_times']["AP_times_last_node"])
    else:
        n_responses = len(axon_result['AP_times']["AP_times_first_node"])
    if n_responses == 0:
        return []
    spike_init_nodes = [] * n_responses
    spike_times_per_train = [[] for i in range(n_responses)]
    spike_nodesinds_per_train = [[] for i in range(n_responses)]
    nodes_keys = list(axon_result['AP_times'].keys())
    node_inds_in_segments = np.argwhere(np.array(axon_result['segment_types']) == "n")[:, 0]
    if not nodes_keys[0] == "AP_times_first_node":
        start_index = nodes_keys.index("AP_times_first_node")
        nodes_keys = nodes_keys[start_index:]
        node_inds_in_segments = node_inds_in_segments[start_index:]
        first_node_coord = axon_result['segment_midpoints'][node_inds_in_segments[0]]
    if not nodes_keys[-1] == "AP_times_last_node":
        end_index = nodes_keys.index("AP_times_last_node")
        nodes_keys = nodes_keys[:end_index+1]
        node_inds_in_segments = node_inds_in_segments[:end_index+1]
    last_node_coord = axon_result['segment_midpoints'][node_inds_in_segments[-1]]
        
    for r_i in range(n_responses):
        if reference == "last":
            spike_times_per_train[r_i].append(axon_result['AP_times']["AP_times_last_node"][r_i])
            spike_nodesinds_per_train[r_i].append(len(nodes_keys)-1)
        else:
            spike_times_per_train[r_i].append(axon_result['AP_times']["AP_times_first_node"][r_i])
            spike_nodesinds_per_train[r_i].append(0)
    if reference == "last": 
        iter_items = list(reversed(list(enumerate(nodes_keys))))
    else:
        iter_items = list(enumerate(nodes_keys))
    for ni, key in iter_items:
        # skip node 0 or last node
        if reference=="last" and ni == len(nodes_keys)-1:
            continue
        elif reference=="first" and ni == 0:
            continue
        if len(axon_result['AP_times'][key]) > 0:
            spike_times_current_node = np.copy(axon_result['AP_times'][key])
            # print(f"\t\t Node {ni} has {len(spike_times_current_node)} spikes")
            spike_times_last_detectable = [spike_times[-1] for spike_times in spike_times_per_train]
            responses_time_diffs = np.zeros(n_responses)
            for spk_i, spk_time in enumerate(spike_times_current_node):
                # find the closest spike time to the last spike time
                closest_index = np.argmin(np.abs(np.array(spike_times_last_detectable) - spk_time))
                # check if another spike was already assigned to this response
                if responses_time_diffs[closest_index] > 0:
                    # if yes, check if the current spike time is closer to the last spike time
                    if np.abs(spk_time - spike_times_last_detectable[closest_index]) < responses_time_diffs[closest_index]:
                        # if yes, update the spike time
                        spike_times_per_train[closest_index][-1] = spk_time
                        spike_nodesinds_per_train[closest_index][-1] = ni
                        responses_time_diffs[closest_index] = np.abs(spk_time - spike_times_last_detectable[closest_index])
                    else:
                        # if no, skip this spike time
                        continue
                else:
                    spike_times_per_train[closest_index].append(spk_time)
                    spike_nodesinds_per_train[closest_index].append(ni)
                    responses_time_diffs[closest_index] = np.abs(spk_time - spike_times_last_detectable[closest_index])
    # for each spike train check where is the earliest spike time
    for r_i in range(n_responses):
        earliest_spike_index = np.argmin(spike_times_per_train[r_i])
        # calculate conduction velocity
        spike_seg_index = node_inds_in_segments[spike_nodesinds_per_train[r_i][earliest_spike_index]]
        if reference == "last":
            dist = np.linalg.norm(last_node_coord - axon_result['segment_midpoints'][spike_seg_index])
            time_diff = axon_result['AP_times']["AP_times_last_node"][r_i] - spike_times_per_train[r_i][earliest_spike_index]
        else:
            dist = np.linalg.norm(first_node_coord - axon_result['segment_midpoints'][spike_seg_index])
            time_diff = axon_result['AP_times']["AP_times_first_node"][r_i] - spike_times_per_train[r_i][earliest_spike_index]
        # dist from um to m
        dist *= 1e-6
        # time from ms to s
        time_diff *= 1e-3
        if time_diff == 0:
            conduction_velocity = default_cond_v
        else:
            conduction_velocity = dist / np.abs(time_diff)
        
        spike_init_nodes.append({
            "node": nodes_keys[spike_nodesinds_per_train[r_i][earliest_spike_index]].replace("AP_times_", ""),
            "time": spike_times_per_train[r_i][earliest_spike_index],
            "time_at_last_node": axon_result['AP_times']["AP_times_last_node"][r_i],
            "conduction_velocity": conduction_velocity,
        })
    return spike_init_nodes

def get_ap_times_at_mn(projecting_axons, mn_coord):
    # convert mn_coord from um to m
    mn_coord = mn_coord * 1e-6
    ap_times_per_axon = {}
    for axon_name, axon_res in projecting_axons.items():
        afferent_last_seg_coord = axon_res["segment_midpoints"][-1] * 1e-6
        dist_to_mn = np.linalg.norm(afferent_last_seg_coord - mn_coord)
        ap_init_sites = axon_res['AP_init_sites']
        # conduction_velocity is in m/s
        # ap['time_at_last_node'] is in ms
        # total delay time should be in ms
        ap_times_at_mn = [ap['time_at_last_node']+((dist_to_mn/ap['conduction_velocity'])*1e3) for ap in ap_init_sites]
        ap_times_per_axon[axon_name] = ap_times_at_mn
    return ap_times_per_axon