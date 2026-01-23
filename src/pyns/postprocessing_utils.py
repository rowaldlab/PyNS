
import numpy as np

def get_ap_init_node_ratio(ap_node_str, n_nodes, passive_end_nodes=False):
    """
    Convert action potential initiation node string to a ratio between 0 and 1.
    Parameters:
    ap_node_str (str): String indicating the node where the action potential was initiated.
                          Can be "first_node", "last_node", or "node_X" where X is the node index (0-based).
    n_nodes (int): Total number of nodes in the axon.
    passive_end_nodes (bool): Whether the axon has passive end nodes. Defaults to False.
    Returns:
    float: Ratio of the action potential initiation node (0.0 for first node, 1.0 for last node).
    """
    if ap_node_str == "first_node":
        if passive_end_nodes:
            return 1.0 / n_nodes
        return 0.0
    elif ap_node_str == "last_node":
        if passive_end_nodes:
            return (n_nodes - 1) / n_nodes
        return 1.0
    else:
        node_index = int(ap_node_str.split("_")[-1])
        return (node_index+1)/n_nodes

def compute_recruitment_curves(tittion_results, group_names, separator=None, stim_factor_step=None, n_stim_factors=None):
    """
    Compute recruitment curves from titration results.

    Parameters:
    tittion_results (dict): Dictionary with axon names as keys and titration results as values.
    group_names (list): List of group names to compute recruitment curves for.
    separator (str, optional): Separator used to split group names into keywords. Defaults to None.
    stim_factor_step (float, optional): Step size for stimulation factors. If provided, generates a range from 0 to max stim factor with this step. Defaults to None.
    n_stim_factors (int, optional): Number of stimulation factors to generate linearly spaced between 0 and max stim factor. Defaults to None.
    Returns:
    dict: Dictionary with recruitment curves and action potential initiation sites (0 to 1) for each group.
    """

    # either stim_factor_step or n_stim_factors can be given, not both
    if stim_factor_step is not None and n_stim_factors is not None:
        raise ValueError("Either stim_factor_step or n_stim_factors can be given, not both!")

    recruitment_curves = {}
    for group_name in group_names:
        if separator is not None:
            group_keywords = group_name.split(separator)
        else:
            group_keywords = [group_name]
        group_results = {axon_name: res for axon_name, res in tittion_results.items() if all(kw in axon_name for kw in group_keywords)}
        if len(group_results) == 0:
            continue
        # get all stim factors that led to spikes
        stim_factors = [axon_res["spike"]["stim_factor"] for axon_res in group_results.values() if "spike" in axon_res]
        recruitment_stim_factors = np.unique(np.round(stim_factors, 2))
        # if stim_factor_step is given, create a range from 0 to max with that step
        if stim_factor_step is not None:
            max_stim = np.max(recruitment_stim_factors)
            recruitment_stim_factors = np.arange(0.0, max_stim + stim_factor_step, stim_factor_step)
        elif n_stim_factors is not None:
            max_stim = np.max(recruitment_stim_factors)
            recruitment_stim_factors = np.linspace(0.0, max_stim, n_stim_factors)
        else:
            # use existing stim factors
            # add an additional point with the same first step and 0.0 stim factor
            if len(recruitment_stim_factors) == 0:
                recruitment_stim_factors = np.array([0.0])
            else:
                step = recruitment_stim_factors[1] - recruitment_stim_factors[0]
                recruitment_stim_factors = np.insert(recruitment_stim_factors, 0, recruitment_stim_factors[0]-step)
                # add an additional point with the same last step
                step = recruitment_stim_factors[-1] - recruitment_stim_factors[-2]
                recruitment_stim_factors = np.append(recruitment_stim_factors, recruitment_stim_factors[-1]+step)
        n_axons = len(group_results)
        axons_with_spikes = {axon_name: axon_result for axon_name, axon_result in group_results.items() if "spike" in axon_result}
        recruitment_percentage = [(len([axon_name for axon_name in axons_with_spikes if group_results[axon_name]["spike"]["stim_factor"] <= recruitment_stim_factor])/n_axons)*100.0 for recruitment_stim_factor in recruitment_stim_factors]
        
        ap_init_sites = []
        for axon_name, axon_res in axons_with_spikes.items():
            if "spike_at_node" in axon_res["spike"].keys():
                loc = (axon_res["spike"]["spike_at_node"])/(len(np.argwhere(np.array(axon_res["segment_types"]) == "n")[:, 0])-1)
            elif "spike_sec_idx" in axon_res["spike"].keys():
                loc = (axon_res["spike"]["spike_sec_idx"])/(axon_res["nsecs"]-1)
            else:
                loc = None
            ap_init_sites.append(loc)
        
        recruitment_curves[group_name] = {
            "stim_factors": recruitment_stim_factors,
            "axon_names": list(group_results.keys()),
            "ap_init_sites": ap_init_sites,
            "recruitment_percentage": recruitment_percentage
        }

    return recruitment_curves

def compute_air_eir_curves(sim_results, group_names, separator=None, passive_end_nodes=False):
    """
    Compute AIR and EIR curves from simulation results.

    Parameters:
    sim_results (dict): Dictionary with axon names as keys and simulation results as values.
    group_names (list): List of group names to compute AIR and EIR curves for.
    separator (str, optional): Separator used to split group names into keywords. Defaults to None.
    passive_end_nodes (bool, optional): Whether the axons have passive end nodes. Defaults to False.
    Returns: dict of dicts:
        - For efferent groups:
            dict: Dictionary with recruitment curves of afferent- and efferent-initiated responses (AIR and EIR) and action potential initiation sites (0 to 1).
        - For afferent groups:
            dict: Dictionary with recruitment curves and action potential initiation sites (0 to 1).
    """

    air_eir_curves = {}
    for group_name in group_names:
        if separator is not None:
            group_keywords = group_name.split(separator)
        else:
            group_keywords = [group_name]
        group_results = {axon_name: res for axon_name, res in sim_results.items() if all(kw in axon_name for kw in group_keywords)}
        afferent_results = {axon_name: axon_res for axon_name, axon_res in group_results.items() if not axon_res["connected_to_mn"]}
        
        # Process afferents
        if len(afferent_results) > 0:
            stim_factors = np.sort(
                list(
                    set(
                        [
                            stim_factor
                            for axon_res in afferent_results.values()
                            for stim_factor in axon_res["results"].keys()
                        ]
                    )
                )
            )
            n_afferent_aps = np.zeros(len(stim_factors))
            n_activated_afferents = np.zeros(len(stim_factors))
            afferents_ap_init_sites = [[] for _ in range(len(stim_factors))]
            for stim_factor_i, stim_factor in enumerate(stim_factors):
                if stim_factor not in afferent_results[list(afferent_results.keys())[0]]["results"]:
                    continue
                n_afferent_aps[stim_factor_i] = np.sum([len([spike_t for spike_t in axon_res["results"][stim_factor]["AP_times"]["AP_times_last_node"]]) 
                                                                    for axon_res in afferent_results.values() if stim_factor in axon_res["results"]])
                n_activated_afferents[stim_factor_i] = len([axon_res for axon_res in afferent_results.values() if stim_factor in axon_res["results"] and len(axon_res["results"][stim_factor]["AP_times"]["AP_times_last_node"]) > 0])
                afferents_ap_init_sites[stim_factor_i] = [get_ap_init_node_ratio(ap['node'], axon_res['nnodes'], passive_end_nodes=passive_end_nodes) for axon_res in afferent_results.values() if stim_factor in axon_res["results"] for ap in axon_res["results"][stim_factor]["AP_init_sites"]]

            air_eir_curves[f"{group_name}_Afferents"] = {
                "stim_factors": stim_factors,
                "n_aps": n_afferent_aps,
                "n_activated_axons": n_activated_afferents,
                "ap_init_sites": afferents_ap_init_sites,
                "axon_names": list(afferent_results.keys()),
                "recruitment_percentage": (n_activated_afferents / len(afferent_results)) * 100.0,
            }
        

        # Process efferents
        efferent_results = {axon_name: axon_res for axon_name, axon_res in group_results.items() if axon_res["connected_to_mn"]}
        if len(efferent_results) > 0:
            efferents_ap_init_sites = [[] for _ in range(len(stim_factors))]
            n_responses = np.zeros(len(stim_factors))
            n_eir = np.zeros(len(stim_factors))
            n_air = np.zeros(len(stim_factors))
            n_directly_activated_efferents = np.zeros(len(stim_factors))
            for stim_factor_i, stim_factor in enumerate(stim_factors):
                if stim_factor not in efferent_results[list(efferent_results.keys())[0]]["results"]:
                    continue
                n_responses[stim_factor_i] = np.sum([axon_res['results'][stim_factor]['responses_classified']['n_responses'] for axon_res in efferent_results.values() if stim_factor in axon_res["results"]])
                n_eir[stim_factor_i] = np.sum([axon_res['results'][stim_factor]['responses_classified']['n_direct_activ'] for axon_res in efferent_results.values() if stim_factor in axon_res["results"]])
                n_air[stim_factor_i] = np.sum([axon_res['results'][stim_factor]['responses_classified']['n_reflex_activ'] for axon_res in efferent_results.values() if stim_factor in axon_res["results"]])
                axon_res_with_direct_activ = [axon_res for axon_res in efferent_results.values() if stim_factor in axon_res["results"] and axon_res['results'][stim_factor]['responses_classified']['n_direct_activ'] > 0]
                # get n_directly_activated_efferents and ap initiation sites
                n_directly_activated_efferents[stim_factor_i] = len(axon_res_with_direct_activ)
                efferents_ap_init_sites[stim_factor_i] = [get_ap_init_node_ratio(ap['node'], axon_res['nnodes'], passive_end_nodes=passive_end_nodes) for axon_res in axon_res_with_direct_activ for ap in axon_res['results'][stim_factor]['responses_classified']['direct_activ_init_nodes']]

            air_eir_curves[f"{group_name}_Efferents"] = {
                "stim_factors": stim_factors,
                "n_responses": n_responses,
                "n_eir": n_eir,
                "n_air": n_air,
                "n_directly_activated_efferents": n_directly_activated_efferents,
                "ap_init_sites": efferents_ap_init_sites,
                "axon_names": list(efferent_results.keys()),
                "recruitment_percentage": (n_directly_activated_efferents / len(efferent_results)) * 100.0,
            }

    return air_eir_curves