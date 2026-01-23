'''
###########################################
# File: pyns/arguments_parsers.py
# Project: pyns
# Author: Abdallah Alashqar (abdallah.j.alashqar@fau.de)
# -----
# PI: Andreas Rowald, PhD (andreas.rowald@fau.de)
# Associate Professor for Digital Health
# Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
https://www.pdh.med.fau.de/
###########################################
'''

import argparse
import yaml
try:
    from mpi4py import MPI
except Exception:
    MPI = None
    print(
        "Warning: MPI (mpi4py) is not installed; parallel processing is disabled. Using rank=0, size=1.",
        flush=True,
    )
import sys
from .config import TitrationsConfig, DiscreteSimulationsConfig
from .utils import DummyComm

def parse_titrations_arguments():
    """Parse arguments for titrations simulations using TitrationsConfig.
    
    Returns:
        TitrationsConfig: Configuration object with all parameters
    """
    comm = MPI.COMM_WORLD if MPI else DummyComm()
    rank = comm.Get_rank()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path", type=str, required=False, default=None, help="Path to YAML configuration file")
    parser.add_argument("--field_path", type=str, required=False, default=None, help="Path to the extracellular field data file. Units should be in V for potential values and m for coordinates. If not provided, uses default example file from package.")
    parser.add_argument("--axons_path", type=str, required=False, default=None, help="Path to the axons trajectories file. Coordinates should be in mm. If not provided, uses default example file from package.")
    parser.add_argument("--results_dir", type=str, required=False, default=None, help="Directory to save the results. If not provided, saves to ./results in current working directory")
    parser.add_argument("--results_dir_suffix", type=str, required=False, default=None, help="Suffix to append to the results directory name")
    parser.add_argument("--record_v", action="store_true", default=None, help="Flag to record membrane potentials")
    parser.add_argument("--recorded_v_nodes", nargs='+', type=int, required=False, default=None, help="List of node indices to record membrane potentials for")
    parser.add_argument("--recorded_ap_times_nodes", nargs='+', type=int, required=False, default=None, help="List of node indices to record action potential times for")
    parser.add_argument("--debug", action="store_true", default=None, help="Enable debug mode (simulate only 5 axons) with verbose output")
    parser.add_argument("--exclude_end_node", action="store_true", default=None, help="Treat end node activation as non-activation during titrations")
    parser.add_argument("--initial_stim_factor", type=float, required=False, default=None, help="Initial stimulation factor for titration")
    parser.add_argument("--titration_conv_perc", type=float, required=False, default=None, help="Convergence percentage for titration")
    parser.add_argument("--initial_reduction_factor", type=float, required=False, default=None, help="Initial reduction factor for titration")
    parser.add_argument("--initial_increment_factor", type=float, required=False, default=None, help="Initial increment factor for titration")
    parser.add_argument("--number_of_req_spikes", type=int, required=False, default=None, help="Number of required spikes during titration")
    parser.add_argument("--max_stim_factor", type=float, required=False, default=None, help="Maximum stimulation factor for titration")
    parser.add_argument("--min_stim_factor", type=float, required=False, default=None, help="Minimum stimulation factor for titration")
    parser.add_argument("--sim_dur", type=float, required=False, default=None, help="Total simulation duration")
    parser.add_argument("--time_step", type=float, required=False, default=None, help="Simulation time step")
    parser.add_argument("--pulse_silence_period", type=float, required=False, default=None, help="Silence period before the onset of the pulse")
    parser.add_argument("--pulse_path", type=str, required=False, default=None, help="Path to the pulse waveform txt file (Example: './examples/example_pulse.txt')")
    parser.add_argument("--pulse_shape", type=str, required=False, default=None, help="Shape of the pulse waveform ('monophasic' or 'biphasic')")
    parser.add_argument("--pulse_amplitude", type=float, required=False, default=None, help="Amplitude of the pulse waveform (1.0 or -1.0)")
    parser.add_argument("--pulse_width", type=float, required=False, default=None, help="Width of the pulse waveform in ms")
    parser.add_argument("--cont_stim_waveform", action="store_true", default=None, help="Flag to use continuous stimulation waveform")
    parser.add_argument("--cont_stim_freq", type=float, required=False, default=None, help="Frequency of the continuous stimulation waveform in Hz")
    parser.add_argument("--cont_stim_carrier_freq", type=float, required=False, default=None, help="Carrier frequency of the continuous stimulation waveform in Hz")
    parser.add_argument("--passive_end_nodes", action="store_true", default=None, help="Replace end nodes with passive segments in the axon models (no ion channels)")
    parser.add_argument("--prepassive_nodes_as_endnodes", action="store_true", default=None, help="Treat pre-passive nodes as end nodes during titrations (has an effect only if --passive_end_nodes and --exclude_end_node are set)")
    parser.add_argument("--no_motoneuron", action="store_true", default=None, help="Whether to disable motoneuron creation for motor efferents")
    parser.add_argument("--model_variant", type=str, required=False, default=None, help="Axon model variant to use ('Alashqar', 'Gaines', or 'MRG')")
    parser.add_argument("--paramfit_method", type=str, required=False, default=None, help="Parameter fitting method to use ('discrete' or 'continuous')")
    parser.add_argument("--min_fiber_length", type=float, required=False, default=None, help="Minimum fiber length (in mm) to include in the simulations")
    parser.add_argument("--axons_kws_any", nargs='+', type=str, required=False, default=None, help="List of keywords to filter axons by (any match)")
    parser.add_argument("--afferent_kws_any", nargs='+', type=str, required=False, default=None, help="List of keywords to filter afferent axons by (any match)")
    parser.add_argument("--efferent_kws_any", nargs='+', type=str, required=False, default=None, help="List of keywords to filter efferent axons by (any match)")
    parser.add_argument("--axons_to_plot", nargs='+', default=None, help="List of axon names to plot membrane potentials for")
    
    args_parsed = parser.parse_args()

    # Create config object and load defaults
    config = TitrationsConfig()

    # Load from YAML config file if provided
    if args_parsed.config_path:
        try:
            config.load_from_yaml(args_parsed.config_path)
            if rank == 0:
                print(f"Config file {args_parsed.config_path} loaded!")
        except Exception as e:
            if rank == 0:
                print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Update config from command line arguments (check for overlap with YAML)
    try:
        config.update_from_arguments(vars(args_parsed), check_overlap=True)
    except ValueError as e:
        if rank == 0:
            print(f"Error: {e}")
        sys.exit(1)
    
    # Validate configuration
    try:
        config.validate(raise_errors=True)
    except ValueError as e:
        if rank == 0:
            print(f"Configuration validation error: {e}")
        sys.exit(1)
    
    if rank == 0:
        print("Configuration loaded successfully!")
        print("Parameters:")
        for key, value in config.to_dict().items():
            print(f"\t{key}: {value}")
    
    return config
    

def parse_discrete_simulations_arguments():
    """Parse arguments for discrete simulations using DiscreteSimulationsConfig.
    
    Returns:
        DiscreteSimulationsConfig: Configuration object with all parameters
    """
    comm = MPI.COMM_WORLD if MPI else DummyComm()
    rank = comm.Get_rank()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, required=False, default=None, help="Path to YAML configuration file")
    parser.add_argument("--field_path", type=str, required=False, default=None, help="Path to the extracellular field data file. Units should be in V for potential values and m for coordinates. If not provided, uses default example file from package.")
    parser.add_argument("--axons_path", type=str, required=False, default=None, help="Path to the axons trajectories file. Coordinates should be in mm. If not provided, uses default example file from package.")
    parser.add_argument("--results_dir", type=str, required=False, default=None, help="Directory to save the results. If not provided, saves to ./results in current working directory")
    parser.add_argument("--results_dir_suffix", type=str, required=False, default=None, help="Suffix to append to the results directory name")
    parser.add_argument("--record_v", action="store_true", default=None, help="Flag to record membrane potentials")
    parser.add_argument("--recorded_v_nodes", nargs='+', type=int, required=False, default=None, help="List of node indices to record membrane potentials for")
    parser.add_argument("--recorded_ap_times_nodes", nargs='+', type=int, required=False, default=None, help="List of node indices to record action potential times for")
    parser.add_argument("--debug", action="store_true", default=None, help="Enable debug mode (simulate only 5 afferents and 5 efferents) with verbose output")
    parser.add_argument("--stim_amplitudes", nargs='+', type=float, required=False, default=None, help="List of stimulation amplitudes (factors) to simulate")
    parser.add_argument("--initial_stim_factor", type=float, required=False, default=None, help="Initial stimulation factor for the discrete simulations")
    parser.add_argument("--stim_factor_step", type=float, required=False, default=None, help="Stimulation factor step size for the discrete simulations")
    parser.add_argument("--max_stim_factor", type=float, required=False, default=None, help="Maximum stimulation factor for the discrete simulations")
    parser.add_argument("--sim_dur", type=float, required=False, default=None, help="Total simulation duration (default to fall back durations based on axon types)")
    parser.add_argument("--sim_dur_afferent", type=float, required=False, default=None, help="Total simulation duration for afferent axons")
    parser.add_argument("--sim_dur_efferent", type=float, required=False, default=None, help="Total simulation duration for efferent axons")
    parser.add_argument("--sim_dur_other", type=float, required=False, default=None, help="Total simulation duration for axons other than afferent/efferent")
    parser.add_argument("--time_step", type=float, required=False, default=None, help="Simulation time step")
    parser.add_argument("--pulse_path", type=str, required=False, default=None, help="Path to the pulse waveform txt file (Example: './examples/example_pulse.txt')")
    parser.add_argument("--pulse_silence_period", type=float, required=False, default=None, help="Silence period before the onset of the pulse")
    parser.add_argument("--pulse_shape", type=str, required=False, default=None, help="Shape of the pulse waveform ('monophasic' or 'biphasic')")
    parser.add_argument("--pulse_amplitude", type=float, required=False, default=None, help="Amplitude of the pulse waveform (1.0 or -1.0)")
    parser.add_argument("--pulse_width", type=float, required=False, default=None, help="Width of the pulse waveform in ms")
    parser.add_argument("--cont_stim_waveform", action="store_true", default=None, help="Flag to use continuous stimulation waveform")
    parser.add_argument("--cont_stim_freq", type=float, required=False, default=None, help="Frequency of the continuous stimulation waveform in Hz")
    parser.add_argument("--cont_stim_carrier_freq", type=float, required=False, default=None, help="Carrier frequency of the continuous stimulation waveform in Hz")
    parser.add_argument("--passive_end_nodes", action="store_true", default=None, help="Replace end nodes with passive segments in the axon models (no ion channels)")
    parser.add_argument("--prepassive_nodes_as_endnodes", action="store_true", default=None, help="Treat pre-passive nodes as end nodes during simulations (has an effect only if --passive_end_nodes is set)")
    parser.add_argument("--model_variant", type=str, required=False, default=None, help="Axon model variant to use ('Alashqar', 'Gaines', or 'MRG')")
    parser.add_argument("--disable_extracellular_efferent", action="store_true", default=None, help="Disable extracellular stimulation for efferent axons")
    parser.add_argument("--afferents_only", action="store_true", default=None, help="Simulate only afferent axons")
    parser.add_argument("--efferents_only", action="store_true", default=None, help="Simulate only efferent axons")
    parser.add_argument("--other_axons_only", action="store_true", default=None, help="Simulate only axons other than afferent/efferent")
    parser.add_argument("--skip_other_axons", action="store_true", default=None, help="Skip simulation of axons other than afferent/efferent")
    parser.add_argument("--no_motoneuron", action="store_true", default=None, help="Whether to disable motoneuron creation for motor efferents")
    parser.add_argument("--paramfit_method", type=str, required=False, default=None, help="Parameter fitting method to use ('discrete' or 'continuous')")
    parser.add_argument("--min_fiber_length", type=float, required=False, default=None, help="Minimum fiber length (in mm) to include in the simulations")
    parser.add_argument("--axons_kws_any", nargs='+', type=str, required=False, default=None, help="List of keywords to filter axons by (any match)")
    parser.add_argument("--root_kws_any", nargs='+', type=str, required=False, default=None, help="List of keywords to filter root/anastomosis axons by (any match)")
    parser.add_argument("--afferents_results_path", type=str, required=False, default=None, help="Path to presimulated afferent axons results file to reuse for synaptic transmission simulations on efferents")
    parser.add_argument("--syn_afferent_kws_all", nargs='+', type=str, required=False, default=None, help="List of keywords that all must match to identify afferent axons for synaptic transmission")
    parser.add_argument("--syn_efferent_kws_all", nargs='+', type=str, required=False, default=None, help="List of keywords that all must match to identify efferent axons for synaptic transmission")
    parser.add_argument("--afferent_kws_any", nargs='+', type=str, required=False, default=None, help="List of keywords to filter afferent axons by (any match)")
    parser.add_argument("--efferent_kws_any", nargs='+', type=str, required=False, default=None, help="List of keywords to filter efferent axons by (any match)")
    parser.add_argument("--save_only_processed_responses", action="store_true", default=None, help="Flag to save only processed responses (for afferents, only AP times at the central end node; for efferents only AP initiations and response classes (synaptically induced or direct activation))")
    parser.add_argument("--enable_synaptic_transmission", action="store_true", default=None, help="Flag to enable synaptic transmission from afferent to efferent axons")
    parser.add_argument("--syn_weight", type=float, required=False, default=None, help="Synaptic weight to use for synaptic transmission")
    parser.add_argument("--proj_freq", type=float, required=False, default=None, help="Afferent-to-efferent projection frequency in percent for synaptic transmission (percentage of efferents receiving synaptic input from afferents)")
    parser.add_argument("--axons_to_plot", nargs='+', default=None, help="List of axon names to plot membrane potentials for")
    
    args_parsed = parser.parse_args()
    
    # Create config object and load defaults
    config = DiscreteSimulationsConfig()

    # Load from YAML config file if provided
    if args_parsed.config_path:
        try:
            config.load_from_yaml(args_parsed.config_path)
            if rank == 0:
                print(f"Config file {args_parsed.config_path} loaded!")
        except Exception as e:
            if rank == 0:
                print(f"Error loading config file: {e}")
            sys.exit(1)

    # Update config from command line arguments (check for overlap with YAML)
    try:
        config.update_from_arguments(vars(args_parsed), check_overlap=True)
    except ValueError as e:
        if rank == 0:
            print(f"Error: {e}")
        sys.exit(1)
    
    # Validate configuration
    try:
        config.validate(raise_errors=True)
    except ValueError as e:
        if rank == 0:
            print(f"Configuration validation error: {e}")
        sys.exit(1)
    
    if rank == 0:
        print("Configuration loaded successfully!")
        print("Parameters:")
        for key, value in config.to_dict().items():
            print(f"\t{key}: {value}")
    
    return config