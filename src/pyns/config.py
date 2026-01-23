"""
Configuration classes for PyNS simulations.

Provides default configuration values for titrations and discrete simulations
without relying on YAML files.
"""

import yaml
import os
from typing import Dict, Any, Optional


def get_package_data_dir() -> str:
    """Get the absolute path to the package data directory.
    
    Returns:
        str: Absolute path to the pyns package directory
    """
    return os.path.dirname(os.path.abspath(__file__))


class BaseConfig:
    """Base configuration class with common functionality for PyNS simulations."""
    
    def __init__(self):
        """Initialize base configuration with common parameters."""
        # Get package data directory for absolute paths
        pkg_dir = get_package_data_dir()
        
        # File paths (using absolute paths from package directory)
        self.field_path = os.path.join(pkg_dir, "test_dataset", "lumbar-tSCS_cathode_T11-T12_anode_navel-sides_units_V_m_cropped.h5")
        self.axons_path = os.path.join(pkg_dir, "test_dataset", "RightSoleusAxons_diams_from_Schalow1992_cropped.npy")
        self.init_hoc_path = os.path.join(pkg_dir, "init_diff_v.hoc")
        self.results_dir = os.path.join(os.getcwd(), "results")
        self.results_dir_suffix = ""
        
        # Recording options
        self.record_v = False
        self.recorded_v_nodes = None
        self.recorded_ap_times_nodes = None
        
        # Debug and simulation flags
        self.debug = False
        self.passive_end_nodes = False
        self.prepassive_nodes_as_endnodes = False
        self.no_motoneuron = False
        
        # Stimulation parameters
        self.initial_stim_factor = 30.0
        self.max_stim_factor = 300.0
        
        # Simulation timing
        self.sim_dur = 5.0
        self.time_step = 0.005
        
        # Waveform parameters
        self.pulse_path = None
        self.pulse_silence_period = 1.0
        self.pulse_shape = "biphasic"  # Options: "monophasic", "biphasic"
        self.pulse_amplitude = 1.0
        self.pulse_width = 2.0
        # Continuous stimulation parameters
        self.cont_stim_waveform = False
        self.cont_stim_freq = 40.0
        self.cont_stim_carrier_freq = 10000.0
        
        # Model parameters
        self.model_variant = "Alashqar"  # Options: "Alashqar", "Gaines", "MRG"
        self.paramfit_method = "continuous"
        
        # Axon filtering keywords (any match)
        self.axons_kws_any = None
        self.min_fiber_length = 40
        self.afferent_kws_any = ["sensory", "_Aalpha", "_DR", "_DL", "dorsal", "afferent"]
        self.efferent_kws_any = ["motor", "_alpha", "_VR", "_VL", "ventral", "efferent"]
        
        # Plotting (axon names to plot membrane potentials for)
        self.axons_to_plot = None
        
        # Track which attributes were loaded from YAML for overlap detection
        self._yaml_loaded_keys = set()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create a config object from a dictionary (typically from YAML).
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            Config object with values from dictionary
        """
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
                config._yaml_loaded_keys.add(key)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration (excludes _yaml_loaded_keys)
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result
    
    def load_from_yaml(self, yaml_path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML contains unknown configuration parameters
            yaml.YAMLError: If YAML is invalid
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if config_dict is None:
            config_dict = {}
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self._yaml_loaded_keys.add(key)
            else:
                raise ValueError(f"Unknown configuration parameter in YAML: {key}")
    
    def update_from_arguments(self, args_dict: Dict[str, Any], check_overlap: bool = True) -> None:
        """Update configuration from parsed command-line arguments.
        
        Args:
            args_dict: Dictionary of arguments (typically from argparse.Namespace.__dict__)
            check_overlap: If True, warn about parameters set in both YAML and CLI
            
        Raises:
            ValueError: If same parameter is set in both YAML and CLI when check_overlap=True
        """
        for key, value in args_dict.items():
            if value is None or key.startswith('_'):
                continue
            
            if hasattr(self, key):
                if check_overlap and key in self._yaml_loaded_keys:
                    raise ValueError(
                        f"Parameter '{key}' was set in both YAML config and command-line arguments. "
                        f"Use one source or the other, not both."
                    )
                setattr(self, key, value)
            else:
                # Skip unknown arguments (e.g., config_path itself)
                pass
    
    def _validate_common_parameters(self, errors: Dict[str, str], raise_errors: bool) -> None:
        """Validate parameters common to all simulation types.
        
        Args:
            errors: Dictionary to collect errors (modified in place)
            raise_errors: If True, raise on first error; otherwise collect errors
        """
        # min_fiber_length must be non-negative
        if self.min_fiber_length < 0:
            msg = "min_fiber_length must be non-negative!"
            if raise_errors:
                raise ValueError(msg)
            errors['min_fiber_length'] = msg
        
        # time_step must be positive
        if self.time_step <= 0:
            msg = "time_step must be positive!"
            if raise_errors:
                raise ValueError(msg)
            errors['time_step'] = msg
        
        # sim_dur must be positive
        if self.sim_dur <= 0:
            msg = "sim_dur must be positive!"
            if raise_errors:
                raise ValueError(msg)
            errors['sim_dur'] = msg
        
        # pulse_shape must be 'biphasic' or 'monophasic'
        if self.pulse_shape not in ["biphasic", "monophasic"]:
            msg = f"Invalid pulse_shape: {self.pulse_shape}. Accepted: 'biphasic', 'monophasic'"
            if raise_errors:
                raise ValueError(msg)
            errors['pulse_shape'] = msg
        
        # pulse_width must be positive
        if self.pulse_width <= 0:
            msg = "pulse_width must be positive!"
            if raise_errors:
                raise ValueError(msg)
            errors['pulse_width'] = msg
        
        # model_variant must be valid
        if self.model_variant.lower() not in ["alashqar", "gaines", "mrg"]:
            msg = f"Invalid model_variant: {self.model_variant}. Accepted: 'Alashqar', 'Gaines', 'MRG'"
            if raise_errors:
                raise ValueError(msg)
            errors['model_variant'] = msg
        
        # afferent_kws_any and efferent_kws_any should not have common keywords
        if self.afferent_kws_any and self.efferent_kws_any:
            common_kws = set(self.afferent_kws_any).intersection(set(self.efferent_kws_any))
            if common_kws:
                msg = f"afferent_kws_any and efferent_kws_any have common keywords: {common_kws}"
                if raise_errors:
                    raise ValueError(msg)
                errors['keyword_overlap'] = msg
    
    def _validate_specific_parameters(self, errors: Dict[str, str], raise_errors: bool) -> None:
        """Validate parameters specific to simulation type.
        
        This method is meant to be overridden by subclasses.
        
        Args:
            errors: Dictionary to collect errors
            raise_errors: If True, raise on first error; otherwise collect errors
        """
        pass
    
    def validate(self, raise_errors: bool = True) -> Dict[str, str]:
        """Validate all configuration parameters.
        
        Uses template method pattern: validates common parameters first,
        then delegates to subclass for type-specific validation.
        
        Args:
            raise_errors: If True, raise on first error; otherwise collect all errors
            
        Returns:
            Dictionary of collected errors (if raise_errors=False)
            
        Raises:
            ValueError: If raise_errors=True and validation fails
        """
        errors = {}
        self._validate_common_parameters(errors, raise_errors)
        self._validate_specific_parameters(errors, raise_errors)
        return errors


class TitrationsConfig(BaseConfig):
    """Configuration for titrations simulations."""
    
    def __init__(self):
        """Initialize titrations configuration."""
        super().__init__()
        
        # Debug and simulation flags specific to titrations
        self.exclude_end_node = False
        
        # Titration parameters
        self.titration_conv_perc = 1.0
        self.initial_reduction_factor = 0.5
        self.initial_increment_factor = 1.5
        self.number_of_req_spikes = 1
        self.min_stim_factor = 0.01
        
        # Axon filtering keywords
        self.axons_kws_any = ["rlet", "anstm"]
    
    def _validate_specific_parameters(self, errors: Dict[str, str], raise_errors: bool) -> None:
        """Validate parameters specific to titrations.
        
        Args:
            errors: Dictionary to collect errors (modified in place)
            raise_errors: If True, raise on first error; otherwise collect errors
        """
        # titration_conv_perc must be between 0 and 100
        if self.titration_conv_perc <= 0 or self.titration_conv_perc > 100.0:
            msg = "titration_conv_perc must be between 0 and 100!"
            if raise_errors:
                raise ValueError(msg)
            errors['titration_conv_perc'] = msg
        
        # number_of_req_spikes must be positive
        if self.number_of_req_spikes <= 0:
            msg = "number_of_req_spikes must be positive!"
            if raise_errors:
                raise ValueError(msg)
            errors['number_of_req_spikes'] = msg
        
        # max_stim_factor must be greater than min_stim_factor
        if self.max_stim_factor <= self.min_stim_factor:
            msg = "max_stim_factor must be greater than min_stim_factor!"
            if raise_errors:
                raise ValueError(msg)
            errors['stim_factors'] = msg
        
        # initial_reduction_factor must be between 0 and 1
        if self.initial_reduction_factor <= 0 or self.initial_reduction_factor >= 1:
            msg = "initial_reduction_factor must be between 0 and 1!"
            if raise_errors:
                raise ValueError(msg)
            errors['initial_reduction_factor'] = msg
        
        # initial_increment_factor must be greater than 1
        if self.initial_increment_factor <= 1:
            msg = "initial_increment_factor must be greater than 1!"
            if raise_errors:
                raise ValueError(msg)
            errors['initial_increment_factor'] = msg


class DiscreteSimulationsConfig(BaseConfig):
    """Configuration for discrete simulations."""
    
    def __init__(self):
        """Initialize discrete simulations configuration."""
        super().__init__()
        
        # Debug and simulation flags specific to discrete simulations
        self.afferents_only = False
        self.efferents_only = False
        self.other_axons_only = False
        self.skip_other_axons = False
        self.disable_extracellular_efferent = False
        self.save_only_processed_responses = False
        
        # Stimulation parameters specific to discrete simulations
        self.stim_amplitudes = None
        self.initial_stim_factor = 2.0
        self.max_stim_factor = 20.0
        self.stim_factor_step = 2.0
        
        # Simulation timing specific to discrete simulations
        self.sim_dur_afferent = 5.0
        self.sim_dur_efferent = 10.0
        self.sim_dur_other = 5.0
        
        # Axon filtering keywords
        self.axons_kws_any = None
        self.root_kws_any = ["rootlet", "root", "segment", "anastomosis", "anastomoses", "rlet", "anstm"]
        
        # Synaptic transmission parameters
        self.afferents_results_path = None
        self.syn_afferent_kws_all = None  # Keywords that all must match for synaptic afferents
        self.syn_efferent_kws_all = None  # Keywords that all must match for synaptic efferents
        self.enable_synaptic_transmission = False
        self.syn_weight = 0.0056994
        self.proj_freq = 50.0
    
    def _validate_specific_parameters(self, errors: Dict[str, str], raise_errors: bool) -> None:
        """Validate parameters specific to discrete simulations.
        
        Args:
            errors: Dictionary to collect errors (modified in place)
            raise_errors: If True, raise on first error; otherwise collect errors
        """
        # afferents_only, efferents_only, other_axons_only are mutually exclusive
        exclusive_flags = [self.afferents_only, self.efferents_only, self.other_axons_only]
        if sum(exclusive_flags) > 1:
            msg = "afferents_only, efferents_only, and other_axons_only flags are mutually exclusive!"
            if raise_errors:
                raise ValueError(msg)
            errors['exclusive_flags'] = msg
        
        # If provided, syn_afferent_kws_all and syn_efferent_kws_all should not have common keywords
        if self.syn_afferent_kws_all is not None and self.syn_efferent_kws_all is not None:
            common_kws_all = set(self.syn_afferent_kws_all).intersection(set(self.syn_efferent_kws_all))
            if common_kws_all:
                msg = f"syn_afferent_kws_all and syn_efferent_kws_all have common keywords: {common_kws_all}"
                if raise_errors:
                    raise ValueError(msg)
                errors['syn_keyword_overlap'] = msg
        
        # If enable_synaptic_transmission is True, no_motoneuron cannot be True
        if self.enable_synaptic_transmission and self.no_motoneuron:
            msg = "enable_synaptic_transmission cannot be True when no_motoneuron is True!"
            if raise_errors:
                raise ValueError(msg)
            errors['syn_no_motoneuron_conflict'] = msg
