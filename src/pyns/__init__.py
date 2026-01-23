"""
PyNS package initializer.

Exports commonly used submodules for convenience imports:

	import pyns
	pyns.sim_utils.run_simulation(...)
"""

from . import (
	arguments_parsers,
	axon_models,
	cli,
	compute_properties,
	config,
	morphological_params,
	postprocessing_scripts,
	postprocessing_utils,
	sim_analysis_utils,
	sim_utils,
	titration_utils,
	utils,
)

__all__ = [
	"arguments_parsers",
	"axon_models",
	"cli",
	"compute_properties",
	"config",
	"morphological_params",
	"postprocessing_scripts",
	"postprocessing_utils",
	"sim_analysis_utils",
	"sim_utils",
	"titration_utils",
	"utils",
]

# add also package directory as a variable of the package
import os
pyns_root_dir = os.path.dirname(os.path.abspath(__file__)).rsplit('/', 1)[0]
__pyns_root_dir__ = pyns_root_dir

