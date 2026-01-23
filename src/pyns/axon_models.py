'''
###########################################
# File: pyns/axon_models.py
# Project: pyns
# Author: Abdallah Alashqar (abdallah.j.alashqar@fau.de)
# -----
# PI: Andreas Rowald, PhD (andreas.rowald@fau.de)
# Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)
https://www.pdh.med.fau.de/
#############################################
'''

import numpy as np
import math
from neuron import h
import time
import random as rnd

err = h.load_file("noload.hoc")
from .utils import interpolate_3d, get_arcline_length
import matplotlib.pyplot as plt

from .morphological_params import MRG_discrete_params, small_fiber_diam_fits, myelin_thickness_fits

class Axon(object):
    def __init__(
            self,
            fiber_diameter=10.0,
            axon_name=None,
            axon_coords=None,
            ):
        """Parent class for different axon models"""
        if any([axon_name is None, axon_coords is None, fiber_diameter is None]):
            raise ValueError(
                "axon_name, axon_coords and fiber_diameter must be provided to initialize an axon object if no discretized dict is provided!"
            )
        self.fiberD = fiber_diameter
        self.name = axon_name
        self.axon_coords = axon_coords
        self.sections_ext_v = []

    def _discretize(self):
        """Discretize the axon into segments based on fiber diameter (different for myelinated and unmyelinated)"""
        pass

    def to_dict(self):
        """Convert the object to a dictionary"""
        pass

    def initialize_neuron(self):
        """Initialize the axon model in NEURON"""
        pass

    def setup_recorders(self):
        """Setup recorders for the axon model in NEURON"""
        pass

    def interpolate_v_on_sections(self, field_dict=None):
        """Interpolate the extracellular voltage on the axon sections"""
        xyz_all_segs = np.copy(self.segments_midpoints)
        if field_dict is not None:
            self.sections_ext_v = interpolate_3d(field_dict, xyz_all_segs)
        else:
            print("\t\t!!! WARNING: No field_dict is provided. Assigning zeros to v_ext of all sections!!!")
            self.sections_ext_v = np.zeros(len(xyz_all_segs))

    def assign_v_ext(self, field_dict=None):
        """Assign extracellular voltage to the axon sections"""
        if len(self.sections_ext_v) == 0:
            self.interpolate_v_on_sections(field_dict)

        # assuming voltage is in V, convert V to mV and multiply by 1e-6 to get value in MegaOhm considering a current of 1mA (see xtra.mod)
        for seg_i, seg_ext_v in enumerate(self.sections_ext_v):
            self.sections_list[seg_i].rx_xtra = seg_ext_v * 1e3 * 1e-6

    def run_simulation(self):
        """Run the simulation for the axon model in NEURON"""
        pass

    def plot_membrane_potential(
            self,
            save_path=None,
            stim_pulse=None,
            stacked=True,
            xlims=None,
            flip_yaxis=False,
            plot_only=None,
            ):
        """Plot the membrane potential for the axon model"""

        # check whether axon has nodes or sections (myelinated or unmyelinated)
        if hasattr(self, "axonnodes"):
            sec_name = "Node"
            n_secs = self.axonnodes
        elif hasattr(self, "n_secs"):
            sec_name = "Sec"
            n_secs = self.n_secs
        else:
            raise ValueError("Axon model does not have nodes or sections defined!")
        
        # check plot_only validity
        if plot_only is not None:
            # must be a list or a numpy array
            if not isinstance(plot_only, list) and not isinstance(plot_only, np.ndarray):
                raise ValueError("plot_only must be a list or a numpy array of section indices!")

            if any([po < 0 or po >= n_secs for po in plot_only]):
                raise ValueError("plot_only contains invalid section indices!")

        spike_prefix = f"spk_{sec_name.lower()}_"
        v_key_prefix = f"v_{sec_name.lower()}_"
        
        # find the node with the earliest spike
        spike_times = [
            self.recorders[f"{spike_prefix}{node_i}"].time
            for node_i in range(0, n_secs)
            if self.recorders[f"{spike_prefix}{node_i}"].n > 0
        ]
        spiking_secs = [
            node_i
            for node_i in range(0, n_secs)
            if self.recorders[f"{spike_prefix}{node_i}"].n > 0
        ]
        spike_sec_earliest = -1
        if len(spike_times) > 0:
            spike_times = np.array(spike_times)
            spiking_secs = np.array(spiking_secs)
            spike_sec_earliest = spiking_secs[np.argmin(spike_times)]
        tv = np.array(self.recorders["t"])
        if flip_yaxis:
            plot_sign = -1
        else:
            plot_sign = 1
        
        spike_secs_to_plot = range(0, n_secs)
        if plot_only is not None:
            spike_secs_to_plot = plot_only
        if stacked:
            plt.figure(figsize=[20, int(n_secs * 0.2)])
            for si in spike_secs_to_plot:
                color = "k"
                if si == spike_sec_earliest:
                    color = "r"
                plt.plot(
                    tv,
                    10 * plot_sign * si + np.array(self.recorders[f"{v_key_prefix}{si}"]),
                    color=color,
                    linewidth=2,
                )
        else:
            plt.figure(figsize=[20, 10])
            for si in spike_secs_to_plot:
                color = "k"
                if si == spike_sec_earliest:
                    color = "r"
                plt.plot(tv, np.array(self.recorders[f"{v_key_prefix}{si}"]), color=color, linewidth=2)
        # dashed line at start and end of stim. pulse
        if not stim_pulse is None:
            first_non_zero = np.argwhere(stim_pulse != 0)[0][0]
            last_non_zero = np.argwhere(stim_pulse != 0)[-1][0]
            plt.axvline(x=tv[first_non_zero], color="k", linestyle="--")
            plt.axvline(x=tv[last_non_zero], color="k", linestyle="--")
        if not xlims is None:
            plt.xlim(xlims)
        
        # set y ticks
        if stacked:
            node_inds = range(0, n_secs)
            if plot_only is not None:
                node_inds = plot_only
            yticks = np.array([10 * ni * plot_sign for ni in node_inds]) + self.v_init
            ytick_labels = [f"{sec_name} {ni}" for ni in node_inds]
            plt.yticks(yticks, ytick_labels, fontsize=12)
        plt.xlabel("Time (ms)", fontsize=16)
        plt.ylabel("Membrane Potential (mV)", fontsize=16)
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
    
    def delete_sections(self):
        """Delete all sections related to the current axon from NEURON memory"""
        if hasattr(self, "sections_list"):
            for sec in self.sections_list:
                try:
                    h.delete_section(sec=sec)
                except:
                    for seg in sec:
                        seg = None
                try:
                    sec = None
                except:
                    pass
    
    def delete_recorders(self):
        """Delete all recorders related to the current axon from NEURON memory"""
        if hasattr(self, "recorders"):
            for k in self.recorders.keys():
                self.recorders[k] = None
    

class UnmyelinatedAxon(Axon):
    def __init__(
            self,
            fiber_diameter=0.8,
            seg_length=5000/100,
            axon_name=None,
            axon_coords=None,
            discretized_dict=None,
            model="sundt",
            ):
        """Generate an unmyelinated axon model based on the discretized_dict or the supplied parameters"""
        # if discretized_dict is provided, use it to initialize the model
        if discretized_dict is not None:
            self.name = discretized_dict["name"]
            self.axon_coords = discretized_dict["axon_coords"]
            self.fiberD = discretized_dict["fiberD"]
            self.n_seg_per_sec = discretized_dict["n_seg_per_sec"]
            self.sec_length = discretized_dict["sec_length"]
            self.seg_length = discretized_dict["seg_length"]
            self.n_secs = discretized_dict["n_secs"]
            self.segments_midpoints = discretized_dict["segments_midpoints"]
            self.total_length = discretized_dict["total_length"]
            self.model = discretized_dict["model"]
            if "sections_ext_v" in discretized_dict.keys():
                self.sections_ext_v = discretized_dict["sections_ext_v"]
        else:
            # first call parent constructor
            super(UnmyelinatedAxon, self).__init__(fiber_diameter=fiber_diameter, axon_name=axon_name, axon_coords=axon_coords)
            self.seg_length = seg_length
            self.model = model
            self._discretize()

    def _discretize(self, sec_length=50):
        self.total_length, points_lengths_cum = get_arcline_length(self.axon_coords, return_length_per_point=True)
        self.n_seg_per_sec = int(np.floor(sec_length/self.seg_length))
        self.sec_length = sec_length
        n_seg = int(np.floor(self.total_length/self.seg_length))
        self.n_secs = int(np.floor(n_seg/self.n_seg_per_sec))

        self.segments_midpoints = np.zeros((int(self.n_secs), 3))
        # interpolate segments coords
        segments_lengths = np.ones(int(self.n_secs)) * self.seg_length
        segments_lengths_cum = np.cumsum(segments_lengths)
        segment_start = self.axon_coords[0]
        self.first_segment_start = segment_start
        for segment_idx in range(
                self.n_secs
            ):
            # find the coordinate before the segment_length (last one less than segment_length)
            coord_prev_idx = np.argwhere(
                points_lengths_cum < segments_lengths_cum[segment_idx]
            )
            coord_prev_idx = coord_prev_idx[-1][0]

            # find the coordinate after the segment_length (first one greater than segment_length)
            coord_next_idx = np.argwhere(
                points_lengths_cum >= segments_lengths_cum[segment_idx]
            )
            if len(coord_next_idx) == 0:
                direction = self.axon_coords[-1] - self.axon_coords[-2]
                direction /= np.linalg.norm(direction)
                # move by segment_length in direction
                segment_end = segment_start + direction * sec_length
                self.segments_midpoints[segment_idx] = (
                    segment_start + (segment_end - segment_start) / 2
                )
                segment_start = segment_end
            else:
                coord_next_idx = coord_next_idx[0][0]
                direction = (
                    self.axon_coords[coord_next_idx] - self.axon_coords[coord_prev_idx]
                )
                n_points = int(
                    np.ceil(np.linalg.norm(direction) / (sec_length * 0.01))
                )
                points_along_axon = np.linspace(
                    self.axon_coords[coord_prev_idx],
                    self.axon_coords[coord_next_idx],
                    n_points,
                )
                points_lengths = np.linalg.norm(
                    np.diff(points_along_axon, axis=0), axis=1
                )
                points_cum_lengths = np.cumsum(points_lengths)
                points_cum_lengths = np.append(0, points_cum_lengths)
                points_cum_lengths += points_lengths_cum[coord_prev_idx]
                # get the closest one to segment_length
                closest_point_idx = np.argmin(
                    np.abs(points_cum_lengths - segments_lengths_cum[segment_idx])
                )
                segment_end = points_along_axon[closest_point_idx]
                self.segments_midpoints[segment_idx] = (
                    segment_start + (segment_end - segment_start) / 2
                )
                segment_start = segment_end
        self.last_segment_end = segment_end
    
    def to_dict(self):
        """Convert the object to a dictionary"""
        obj_dict = {
            "name": self.name,
            "axon_coords": self.axon_coords,
            "fiberD": self.fiberD,
            "n_seg_per_sec": self.n_seg_per_sec,
            "sec_length": self.sec_length,
            "seg_length": self.seg_length,
            "n_secs": self.n_secs,
            "segments_midpoints": self.segments_midpoints,
            "total_length": self.total_length,
            "sections_ext_v": self.sections_ext_v,
            "model": self.model,
        }
        return obj_dict

    def initialize_neuron(
        self,
        mod_params=None,
        decreasing_diam_from_sec=None,
        temp_c=37,
        ):
        if self.model == "sundt":
            self.v_init = -60
        elif self.model == "schild":
            self.v_init = -48
        elif self.model == "tigerholm":
            self.v_init = -55
        # delete previous sections
        self.delete_sections()
        kcnq = 0.0004
        nav = 0.04
        kv = 0.04
        self.temp_c = temp_c
        self.sections_list = []
        self.branches_list = []

        # Physical constants
        R = 8314  # Gas constant
        F = 96500  # Faraday constant
        for sec_i in range(self.n_secs):
            # Set basic section properties
            sec = h.Section(name='sec[%d]' % sec_i)
            sec.nseg = self.n_seg_per_sec
            sec.diam = self.fiberD
            if decreasing_diam_from_sec is not None:
                if (self.n_secs - sec_i) <= decreasing_diam_from_sec:
                    print("Decreasing diameter for section %d" % sec_i, flush=True)
                    sec.diam = self.fiberD * ((self.n_secs - sec_i) / decreasing_diam_from_sec)
                    print("New diameter: %.2f" % sec.diam, flush=True)
            sec.L = self.sec_length

            if self.model == "sundt":
                
                sec.cm = 1

                sec.insert('nahh')
                sec.gnabar_nahh = nav
                sec.mshift_nahh = -6
                sec.hshift_nahh = 6
                
                sec.insert('borgkdr')
                sec.gkdrbar_borgkdr = kv
                sec.ek = -90
                
                sec.insert('pas')
                sec.g_pas = 1/10000
                sec.v = self.v_init

                sec.Ra = 100
                sec.e_pas = sec.v + (sec.ina - sec.ik)/sec.g_pas

            elif self.model == "schild":
                
                # Insert all mechanisms from Schild 1994
                sec.insert('leakSchild')
                sec.insert('kd')
                sec.insert('ka')
                sec.insert('can')
                sec.insert('cat')
                sec.insert('kds')
                sec.insert('kca')
                sec.insert('caextscale')
                sec.insert('caintscale')
                sec.insert('CaPump')
                sec.insert('NaCaPump')
                sec.insert('NaKpumpSchild')
                
                # Insert sodium channels (97 version)
                sec.insert('naf97mean')
                sec.insert('nas97mean')
                
                # Set ionic concentrations
                # Calcium concentrations

                h.cao0_ca_ion = 2.0        # [mM] Initial Cao Concentration
                h.cai0_ca_ion = 0.000117   # [mM] Initial Cai Concentrations
                # sec.cao0_ca_ion = 2.0        # [mM] Initial Cao Concentration
                # sec.cai0_ca_ion = 0.000117   # [mM] Initial Cai Concentrations
                
                # Potassium concentrations
                ko = 5.4    # [mM] External K Concentration
                ki = 145.0  # [mM] Internal K Concentration
                
                # Set potassium ion style and calculate reversal potential
                k_style = h.ion_style("k_ion", 1, 2, 0, 0, 0)  # Allows ek to be calculated manually
                ek = ((R * (h.celsius + 273.15)) / F) * math.log(ko / ki)  # Manual calculation of ek
                sec.ek = ek
                
                # Sodium concentrations
                nao = 154.0  # [mM] External Na Concentration
                nai = 8.9    # [mM] Internal Na Concentration
                
                # Set sodium ion style and calculate reversal potential
                na_style = h.ion_style("na_ion", 1, 2, 0, 0, 0)  # Allows ena to be calculated manually
                ena = ((R * (h.celsius + 273.15)) / F) * math.log(nao / nai)  # Manual calculation of ena
                sec.ena = ena
                
                # Set conductance values from Schild 1997
                sec.gbar_naf97mean = 0.022434928    # [S/cm^2]
                sec.gbar_nas97mean = 0.022434928
                sec.gbar_kd = 0.001956534
                sec.gbar_ka = 0.001304356
                sec.gbar_kds = 0.000782614
                sec.gbar_kca = 0.000913049
                sec.gbar_can = 0.000521743
                sec.gbar_cat = 0.00018261
                sec.gbna_leakSchild = 1.8261E-05
                sec.gbca_leakSchild = 9.13049E-06
                
                # Set passive properties
                sec.Ra = 100
                sec.cm = 1.326291192
                
                # Set initial voltage
                sec.v = self.v_init

                sec.L_caintscale = sec.L
                sec.nseg_caintscale = sec.nseg
                sec.L_caextscale = sec.L
                sec.nseg_caextscale = sec.nseg

            elif self.model == "tigerholm":
                # Insert potassium channels
                sec.insert('ks')
                sec.gbar_ks = 0.0069733
                
                sec.insert('kf')
                sec.gbar_kf = 0.012756
                
                # Insert hyperpolarization-activated channel
                sec.insert('h')
                sec.gbar_h = 0.0025377
                
                # Insert TTX-sensitive sodium channels
                sec.insert('nattxs')
                sec.gbar_nattxs = 0.10664
                
                # Insert Nav1.8 sodium channels
                sec.insert('nav1p8')
                sec.gbar_nav1p8 = 0.24271
                
                # Insert Nav1.9 sodium channels
                sec.insert('nav1p9')
                sec.gbar_nav1p9 = 9.4779e-05
                
                # Insert sodium-potassium pump
                sec.insert('nakpump')
                sec.smalla_nakpump = -0.0047891
                
                # Insert delayed rectifier potassium channel (Tiger model)
                sec.insert('kdrTiger')
                sec.gbar_kdrTiger = 0.018002
                
                # Insert sodium-activated potassium channel
                sec.insert('kna')
                sec.gbar_kna = 0.00042
                
                # Insert sodium osmotic imbalance mechanism
                sec.insert('naoi')
                sec.theta_naoi = 0.029
                
                # Insert potassium osmotic imbalance mechanism
                sec.insert('koi')
                sec.theta_koi = 0.029
                
                # Insert leak and extrapump mechanisms
                sec.insert('leak')
                sec.insert('extrapump')
                
                # Set passive properties
                sec.Ra = 35.4  # Axial resistance
                sec.cm = 1     # Membrane capacitance

                sec.celsiusT_ks 	= self.temp_c
                sec.celsiusT_kf 	= self.temp_c
                sec.celsiusT_h 	= self.temp_c
                sec.celsiusT_nattxs = self.temp_c
                sec.celsiusT_nav1p8 = self.temp_c
                sec.celsiusT_nav1p9 = self.temp_c
                sec.celsiusT_nakpump = self.temp_c
                sec.celsiusT_kdrTiger 	= self.temp_c

                sec.v = self.v_init      

            if mod_params is not None:
                for param, value in mod_params.items():
                    if hasattr(sec, param):
                        # if sec_i == 0:
                            # print(f"Setting {param} to {value} in section {sec_i}", flush=True)
                        setattr(sec, param, value)
                    else:
                        print(f"Warning: Section does not have parameter {param}", flush=True)
            sec.insert('extracellular')
            sec.insert('xtra')
            sec.xg[0] = 1e10
            sec.xc[0] = 0
            h.setpointer(sec(0.5)._ref_i_membrane, 'im', sec(0.5).xtra)
            h.setpointer(sec(0.5)._ref_e_extracellular, 'ex', sec(0.5).xtra)
            # for seg in sec:
            #     h.setpointer(seg._ref_i_membrane, 'im', seg.xtra)
            #     h.setpointer(seg._ref_e_extracellular, 'ex', seg.xtra)
            if sec_i > 0:
                sec.connect(self.sections_list[sec_i-1](1), 0)
            self.sections_list.append(sec)

    def setup_recorders(self, dt=0.005, record_v=False, thresh=-20, recorded_v_secs=None, recorded_ap_times_secs=None):
        # delete old recorders first
        self.delete_recorders()
        self.recorders = {}
        tv = h.Vector()
        tv.record(h._ref_t, dt)
        self.recorders["t"] = tv
        for sec_i in range(self.n_secs):
            if record_v and (recorded_v_secs is None or sec_i in recorded_v_secs):
                self.recorders[f"v_sec_{sec_i}"] = h.Vector()
                self.recorders[f"v_sec_{sec_i}"].record(self.sections_list[sec_i](0.5)._ref_v, dt)
            self.recorders[f"spk_sec_{sec_i}"] = h.APCount(self.sections_list[sec_i](0.5))
            self.recorders[f"spk_sec_{sec_i}"].thresh = thresh
            if recorded_ap_times_secs is None or sec_i in recorded_ap_times_secs:
                self.recorders[f"spk_times_sec_{sec_i}"] = h.Vector()
                self.recorders[f"spk_sec_{sec_i}"].record(self.recorders[f"spk_times_sec_{sec_i}"])

    def run_simulation(
            self,
            stim_factor,
            stim_pulse,
            dt=0.005,
            tstop=5.0,
            verbose=False,
            min_n_spikes_per_sec=1,
            return_only_spiking=False,
            delete_hoc_objects=True,
            ):
        h.celsius = self.temp_c
        h.dt = dt
        h.tstop = tstop
        h.v_init = self.v_init
        # integrator
        h.cvode_active(0)

        # Initialize stimulation vector
        svec = h.Vector(stim_factor*stim_pulse)
        svec.play(h._ref_is_xtra, h.dt)

        h.finitialize()
        h.fcurrent()

        if verbose:
            print(f"    Running simulation for {self.name} with {self.n_secs} sections", flush=True)       # Run simulation
        t1 = time.time()
        h.run(h.tstop)
        t2 = time.time()
        if verbose:
            print(f"    Simulation took: {t2-t1} seconds", flush=True)

        axon_results = {
            "segment_midpoints": self.segments_midpoints,
            "diameter": self.fiberD,
            "axon_name": self.name,
            "nsecs": self.n_secs,
            "membrane_potential": {},
        }
        tv = np.array(self.recorders["t"])
        # return axon_results
        v_keys = [k for k in self.recorders.keys() if k.startswith("v_sec_")]
        if len(v_keys) > 0:
            axon_results["time_vector"] = tv
        for k in v_keys:
            axon_results["membrane_potential"][k] = np.array(self.recorders[k])
        
        max_n_spikes_per_sec = 0

        spike_times_recs = [k for k in self.recorders.keys() if "times" in k]
        # for k in spike_times_recs:
            # print(f"k: {k}, n spikes: {self.recorders[k]}", flush=True)
        spike_times_recorders = {k: np.array(self.recorders[k]) if self.recorders[k] is not None and len(self.recorders[k]) > 0 else np.array([]) for k in spike_times_recs}
        axon_results.update({"AP_times": spike_times_recorders})
        spk_times_earliest = {k: np.min(v) if len(v) > 0 else np.inf for k, v in spike_times_recorders.items()}
        sec_names = [k.split("spk_times_")[-1] for k in spike_times_recs]
        earliest_t_ind = np.argmin(list(spk_times_earliest.values()))
        earliest_sec_name = sec_names[earliest_t_ind]
        earliest_spike_time = spk_times_earliest[f"spk_times_{earliest_sec_name}"]
        if earliest_spike_time != np.inf:
            spike = {
                "earliest_spike_time": earliest_spike_time,
                "earliest_spiking_section": earliest_sec_name,
                "spike_sec_idx": int(earliest_sec_name.split("_")[-1]),
                "stim_factor": stim_factor,
            }
            axon_results.update({"spike": spike})
            max_n_spikes_per_sec = np.max([self.recorders[f"spk_{sec_name}"].n for sec_name in sec_names])

        # delete all
        if delete_hoc_objects:
            svec = None
            self.delete_sections()
            self.delete_recorders()

        if not "spike" in axon_results and return_only_spiking:
            return {}
        elif return_only_spiking:
            # check number of spikes per node
            if max_n_spikes_per_sec >= min_n_spikes_per_sec:
                return axon_results
            else:
                return {}
        return axon_results

class MyelinatedAxon(Axon):
    def __init__(
        self,
        discretized_dict=None,
        axon_name=None,
        axon_coords=None,
        fiber_diameter=None,
        axon_inner_diameter=None,
        n_extra_nodes=1,
        end_with_node=True,
        model_type=None,
        params_fit_method="continuous",
        node_params=None,
        tuned_model=False,
        afferent_kws_any=["sensory", "_Aalpha", "_DR", "_DL", "dorsal", "afferent"],
        efferent_kws_any=["motor", "_alpha", "_VR", "_VL", "ventral", "efferent"],
    ):
        """Generate a myelinated axon model based on the discretized_dict or the supplied parameters"""

        # if discretized_dict is provided, use it to initialize the model
        self.params_fit_method = params_fit_method
        self.node_params = node_params
        self.axon_inner_diameter = axon_inner_diameter
        self.sections_ext_v = []
        if discretized_dict is not None:
            self.name = discretized_dict["name"]
            self.axon_coords = discretized_dict["axon_coords"]
            self.fiberD = discretized_dict["fiberD"]
            if "axon_inner_diameter" in discretized_dict.keys():
                self.axon_inner_diameter = discretized_dict["axon_inner_diameter"]
            else:
                self.axon_inner_diameter = None
            self.end_with_node = discretized_dict["end_with_node"]
            self.g = discretized_dict["g"]
            self.axonD = discretized_dict["axonD"]
            self.nodeD = discretized_dict["nodeD"]
            self.paraD1 = discretized_dict["paraD1"]
            self.paraD2 = discretized_dict["paraD2"]
            self.deltax = discretized_dict["deltax"]
            self.paralength2 = discretized_dict["paralength2"]
            self.nl = discretized_dict["nl"]
            self.axonnodes = discretized_dict["axonnodes"]
            self.paranodes1 = discretized_dict["paranodes1"]
            self.paranodes2 = discretized_dict["paranodes2"]
            self.axoninter = discretized_dict["axoninter"]
            self.segments_types = discretized_dict["segments_types"]
            self.segments_midpoints = discretized_dict["segments_midpoints"]
            self.total_length = discretized_dict["total_length"]
            self.model_type = discretized_dict["model_type"]
            self.tuned_model = discretized_dict["tuned_model"]
            if "sections_ext_v" in discretized_dict.keys():
                self.sections_ext_v = discretized_dict["sections_ext_v"]
        else:
            # first call parent constructor
            super(MyelinatedAxon, self).__init__(fiber_diameter=fiber_diameter, axon_name=axon_name, axon_coords=axon_coords)
            if fiber_diameter < 5.7 and self.params_fit_method == "continuous":
                self.params_fit_method = "small"
            self.end_with_node = end_with_node

            if model_type is not None:
                self.model_type = model_type
            else:
                if afferent_kws_any is not None and any([kw.lower() in self.name.lower() for kw in afferent_kws_any]):
                    self.model_type = "sensory"
                elif efferent_kws_any is not None and any([kw.lower() in self.name.lower() for kw in efferent_kws_any]):
                    self.model_type = "motor"
                else:
                    # setting sensory as default
                    self.model_type = "sensory"

            self.tuned_model = tuned_model
            self._discretize(n_extra_nodes)

    def _fit_params(self, fiber_diam=None):
        param_names = ["g", "axonD", "nodeD", "paraD1", "paraD2", "deltax", "paralength2", "nl"]
        # linear interpolation of properties from McIntyre, Richardson, and Grill (2002) for any fiber diameter between 5.7 and 16 um
        if fiber_diam is None:
            fiber_diam = self.fiberD
        if self.params_fit_method == "discrete":
            # MRG discrete values
            # first get fiberD in discrete values
            fiberD_allowed = np.array(list(MRG_discrete_params.keys()))
            fiberD_allowed = np.sort(fiberD_allowed)
            # find the closest value to the fiber diameter
            fiberD_idx = np.argmin(np.abs(fiberD_allowed - fiber_diam))
            fiberD_closest = fiberD_allowed[fiberD_idx]
            g, axonD, nodeD, paraD1, paraD2, deltax, paralength2, nl = (
                MRG_discrete_params[fiberD_closest]
            )
        elif self.params_fit_method == "continuous":
            # Gaines fitting
            g = 0.0172 * (fiber_diam) + 0.5076  # conductace
            axonD = 0.889 * (fiber_diam) - 1.9104  # diameter of the axon
            nodeD = 0.3449 * (fiber_diam) - 0.1484  # diameter of the node
            paraD1 = 0.3527 * (fiber_diam) - 0.1804  # diameter of paranode 1
            paraD2 = 0.889 * (fiber_diam) - 1.9104  # diameter of paranode 2
            deltax = (
                969.3 * np.log(fiber_diam) - 1144.6
            )  # total length between nodes (including 1/2 the node on each side)
            paralength2 = 2.5811 * (fiber_diam) + 19.59  # length of paranode2
            nl = 65.897 * np.log(fiber_diam) - 32.666  # number of lamella
        elif self.params_fit_method == "small":
            # use fiber_diam_fits to get the parameters
            params_dict = {"g": None}
            for param_name, (slope, intercept) in small_fiber_diam_fits.items():
                params_dict[param_name] = slope * fiber_diam + intercept
                # print(f"{param_name}: {params_dict[param_name]}")
            if self.axon_inner_diameter is not None:
                # use myelin_thickness_fits to get paraD1, paraD2, and nl
                myelin_thickness = (fiber_diam - self.axon_inner_diameter)/2
                for param_name, (slope, intercept) in myelin_thickness_fits.items():
                    params_dict[param_name] = slope * myelin_thickness + intercept
                # replace axonD with axon_inner_diameter
                params_dict["axonD"] = self.axon_inner_diameter
            return params_dict["g"], params_dict["axonD"], params_dict["nodeD"], params_dict["paraD1"], params_dict["paraD2"], params_dict["deltax"], params_dict["paralength2"], params_dict["nl"]
        else:
            raise ValueError(
                f"params_fit_method {self.params_fit_method} is not supported! Use 'discrete', 'continuous' or 'small'."
            )
        return g, axonD, nodeD, paraD1, paraD2, deltax, paralength2, nl

    def _set_segment_types(self):
        segment_sequence = ["n", "m", "f", "s", "s", "s", "s", "s", "s", "f", "m"]
        axon_segments_list = []
        for i in range(self.axonnodes - 1):
            axon_segments_list.extend(segment_sequence)
        if self.end_with_node:
            axon_segments_list.append("n")
        else:
            # remove the first node
            axon_segments_list = axon_segments_list[1:]
        self.segments_types = axon_segments_list

    def _set_segments_coords(self):
        paralength1 = 3  # mysa length
        nodelength = 1.0  # node length
        interlength = (
            self.deltax - nodelength - (2 * paralength1) - (2 * self.paralength2)
        ) / 6
        segment_type_to_length = {
            "n": nodelength,
            "m": paralength1,
            "f": self.paralength2,
            "s": interlength,
        }
        segments_midpoints = np.zeros((len(self.segments_types), 3))

        # get segments lengths
        segments_lengths = []
        for segment_idx, segment in enumerate(self.segments_types):
            segment_length = segment_type_to_length[segment]
            segments_lengths.append(segment_length)
        segment_lengths = np.array(segments_lengths)
        segments_lengths_cum = np.cumsum(segment_lengths)

        segment_start = self.axon_coords[0]
        points_lengths = np.linalg.norm(np.diff(self.axon_coords, axis=0), axis=1)
        points_lengths = np.append(0, points_lengths)
        points_lengths_cum = np.cumsum(points_lengths)

        for segment_idx, (segment, segment_length) in enumerate(
            zip(self.segments_types, segments_lengths)
        ):
            # find the coordinate before the segment_length (last one less than segment_length)
            coord_prev_idx = np.argwhere(
                points_lengths_cum < segments_lengths_cum[segment_idx]
            )
            coord_prev_idx = coord_prev_idx[-1][0]

            # find the coordinate after the segment_length (first one greater than segment_length)
            coord_next_idx = np.argwhere(
                points_lengths_cum >= segments_lengths_cum[segment_idx]
            )
            if len(coord_next_idx) == 0:
                direction = self.axon_coords[-1] - self.axon_coords[-2]
                direction /= np.linalg.norm(direction)
                # move by segment_length in direction
                segment_end = segment_start + direction * segment_length
                segments_midpoints[segment_idx] = (
                    segment_start + (segment_end - segment_start) / 2
                )
                segment_start = segment_end
            else:
                coord_next_idx = coord_next_idx[0][0]
                direction = (
                    self.axon_coords[coord_next_idx] - self.axon_coords[coord_prev_idx]
                )
                n_points = int(
                    np.ceil(np.linalg.norm(direction) / (segment_length * 0.01))
                )
                points_along_axon = np.linspace(
                    self.axon_coords[coord_prev_idx],
                    self.axon_coords[coord_next_idx],
                    n_points,
                )
                points_lengths = np.linalg.norm(
                    np.diff(points_along_axon, axis=0), axis=1
                )
                points_cum_lengths = np.cumsum(points_lengths)
                points_cum_lengths = np.append(0, points_cum_lengths)
                points_cum_lengths += points_lengths_cum[coord_prev_idx]
                # get the closest one to segment_length
                closest_point_idx = np.argmin(
                    np.abs(points_cum_lengths - segments_lengths_cum[segment_idx])
                )
                segment_end = points_along_axon[closest_point_idx]
                segments_midpoints[segment_idx] = (
                    segment_start + (segment_end - segment_start) / 2
                )
                segment_start = segment_end
        self.segments_midpoints = segments_midpoints

    def _discretize(self, n_extra_nodes=1):
        (
            self.g,
            self.axonD,
            self.nodeD,
            self.paraD1,
            self.paraD2,
            self.deltax,
            self.paralength2,
            self.nl,
        ) = self._fit_params()
        self.total_length = get_arcline_length(self.axon_coords)
        self.axonnodes = (
            int(math.floor(self.total_length / self.deltax)) + n_extra_nodes
        )
        self.paranodes1 = (
            self.axonnodes - 1
        ) * 2  # number of mysa segments in the axon model
        self.paranodes2 = (
            self.axonnodes - 1
        ) * 2  # number of flut segments in the axon model
        self.axoninter = (
            self.axonnodes - 1
        ) * 6  # number of internodal segments in the axon model
        self._set_segment_types()
        if not self.end_with_node:
            self.axonnodes -= 2
        self._set_segments_coords()

    def to_dict(self):
        """Convert the object to a dictionary"""
        obj_dict = {
            "name": self.name,
            "axon_coords": self.axon_coords,
            "fiberD": self.fiberD,
            "axon_inner_diameter": self.axon_inner_diameter,
            "end_with_node": self.end_with_node,
            "g": self.g,
            "axonD": self.axonD,
            "nodeD": self.nodeD,
            "paraD1": self.paraD1,
            "paraD2": self.paraD2,
            "deltax": self.deltax,
            "paralength2": self.paralength2,
            "nl": self.nl,
            "axonnodes": self.axonnodes,
            "paranodes1": self.paranodes1,
            "paranodes2": self.paranodes2,
            "axoninter": self.axoninter,
            "segments_types": self.segments_types,
            "segments_midpoints": self.segments_midpoints,
            "total_length": self.total_length,
            "model_type": self.model_type,
            "tuned_model": self.tuned_model,
            "sections_ext_v": self.sections_ext_v,
        }
        return obj_dict
    
    def initialize_neuron(
        self,
        passive_end_nodes=False,
        end_connected_to_mn=False,
        node_mechanism=None,
        mysa_mechanism=None,
        flut_mechanism=None,
        stin_mechanism=None,
        add_pas_to_mechanism=False,
        v_init=None,
        mod_params_node=None,
        mod_params_mysa=None,
        mod_params_flut=None,
        mod_params_stin=None,
        temp_c=37,
        ):
        """Initialize the NEURON model of the myelinated axon"""
        
        self.temp_c = temp_c

        self.passive_end_nodes = passive_end_nodes
        self.end_connected_to_mn = end_connected_to_mn

        # delete previous sections
        self.delete_sections()

        rhoa = 0.7e6  # Ohm-um
        mycm = 0.1  # uF/cm2/lamella membrane
        mygm = 0.001  # S/cm2/lamella membrane

        # Define morphological parameters
        paralength1 = 3  # mysa length
        nodelength = 1.0  # node length
        space_p1 = 0.002  # mysa periaxonal space width
        space_p2 = 0.004  # flut periaxonal space width
        space_i = 0.004  # stin periaxonal space width

        # Calculate dependent variables
        Rpn0 = (rhoa * 0.01) / (np.pi * ((((self.nodeD / 2) + space_p1) ** 2) - ((self.nodeD / 2) ** 2)))
        Rpn1 = (rhoa * 0.01) / (np.pi * ((((self.paraD1 / 2) + space_p1) ** 2) - ((self.paraD1 / 2) ** 2)))
        Rpn2 = (rhoa * 0.01) / (np.pi * ((((self.paraD2 / 2) + space_p2) ** 2) - ((self.paraD2 / 2) ** 2)))
        Rpx = (rhoa * 0.01) / (np.pi * ((((self.axonD / 2) + space_i) ** 2) - ((self.axonD / 2) ** 2)))
        interlength = (self.deltax - nodelength - (2 * paralength1) - (2 * self.paralength2)) / 6

        if v_init is not None:
            self.v_init = v_init
        else:
            # determine v_init basd on model type
            if self.model_type == "MRG":
                self.v_init = -80
            elif self.model_type == "sensory":
                if self.tuned_model:
                    self.v_init = -78.05
                else:
                    # Gaines Sensory MRG is assumed [Gaines et al. 2018]
                    self.v_init = -79.3565
            elif self.model_type == "motor":
                if self.tuned_model:
                    self.v_init = -79.5
                else:
                    # Gaines Motor MRG is assumed [Gaines et al. 2018]
                    self.v_init = -85.9411
            else:
                raise NotImplementedError(f"Model type {self.model_type} is not implemented!")
        
        # Get the mechanism names
        if node_mechanism is not None:
            self.node_name = node_mechanism
        elif self.model_type == "motor":
            if self.tuned_model:
                self.node_name = 'axnode_motor_t'
            else:
                self.node_name = 'node_motor'
        elif self.model_type == "sensory":
            if self.tuned_model:
                self.node_name = 'axnode_sensory_t'
            else:
                self.node_name = 'node_sensory'
        elif self.model_type == "MRG":
            self.node_name = 'axnode'

        if mysa_mechanism is not None:
            self.mysa_name = mysa_mechanism
        elif self.model_type == "motor" and not self.tuned_model:
            self.mysa_name = 'mysa_motor'
        elif self.model_type == "sensory" and not self.tuned_model:
            self.mysa_name = 'mysa_sensory'
        else:
            self.mysa_name = None

        if flut_mechanism is not None:
            self.flut_name = flut_mechanism
        elif self.model_type == "motor":
            if self.tuned_model:
                self.flut_name = 'flut_motor_t'
            else:
                self.flut_name = 'flut_motor'
        elif self.model_type == "sensory":
            if self.tuned_model:
                self.flut_name = 'flut_sensory_t'
            else:
                self.flut_name = 'flut_sensory'
        else:
            self.flut_name = None
        
        if stin_mechanism is not None:
            self.stin_name = stin_mechanism
        elif self.model_type == "motor" and not self.tuned_model:
            self.stin_name = 'stin_motor'
        elif self.model_type == "sensory" and not self.tuned_model:
            self.stin_name = 'stin_sensory'
        else:
            self.stin_name = None

        # Initialize sections
        node_i = 0
        mysa_i = 0
        flut_i = 0
        stin_i = 0
        self.sections_list = []
        for sec_i, sec_type in enumerate(self.segments_types):
            if sec_type == "n":
                sec = h.Section(name='node[%d]' % node_i)
                sec.nseg = 1
                sec.diam = self.nodeD
                sec.L = nodelength
                sec.Ra = rhoa / 10000
                sec.cm = 2
                if passive_end_nodes and node_i in [0, self.axonnodes - 1]:
                    if self.end_connected_to_mn and node_i == self.axonnodes - 1:
                        sec.insert(self.node_name)
                        if self.fiberD < 5.16 and self.tuned_model and self.model_type == "sensory":    
                        # if self.fiberD < 5.7 and self.tuned_model and self.model_type == "sensory":
                            asB_n = sec.asB_axnode_sensory_t + 4
                            sec.asB_axnode_sensory_t = asB_n
                        if mod_params_node is not None:
                            for param, value in mod_params_node.items():
                                setattr(sec, param, value)
                    else:
                        sec.insert('pas')
                        sec.g_pas = 0.001 * self.nodeD/self.fiberD
                        sec.e_pas = self.v_init
                else:
                    sec.insert(self.node_name)
                    if self.fiberD < 5.16 and self.tuned_model and self.model_type == "sensory":
                        asB_n = sec.asB_axnode_sensory_t + 4
                        sec.asB_axnode_sensory_t = asB_n
                    if mod_params_node is not None:
                        for param, value in mod_params_node.items():
                            setattr(sec, param, value)
                sec.insert('extracellular')
                sec.insert('xtra')
                sec.xraxial[0] = Rpn0
                sec.xg[0] = 1e10
                sec.xc[0] = 0
                node_i += 1
            elif sec_type == "m":
                sec = h.Section(name='MYSA[%d]' % mysa_i)
                sec.nseg = 1
                sec.diam = self.fiberD
                sec.L = paralength1
                sec.Ra = rhoa * (1 / (self.paraD1 / self.fiberD) ** 2) / 10000
                sec.cm = 2 * self.paraD1 / self.fiberD
                if self.mysa_name is not None:
                    sec.insert(self.mysa_name)
                    if mod_params_mysa is not None:
                        for param, value in mod_params_mysa.items():
                            setattr(sec, param, value)
                if (self.mysa_name is None) or add_pas_to_mechanism:
                        sec.insert('pas')
                        sec.g_pas = 0.001 * self.paraD1 / self.fiberD
                        sec.e_pas = self.v_init
                sec.insert('extracellular')
                sec.insert('xtra')
                sec.xraxial[0] = Rpn1
                sec.xg[0] = mygm / (self.nl * 2)
                sec.xc[0] = mycm / (self.nl * 2)
                mysa_i += 1
            elif sec_type == "f":
                sec = h.Section(name='FLUT[%d]' % flut_i)
                sec.nseg = 1
                sec.diam = self.fiberD
                sec.L = self.paralength2
                sec.Ra = rhoa * (1 / (self.paraD2 / self.fiberD) ** 2) / 10000
                sec.cm = 2 * self.paraD2 / self.fiberD
                if self.flut_name is not None:
                    sec.insert(self.flut_name)
                    if mod_params_flut is not None:
                        for param, value in mod_params_flut.items():
                            setattr(sec, param, value)
                if (self.flut_name is None) or add_pas_to_mechanism or self.tuned_model:
                    sec.insert('pas')
                    sec.g_pas = 0.0001 * self.paraD2 / self.fiberD
                    sec.e_pas = self.v_init
                sec.insert('extracellular')
                sec.insert('xtra')
                sec.xraxial[0] = Rpn2
                sec.xg[0] = mygm / (self.nl * 2)
                sec.xc[0] = mycm / (self.nl * 2)
                flut_i += 1
            elif sec_type == "s":
                sec = h.Section(name='STIN[%d]' % stin_i)
                sec.nseg = 1
                sec.diam = self.fiberD
                sec.L = interlength
                sec.Ra = rhoa * (1 / (self.axonD / self.fiberD) ** 2) / 10000
                sec.cm = 2 * self.axonD / self.fiberD
                if self.stin_name is not None:
                    sec.insert(self.stin_name)
                    if mod_params_stin is not None:
                        for param, value in mod_params_stin.items():
                            setattr(sec, param, value)
                if (self.stin_name is None) or add_pas_to_mechanism:
                    sec.insert('pas')
                    sec.g_pas = 0.0001 * self.axonD / self.fiberD
                    sec.e_pas = self.v_init
                sec.insert('extracellular')
                sec.insert('xtra')
                sec.xraxial[0] = Rpx
                sec.xg[0] = mygm / (self.nl * 2)
                sec.xc[0] = mycm / (self.nl * 2)
                stin_i += 1
            else:
                raise ValueError(f"Unknown segment type: {sec_type}")
            h.setpointer(sec(0.5)._ref_i_membrane, 'im', sec(0.5).xtra)
            h.setpointer(sec(0.5)._ref_e_extracellular, 'ex', sec(0.5).xtra)
            if sec_i > 0:
                sec.connect(self.sections_list[sec_i-1](1), 0)
            self.sections_list.append(sec)

    def setup_recorders(self, dt=0.005, record_v=False, record_node_variables=None, recorded_v_nodes=None, recorded_ap_times_nodes=None, thresh=-20):
        # delete old recorders first
        self.delete_recorders()
        self.recorders = {}
        tv = h.Vector()  # time stamp vector
        tv.record(h._ref_t, dt)
        self.recorders["t"] = tv
        node_indices = [i for i, sec_type in enumerate(self.segments_types) if sec_type == "n"]
        if recorded_ap_times_nodes is not None:
            if self.passive_end_nodes:
                recorded_ap_times_nodes = [self.axonnodes - 2 if node_i == -1 else 1 if node_i == 0 else node_i for node_i in recorded_ap_times_nodes]
            else:
                recorded_ap_times_nodes = [self.axonnodes - 1 if node_i == -1 else node_i for node_i in recorded_ap_times_nodes]
        if recorded_v_nodes is not None:
            recorded_v_nodes = [node_i if node_i != -1 else self.axonnodes - 1 for node_i in recorded_v_nodes]
        for ni, node_i in enumerate(node_indices):
            if record_v:
                if recorded_v_nodes is not None and ni in recorded_v_nodes:
                    self.recorders[f"v_node_{ni}"] = h.Vector()
                    self.recorders[f"v_node_{ni}"].record(self.sections_list[node_i](0.5)._ref_v, dt)
                elif recorded_v_nodes is None:
                    self.recorders[f"v_node_{ni}"] = h.Vector()
                    self.recorders[f"v_node_{ni}"].record(self.sections_list[node_i](0.5)._ref_v, dt)
            self.recorders[f"spk_node_{ni}"] = h.APCount(self.sections_list[node_i](0.5))
            self.recorders[f"spk_node_{ni}"].thresh = thresh
            if (recorded_ap_times_nodes is not None and ni in recorded_ap_times_nodes) or recorded_ap_times_nodes is None:
                if ni == 0 and not self.passive_end_nodes:
                    key_name = "AP_times_first_node"
                elif ni == 1 and self.passive_end_nodes:
                    key_name = "AP_times_first_node"
                elif (ni == len(node_indices) - 1 and not self.passive_end_nodes) or (ni == len(node_indices) - 1 and self.end_connected_to_mn):
                    key_name = "AP_times_last_node"
                elif ni == len(node_indices) - 2 and self.passive_end_nodes and not self.end_connected_to_mn:
                    key_name = "AP_times_last_node"
                else:
                    key_name = f"AP_times_node_{ni}"
                self.recorders[key_name] = h.Vector()
                self.recorders[f"spk_node_{ni}"].record(self.recorders[key_name])
            self.recorders["time_vector"] = tv
            if record_node_variables is not None and len(record_node_variables) > 0:
                for variable_name in record_node_variables:
                    self.recorders[f"{variable_name}_node_{ni}"] = h.Vector()
                    exec(f"recs[f'{variable_name}_node_{ni}'].record(self.sections_list[node_i](0.5)._ref_{variable_name}_{self.node_name}, dt)")

    def interpolate_v_on_sections(self, field_dict=None):
        xyz_all_segs = np.copy(self.segments_midpoints)
        if field_dict is not None:
            self.sections_ext_v = interpolate_3d(field_dict, xyz_all_segs)
        else:
            print("\t\t!!! WARNING: No field_dict is provided. Assigning zeros to v_ext of all sections!!!")
            self.sections_ext_v = np.zeros(len(xyz_all_segs))

    def assign_v_ext(self, field_dict=None):
        if len(self.sections_ext_v) == 0:
            self.interpolate_v_on_sections(field_dict)

        # assuming voltage is in V, convert V to mV and multiply by 1e-6 to get value in MegaOhm considering a current of 1mA (see xtra.mod)
        for seg_i, seg_ext_v in enumerate(self.sections_ext_v):
            self.sections_list[seg_i].rx_xtra = seg_ext_v * 1e3 * 1e-6

    def run_simulation(
            self,
            stim_factor,
            stim_pulse,
            dt=0.005,
            tstop=5.0,
            verbose=False,
            prepassive_nodes_as_endnodes=False,
            exclude_end_node=False,
            min_n_spikes_per_node=1,
            return_only_spiking=False,
            output_path=None,
            delete_hoc_objects=True,
            init_hoc_path=None,
            intracellular_stims=None,
            ):
        h.celsius = self.temp_c
        h.dt = dt
        h.tstop = tstop
        h.v_init = self.v_init
        # integrator
        h.cvode_active(0)
        self.intracellularIClamps = []
        if intracellular_stims is not None:
            for intracellular_stim in intracellular_stims:
                if "node" not in intracellular_stim.keys():
                    intracellular_stim["node"] = 0
                if "amp" not in intracellular_stim.keys():
                    intracellular_stim["amp"] = 1.0
                if "delay" not in intracellular_stim.keys():
                    intracellular_stim["delay"] = 0.1
                if "dur" not in intracellular_stim.keys():
                    intracellular_stim["dur"] = 1.0
                if intracellular_stim["node"] == -1:
                    intracellular_stim["node"] = self.axonnodes - 1
                if self.passive_end_nodes:
                    if intracellular_stim["node"] == 0:
                        intracellular_stim["node"] = 1
                    elif intracellular_stim["node"] == self.axonnodes - 1:
                        intracellular_stim["node"] = self.axonnodes - 2
                stim_sec_idx = [i for i, sec_type in enumerate(self.segments_types) if sec_type == "n"][intracellular_stim["node"]]
                stim_sec = self.sections_list[stim_sec_idx]
                stim = h.IClamp(stim_sec(0.5))
                stim.delay = intracellular_stim["delay"]
                stim.dur = intracellular_stim["dur"]
                stim.amp = intracellular_stim["amp"]
                self.intracellularIClamps.append(stim)
                if verbose:
                    print(f"\tInjecting {intracellular_stim['amp']} nA in node {intracellular_stim['node']} (section {stim_sec.name()}) for {intracellular_stim['dur']} ms starting at {intracellular_stim['delay']} ms")

        # Initialize stimulation vector
        svec = h.Vector(stim_factor*stim_pulse)
        svec.play(h._ref_is_xtra, h.dt)

        # Initialize simulation
        if init_hoc_path is not None:
            h.load_file(init_hoc_path)
        else:
            h.finitialize()
            h.fcurrent()

        # Run simulation
        t1 = time.time()
        h.run(h.tstop)
        t2 = time.time()
        if verbose:
            print(f"    Simulation took: {t2-t1} seconds")

        tv = np.array(self.recorders["time_vector"])
        axon_results = {
            "segment_types": self.segments_types,
            "segment_midpoints": self.segments_midpoints,
            "diameter": self.fiberD,
            "axon_name": self.name,
            "nnodes": self.axonnodes,
            "membrane_potential": {},
            "axon_nodetonode_dist": self.deltax,
        }
        node_inds_in_segments = np.argwhere(np.array(self.segments_types) == "n")[:, 0]
        v_keys = [k for k in self.recorders.keys() if k.startswith("v_node_")]
        if len(v_keys) > 0:
            axon_results["time_vector"] = tv
        for k in v_keys:
            axon_results["membrane_potential"][k] = np.array(self.recorders[k])
            
        
        spike_times_recs = [k for k in self.recorders.keys() if "times" in k]
        spike_times_recorders = {k: np.array(self.recorders[k]) if len(self.recorders[k]) > 0 else np.array([]) for k in spike_times_recs}
        axon_results.update({"AP_times": spike_times_recorders})
        
        # Note: the following spike detection works only if each node spikes once (useful for simple titration)
        # For more complex spike detection, please use the AP_times dictionary
        spike_times = [
            self.recorders[f"spk_node_{node_i}"].time
            for node_i in range(0, self.axonnodes)
            if self.recorders[f"spk_node_{node_i}"].n > 0
        ]
        spiking_nodes = [
            node_i
            for node_i in range(0, self.axonnodes)
            if self.recorders[f"spk_node_{node_i}"].n > 0
        ]
        node_idx_earliest_spike = -1
        spikes_list = []
        if len(spike_times) > 0:
            spike_times = np.array(spike_times)
            spiking_nodes = np.array(spiking_nodes)
            node_idx_earliest_spike = spiking_nodes[np.argmin(spike_times)]
            spike_time_earliest_spike = spike_times[np.argmin(spike_times)]
            spike_seg_idx = node_inds_in_segments[node_idx_earliest_spike]
            if self.passive_end_nodes and prepassive_nodes_as_endnodes:
                end_node_inds = [
                    0, 1, self.axonnodes - 2, self.axonnodes - 1
                ]
            else:
                end_node_inds = [0, self.axonnodes - 1]
            end_node_check = any(
                [node_idx_earliest_spike == end_node_ind for end_node_ind in end_node_inds]
            )
            spike_dict = {
                "spike": {
                    "spike_at_node": node_idx_earliest_spike,
                    "spike_seg_idx": spike_seg_idx,
                    "spike_time": spike_time_earliest_spike,
                    "stim_factor": stim_factor,
                    "end_node": end_node_check,
                    "axon_had_end_node": end_node_check,
                }
            }
            spikes_list.append(
                {
                    "spike_at_node": node_idx_earliest_spike,
                    "spike_seg_idx": spike_seg_idx,
                    "spike_time": spike_time_earliest_spike,
                    "stim_factor": stim_factor,
                    "end_node": end_node_check,
                }
            )
            # look for another location
            if len(spike_times) > 1:
                spike_times_diff = np.diff(spike_times)
                spike_times_diff = np.append(spike_times_diff, spike_times_diff[-1]).astype(
                    np.float32
                )
                zero_crossings = np.where(np.diff(np.signbit(spike_times_diff)))[0]
                zero_crossings = np.array(
                    [
                        zc
                        for zc in zero_crossings
                        if spiking_nodes[zc + 1] not in end_node_inds
                        and spike_times_diff[zc] != 0
                    ]
                )
                # print(f"Zero crossings: {zero_crossings}")
                if len(zero_crossings) > 0:
                    # get the node with the earliest spike
                    if max(zero_crossings + 1) < len(spiking_nodes):
                        for zc in zero_crossings:
                            spike_time_zc = spike_times[zc + 1]
                            spike_node_zc = spiking_nodes[zc + 1]
                            spike_seg_idx = node_inds_in_segments[spike_node_zc]
                            # make sure it is a spike initiation node by checking spike time of preceeding and following 2 nodes
                            spike_init = True
                            node_inds_to_check = [spike_node_zc - 2, spike_node_zc - 1, spike_node_zc + 1, spike_node_zc + 2]
                            for node_i in node_inds_to_check:
                                if node_i < 0 or node_i >= self.axonnodes:
                                    continue
                                if self.recorders[f"spk_node_{node_i}"].n == 0 or self.recorders[f"spk_node_{node_i}"].time < spike_time_zc:
                                    spike_init = False
                                    break
                            if spike_init:
                                # check if this spike was added before
                                spike_exists = any(
                                    [
                                        spike["spike_at_node"] == spike_node_zc
                                        for spike in spikes_list
                                    ]
                                )
                                if not spike_exists:
                                    spikes_list.append(
                                        {
                                            "spike_at_node": spike_node_zc,
                                            "spike_seg_idx": spike_seg_idx,
                                            "spike_time": spike_time_zc,
                                            "stim_factor": stim_factor,
                                            "end_node": False,
                                        }
                                    )
                        if exclude_end_node and end_node_check:
                            earliest_spike_idx = np.argmin(spike_times[zero_crossings + 1])
                            node_idx_earliest_spike = spiking_nodes[zero_crossings + 1][earliest_spike_idx]
                            spike_time_earliest_spike = spike_times[zero_crossings + 1][earliest_spike_idx]
                            spike_seg_idx = node_inds_in_segments[node_idx_earliest_spike]
                            spike_dict = {
                                "spike": {
                                    "spike_at_node": node_idx_earliest_spike,
                                    "spike_seg_idx": spike_seg_idx,
                                    "spike_time": spike_time_earliest_spike,
                                    "stim_factor": stim_factor,
                                    "end_node": False,
                                    "axon_had_end_node": end_node_check,
                                }
                            }
                            axon_results.update(spike_dict)
                        else:
                            axon_results.update(spike_dict)
                else:
                    if not (exclude_end_node and end_node_check):
                        axon_results.update(spike_dict)
            else:
                if not (exclude_end_node and end_node_check):
                    axon_results.update(spike_dict)
        axon_results.update({"spikes_list": spikes_list})
        if output_path is not None:
            np.save(output_path, axon_results, allow_pickle=True)

        max_n_spikes_per_node = 0
        if "spike" in axon_results:
            max_n_spikes_per_node = np.max([self.recorders[f"spk_node_{node_i}"].n for node_i in range(0, self.axonnodes)])
        # add max_n_spikes_per_node to results
        axon_results.update({"max_n_spikes_per_node": max_n_spikes_per_node})

        # delete all
        svec = None
        if delete_hoc_objects:
            self.delete_sections()
            self.delete_recorders()

        if not "spike" in axon_results and return_only_spiking:
            # print("FINISHED SIM ONE AXON! NO SPIKE!")
            return {}
        elif return_only_spiking:
            # check number of spikes per node
            if max_n_spikes_per_node >= min_n_spikes_per_node:
                return axon_results
            else:
                return {}
        return axon_results

    def delete_sections(self):
        # delete all sections (set to None if direct deletion does not work)
        if hasattr(self, "sections_list"):
            for sec in self.sections_list:
                try:
                    h.delete_section(sec=sec)
                except:
                    for seg in sec:
                        seg = None
                try:
                    sec = None
                except:
                    pass
    
    def delete_recorders(self):
        if hasattr(self, "recorders"):
            for k in self.recorders.keys():
                self.recorders[k] = None

class Motoneuron:
    """ Neuron Biophysical cat motoneuron model.
    A Modified and simplified version of the motoneuron presented in:
    - Moraud et al. 2016, Mechanisms Underlying the Neuromodulation of Spinal Circuits for Correcting Gait and Balance Deficits after Spinal Cord Injury
    - Formento et al. 2018, Electrical spinal cord stimulation must preserve proprioception to enable locomotion in humans with spinal cord injury. Nature Neuroscience

    The model integrates a motoneuron soma developed by McIntyre 2002.
    The soma geometry is modified to match the human dimension.
    """

    def __init__(self, name="motoneuron", init_seg_diam=10, soma_diam=53.04, initseg_length=1000, soma_length=53.04, soma_coord=None, initseg_coord=None):
        """ Object initialization.

        Keyword arguments:
        name -- name of the cell (default "motoneuron")
        """

        # Define parameters
        self.name = name
        self.synapses = []
        self.netcons = []
        self.Ia_afferent_inputs = {}

        self.initseg_diam = init_seg_diam
        self.soma_diam = soma_diam # 53.04 from Cullheim et al. 1987
        self.initseg_length = initseg_length # 1000 from Moraud et al. 2016
        self.soma_length = soma_length # 53.04 from Cullheim et al. 1987

        self.soma_coord = soma_coord
        self.initseg_coord = initseg_coord
        
        self._create_sections()
        self._define_biophysics()
        self._build_topology()


    """
    Specific Methods of this class
    """

    def _create_sections(self):
        """ Create the sections of the cell. """
        self.soma = h.Section(name='soma', cell=self)
        self.initSegment = h.Section(name='initsegment', cell=self)

    def _calc_Rpn(self, diam):
        rhoa = 0.7e6  # Ohm-um
        space_p1 = 0.002  # mysa periaxonal space width
        return (rhoa * 0.01) / (np.pi * ((((diam / 2) + space_p1) ** 2) - ((diam / 2) ** 2)))

    def _define_biophysics(self):
        """ Assign geometry and membrane properties across the cell. """
        
        soma_diam = self.soma_diam
        initseg_diam = self.initseg_diam
        Rpn_soma = self._calc_Rpn(soma_diam)
        Rpn_initseg = self._calc_Rpn(initseg_diam)
        
        self.soma.nseg = 1
        self.soma.L = self.soma_length
        self.soma.diam = soma_diam
        self.soma.cm = 2
        self.soma.Ra = 200
        self.soma.insert('motoneuron') # Insert the Neuron motoneuron mechanism developed by McIntyre 2002
        # if self._drug: self.soma.gcak_motoneuron *= 0.6 # Add the drug effect as in Booth et al 1997
        self.soma.insert('extracellular')
        self.soma.insert('xtra')
        self.soma.xraxial[0] = Rpn_soma
        self.soma.xg[0] = 1e10
        self.soma.xc[0] = 0
        for seg in self.soma:
            h.setpointer(seg._ref_i_membrane, 'im', seg.xtra)
            h.setpointer(seg._ref_e_extracellular, 'ex', seg.xtra)
        
        self.initSegment.nseg = 5
        self.initSegment.L = self.initseg_length
        self.initSegment.diam = initseg_diam
        self.initSegment.insert('initial') # Insert the Neuorn initial mechanism developed by McIntyre
        self.initSegment.gnap_initial = 0
        self.initSegment.Ra = 200
        self.initSegment.cm = 2
        self.initSegment.insert('extracellular')
        self.initSegment.insert('xtra')
        self.soma.xraxial[0] = Rpn_initseg
        self.initSegment.xg[0] = 1e10
        self.initSegment.xc[0] = 0
        for seg in self.initSegment:
            h.setpointer(seg._ref_i_membrane, 'im', seg.xtra)
            h.setpointer(seg._ref_e_extracellular, 'ex', seg.xtra)

    def _build_topology(self):
        """ Connect the sections together. """
        #childSection.connect(parentSection, [parentX], [childEnd])
        self.soma.connect(self.initSegment, 0, 0)

    def set_coords(self, soma_coord, initseg_coord):
        """ Set the coordinates of the cell. """
        self.soma_coord = soma_coord
        self.initseg_coord = initseg_coord

    def assign_v_ext(self, v_ext_soma=None, v_ext_initseg=None, field_dict=None):
        """ Assign the external voltage to the soma and initsegment sections. """

        if (v_ext_soma is None or v_ext_initseg is None) and field_dict is None:
            raise ValueError("Either v_ext_soma and v_ext_initseg or field_dict must be provided.")
        elif (v_ext_soma is None or v_ext_initseg is None) and field_dict is not None:
            if self.soma_coord is None or self.initseg_coord is None:
                raise ValueError("soma_coord and initseg_coord must be set before using field_dict.")
            # interpolate the field_dict to get the external voltage at the soma and initsegment coordinates
            mn_coords = np.concatenate((self.initseg_coord, self.soma_coord))
            extSegPot_mn = interpolate_3d(field_dict, mn_coords)
            v_ext_initseg = extSegPot_mn[0]
            v_ext_soma = extSegPot_mn[1]
        self.soma.rx_xtra = v_ext_soma * 1e3 * 1e-6
        self.initSegment.rx_xtra = v_ext_initseg * 1e3 * 1e-6

    def create_synapse(self, type="excitatory", x=None, verbose=False):
        """ Create and return a synapse that links motoneuron state variables to external events.

        The created synapse is also appended to a list containg all synapses the this motoneuron has.

        Keyword arguments:
        type -- type of synapse to be created. This could be:
        1) "excitatory" to create an excitatory synapse positioned on the dendritic tree
        2) "inhibitory" to create an inhibitory synapse positioned on the soma
        3) "ees" to create a synapse that mimic the recruitmend induced by electrical
        stimulation; in this case the synapse is positioned on the axon.
        """

        if type=="excitatory":
            if x is None:
                x = rnd.random()
            if verbose:
                print(f"Creating a synapse on the soma. Position: {x}")
            syn = h.ExpSyn(self.soma(x))
            syn.tau = 0.5
            syn.e = 0
            self.synapses.append(syn)
        else:
            raise NotImplementedError(f"Synapse type {type} is not implemented!")

        return syn

    def setup_recorders(self, record_v=False):
        """Setup the recorders for the motoneuron model."""
        self.recorders = {}
        self.recorders['t'] = h.Vector()
        self.recorders['t'].record(h._ref_t)
        if record_v:
            self.recorders['v_soma'] = h.Vector()
            self.recorders['v_soma'].record(self.soma(0.5)._ref_v)
            self.recorders['v_initseg'] = h.Vector()
            self.recorders['v_initseg'].record(self.initSegment(0.5)._ref_v)
        # AP recorders
        self.recorders['ap_soma'] = h.APCount(self.soma(0.5))
        self.recorders['ap_soma'].thresh = 0
        self.recorders["ap_soma_times"] = h.Vector()
        self.recorders["ap_soma"].record(self.recorders["ap_soma_times"])
        self.recorders['ap_initseg'] = h.APCount(self.initSegment(0.5))
        self.recorders['ap_initseg'].thresh = 0
        self.recorders["ap_initseg_times"] = h.Vector()
        self.recorders["ap_initseg"].record(self.recorders["ap_initseg_times"])
    
    def get_recorders_npy(self):
        """Get the recorded variables as numpy arrays. """
        recorders_extracted = {}
        for k, v in self.recorders.items():
            if "v_" in k or k == "t":
                recorders_extracted[k] = np.array(v)
            elif "ap_" in k and "times" in k:
                if len(v) > 0:
                    recorders_extracted[k] = np.array(v)
                else:
                    recorders_extracted[k] = np.array([])
        return recorders_extracted

    def plot_membrane_potential(self, fig_path=None):
        """ Plot the membrane potential of the soma and the dendrites. """
        plt.figure(figsize=[20, 10])
        plt.plot(self.recorders['t'], self.recorders['v_soma'],label='soma')
        plt.plot(self.recorders['t'], self.recorders['v_initseg'],label='initseg')
        plt.legend()
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        if fig_path is not None:
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_epsp(self, syn_t, plot=False, fig_path=None):
        # check if recorders are set up
        # if 'v_dendrite_0' not in self.recorders:
        # 	raise ValueError("Recorders are not set up. Run setup_recorders() and a simulation first.")
        # v_dendrite = np.array(self.recorders[f'v_dendrite_{dendrite_index}'])
        t = self.recorders['t']
        t_index = np.argmin(np.abs(t - syn_t))
        v_soma = np.array(self.recorders['v_soma'])
        epsp_soma = (np.max(v_soma[t_index:]) - v_soma[t_index])
        print(f"EPSP at soma: {epsp_soma} uV")
        if plot:
            plt.figure()
            plt.plot(t, v_soma)
            plt.axvline(x=syn_t, color='r', linestyle='--')
            # plot horizontal line at the resting potential at the synapse time
            plt.axhline(y=v_soma[t_index], color='g', linestyle='--')
            # plot horizontal line at the peak potential after the synapse time
            plt.axhline(y=np.max(v_soma[t_index:]), color='g', linestyle='--')
            if fig_path is not None:
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        # epsp is (v_max after syn_t - v at syn_t)
        return epsp_soma

    def connect_to_source(self, source, weight=0.001, delay=0, synapses_index=0, threshold=None):
        """ Connect the current cell to a source cell and return the netCon object.

        Keyword arguments:
        source -- the source object to which we want to connect
        weight -- the weight of the connection (default 0.001: depending on the fiber diameter, this weight determines the EPSP amplitude at the motoneuron soma)

        delay -- communication time delay in ms (default 1)
        """
        if len(self.synapses) == 0:
            raise ValueError("No synapses created yet. Please create a synapse first!")

        nc = h.NetCon(source, self.synapses[synapses_index])
        nc.delay = delay
        nc.weight[0] = weight
        if threshold is not None:
            nc.threshold = threshold
        self.netcons.append(nc)
        return nc
    
    def set_Ia_afferent_inputs(self, Ia_afferent_inputs, n_synapses=1, syn_weight=0.001653909):
        """Set Ia afferent inputs to the motoneuron."""
        self.vecstims = []
        self.Ia_afferent_inputs = Ia_afferent_inputs
        for axon_name, spike_times_list in Ia_afferent_inputs.items():
            if len(spike_times_list) == 0:
                continue
            axon_spike_times = np.sort(spike_times_list)
            for i in range(n_synapses):
                spike_times = np.copy(axon_spike_times)
                syn = self.create_synapse(type="excitatory", x=0.5)
                spiketime_v = h.Vector(spike_times)				#Create a h.Vector() and insert the time element in the vector.
                vecstim = h.VecStim()
                vecstim.play(spiketime_v)				#Play the spiketime.
                self.vecstims.append(vecstim)
        for syn_i, vecstim in enumerate(self.vecstims):
            nc = self.connect_to_source(vecstim, synapses_index=syn_i, weight=syn_weight)

    def delete_recorders(self):
        """ Delete all recorders. """
        if hasattr(self, "recorders"):
            for k in self.recorders.keys():
                self.recorders[k] = None
            self.recorders = {}

    def delete_all(self):
        """ Delete all sections, recorders and synapses. """
        h.delete_section(sec=self.soma)
        h.delete_section(sec=self.initSegment)
        self.soma = None
        self.initSegment = None
        self.delete_recorders()
        self.synapses = []
        self.netcons = []
        self.vecstims = []
        self.Ia_afferent_inputs = {}

     