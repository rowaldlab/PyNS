# MRG discrete parameters from:
# - diameter=1.0: Pelot et al. 2017 (https://doi.org/10.1088/1741-2552/aa6a5f) and corrected deltax extrapolation
# - diameter=2.0: McIntyre et al. 2003 (https://doi.org/10.1152/jn.00989.2003)
# - diameters >= 5.7: McIntyre, Richardson, and Grill 2002 (https://doi.org/10.1152/jn.00353.2001)
MRG_discrete_params = {
    1.0: (None, 0.8, 0.7, 0.7, 0.8, 186.6, 5, 15), # Pelot et al. 2017 (https://doi.org/10.1088/1741-2552/aa6a5f) and corrected deltax extrapolation
    2.0: (None, 1.6, 1.4, 1.4, 1.6, 373.2, 10, 30), # McIntyre et al. 2003 (https://doi.org/10.1152/jn.00989.2003)
    5.7: (0.605, 3.4, 1.9, 1.9, 3.4, 500, 35, 80),
    7.3: (0.630, 4.6, 2.4, 2.4, 4.6, 750, 38, 100),
    8.7: (0.661, 5.8, 2.8, 2.8, 5.8, 1000, 40, 110),
    10.0: (0.690, 6.9, 3.3, 3.3, 6.9, 1150, 46, 120),
    11.5: (0.700, 8.1, 3.7, 3.7, 8.1, 1250, 50, 130),
    12.8: (0.719, 9.2, 4.2, 4.2, 9.2, 1350, 54, 135),
    14.0: (0.739, 10.4, 4.7, 4.7, 10.4, 1400, 56, 140),
    15.0: (0.767, 11.5, 5.0, 5.0, 11.5, 1450, 58, 145),
    16.0: (0.791, 12.7, 5.5, 5.5, 12.7, 1500, 60, 150),
}

# fiber diameter to morphology linear fits based on data of small fibers (2, 3, and 5.7) from:
# - Mcintyre et al. 2003 (https://doi.org/10.1152/jn.00989.2003)
# - Mirzakhalili et al. 2020 (https://doi.org/10.1016/j.cels.2020.10.004)
# for each parameter: (slope, intercept)
small_fiber_diam_fits = {
    'axonD': (0.5350734094616638, 0.3816204458945085),
    'nodeD': (0.22267536704730825, 0.687574768896139),
    'paraD1': (0.22267536704730825, 0.687574768896139),
    'paraD2': (0.5350734094616638, 0.3816204458945085),
    'deltax': (57.87765089722674, 185.42147906470913),
    'paralength2': (6.484502446982054, -2.1383904295812894),
    'nl': (13.743882544861334, 1.8094072865688047),
    }

# myelin thickness to nl, paraD1 and paraD2 linear fits based on data of small fibers (2, 3, and 5.7) from:
# - Mcintyre et al. 2003 (https://doi.org/10.1152/jn.00989.2003)
# - Mirzakhalili et al. 2020 (https://doi.org/10.1016/j.cels.2020.10.004)
# for each parameter: (slope, intercept)
myelin_thickness_fits = {
    'nl': (58.18858561, 13.54218362),
    'paraD1': (0.89578164, 0.90037221),
    'paraD2': (2.24317618, 0.84913151)
    }