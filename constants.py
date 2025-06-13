import numpy as np

# Temperature limitation (10 and 5 are safety margins)
t_hh_max = 750 + 273.15 - 10
t_lh_min = 450 + 273.15 + 10
t_hc_max = 65 + 273.15 - 5
t_lc_min = -98 + 273.15 + 5

t_amb = 25 + 273.15

# Efficiencies
eff_mg = 0.985  # motor-generator
eff_com_0 = 0.87  # compressor
eff_exp_0 = 0.93  # expander

# Helium
kappa = 0.4 # (gamma - 1) / gamma = (1.666 - 1) / 1.666 = 0.4
cp = 5190 # J/kg*K
r = 2077

# NTUs and specific heat flow rate ratio of heat exchangers
ntu_hs_0 = 9.0 # heat storage
ntu_re_0 = 49.0 # recuperator
ntu_hr_0 = 3.4 # heat rejection
ntu_cs_0 = 49.0 # cold storage
capacity_ratio_hs = 1.0
capacity_ratio_re = 1.0
capacity_ratio_hr = 0.5
capacity_ratio_cs = 1.0

# Coefficient for surrogate heat exchanger model
coe_hs = -0.25  # heat storage
coe_re = -0.55  # recuperator
coe_hr = -0.25  # heat rejection
coe_cs = -0.55  # cold storage

heat_loss = 1e-4 # per hour

# Constants for analytical model
d1 = 1.8
d2 = 1.8

# Bounds for n, p_max
lower_bounds = np.array([0.8, 0.1])
upper_bounds = np.array([1, 1])

# Cost parameter
c_vom = 0
c_start = 0
# c_vom = 0
# c_start = 0

penalty_factor = 1e5