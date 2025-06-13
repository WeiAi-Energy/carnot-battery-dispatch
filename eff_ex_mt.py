import os
import numpy as np
import pandas as pd
from ptes_model_mt import PTES_mt

# Create turbo_map directory if it doesn't exist
if not os.path.exists('turbo_map'):
    os.makedirs('turbo_map')

# Initialize PTES object
ptes = PTES_mt(design_file='ptes_design_200MW_10h.csv')

# Define range of normalized mass flow rates
m_norm_values = np.arange(0.1, 1.01, 0.01)

# Initialize data structures for results
char_results = {
    'm_norm': [],
    'beta': [],
    'eff_ex': []
}

dis_results = {
    'm_norm': [],
    'beta': [],
    'eff_ex': []
}

# Calculate performance for each normalized mass flow rate
for m_norm in m_norm_values:
    # Initialize store temperatures
    store_temps = {
        "t_hh": ptes.t_hh_0,
        "t_lh": ptes.t_lh_0,
        "t_hc": ptes.t_hc_0,
        "t_lc": ptes.t_lc_0
    }

    # Calculate performance for charging
    char_perf = ptes.calculate_performance('char', m_norm, store_temps)

    # Store results for charging
    char_results['m_norm'].append(m_norm)
    char_results['eff_ex'].append(char_perf['eff_ex'])
    char_results['beta'].append(char_perf['beta']/1.859746932866164)

    # Calculate performance for discharging
    dis_perf = ptes.calculate_performance('dis', m_norm, store_temps)

    # Store results for discharging
    dis_results['m_norm'].append(m_norm)
    dis_results['eff_ex'].append(dis_perf['eff_ex'])
    dis_results['beta'].append(dis_perf['beta']/2.1021738680997384)

# Convert results to DataFrames
char_df = pd.DataFrame(char_results)
dis_df = pd.DataFrame(dis_results)

# Save results to CSV files
char_df.to_csv('turbo_map/mt_char_eff_ex.csv', index=False)
dis_df.to_csv('turbo_map/mt_dis_eff_ex.csv', index=False)

print("Calculation complete. Results saved to turbo_map folder.")