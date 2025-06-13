import numpy as np
import matplotlib.pyplot as plt
from ptes_model_our import PTES_our  # Using your model
from constants import *
import multiprocessing as mp
from functools import partial
import random
import time
from tqdm import tqdm
import pandas as pd
import os


def calculate_point(args):
    """Calculate PTES performance for a single point in the parameter space"""
    ptes, beta_norm, m_norm, phase, idx = args

    # Set initial temperatures
    current_temps = {
        't_hh': ptes.t_hh_0,
        't_lh': ptes.t_lh_0,
        't_hc': ptes.t_hc_0,
        't_lc': ptes.t_lc_0
    }

    # Calculate performance
    results = ptes.calculate_performance(
        phase=phase,
        m_norm=m_norm,
        beta_norm=beta_norm,
        current_temps=current_temps
    )

    # Check if calculation is valid
    is_valid = 1
    if results['vio_choke_com'] > 1e-3 or results['vio_choke_exp'] > 1e-3 or results['vio_surge'] > 1e-3 or results[
        'vio_beta'] > 1e-3:
        is_valid = 0
        eff_ex = np.nan
        n = np.nan
        p_exp = np.nan
        alpha_com = np.nan
        alpha_exp = np.nan
    else:
        eff_ex = results['eff_ex']
        n = results['n']
        p_exp = results['p_exp']
        alpha_com = results['alpha_com']
        alpha_exp = results['alpha_exp']

    return idx, beta_norm, m_norm, eff_ex, n, p_exp, alpha_com, alpha_exp, is_valid


def process_results_chunk(completed, total, results, beta_norm_values, m_norm_values,
                          eff_ex_array, n_array, p_exp_array, alpha_com_array, alpha_exp_array, valid_array):
    """Process a chunk of completed results and update progress"""
    for result in results:
        idx, beta_norm, m_norm, eff_ex, n, p_exp, alpha_com, alpha_exp, is_valid = result
        i = np.where(beta_norm_values == beta_norm)[0][0]
        j = np.where(m_norm_values == m_norm)[0][0]
        eff_ex_array[i, j] = eff_ex
        n_array[i, j] = n
        p_exp_array[i, j] = p_exp
        alpha_com_array[i, j] = alpha_com
        alpha_exp_array[i, j] = alpha_exp
        valid_array[i, j] = is_valid

    # Update progress
    progress = completed / total * 100
    print(f"Progress: {completed}/{total} ({progress:.1f}%)")


def save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, param_array, valid_array, param_name, phase):
    """Save a specific parameter to CSV file in matrix format for Origin"""
    # Apply mask for invalid points
    masked_param = np.ma.masked_where(valid_array == 0, param_array)
    masked_param = np.ma.filled(masked_param, np.nan)  # Replace masked values with NaN

    # Create DataFrame with the matrix structure
    df = pd.DataFrame(
        masked_param,
        index=beta_norm_values,
        columns=m_norm_values
    )

    # Add row/column labels
    df.index.name = 'beta_norm'
    df.columns.name = 'm_norm'

    # Ensure turbo_map directory exists
    os.makedirs('turbo_map', exist_ok=True)

    # Save to CSV in the turbo_map folder
    filename = os.path.join('turbo_map', f'ptes_{phase}_{param_name}_matrix.csv')
    df.to_csv(filename)
    print(f"{param_name} matrix data for {phase} saved to {filename}")

    return df


def create_parameter_heatmap(M_norm, Beta_norm, param_array, valid_array, param_name, phase, cmap='viridis',
                             invert=False):
    """Create and save a heatmap for a specific parameter"""
    # Create figure for the parameter
    plt.figure(figsize=(10, 8))

    # Mask invalid points
    masked_param = np.ma.masked_where(valid_array == 0, param_array)

    # Choose colormap based on parameter
    if param_name == 'eff_ex':
        if phase == 'charging':
            cmap = 'Blues_r'  # For charging efficiency, higher (more negative) is better
        else:
            cmap = 'Reds'  # For discharging efficiency, higher (more positive) is better

    # Plot heatmap
    heatmap = plt.pcolormesh(M_norm, Beta_norm, masked_param, cmap=cmap)
    plt.title(f'{phase.capitalize()} {param_name} vs m_norm and beta_norm', fontsize=14)
    plt.xlabel('Mass Flow Rate Ratio (m_norm)', fontsize=12)
    plt.ylabel('Pressure Ratio Ratio (beta_norm)', fontsize=12)
    cbar = plt.colorbar(heatmap, label=f'{param_name}')

    # Add contour lines
    valid_values = masked_param.compressed()
    if len(valid_values) > 0:
        levels = np.linspace(np.nanmin(valid_values), np.nanmax(valid_values), 8)
        CS = plt.contour(M_norm, Beta_norm, masked_param, levels=levels, colors='black', alpha=0.6, linewidths=0.7)
        plt.clabel(CS, inline=True, fontsize=8, fmt='%.2f')

    # Mark the reference point (m_norm=1, beta_norm=1)
    plt.plot(1, 1, 'o', color='lime', markersize=10, markeredgecolor='black')

    # Save figure
    os.makedirs('turbo_map', exist_ok=True)
    plt.savefig(os.path.join('turbo_map', f'ptes_{phase}_{param_name}_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Heatmap for {param_name} in {phase} phase saved to turbo_map/ptes_{phase}_{param_name}_heatmap.png")
    plt.close()


def main():
    # Initialize the PTES model
    ptes = PTES_our(design_file="ptes_design_200MW_10h.csv")

    # Test a single calculation point
    print("Testing a single calculation point...")
    test_result = calculate_point([ptes, 1.1, 1.1, 'char', 0])
    print(f"Test result: {test_result}")

    # Define the range of m_norm and beta_norm values to explore
    m_norm_values = np.linspace(0.02, 1.5, 75)  # Expanded range to capture more operating conditions
    beta_norm_values = np.linspace(0.6, 1.4, 81)  # More resolution

    # Create meshgrid for heatmap
    M_norm, Beta_norm = np.meshgrid(m_norm_values, beta_norm_values)

    # Initialize results arrays
    eff_ex_char = np.zeros_like(M_norm)
    eff_ex_dis = np.zeros_like(M_norm)
    n_char = np.zeros_like(M_norm)
    n_dis = np.zeros_like(M_norm)
    p_exp_char = np.zeros_like(M_norm)
    p_exp_dis = np.zeros_like(M_norm)
    alpha_com_char = np.zeros_like(M_norm)
    alpha_com_dis = np.zeros_like(M_norm)
    alpha_exp_char = np.zeros_like(M_norm)
    alpha_exp_dis = np.zeros_like(M_norm)
    valid_char = np.ones_like(M_norm)
    valid_dis = np.ones_like(M_norm)

    # Prepare parameter combinations for parallel processing
    params_char = [(ptes, beta, m, "char", idx) for idx, (beta, m) in enumerate(zip(
        np.repeat(beta_norm_values, len(m_norm_values)),
        np.tile(m_norm_values, len(beta_norm_values))
    ))]

    params_dis = [(ptes, beta, m, "dis", idx) for idx, (beta, m) in enumerate(zip(
        np.repeat(beta_norm_values, len(m_norm_values)),
        np.tile(m_norm_values, len(beta_norm_values))
    ))]

    # Randomly shuffle the parameter combinations to balance workload
    random.shuffle(params_char)
    random.shuffle(params_dis)

    # Set up parallel processing
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing")

    # Execute calculations in parallel with progress tracking
    total_points = len(params_char)
    chunk_size = max(1, total_points // 50)  # Report progress after each 2% completion

    # Process charging phase
    print("\nStarting charging phase calculations...")
    start_time = time.time()

    with mp.Pool(processes=num_cores) as pool:
        # Process charging phase with progress tracking
        completed = 0
        results_char = []

        for result in tqdm(pool.imap_unordered(calculate_point, params_char), total=total_points):
            results_char.append(result)
            completed += 1

            # Process results in chunks to avoid too frequent updates
            if completed % chunk_size == 0 or completed == total_points:
                process_results_chunk(
                    completed, total_points,
                    results_char[-chunk_size:],
                    beta_norm_values, m_norm_values,
                    eff_ex_char, n_char, p_exp_char, alpha_com_char, alpha_exp_char, valid_char
                )

    charging_time = time.time() - start_time
    print(f"Charging phase completed in {charging_time:.2f} seconds")

    # Process discharging phase
    print("\nStarting discharging phase calculations...")
    start_time = time.time()

    with mp.Pool(processes=num_cores) as pool:
        # Process discharging phase with progress tracking
        completed = 0
        results_dis = []

        for result in tqdm(pool.imap_unordered(calculate_point, params_dis), total=total_points):
            results_dis.append(result)
            completed += 1

            # Process results in chunks to avoid too frequent updates
            if completed % chunk_size == 0 or completed == total_points:
                process_results_chunk(
                    completed, total_points,
                    results_dis[-chunk_size:],
                    beta_norm_values, m_norm_values,
                    eff_ex_dis, n_dis, p_exp_dis, alpha_com_dis, alpha_exp_dis, valid_dis
                )

    discharging_time = time.time() - start_time
    print(f"Discharging phase completed in {discharging_time:.2f} seconds")

    # Save the raw data to numpy files for later use
    # Ensure parameter_maps directory exists
    os.makedirs('turbo_map', exist_ok=True)

    # Save NPZ file with all data
    np.savez(os.path.join('turbo_map', 'ptes_all_data.npz'),
             m_norm=m_norm_values,
             beta_norm=beta_norm_values,
             eff_ex_char=eff_ex_char,
             eff_ex_dis=eff_ex_dis,
             n_char=n_char,
             n_dis=n_dis,
             p_exp_char=p_exp_char,
             p_exp_dis=p_exp_dis,
             alpha_com_char=alpha_com_char,
             alpha_com_dis=alpha_com_dis,
             alpha_exp_char=alpha_exp_char,
             alpha_exp_dis=alpha_exp_dis,
             valid_char=valid_char,
             valid_dis=valid_dis)

    print("Raw data saved to turbo_map/ptes_all_data.npz")

    # Save parameter data in matrix format for Origin
    # Charging phase
    print("\nSaving charging phase parameter data in matrix format...")
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, eff_ex_char, valid_char, 'eff_ex', 'charging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, n_char, valid_char, 'n', 'charging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, p_exp_char, valid_char, 'p_exp', 'charging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, alpha_com_char, valid_char, 'alpha_com', 'charging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, alpha_exp_char, valid_char, 'alpha_exp', 'charging')

    # Discharging phase
    print("\nSaving discharging phase parameter data in matrix format...")
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, eff_ex_dis, valid_dis, 'eff_ex', 'discharging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, n_dis, valid_dis, 'n', 'discharging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, p_exp_dis, valid_dis, 'p_exp', 'discharging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, alpha_com_dis, valid_dis, 'alpha_com', 'discharging')
    save_parameter_to_csv_matrix(m_norm_values, beta_norm_values, alpha_exp_dis, valid_dis, 'alpha_exp', 'discharging')

    # Create heatmaps for visualization
    create_parameter_heatmap(M_norm, Beta_norm, eff_ex_char, valid_char, 'eff_ex', 'charging', 'Blues_r')
    create_parameter_heatmap(M_norm, Beta_norm, n_char, valid_char, 'n', 'charging')
    create_parameter_heatmap(M_norm, Beta_norm, p_exp_char, valid_char, 'p_exp', 'charging')
    create_parameter_heatmap(M_norm, Beta_norm, alpha_com_char, valid_char, 'alpha_com', 'charging')
    create_parameter_heatmap(M_norm, Beta_norm, alpha_exp_char, valid_char, 'alpha_exp', 'charging')

    create_parameter_heatmap(M_norm, Beta_norm, eff_ex_dis, valid_dis, 'eff_ex', 'discharging', 'Reds')
    create_parameter_heatmap(M_norm, Beta_norm, n_dis, valid_dis, 'n', 'discharging')
    create_parameter_heatmap(M_norm, Beta_norm, p_exp_dis, valid_dis, 'p_exp', 'discharging')
    create_parameter_heatmap(M_norm, Beta_norm, alpha_com_dis, valid_dis, 'alpha_com', 'discharging')
    create_parameter_heatmap(M_norm, Beta_norm, alpha_exp_dis, valid_dis, 'alpha_exp', 'discharging')

    # Create figure with two subplots for eff_ex
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot charging eff_ex heatmap
    masked_eff_ex_char = np.ma.masked_where(valid_char == 0, eff_ex_char)
    char_heatmap = ax1.pcolormesh(M_norm, Beta_norm, masked_eff_ex_char, cmap='Blues_r')
    ax1.set_title('Charging Exergy Efficiency vs m_norm and beta_norm', fontsize=14)
    ax1.set_xlabel('Mass Flow Rate Ratio (m_norm)', fontsize=12)
    ax1.set_ylabel('Pressure Ratio Ratio (beta_norm)', fontsize=12)
    char_cbar = fig.colorbar(char_heatmap, ax=ax1, label='Exergy Efficiency')

    # Plot discharging eff_ex heatmap
    masked_eff_ex_dis = np.ma.masked_where(valid_dis == 0, eff_ex_dis)
    dis_heatmap = ax2.pcolormesh(M_norm, Beta_norm, masked_eff_ex_dis, cmap='Reds')
    ax2.set_title('Discharging Exergy Efficiency vs m_norm and beta_norm', fontsize=14)
    ax2.set_xlabel('Mass Flow Rate Ratio (m_norm)', fontsize=12)
    ax2.set_ylabel('Pressure Ratio Ratio (beta_norm)', fontsize=12)
    dis_cbar = fig.colorbar(dis_heatmap, ax=ax2, label='Exergy Efficiency')

    # Add contour lines for eff_ex levels
    valid_char_values = masked_eff_ex_char.compressed()
    valid_dis_values = masked_eff_ex_dis.compressed()

    if len(valid_char_values) > 0:
        levels_char = np.linspace(np.nanmin(valid_char_values), np.nanmax(valid_char_values), 8)
        CS1 = ax1.contour(M_norm, Beta_norm, masked_eff_ex_char, levels=levels_char, colors='black', alpha=0.6,
                          linewidths=0.7)
        ax1.clabel(CS1, inline=True, fontsize=8, fmt='%.2f')

    if len(valid_dis_values) > 0:
        levels_dis = np.linspace(np.nanmin(valid_dis_values), np.nanmax(valid_dis_values), 8)
        CS2 = ax2.contour(M_norm, Beta_norm, masked_eff_ex_dis, levels=levels_dis, colors='black', alpha=0.6,
                          linewidths=0.7)
        ax2.clabel(CS2, inline=True, fontsize=8, fmt='%.2f')

    # Mark the reference point (m_norm=1, beta_norm=1)
    ax1.plot(1, 1, 'o', color='lime', markersize=10, markeredgecolor='black')
    ax2.plot(1, 1, 'o', color='lime', markersize=10, markeredgecolor='black')

    # Add annotations for the reference point
    ref_char = ptes.calculate_performance(phase="char", m_norm=1, beta_norm=1,
                                          current_temps={'t_hh': ptes.t_hh_0, 't_lh': ptes.t_lh_0,
                                                         't_hc': ptes.t_hc_0, 't_lc': ptes.t_lc_0})
    ref_dis = ptes.calculate_performance(phase="dis", m_norm=1, beta_norm=1,
                                         current_temps={'t_hh': ptes.t_hh_0, 't_lh': ptes.t_lh_0,
                                                        't_hc': ptes.t_hc_0, 't_lc': ptes.t_lc_0})

    char_eff_ex = ref_char['eff_ex']
    dis_eff_ex = ref_dis['eff_ex']

    ax1.annotate(f"{char_eff_ex:.3f}",
                 xy=(1, 1), xytext=(1.05, 0.95),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    ax2.annotate(f"{dis_eff_ex:.3f}",
                 xy=(1, 1), xytext=(1.05, 0.95),
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join('turbo_map', 'ptes_eff_ex_heatmaps.png'), dpi=300, bbox_inches='tight')
    print("Combined exergy efficiency heatmap saved to turbo_map/ptes_eff_ex_heatmaps.png")
    plt.close()

    # Print summary information
    print("\n=== Summary Information ===")
    print("\nCharging Phase:")
    if len(valid_char_values) > 0:
        print(f"Exergy efficiency range: {np.nanmin(valid_char_values):.3f} to {np.nanmax(valid_char_values):.3f}")
    print(f"Reference point (m_norm=1, beta_norm=1): {char_eff_ex:.3f}")
    print(f"Points with constraint violations: {np.sum(valid_char == 0)} out of {eff_ex_char.size}")

    print("\nDischarging Phase:")
    if len(valid_dis_values) > 0:
        print(f"Exergy efficiency range: {np.nanmin(valid_dis_values):.3f} to {np.nanmax(valid_dis_values):.3f}")
    print(f"Reference point (m_norm=1, beta_norm=1): {dis_eff_ex:.3f}")
    print(f"Points with constraint violations: {np.sum(valid_dis == 0)} out of {eff_ex_dis.size}")

    # Find optimal operating points (for charging: most negative eff_ex, for discharging: highest positive eff_ex)
    if len(valid_char_values) > 0:
        # For charging, find the most negative eff_ex (best performance)
        min_char_idx = np.unravel_index(np.nanargmin(masked_eff_ex_char), masked_eff_ex_char.shape)
        print("\nOptimal Charging Point (Highest Exergy Efficiency - most negative):")
        print(f"beta_norm={beta_norm_values[min_char_idx[0]]:.2f}, " +
              f"m_norm={m_norm_values[min_char_idx[1]]:.2f}, eff_ex={masked_eff_ex_char[min_char_idx]:.3f}")
        print(f"Normalized rotational speed (n): {n_char[min_char_idx]:.3f}")
        print(f"Expander pressure (p_exp): {p_exp_char[min_char_idx]:.3f}")
        print(f"Compressor angle (alpha_com): {alpha_com_char[min_char_idx]:.3f}")
        print(f"Expander angle (alpha_exp): {alpha_exp_char[min_char_idx]:.3f}")

    if len(valid_dis_values) > 0:
        # For discharging, find the highest positive eff_ex (best performance)
        max_dis_idx = np.unravel_index(np.nanargmax(masked_eff_ex_dis), masked_eff_ex_dis.shape)
        print("\nOptimal Discharging Point (Highest Exergy Efficiency - most positive):")
        print(f"beta_norm={beta_norm_values[max_dis_idx[0]]:.2f}, " +
              f"m_norm={m_norm_values[max_dis_idx[1]]:.2f}, eff_ex={masked_eff_ex_dis[max_dis_idx]:.3f}")
        print(f"Normalized rotational speed (n): {n_dis[max_dis_idx]:.3f}")
        print(f"Expander pressure (p_exp): {p_exp_dis[max_dis_idx]:.3f}")
        print(f"Compressor angle (alpha_com): {alpha_com_dis[max_dis_idx]:.3f}")
        print(f"Expander angle (alpha_exp): {alpha_exp_dis[max_dis_idx]:.3f}")

    # Save optimal points to a separate CSV file
    if len(valid_char_values) > 0 and len(valid_dis_values) > 0:
        optimal_points = pd.DataFrame({
            'Parameter': ['beta_norm', 'm_norm', 'eff_ex', 'n', 'p_exp', 'alpha_com', 'alpha_exp'],
            'Charging': [
                beta_norm_values[min_char_idx[0]],
                m_norm_values[min_char_idx[1]],
                masked_eff_ex_char[min_char_idx],
                n_char[min_char_idx],
                p_exp_char[min_char_idx],
                alpha_com_char[min_char_idx],
                alpha_exp_char[min_char_idx]
            ],
            'Discharging': [
                beta_norm_values[max_dis_idx[0]],
                m_norm_values[max_dis_idx[1]],
                masked_eff_ex_dis[max_dis_idx],
                n_dis[max_dis_idx],
                p_exp_dis[max_dis_idx],
                alpha_com_dis[max_dis_idx],
                alpha_exp_dis[max_dis_idx]
            ]
        })

        optimal_points.to_csv(os.path.join('turbo_map', 'optimal_points.csv'), index=False)
        print("Optimal points saved to turbo_map/optimal_points.csv")

    print(f"\nTotal calculation time: {charging_time + discharging_time:.2f} seconds")


if __name__ == "__main__":
    main()