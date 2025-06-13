import numpy as np
import pandas as pd
import os

# Constants from the PTES model
d1 = 1.8
d2 = 1.8

# Set fixed beta_0 as reference
beta_0 = 2.0  # Fixed reference pressure ratio
eff_com_0 = 0.87  # Base compressor efficiency
eff_exp_0 = 0.93  # Base expander efficiency


def generate_compressor_map(G_dots, n_dots, alphas):
    """
    Generate compressor map data with surge and choke constraints.

    Parameters:
    -----------
    G_dots : array
        Array of normalized mass flow rates
    n_dots : array
        Array of normalized rotational speeds
    alphas : array
        Array of IGV angles

    Returns:
    --------
    results : dict
        Dictionary containing the computed map data
    """
    results = {
        'G_dots': [],
        'n_dots': [],
        'alphas': [],
        'betas': [],
        'effs': [],
        'is_valid': [],  # Track valid points
        'surge_line': {'G_dots': [], 'betas': [], 'n_dots': [], 'alphas': []},
        'choke_line': {'G_dots': [], 'betas': [], 'n_dots': [], 'alphas': []}
    }

    # Calculate for each speed and alpha combination
    for n_dot in n_dots:
        for alpha in alphas:
            # Calculate coefficients for compressor map
            c1 = n_dot / (d1 * (1 - d2 / n_dot) + n_dot * (n_dot - d2) ** 2)
            c2 = (d1 - 2 * d2 * n_dot ** 2) / (d1 * (1 - d2 / n_dot) + n_dot * (n_dot - d2) ** 2)
            c3 = -(d1 * d2 * n_dot - d2 ** 2 * n_dot ** 3) / (d1 * (1 - d2 / n_dot) + n_dot * (n_dot - d2) ** 2)

            # First calculate the reference surge point
            G_s_dot_map_n1 = ((1.18 - c2) - np.sqrt((c2 - 1.18) ** 2 - 4 * c1 * c3)) / 2 / c1
            eff_s_map_n1 = (1 - 0.3 * (1 - n_dot) ** 2) * (n_dot / G_s_dot_map_n1) * (
                    2 - n_dot / G_s_dot_map_n1) * eff_com_0
            eff_s_map = (1 - 0.7 * (1 - n_dot)) ** 2 * eff_s_map_n1

            # Now find the actual surge point by solving where eff_map equals eff_s_map
            # The equation is:
            # (1 - 0.3*(1-n_dot)^2) * (n_dot/G_dot_map) * (2 - n_dot/G_dot_map) * eff_com_0 = eff_s_map
            # Let x = n_dot/G_dot_map
            efficiency_factor = (1 - 0.3 * (1 - n_dot) ** 2)
            a = efficiency_factor
            b = -2 * efficiency_factor
            c = eff_s_map / eff_com_0

            # Solve quadratic equation for x = n_dot/G_dot_map
            discriminant = b ** 2 - 4 * a * c
            if discriminant >= 0:
                # We want the larger x solution, which would be in the surge region
                surge_G_dot_map = n_dot / ((-b + np.sqrt(discriminant)) / (2 * a))

                # Ensure it's in the valid range (less than n_dot for surge)
                if 0 < surge_G_dot_map < n_dot:
                    # Convert map G_dot to actual G_dot
                    G_s_dot = surge_G_dot_map * (1 + alpha / 100)

                    # Calculate beta at this surge point
                    beta_map = (c1 * surge_G_dot_map ** 2 + c2 * surge_G_dot_map + c3) * beta_0
                    beta_surge = (beta_map - 1) * (1 + 0.01 * alpha) + 1

                    results['surge_line']['G_dots'].append(G_s_dot)
                    results['surge_line']['betas'].append(beta_surge / beta_0)
                    results['surge_line']['n_dots'].append(n_dot)
                    results['surge_line']['alphas'].append(alpha)

            # Find choke line point
            # Choke occurs when G_dot_map > 1 and efficiency falls below 0.85 * eff_com_0
            # We'll find the G_dot where efficiency equals 0.85 * eff_com_0

            # First, let's find the G_dot_map where eff_map = 0.85 * eff_com_0
            # The equation is: (1 - 0.3 * (1 - n_dot)^2) * (n_dot / G_dot_map) * (2 - n_dot / G_dot_map) = 0.85
            # This is a quadratic equation in G_dot_map

            efficiency_factor = (1 - 0.3 * (1 - n_dot) ** 2)
            a = efficiency_factor
            b = -2 * efficiency_factor
            c = 0.85

            # Solve quadratic equation
            discriminant = b ** 2 - 4 * a * c
            if discriminant >= 0:
                # We want the smaller x solution, which would be in the choke region
                G_choke_dot_map = n_dot / ((-b - np.sqrt(discriminant)) / (2 * a))

                if G_choke_dot_map > n_dot:  # Only record if it's in the choke region
                    G_choke = G_choke_dot_map * (1 + alpha / 100)  # Convert to actual G_dot
                    beta_map = (c1 * G_choke_dot_map ** 2 + c2 * G_choke_dot_map + c3) * beta_0
                    beta_choke = (beta_map - 1) * (1 + 0.01 * alpha) + 1

                    results['choke_line']['G_dots'].append(G_choke)
                    results['choke_line']['betas'].append(beta_choke / beta_0)
                    results['choke_line']['n_dots'].append(n_dot)
                    results['choke_line']['alphas'].append(alpha)

            # Process each G_dot for this n_dot and alpha
            for G_dot in G_dots:
                # Adjust G_dot for alpha (IGV angle)
                G_dot_map = G_dot / (1 + alpha / 100)

                # Calculate beta for this point
                beta_map = (c1 * G_dot_map ** 2 + c2 * G_dot_map + c3) * beta_0
                beta = (beta_map - 1) * (1 + 0.01 * alpha) + 1

                # Calculate efficiency
                eff_map = (1 - 0.3 * (1 - n_dot) ** 2) * (n_dot / G_dot_map) * (2 - n_dot / G_dot_map) * eff_com_0
                eff = eff_map * (1 - alpha ** 2 / 10000)

                # Check if point is valid (not in surge or choke region)
                is_valid = True

                # Check choke constraint
                if G_dot_map > n_dot:
                    if eff_map < 0.85 * eff_com_0:
                        is_valid = False

                # Check surge constraint by comparing efficiencies
                # In the surge region (G_dot_map < 1), the point is invalid if eff_s_map > eff_map
                if G_dot_map < n_dot:
                    if eff_s_map > eff_map:
                        is_valid = False

                # Store results
                results['G_dots'].append(G_dot)
                results['n_dots'].append(n_dot)
                results['alphas'].append(alpha)
                results['betas'].append(beta / beta_0)  # Store true pressure ratio
                results['effs'].append(eff / eff_com_0)  # Store true efficiency
                results['is_valid'].append(is_valid)

    return results


def generate_expander_map(G_dots, n_dots, alphas):
    """
    Generate expander map data with choke constraints.

    Parameters:
    -----------
    G_dots : array
        Array of normalized mass flow rates
    n_dots : array
        Array of normalized rotational speeds
    alphas : array
        Array of NGV angles

    Returns:
    --------
    results : dict
        Dictionary containing the computed map data
    """
    results = {
        'G_dots': [],
        'n_dots': [],
        'alphas': [],
        'betas': [],
        'effs': [],
        'is_valid': [],  # Track valid points
        'choke_line': {'G_dots': [], 'betas': [], 'n_dots': [], 'alphas': []}
    }

    # Calculate for each combination
    for n_dot in n_dots:
        for alpha in alphas:
            efficiency_factor = (1 - 0.3 * (1 - n_dot) ** 2)
            a = efficiency_factor
            b = -2 * efficiency_factor
            c = 0.85

            # Solve quadratic equation
            discriminant = b ** 2 - 4 * a * c
            if discriminant >= 0:
                # We want the smaller x solution, which would be in the choke region
                G_choke_dot_map = n_dot / ((-b - np.sqrt(discriminant)) / (2 * a))

                if G_choke_dot_map > n_dot:  # Only record if it's in the choke region
                    G_choke_dot = G_choke_dot_map * (1.1875 - 3 / 10000 * (alpha - 25) ** 2)
                    beta_choke = np.sqrt(
                        (G_choke_dot_map / np.sqrt(1.4 - 0.4 * n_dot)) ** 2 * (
                                beta_0 ** 2 - 1) + 1)

                    results['choke_line']['G_dots'].append(G_choke_dot)
                    results['choke_line']['betas'].append(beta_choke / beta_0)
                    results['choke_line']['n_dots'].append(n_dot)
                    results['choke_line']['alphas'].append(alpha)

            # Process each G_dot for this n_dot and alpha
            for G_dot in G_dots:
                # Adjust G_dot for alpha (NGV angle)
                G_dot_map = G_dot / (1.1875 - 3 / 10000 * (alpha - 25) ** 2)

                # Calculate beta for this point using formula matching the PTES model
                correction_factor = np.sqrt(1.4 - 0.4 * n_dot)
                beta = np.sqrt((G_dot_map / correction_factor) ** 2 * (beta_0 ** 2 - 1) + 1)

                # Calculate efficiency
                eff_map = (1 - 0.3 * (1 - n_dot) ** 2) * (n_dot / G_dot_map) * (2 - n_dot / G_dot_map) * eff_exp_0
                eff = eff_map * (1 - 0.9 * alpha ** 2 / 10000)

                # Check if point is valid (not in choke region)
                is_valid = True
                if G_dot_map > n_dot:
                    if eff_map < 0.85 * eff_exp_0:
                        is_valid = False

                # Store results
                results['G_dots'].append(G_dot)
                results['n_dots'].append(n_dot)
                results['alphas'].append(alpha)
                results['betas'].append(beta / beta_0)
                results['effs'].append(eff / eff_exp_0)
                results['is_valid'].append(is_valid)

    return results


def export_to_csv_by_ndot(data, base_filename, folder="turbo_curve"):
    """
    Export the generated data to CSV files, with all data in a single file
    organized with G_dot as rows and columns for each (n_dot, alpha) combination.
    Values outside of valid operating region are replaced with NaN.

    Parameters:
    -----------
    data : dict
        Dictionary containing the data to export
    base_filename : str
        Base filename to use for the CSV files
    folder : str
        Folder to save the CSV files in
    """
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Convert main data to DataFrame
    all_data = pd.DataFrame({
        'G_dot': data['G_dots'],
        'n_dot': data['n_dots'],
        'alpha': data['alphas'],
        'beta': data['betas'],
        'efficiency': data['effs'],
        'is_valid': data['is_valid']
    })

    # Get unique n_dot, alpha, and G_dot values
    unique_n_dots = sorted(set(data['n_dots']))
    unique_alphas = sorted(set(data['alphas']))
    unique_g_dots = sorted(set(data['G_dots']))

    # Create DataFrames with G_dot as rows
    beta_df = pd.DataFrame({'G_dot': unique_g_dots})
    eff_df = pd.DataFrame({'G_dot': unique_g_dots})

    # Add columns for each combination of n_dot and alpha
    for n_dot in unique_n_dots:
        for alpha in unique_alphas:
            # Filter data for this n_dot and alpha combination
            filtered_data = all_data[(all_data['n_dot'] == n_dot) & (all_data['alpha'] == alpha)]
            filtered_data = filtered_data.sort_values('G_dot')

            # Create dictionaries for beta and efficiency values
            beta_dict = {}
            eff_dict = {}

            for _, row in filtered_data.iterrows():
                g_dot = row['G_dot']

                # Store the value or NaN based on whether the point is valid
                if row['is_valid']:
                    beta_dict[g_dot] = row['beta']
                    eff_dict[g_dot] = row['efficiency']
                else:
                    beta_dict[g_dot] = np.nan
                    eff_dict[g_dot] = np.nan

            # Add columns to the DataFrame
            beta_col = f'beta_n{n_dot}_alpha{alpha}'
            eff_col = f'eff_n{n_dot}_alpha{alpha}'

            beta_df[beta_col] = beta_df['G_dot'].map(beta_dict)
            eff_df[eff_col] = eff_df['G_dot'].map(eff_dict)

    # Save to CSV files
    beta_filename = f"{base_filename}_beta.csv"
    eff_filename = f"{base_filename}_efficiency.csv"

    beta_df.to_csv(os.path.join(folder, beta_filename), index=False)
    eff_df.to_csv(os.path.join(folder, eff_filename), index=False)

    # Export special points separately
    if 'surge_line' in data and len(data['surge_line']['G_dots']) > 0:
        surge_line_df = pd.DataFrame({
            'n_dot': data['surge_line']['n_dots'],
            'alpha': data['surge_line']['alphas'],
            'G_dot': data['surge_line']['G_dots'],
            'beta': data['surge_line']['betas']
        })
        surge_line_df.to_csv(os.path.join(folder, f"{base_filename}_surge_line.csv"), index=False)

    if 'choke_line' in data and len(data['choke_line']['G_dots']) > 0:
        choke_line_df = pd.DataFrame({
            'n_dot': data['choke_line']['n_dots'],
            'alpha': data['choke_line']['alphas'],
            'G_dot': data['choke_line']['G_dots'],
            'beta': data['choke_line']['betas']
        })
        choke_line_df.to_csv(os.path.join(folder, f"{base_filename}_choke_line.csv"), index=False)


def generate_and_export_data():
    """
    Generate turbomachinery data and export to CSV files
    """
    # Define parameter ranges
    G_dots = np.linspace(0.1, 2.0, 191)  # Normalized mass flow rate from 0.6 to 1.4
    n_dots = [0.8, 0.9, 1.0]  # Different rotational speeds
    alphas = [-10, 0, 10]  # Different deviation angles

    # Generate the data
    print("Generating compressor map data...")
    comp_data = generate_compressor_map(G_dots, n_dots, alphas)

    print("Generating expander map data...")
    exp_data = generate_expander_map(G_dots, n_dots, alphas)

    # Export the data to CSV
    print("Exporting compressor data to CSV...")
    export_to_csv_by_ndot(comp_data, "compressor")

    print("Exporting expander data to CSV...")
    export_to_csv_by_ndot(exp_data, "expander")

    print("Data generation and export complete!")

    return comp_data, exp_data


if __name__ == "__main__":
    comp_data, exp_data = generate_and_export_data()