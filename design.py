import numpy as np
from sympy import symbols, Eq, nsolve, solve
from constants import *
from scipy.optimize import minimize_scalar
import pandas as pd
import os


def solve_ptes_design(power_0, duration):
    """
    Design optimization of the PTES system to determine the designed charge and discharge pressure ratios and RTE.
    Conditions:
    1. Reach t_hh_max
    2. Maximize RTE
    3. Charge duration = Discharge duration

    Parameters:
    power_0 (float): Power capacity in MW (for discharge)
    duration (float): Duration in hours (for both charge and discharge)

    Returns:
    dict: Dictionary containing all the designed parameters
    """
    # Define the expected CSV filename
    csv_filename = f"ptes_design_{power_0}MW_{duration}h.csv"

    # Variables to store initial guess values
    initial_values = {}

    # Check if the file already exists for initial values
    if os.path.exists(csv_filename):
        print(f"Design file {csv_filename} exists. Loading parameters for initial values.")
        try:
            design_df = pd.read_csv(csv_filename)
            initial_values = design_df.iloc[0].to_dict()
            print("Successfully loaded initial values from existing design file.")
        except Exception as e:
            print(f"Error loading existing design file: {e}")
            print("Will use default initial values.")

    # Initialize parameters
    power_dis_0 = power_0

    # Charge power will be determined based on equal duration constraint

    # Get heat exchanger effectiveness
    def eff_heat_exchanger(name, m=1.0):
        # Get parameters for the specified heat exchanger
        heat_exchanger_data = {
            "hs": {"ntu_0": ntu_hs_0, "coe": coe_hs, "default_capacity_ratio": capacity_ratio_hs},
            "re": {"ntu_0": ntu_re_0, "coe": coe_re, "default_capacity_ratio": capacity_ratio_re},
            "hr": {"ntu_0": ntu_hr_0, "coe": coe_hr, "default_capacity_ratio": capacity_ratio_hr},
            "cs": {"ntu_0": ntu_cs_0, "coe": coe_cs, "default_capacity_ratio": capacity_ratio_cs},
        }
        exchanger = heat_exchanger_data[name]
        ntu = exchanger["ntu_0"] * m ** exchanger["coe"]
        capacity_ratio = exchanger["default_capacity_ratio"]

        # Calculate the effectiveness based on NTU and capacity ratio
        if abs(capacity_ratio - 1.0) < 1e-6:  # Avoid numerical issues
            eff = ntu / (1 + ntu)
        else:
            eff = ((1 - np.exp(-ntu * (1 - capacity_ratio))) /
                   (1 - capacity_ratio * np.exp(-ntu * (1 - capacity_ratio))))
        return eff

    eff_com = eff_com_0
    eff_exp = eff_exp_0

    t13 = t_hh_max

    # Variables to store the designed parameters
    design_params = {}

    def obj(beta_dis):
        """
        Define the objective function with input beta_dis.
        It performs the charge-discharge iterative loop and returns the converged RTE.
        """
        tol = 1e-4
        iter_max = 100

        # Initialize with values from file or default
        if 't12_dis' in initial_values and 't9_dis' in initial_values:
            t12_old = initial_values['t12_dis']
            t9_old = initial_values['t9_dis']
            print(f"Using initial values from file: t12_old={t12_old}, t9_old={t9_old}")
        else:
            t12_old = 761.6  # Default initial values
            t9_old = 297.5
            print(f"Using default initial values: t12_old={t12_old}, t9_old={t9_old}")

        # Initialize m_dis_0 and m_char_0 to default values
        # These will be updated during the iteration
        if 'm_dis_0' in initial_values:
            m_dis_0 = initial_values['m_dis_0']
        else:
            m_dis_0 = 1000.0  # Default initial value

        if 'm_char_0' in initial_values:
            m_char_0 = initial_values['m_char_0']
        else:
            m_char_0 = 1000.0  # Default initial value

        # Previous solution to use as initial guess for next iteration
        prev_sol_char = None
        prev_sol_dis = None

        for i in range(iter_max):
            # Calculate heat exchanger efficiencies for discharge
            # Note: In the design phase, we use m=1.0 as we're at the design point
            eff_hs_dis = eff_heat_exchanger("hs", 1.0)
            eff_re_dis = eff_heat_exchanger("re", 1.0)
            eff_hr_dis = eff_heat_exchanger("hr", 1.0)
            eff_cs_dis = eff_heat_exchanger("cs", 1.0)

            # Calculate heat exchanger efficiencies for charge
            # For charge, the reference flow rate is m_dis_0
            if i > 0 and m_dis_0 > 0 and m_char_0 > 0:
                relative_flow = m_char_0 / m_dis_0
                eff_hs_char = eff_heat_exchanger("hs", relative_flow)
                eff_re_char = eff_heat_exchanger("re", relative_flow)
                eff_hr_char = eff_heat_exchanger("hr", relative_flow)
                eff_cs_char = eff_heat_exchanger("cs", relative_flow)
            else:
                # For the first iteration, use default values
                eff_hs_char = eff_heat_exchanger("hs", 1.0)
                eff_re_char = eff_heat_exchanger("re", 1.0)
                eff_hr_char = eff_heat_exchanger("hr", 1.0)
                eff_cs_char = eff_heat_exchanger("cs", 1.0)

            # Solve the charge process equations
            st1_char, st2_char, st3_char, st4_char, st5_char, st6_char, st7_char, st8_char, sbeta_char = symbols(
                'st1_char st2_char st3_char st4_char st5_char st6_char st7_char st8_char sbeta_char', positive=True
            )
            eqs_char = [
                Eq(st2_char - st1_char * (1 + (sbeta_char ** kappa - 1) / eff_com), 0),
                Eq(st6_char - st5_char * (1 - (1 - sbeta_char ** -kappa) * eff_exp), 0),
                Eq(t13 - eff_hs_char * st2_char - (1 - eff_hs_char) * t12_old, 0),
                Eq(st3_char - eff_hs_char * t12_old - (1 - eff_hs_char) * st2_char, 0),
                Eq(st4_char - eff_re_char * st7_char - (1 - eff_re_char) * st3_char, 0),
                Eq(st1_char - eff_re_char * st3_char - (1 - eff_re_char) * st7_char, 0),
                Eq(st5_char - eff_hr_char * t_amb - (1 - eff_hr_char) * st4_char, 0),
                Eq(st7_char - eff_cs_char * t9_old - (1 - eff_cs_char) * st6_char, 0),
                Eq(st8_char - eff_cs_char * st6_char - (1 - eff_cs_char) * t9_old, 0)
            ]

            # Determine initial guess for charge process
            if i == 0 and all(k in initial_values for k in ['t1_char', 't2_char', 't3_char', 't4_char',
                                                            't5_char', 't6_char', 't7_char', 't8_char', 'beta_char_0']):
                # Use values from file as initial guess for first iteration
                char_initial_guess = (
                    initial_values['t1_char'],
                    initial_values['t2_char'],
                    initial_values['t3_char'],
                    initial_values['t4_char'],
                    initial_values['t5_char'],
                    initial_values['t6_char'],
                    initial_values['t7_char'],
                    initial_values['t8_char'],
                    initial_values['beta_char_0']
                )
                print("Using initial values from file for charge process")
            elif prev_sol_char is not None:
                # Use previous solution as initial guess for subsequent iterations
                char_initial_guess = prev_sol_char
                print(f"Iteration {i}: Using previous solution as initial guess for charge process")
            else:
                # Default initial guess
                char_initial_guess = (300, 600, 500, 400, 280, 260, 290, 270, 2.0)
                print(f"Iteration {i}: Using default initial guess for charge process")

            # Solve charge process with nsolve and initial guess
            try:
                sol_char = nsolve(eqs_char, (st1_char, st2_char, st3_char, st4_char,
                                             st5_char, st6_char, st7_char, st8_char, sbeta_char),
                                  char_initial_guess)

                # Convert the result to a tuple for future use as initial guess
                prev_sol_char = tuple(float(x) for x in sol_char)

                # Extract individual values
                t1_char, t2_char, t3_char, t4_char, t5_char, t6_char, t7_char, t8_char, beta_char = prev_sol_char

            except Exception as e:
                print(f"nsolve failed for charge process: {e}")
                print("Falling back to solve...")
                # If nsolve fails, fall back to solve
                sol_char = solve(eqs_char, (st1_char, st2_char, st3_char, st4_char,
                                            st5_char, st6_char, st7_char, st8_char, sbeta_char), dict=True)
                sol_char = sol_char[0]
                t1_char = float(sol_char[st1_char])
                t2_char = float(sol_char[st2_char])
                t3_char = float(sol_char[st3_char])
                t4_char = float(sol_char[st4_char])
                t5_char = float(sol_char[st5_char])
                t6_char = float(sol_char[st6_char])
                t7_char = float(sol_char[st7_char])
                t8_char = float(sol_char[st8_char])
                beta_char = float(sol_char[sbeta_char])

                # Update prev_sol_char for future iterations
                prev_sol_char = (t1_char, t2_char, t3_char, t4_char, t5_char, t6_char, t7_char, t8_char, beta_char)

            # Solve the discharge process equations using the candidate beta (beta_dis)
            st1_dis, st2_dis, st3_dis, st4_dis, st5_dis, st6_dis, st7_dis, st9_dis, st12_dis = symbols(
                'st1_dis st2_dis st3_dis st4_dis st5_dis st6_dis st7_dis st9_dis st12_dis', positive=True
            )
            eqs_dis = [
                Eq(st1_dis - st2_dis * (1 - (1 - beta_dis ** -kappa) * eff_exp), 0),
                Eq(st5_dis - st6_dis * (1 + (beta_dis ** kappa - 1) / eff_com), 0),
                Eq(st7_dis - eff_re_dis * st4_dis - (1 - eff_re_dis) * st1_dis, 0),
                Eq(st3_dis - eff_re_dis * st1_dis - (1 - eff_re_dis) * st4_dis, 0),
                Eq(st6_dis - eff_cs_dis * t8_char - (1 - eff_cs_dis) * st7_dis, 0),
                Eq(st9_dis - eff_cs_dis * st7_dis - (1 - eff_cs_dis) * t8_char, 0),
                Eq(st4_dis - eff_hr_dis * t_amb - (1 - eff_hr_dis) * st5_dis, 0),
                Eq(st2_dis - eff_hs_dis * t13 - (1 - eff_hs_dis) * st3_dis, 0),
                Eq(st12_dis - eff_hs_dis * st3_dis - (1 - eff_hs_dis) * t13, 0)
            ]

            # Determine initial guess for discharge process
            if i == 0 and all(k in initial_values for k in ['t1_dis', 't2_dis', 't3_dis', 't4_dis',
                                                            't5_dis', 't6_dis', 't7_dis', 't9_dis', 't12_dis']):
                # Use values from file as initial guess for first iteration
                dis_initial_guess = (
                    initial_values['t1_dis'],
                    initial_values['t2_dis'],
                    initial_values['t3_dis'],
                    initial_values['t4_dis'],
                    initial_values['t5_dis'],
                    initial_values['t6_dis'],
                    initial_values['t7_dis'],
                    initial_values['t9_dis'],
                    initial_values['t12_dis']
                )
                print("Using initial values from file for discharge process")
            elif prev_sol_dis is not None:
                # Use previous solution as initial guess for subsequent iterations
                dis_initial_guess = prev_sol_dis
                print(f"Iteration {i}: Using previous solution as initial guess for discharge process")
            else:
                # Default initial guess
                dis_initial_guess = (500, 600, 550, 350, 300, 280, 320, 290, 700)
                print(f"Iteration {i}: Using default initial guess for discharge process")

            # Solve discharge process with nsolve and initial guess
            try:
                sol_dis = nsolve(eqs_dis, (st1_dis, st2_dis, st3_dis, st4_dis, st5_dis,
                                           st6_dis, st7_dis, st9_dis, st12_dis),
                                 dis_initial_guess)

                # Convert the result to a tuple for future use as initial guess
                prev_sol_dis = tuple(float(x) for x in sol_dis)

                # Extract individual values
                t1_dis, t2_dis, t3_dis, t4_dis, t5_dis, t6_dis, t7_dis, t9_dis, t12_dis = prev_sol_dis

            except Exception as e:
                print(f"nsolve failed for discharge process: {e}")
                print("Falling back to solve...")
                # If nsolve fails, fall back to solve
                sol_dis = solve(eqs_dis, (st1_dis, st2_dis, st3_dis, st4_dis, st5_dis,
                                          st6_dis, st7_dis, st9_dis, st12_dis), dict=True)
                sol_dis = sol_dis[0]
                t1_dis = float(sol_dis[st1_dis])
                t2_dis = float(sol_dis[st2_dis])
                t3_dis = float(sol_dis[st3_dis])
                t4_dis = float(sol_dis[st4_dis])
                t5_dis = float(sol_dis[st5_dis])
                t6_dis = float(sol_dis[st6_dis])
                t7_dis = float(sol_dis[st7_dis])
                t9_dis = float(sol_dis[st9_dis])
                t12_dis = float(sol_dis[st12_dis])

                # Update prev_sol_dis for future iterations
                prev_sol_dis = (t1_dis, t2_dis, t3_dis, t4_dis, t5_dis, t6_dis, t7_dis, t9_dis, t12_dis)

            # Calculate mass flow rates based on power and temperature differences
            m_dis_0 = power_dis_0 * 1e6 / (cp * ((t2_dis - t1_dis) - (t5_dis - t6_dis)) * eff_mg)

            # Since duration_char = duration_dis, we have m_char_0 = m_dis_0
            m_char_0 = m_dis_0

            # Calculate the charge power based on equal flow rates (to ensure equal duration)
            power_char_0 = m_char_0 * cp * ((t2_char - t1_char) - (t5_char - t6_char)) / (eff_mg * 1e6)

            # Calculate errors for convergence check
            temp_error = max(abs(t12_dis - t12_old), abs(t9_dis - t9_old))

            # Update values for next iteration
            t12_old = t12_dis
            t9_old = t9_dis

            if temp_error < tol:
                print(f"Converged after {i + 1} iterations")
                break

        if i == iter_max - 1:
            print(f"Warning: Maximum iterations ({iter_max}) reached without convergence. Final error: {temp_error}")

        # Calculate RTE from charge and discharge processes
        rte = ((t2_dis - t1_dis) - (t5_dis - t6_dis)) / ((t2_char - t1_char) - (t5_char - t6_char)) * eff_mg ** 2

        # Store the designed parameters in the dictionary - basic system parameters
        design_params["power_char_0"] = power_char_0
        design_params["power_dis_0"] = power_dis_0
        design_params["duration"] = duration
        design_params["beta_char_0"] = beta_char
        design_params["beta_dis_0"] = beta_dis

        # Store all charge process temperatures
        design_params["t1_char_0"] = t1_char
        design_params["t2_char_0"] = t2_char
        design_params["t3_char_0"] = t3_char
        design_params["t4_char_0"] = t4_char
        design_params["t5_char_0"] = t5_char
        design_params["t6_char_0"] = t6_char
        design_params["t7_char_0"] = t7_char
        design_params["t8_char_0"] = t8_char

        # Store all discharge process temperatures
        design_params["t1_dis_0"] = t1_dis
        design_params["t2_dis_0"] = t2_dis
        design_params["t3_dis_0"] = t3_dis
        design_params["t4_dis_0"] = t4_dis
        design_params["t5_dis_0"] = t5_dis
        design_params["t6_dis_0"] = t6_dis
        design_params["t7_dis_0"] = t7_dis
        design_params["t9_dis_0"] = t9_dis
        design_params["t12_dis_0"] = t12_dis

        # Original interface parameters (for backward compatibility)
        design_params["t_com_char_0"] = t1_char
        design_params["t_exp_char_0"] = t5_char
        design_params["t_com_dis_0"] = t6_dis
        design_params["t_exp_dis_0"] = t2_dis
        design_params["p_com_char_0"] = 1 / beta_char
        design_params["p_exp_char_0"] = 1  # Fixed value
        design_params["p_com_dis_0"] = 1 / beta_dis
        design_params["p_exp_dis_0"] = 1  # Fixed value
        design_params["t_hh_0"] = t13
        design_params["t_lh_0"] = t12_dis
        design_params["t_hc_0"] = t9_dis
        design_params["t_lc_0"] = t8_char

        # Store the calculated mass flow rates
        design_params["m_0"] = m_char_0
        design_params["mass_max"] = m_dis_0 * duration * 3600
        design_params["n_0"] = 1  # Fixed value
        design_params["rte_0"] = rte

        # For backward compatibility
        design_params["eff_hs"] = eff_hs_dis
        design_params["eff_re"] = eff_re_dis
        design_params["eff_hr"] = eff_hr_dis
        design_params["eff_cs"] = eff_cs_dis

        return -rte

    # Optimize beta_dis, using initial value from file if available
    beta_dis_initial = initial_values.get('beta_dis_0', 2.0)
    print(f"Starting optimization with initial beta_dis = {beta_dis_initial}")

    # Set tighter bounds around the initial value if available
    if 'beta_dis_0' in initial_values:
        lower_bound = max(1.5, beta_dis_initial * 0.9)  # Ensure at least 1.5
        upper_bound = min(2.5, beta_dis_initial * 1.1)  # Ensure at most 2.5
    else:
        lower_bound = 1.5
        upper_bound = 2.5

    res = minimize_scalar(obj, bounds=(lower_bound, upper_bound), method='bounded')
    optimal_beta_dis = res.x

    # Run the objective function one more time with the optimal beta_dis to ensure the design_params are set
    obj(optimal_beta_dis)

    # Save the designed parameters to a CSV file
    design_df = pd.DataFrame([design_params])
    csv_filename = f"ptes_design_{power_0}MW_{duration}h.csv"
    design_df.to_csv(csv_filename, index=False)

    print(f"Design optimization completed. Parameters saved to {csv_filename}")
    print(f"Optimal RTE: {design_params['rte_0']:.4f}")
    print(
        f"Charge power: {design_params['power_char_0']:.2f} MW, Discharge power: {design_params['power_dis_0']:.2f} MW")
    print(
        f"Charge mass flow rate: {design_params['m_0']:.2f} kg/s, Discharge mass flow rate: {design_params['m_0']:.2f} kg/s")
    print(
        f"Charge temperatures (t1-t8): {design_params['t1_char_0']:.2f}, {design_params['t2_char_0']:.2f}, {design_params['t3_char_0']:.2f}, {design_params['t4_char_0']:.2f}, {design_params['t5_char_0']:.2f}, {design_params['t6_char_0']:.2f}, {design_params['t7_char_0']:.2f}, {design_params['t8_char_0']:.2f}")
    print(
        f"Discharge temperatures (t1-t7,t9,t12): {design_params['t1_dis_0']:.2f}, {design_params['t2_dis_0']:.2f}, {design_params['t3_dis_0']:.2f}, {design_params['t4_dis_0']:.2f}, {design_params['t5_dis_0']:.2f}, {design_params['t6_dis_0']:.2f}, {design_params['t7_dis_0']:.2f}, {design_params['t9_dis_0']:.2f}, {design_params['t12_dis_0']:.2f}")

    return design_params


if __name__ == "__main__":
    # Example usage
    # Solve for a 300 MW / 10 hour PTES system
    params = solve_ptes_design(236.87, 2000/236.87)