import pandas as pd
import numpy as np
from pyomo.environ import *
import matplotlib.pyplot as plt
from ptes_model_mt import PTES_mt
from constants import *
import time
import os

# Constants for the optimization
LEN_HORIZON = 48  # 2-day lookahead window
k_charge = 329.10961578240665  # Factor for converting mass flow to charge power
k_discharge = 230.88  # Factor for converting mass flow to discharge power
MAX_DURATION = 8.662508662508662  # Maximum storage duration (hours)
MFR_LOWER = 0.1  # Minimum mass flow rate (normalized)
MFR_UPPER = 1.0  # Maximum mass flow rate (normalized)


def build_milp_model(price_data, M_0=0, is_final_day=False, target_final_mass=MAX_DURATION/2):
    """
    Build a MILP model for ESS operation

    Args:
        price_data: DataFrame with price data
        M_0: Initial mass in the storage system

    Returns:
        model: Pyomo concrete model
    """
    model = ConcreteModel()

    # Sets
    model.times = Set(initialize=price_data.index.values)

    # Parameters
    model.pLMP = Param(model.times, initialize=price_data['LMP ($/MWh)'].to_dict())
    model.pMaxDuration = Param(initialize=MAX_DURATION)
    model.pMFRUpper = Param(initialize=MFR_UPPER)
    model.pMFRLower = Param(initialize=MFR_LOWER)
    model.pM0 = Param(initialize=M_0, domain=Any)

    # Variables
    model.vCharge = Var(model.times, within=NonNegativeReals)
    model.vDischarge = Var(model.times, within=NonNegativeReals)
    model.vZcop = Var(model.times, domain=Binary)
    model.vZdop = Var(model.times, domain=Binary)
    model.vMFRopc = Var(model.times, within=NonNegativeReals)
    model.vMFRopd = Var(model.times, within=NonNegativeReals)
    model.vM = Var(model.times, within=NonNegativeReals)

    # Objective function - maximize arbitrage profit
    def objFunc(model):
        return sum((model.vDischarge[t] - model.vCharge[t]) * model.pLMP[t] for t in model.times)

    model.profit = Objective(rule=objFunc, sense=maximize)

    # Constraints
    def ptesCCop(model, t):
        return model.vCharge[t] == k_charge * model.vMFRopc[t]

    model.cop = Constraint(model.times, rule=ptesCCop)

    def ptesDCop(model, t):
        return model.vDischarge[t] == k_discharge * model.vMFRopd[t]

    model.dop = Constraint(model.times, rule=ptesDCop)

    def soc(model, t):
        if t == model.times.first():
            return model.vM[t] == model.pM0 + model.vMFRopc[t] - model.vMFRopd[t]
        else:
            prev_index = model.times.prev(t)
            return model.vM[t] == model.vM[prev_index] + model.vMFRopc[t] - model.vMFRopd[t]

    model.ptesSOC = Constraint(model.times, rule=soc)

    def maxstor(model, t):
        return model.vM[t] <= model.pMaxDuration

    model.ptesMax = Constraint(model.times, rule=maxstor)

    def chargeLow(model, t):
        return model.vMFRopc[t] >= model.pMFRLower * model.vZcop[t]

    model.chargeLowConstraint = Constraint(model.times, rule=chargeLow)

    def chargeUpper(model, t):
        return model.vMFRopc[t] <= model.pMFRUpper * model.vZcop[t]

    model.chargeUpperConstraint = Constraint(model.times, rule=chargeUpper)

    def dischargeLower(model, t):
        return model.vMFRopd[t] >= model.pMFRLower * model.vZdop[t]

    model.dischargeLowerConstraint = Constraint(model.times, rule=dischargeLower)

    def dischargeUpper(model, t):
        return model.vMFRopd[t] <= model.pMFRUpper * model.vZdop[t]

    model.dischargeConstraint = Constraint(model.times, rule=dischargeUpper)

    def onePhase(model, t):
        return model.vZcop[t] + model.vZdop[t] <= 1

    model.op = Constraint(model.times, rule=onePhase)

    if is_final_day:
        # Find the time index corresponding to hour 24 (end of first day)
        day_end_time = list(model.times)[23]  # 0-indexed, so 23 is the 24th hour
        model.final_mass_constraint = Constraint(
            expr=model.vM[day_end_time] == target_final_mass
        )

    return model


def milp_to_operation_params(milp_model, horizon_length):
    """
    Convert MILP solution to operation parameters

    Args:
        milp_model: Solved MILP model
        horizon_length: Length of the horizon (hours)

    Returns:
        modes: List of modes (-1: charge, 0: idle, 1: discharge)
        m_norms: List of mass flow rates
        beta_norms: List of beta values (all set to 1.0)
    """
    modes = []
    m_norms = []
    beta_norms = []

    for t in milp_model.times:
        if t >= horizon_length:
            break

        if milp_model.vZcop[t].value > 0.5:
            modes.append(-1)  # Charge
        elif milp_model.vZdop[t].value > 0.5:
            modes.append(1)  # Discharge
        else:
            modes.append(0)  # Idle

        if modes[-1] == -1:  # Charge
            m_norms.append(milp_model.vMFRopc[t].value)
            beta_norms.append(1.0)  # Fixed beta for charging
        elif modes[-1] == 1:  # Discharge
            m_norms.append(milp_model.vMFRopd[t].value)
            beta_norms.append(1.0)  # Fixed beta for discharging
        else:  # Idle
            m_norms.append(0.0)
            beta_norms.append(0.0)

    return modes, m_norms, beta_norms


def compute_one_horizon(modes, m_norms, beta_norms, mass_norm, store_temps, lmps, ptes):
    """
    Compute operation for one horizon using PTES physical model

    Args:
        modes: List of operation modes (-1: charge, 0: idle, 1: discharge)
        m_norms: List of normalized mass flow rates
        beta_norms: List of normalized beta values
        mass_norm: Initial mass normalized (0-1)
        store_temps: Initial store temperatures
        lmps: Locational marginal prices for the horizon
        ptes: PTES model instance

    Returns:
        df: DataFrame with operation results
        current_mass_norm: Updated mass normalization
        current_store_temps: Updated store temperatures
        total_profit: Total profit for the horizon
        total_penalty: Total penalty for the horizon
    """
    # Initialize data structure
    data = {
        'mode': [],
        'm_norm': [],
        'beta_norm': [],
        'power': [],
        't_hh': [],
        't_lh': [],
        't_hc': [],
        't_lc': [],
        'hourly_profit': [],
        'eff_com': [],
        'eff_exp': [],
        'mass_norm': [],
        'vio_choke': [],
        'vio_surge': [],
        'vio_beta': []
    }

    # Initialize state variables
    current_mass_norm = mass_norm
    current_store_temps = store_temps.copy()

    # Initialize profit and penalty trackers
    total_profit = 0
    total_penalty = 0

    # Compute each hour
    for i in range(len(modes)):
        mode = modes[i]
        if mode == -1:
            phase = "char"
        elif mode == 1:
            phase = "dis"
        else:
            phase = "idle"
            # Ensure m_norm and beta_norm are 0 for idle mode
            m_norms[i] = 0.0
            beta_norms[i] = 0.0

        m_norm = m_norms[i]
        beta_norm = beta_norms[i]
        lmp = lmps[i]

        # Check constraints and adjust if necessary
        if phase == "char" and current_mass_norm + m_norm > MAX_DURATION:
            # Adjust m_norm to not exceed capacity
            m_norm = max(0, min(MAX_DURATION - current_mass_norm, m_norm))
            m_norms[i] = m_norm

            if m_norm < MFR_LOWER:
                phase = "idle"
                m_norm = 0.0
                beta_norm = 0.0
                m_norms[i] = 0.0
                beta_norms[i] = 0.0

        if phase == "dis" and current_mass_norm - m_norm < 0.0:
            # Adjust m_norm to not go below empty
            m_norm = max(0, min(current_mass_norm, m_norm))
            m_norms[i] = m_norm

            if m_norm < MFR_LOWER:
                phase = "idle"
                m_norm = 0.0
                beta_norm = 0.0
                m_norms[i] = 0.0
                beta_norms[i] = 0.0

        # Initialize values
        power = 0
        eff_com = np.nan
        eff_exp = np.nan
        vio_choke = 0
        vio_surge = 0
        vio_beta = 0
        hourly_profit = 0

        # Calculate power and update state using PTES model
        if phase == "char" or phase == "dis":

            # Call PTES model to calculate performance
            perform_results = ptes.calculate_performance(
                phase=phase,
                m_norm=m_norm,
                current_temps=current_store_temps
            )

            # Extract results
            power = perform_results['power']
            beta_actual = perform_results['beta']
            beta_norm = beta_actual / ptes.beta_char_0 if phase == "char" else beta_actual / ptes.beta_dis_0
            eff_com = perform_results['eff_com']
            eff_exp = perform_results['eff_exp']
            vio_choke = perform_results.get('vio_choke', 0)
            vio_surge = perform_results.get('vio_surge', 0)
            vio_beta = perform_results.get('vio_beta', 0)

            # Update mass and temperatures based on operation mode
            if phase == "char":
                # Update temperatures for charging
                t13 = perform_results["temperatures"]["t13"]
                t8 = perform_results["temperatures"]["t8"]

                # Calculate new mass
                mass_norm_new = current_mass_norm + m_norm

                # Update hot high and low cold temperatures
                current_store_temps['t_hh'] = (current_store_temps['t_hh'] * current_mass_norm +
                                               t13 * m_norm) / mass_norm_new
                current_store_temps['t_lc'] = (current_store_temps['t_lc'] * current_mass_norm +
                                               t8 * m_norm) / mass_norm_new

                # Update mass
                current_mass_norm = mass_norm_new

                # Calculate profit and penalty
                penalty = 100 * penalty_factor * (vio_choke + vio_surge + vio_beta)
                hourly_profit = power * lmp

                # Update total profit and penalty
                total_profit += hourly_profit
                total_penalty += penalty

            else:  # discharge
                # Update temperatures for discharging
                t12 = perform_results["temperatures"]["t12"]
                t9 = perform_results["temperatures"]["t9"]

                # Calculate new mass
                mass_norm_new = current_mass_norm - m_norm

                # Update low hot and high cold temperatures
                if mass_norm_new < 1e-6:  # Avoid division by zero
                    current_store_temps['t_lh'] = current_store_temps['t_lh']
                    current_store_temps['t_hc'] = current_store_temps['t_hc']
                else:
                    current_store_temps['t_lh'] = (current_store_temps['t_lh'] * (MAX_DURATION - current_mass_norm) +
                                                   t12 * m_norm) / (MAX_DURATION - mass_norm_new)
                    current_store_temps['t_hc'] = (current_store_temps['t_hc'] * (MAX_DURATION - current_mass_norm) +
                                                   t9 * m_norm) / (MAX_DURATION - mass_norm_new)

                # Update mass
                current_mass_norm = mass_norm_new

                # Calculate profit and penalty
                penalty = 100 * penalty_factor * (vio_choke + vio_surge + vio_beta)
                hourly_profit = power * lmp

                # Update total profit and penalty
                total_profit += hourly_profit
                total_penalty += penalty

        # Apply heat losses to all thermal storage components
        current_store_temps['t_hh'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_hh']
        current_store_temps['t_lh'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_lh']
        current_store_temps['t_hc'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_hc']
        current_store_temps['t_lc'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_lc']

        # Store data for this time step
        data['mode'].append(phase)
        data['m_norm'].append(m_norm)
        data['beta_norm'].append(beta_norm)
        data['mass_norm'].append(current_mass_norm)
        data['power'].append(power)
        data['t_hh'].append(current_store_temps["t_hh"])
        data['t_lh'].append(current_store_temps["t_lh"])
        data['t_hc'].append(current_store_temps["t_hc"])
        data['t_lc'].append(current_store_temps["t_lc"])
        data['hourly_profit'].append(hourly_profit)
        data['eff_com'].append(eff_com)
        data['eff_exp'].append(eff_exp)
        data['vio_choke'].append(vio_choke)
        data['vio_surge'].append(vio_surge)
        data['vio_beta'].append(vio_beta)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Return dataframe, updated state, and profit/penalty
    return df, current_mass_norm, current_store_temps, total_profit, total_penalty


def rolling_optimization_milp(price_data, initial_mass_norm=MAX_DURATION/2, days=7, output_dir="./results_milp/"):
    """
    Perform rolling optimization for specified number of days using pure MILP approach

    Args:
        price_data: DataFrame with price data
        initial_mass_norm: Initial mass normalized (0-1)
        days: Number of days to optimize
        output_dir: Directory to save detailed results

    Returns:
        results_df: DataFrame with combined results
        daily_summary_df: DataFrame with daily summary statistics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize state variables
    current_mass_norm = initial_mass_norm

    # Initialize PTES model
    ptes = PTES_mt(design_file='ptes_design_230.88MW_8.662508662508662h.csv')

    # Initialize store temperatures
    current_store_temps = {
        "t_hh": ptes.t_hh_0,
        "t_lh": ptes.t_lh_0,
        "t_hc": ptes.t_hc_0,
        "t_lc": ptes.t_lc_0
    }

    # List to store results for each day
    all_results = []

    # Dictionary to store daily detailed information
    daily_details = {
        'day': [],
        'date': [],
        'initial_mass': [],
        'final_mass': [],
        'initial_t_hh': [],
        'initial_t_lh': [],
        'initial_t_hc': [],
        'initial_t_lc': [],
        'final_t_hh': [],
        'final_t_lh': [],
        'final_t_hc': [],
        'final_t_lc': [],
        'total_charge_power': [],
        'total_discharge_power': [],
        'charge_hours': [],
        'discharge_hours': [],
        'idle_hours': [],
        'daily_profit': [],
        'daily_penalty': [],
        'daily_net_profit': [],
        'avg_charge_power': [],
        'avg_discharge_power': [],
        'max_violations': [],
        'avg_eff_com': [],
        'avg_eff_exp': [],
        'avg_lmp': [],
        'max_lmp': [],
        'min_lmp': [],
        'optimization_time': []
    }

    # For each day
    for day in range(days):

        is_final_day = (day == days - 1)

        print(f"\nOptimizing day {day + 1}/{days}")
        day_start_time = time.time()

        # Calculate date (assuming price_data has a datetime index)
        if hasattr(price_data.index, 'date'):
            current_date = price_data.index[day * 24].date()
        else:
            current_date = f"Day_{day + 1}"

        # Get price data for the lookahead window (48 hours)
        start_hour = day * 24
        end_hour = start_hour + LEN_HORIZON

        if end_hour > len(price_data):
            print(f"Warning: Not enough price data for day {day + 1}. Stopping.")
            break

        window_prices = price_data.iloc[start_hour:end_hour]
        lmps = window_prices["LMP ($/MWh)"].values

        # Store initial state for daily record
        initial_mass = current_mass_norm
        initial_t_hh = current_store_temps['t_hh']
        initial_t_lh = current_store_temps['t_lh']
        initial_t_hc = current_store_temps['t_hc']
        initial_t_lc = current_store_temps['t_lc']

        print(f"Optimizing from hour {start_hour} to {start_hour + LEN_HORIZON - 1}")
        print(f"Current mass: {current_mass_norm:.4f}")
        print(f"Store temps: t_hh={current_store_temps['t_hh']:.1f}, t_lh={current_store_temps['t_lh']:.1f}, "
              f"t_hc={current_store_temps['t_hc']:.1f}, t_lc={current_store_temps['t_lc']:.1f}")

        # Step 1: Create price data DataFrame for the window
        price_data_window = pd.DataFrame({
            'LMP ($/MWh)': lmps
        })

        # Step 2: Solve the MILP model for the 48-hour horizon
        print("Solving MILP model...")
        milp_model = build_milp_model(price_data_window, current_mass_norm,
                                      is_final_day=is_final_day, target_final_mass=MAX_DURATION/2)

        solver = SolverFactory('gurobi')
        results = solver.solve(milp_model)

        print(f"MILP solution found. Objective value: ${milp_model.profit():.2f}")

        # Convert MILP solution to operation parameters
        modes, m_norms, beta_norms = milp_to_operation_params(milp_model, len(lmps))

        # Get the actual operation results for the first day only
        modes_first_day = modes[:24]
        m_norms_first_day = m_norms[:24]
        beta_norms_first_day = beta_norms[:24]

        df, current_mass_norm, current_store_temps, total_profit, total_penalty = compute_one_horizon(
            modes_first_day,
            m_norms_first_day,
            beta_norms_first_day,
            initial_mass,
            {
                't_hh': initial_t_hh,
                't_lh': initial_t_lh,
                't_hc': initial_t_hc,
                't_lc': initial_t_lc
            },
            lmps[:24],
            ptes
        )

        # Calculate daily statistics
        day_data = df
        charge_hours = len(day_data[day_data['mode'] == 'char'])
        discharge_hours = len(day_data[day_data['mode'] == 'dis'])
        idle_hours = len(day_data[day_data['mode'] == 'idle'])

        total_charge_power = -day_data[day_data['mode'] == 'char']['power'].sum()
        total_discharge_power = day_data[day_data['mode'] == 'dis']['power'].sum()

        avg_charge_power = -day_data[day_data['mode'] == 'char']['power'].mean() if charge_hours > 0 else 0
        avg_discharge_power = day_data[day_data['mode'] == 'dis']['power'].mean() if discharge_hours > 0 else 0

        daily_profit = day_data['hourly_profit'].sum()
        daily_penalty = 100 * penalty_factor * (
                day_data['vio_choke'].sum() +
                day_data['vio_surge'].sum() +
                day_data['vio_beta'].sum()
        )
        daily_net_profit = daily_profit - daily_penalty

        max_violations = max(
            day_data['vio_choke'].max() if not day_data['vio_choke'].empty else 0,
            day_data['vio_surge'].max() if not day_data['vio_surge'].empty else 0,
            day_data['vio_beta'].max() if not day_data['vio_beta'].empty else 0
        )

        avg_eff_com = day_data[day_data['eff_com'].notna()]['eff_com'].mean() if not day_data[
            day_data['eff_com'].notna()].empty else np.nan
        avg_eff_exp = day_data[day_data['eff_exp'].notna()]['eff_exp'].mean() if not day_data[
            day_data['eff_exp'].notna()].empty else np.nan

        avg_lmp = lmps[:24].mean()
        max_lmp = lmps[:24].max()
        min_lmp = lmps[:24].min()

        optimization_time = time.time() - day_start_time

        # Store daily statistics
        daily_details['day'].append(day + 1)
        daily_details['date'].append(current_date)
        daily_details['initial_mass'].append(initial_mass)
        daily_details['final_mass'].append(current_mass_norm)
        daily_details['initial_t_hh'].append(initial_t_hh)
        daily_details['initial_t_lh'].append(initial_t_lh)
        daily_details['initial_t_hc'].append(initial_t_hc)
        daily_details['initial_t_lc'].append(initial_t_lc)
        daily_details['final_t_hh'].append(current_store_temps['t_hh'])
        daily_details['final_t_lh'].append(current_store_temps['t_lh'])
        daily_details['final_t_hc'].append(current_store_temps['t_hc'])
        daily_details['final_t_lc'].append(current_store_temps['t_lc'])
        daily_details['total_charge_power'].append(total_charge_power)
        daily_details['total_discharge_power'].append(total_discharge_power)
        daily_details['charge_hours'].append(charge_hours)
        daily_details['discharge_hours'].append(discharge_hours)
        daily_details['idle_hours'].append(idle_hours)
        daily_details['daily_profit'].append(daily_profit)
        daily_details['daily_penalty'].append(daily_penalty)
        daily_details['daily_net_profit'].append(daily_net_profit)
        daily_details['avg_charge_power'].append(avg_charge_power)
        daily_details['avg_discharge_power'].append(avg_discharge_power)
        daily_details['max_violations'].append(max_violations)
        daily_details['avg_eff_com'].append(avg_eff_com)
        daily_details['avg_eff_exp'].append(avg_eff_exp)
        daily_details['avg_lmp'].append(avg_lmp)
        daily_details['max_lmp'].append(max_lmp)
        daily_details['min_lmp'].append(min_lmp)
        daily_details['optimization_time'].append(optimization_time)

        # Add day information to hourly data
        df["day"] = day + 1
        df["date"] = current_date
        df["hour"] = range(start_hour, start_hour + 24)
        df["global_hour"] = range(start_hour, start_hour + 24)
        df["hour_of_day"] = range(24)

        # Save detailed hourly data for this day
        df.to_csv(f"{output_dir}/day_{day + 1}_hourly_details.csv", index=False)

        # Append results to all_results list
        all_results.append(df)

        print(f"Day {day + 1} completed. New mass: {current_mass_norm:.4f}")
        print(f"New temps: t_hh={current_store_temps['t_hh']:.1f}, t_lh={current_store_temps['t_lh']:.1f}, "
              f"t_hc={current_store_temps['t_hc']:.1f}, t_lc={current_store_temps['t_lc']:.1f}")
        print(f"Daily profit: ${daily_net_profit:.2f}, Optimization time: {optimization_time:.2f}s")

    # Save daily summary statistics
    daily_summary_df = pd.DataFrame(daily_details)
    daily_summary_df.to_csv(f"{output_dir}/daily_summary.csv", index=False)

    # Combine all hourly results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(f"{output_dir}/complete_hourly_details.csv", index=False)
    else:
        results_df = pd.DataFrame()

    return results_df, daily_summary_df


def load_price_data(csv_file="CAISO2022.csv"):
    """
    Load price data from CSV file

    Args:
        csv_file: Path to CSV file with price data

    Returns:
        df: DataFrame with price data
    """
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded price data from {csv_file}")
        print(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None


def analyze_results(results, price_data=None):
    """
    Analyze optimization results in detail with revenue potential calculation

    Args:
        results: DataFrame with optimization results
        price_data: DataFrame with price data (optional, for LMP analysis)

    Returns:
        analysis_dict: Dictionary with analysis results
    """
    # Check if results DataFrame is not empty
    if results.empty:
        return {
            "total_reward": 0,
            "total_reward_potential": 0,
            "avg_daily_reward": 0,
            "charge_hours": 0,
            "discharge_hours": 0,
            "idle_hours": 0,
            "avg_charge_power": 0,
            "avg_discharge_power": 0,
            "avg_charge_power_per_mflow": 0,
            "avg_discharge_power_per_mflow": 0,
            "avg_discharge_lmp": 0,
            "remaining_mass_norm": 0,
            "additional_revenue_potential": 0,
            "total_violations": 0,
            "violation_penalty": 0,
            "profit_without_penalty": 0,
            "temperature_stats": {
                "max_t_hh": 0,
                "min_t_lh": 0,
                "max_t_hc": 0,
                "min_t_lc": 0
            }
        }

    # Calculate basic statistics
    total_reward = results["hourly_profit"].sum()
    days_completed = max(results["day"]) if "day" in results.columns and not results["day"].empty else 1
    avg_daily_reward = total_reward / days_completed

    # Calculate operational statistics
    charge_hours = len(results[results["mode"] == "char"])
    discharge_hours = len(results[results["mode"] == "dis"])
    idle_hours = len(results[results["mode"] == "idle"])

    # Calculate average power
    avg_charge_power = -results[results["mode"] == "char"]["power"].mean() if charge_hours > 0 else 0
    avg_discharge_power = results[results["mode"] == "dis"]["power"].mean() if discharge_hours > 0 else 0

    # Calculate average power/m_norm ratios
    charge_df = results[results["mode"] == "char"].copy() if charge_hours > 0 else pd.DataFrame()
    discharge_df = results[results["mode"] == "dis"].copy() if discharge_hours > 0 else pd.DataFrame()

    avg_charge_power_per_mflow = 0
    if not charge_df.empty:
        # For charging, power is negative, so we use absolute value
        charge_df.loc[:, 'power_per_mflow'] = abs(charge_df["power"]) / charge_df["m_norm"]
        avg_charge_power_per_mflow = charge_df['power_per_mflow'].mean()

    avg_discharge_power_per_mflow = 0
    if not discharge_df.empty:
        discharge_df.loc[:, 'power_per_mflow'] = discharge_df["power"] / discharge_df["m_norm"]
        avg_discharge_power_per_mflow = discharge_df['power_per_mflow'].mean()

    # Calculate average discharge LMP
    avg_discharge_lmp = 0
    if price_data is not None and not discharge_df.empty:
        discharge_hours_list = discharge_df["hour"].values
        discharge_lmps = [price_data.iloc[h]["LMP ($/MWh)"] if h < len(price_data) else 0 for h in discharge_hours_list]
        avg_discharge_lmp = sum(discharge_lmps) / len(discharge_lmps) if len(discharge_lmps) > 0 else 0

    # Get the remaining mass at the end of simulation
    remaining_mass_norm = results["mass_norm"].iloc[-1] if not results.empty else 0

    # Calculate potential additional revenue from remaining mass
    additional_revenue_potential = 0
    if remaining_mass_norm > 0 and avg_discharge_power_per_mflow > 0 and avg_discharge_lmp > 0:
        # Calculate how much power could be generated from remaining mass
        potential_power = remaining_mass_norm * avg_discharge_power_per_mflow
        # Calculate potential revenue
        additional_revenue_potential = potential_power * avg_discharge_lmp

    # Calculate total reward potential
    total_reward_potential = total_reward + additional_revenue_potential

    # Calculate violations statistics
    total_violations = (results["vio_choke"] + results["vio_surge"] + results["vio_beta"]).sum()
    violation_penalty = 100 * penalty_factor * total_violations
    profit_without_penalty = total_reward + violation_penalty

    # Create analysis dictionary
    analysis = {
        "total_reward": total_reward,
        "total_reward_potential": total_reward_potential,
        "avg_daily_reward": avg_daily_reward,
        "charge_hours": charge_hours,
        "discharge_hours": discharge_hours,
        "idle_hours": idle_hours,
        "avg_charge_power": avg_charge_power,
        "avg_discharge_power": avg_discharge_power,
        "avg_charge_power_per_mflow": avg_charge_power_per_mflow,
        "avg_discharge_power_per_mflow": avg_discharge_power_per_mflow,
        "avg_discharge_lmp": avg_discharge_lmp,
        "remaining_mass_norm": remaining_mass_norm,
        "additional_revenue_potential": additional_revenue_potential,
        "total_violations": total_violations,
        "violation_penalty": violation_penalty,
        "profit_without_penalty": profit_without_penalty,
        "temperature_stats": {
            "max_t_hh": results["t_hh"].max(),
            "min_t_lh": results["t_lh"].min(),
            "max_t_hc": results["t_hc"].max(),
            "min_t_lc": results["t_lc"].min()
        }
    }

    return analysis


def plot_results(days, results, price_data=None, output_dir="./results_milp_pricetaker"):
    """
    Plot results of the optimization with larger fonts for better readability

    Args:
        results: DataFrame with optimization results
        price_data: DataFrame with price data
        output_dir: Directory to save plots
    """
    if results.empty:
        print("No results to plot.")
        return

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'lines.antialiased': True,
        'patch.antialiased': True
    })

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

    # Plot 1: Storage Mass Level
    axs[0].plot(results["hour"], results["mass_norm"], 'b-', linewidth=3)
    axs[0].set_title("Storage Mass Level")
    axs[0].set_ylabel("Normalized Mass")
    axs[0].grid(True)
    axs[0].set_ylim(0, 10)  # Set y-axis range to 0-10
    axs[0].tick_params(axis='both', which='major', labelsize=14)

    # Plot 2: Power Output and LMP prices
    ax1 = axs[1]
    ax1.bar(results["hour"], results["power"], width=0.8, color='skyblue', alpha=0.7)
    ax1.set_ylabel("Power (MW)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=14)
    ax1.set_title("Power Output and LMP Prices")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-400, 400)

    # Add nominal power lines
    nominal_charge = -285.0914897629995  # Negative for charging
    nominal_discharge = 200
    ax1.axhline(y=nominal_charge, color='blue', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Nominal Charge Power: {nominal_charge:.2f} MW')
    ax1.axhline(y=nominal_discharge, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label=f'Nominal Discharge Power: {nominal_discharge:.2f} MW')
    ax1.legend(loc='upper left', framealpha=0.9)

    # Add LMP prices on a secondary axis if available
    if price_data is not None:
        ax2 = ax1.twinx()
        hours = results["hour"].values
        lmps = [price_data.iloc[h]["LMP ($/MWh)"] if h < len(price_data) else np.nan for h in hours]
        ax2.plot(hours, lmps, 'r-', linewidth=2, alpha=0.7)
        ax2.set_ylabel("LMP ($/MWh)", color='red')
        ax2.tick_params(axis='y', labelcolor='red', labelsize=14)
        ax2.set_ylim(0, 270)

    # Plot 3: Temperatures
    axs[2].plot(results["hour"], results["t_hh"], 'r-', label='T_hh (Hot High)', linewidth=3)
    axs[2].plot(results["hour"], results["t_lh"], 'r--', label='T_lh (Low High)', linewidth=3)
    axs[2].plot(results["hour"], results["t_hc"], 'b-', label='T_hc (High Cold)', linewidth=3)
    axs[2].plot(results["hour"], results["t_lc"], 'b--', label='T_lc (Low Cold)', linewidth=3)
    axs[2].set_title("Storage Temperatures")
    axs[2].set_ylabel("Temperature (K)")
    axs[2].set_xlabel("Hour", fontsize=16)
    axs[2].legend(loc='best', framealpha=0.9)
    axs[2].grid(True)
    axs[2].set_ylim(220, 1020)  # Set y-axis range to 220-1020
    axs[2].tick_params(axis='both', which='major', labelsize=14)

    # Set x-axis range to 0-168 (7 days) for all subplots
    for ax in axs:
        ax.set_xlim(0, days * 24)

    # Set custom x-tick positions and labels at 24-hour intervals
    xtick_positions = range(0, days * 24, 24)  # 0, 24, 48, ..., 168
    axs[2].set_xticks(xtick_positions)
    axs[2].set_xticklabels([str(x) for x in xtick_positions])

    # Add day separators
    for day in range(days):
        for ax in axs:
            ax.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f"{output_dir}/milp_pricetaker.png", dpi=600, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run the optimization
    """
    # Create results directory
    results_dir = "./results_milp_pricetaker_230.88MW"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load price data
    price_data = load_price_data("representative_week.csv")

    if price_data is None:
        print("Failed to load price data. Exiting.")
        return

    # Run rolling optimization
    print("\nStarting rolling optimization using pure MILP approach...")
    start_time = time.time()
    days = 7
    # Use updated function with results directory
    results, daily_summary = rolling_optimization_milp(
        price_data,
        initial_mass_norm=MAX_DURATION/2,
        days=days,
        output_dir=results_dir
    )

    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)")

    # Call plot_results function here with the results and price data
    plot_results(days, results, price_data, results_dir)

    # Analyze and display results
    analysis = analyze_results(results, price_data)

    # Save summary analysis to CSV
    summary_dict = {
        'metric': [
            'total_reward', 'total_reward_potential', 'avg_daily_reward', 'charge_hours', 'discharge_hours', 'idle_hours',
            'avg_charge_power', 'avg_discharge_power', 'avg_charge_power_per_mflow', 'avg_discharge_power_per_mflow',
            'violation_penalty', 'profit_without_penalty', 'total_violations',
            'max_t_hh', 'min_t_lh', 'max_t_hc', 'min_t_lc'
        ],
        'value': [
            analysis['total_reward'], analysis['total_reward_potential'], analysis['avg_daily_reward'],
            analysis['charge_hours'], analysis['discharge_hours'], analysis['idle_hours'],
            analysis['avg_charge_power'], analysis['avg_discharge_power'],
            analysis['avg_charge_power_per_mflow'], analysis['avg_discharge_power_per_mflow'],
            analysis['violation_penalty'], analysis['profit_without_penalty'], analysis['total_violations'],
            analysis['temperature_stats']['max_t_hh'], analysis['temperature_stats']['min_t_lh'],
            analysis['temperature_stats']['max_t_hc'], analysis['temperature_stats']['min_t_lc']
        ]
    }
    pd.DataFrame(summary_dict).to_csv(f"{results_dir}/analysis_summary.csv", index=False)


if __name__ == "__main__":
    main()