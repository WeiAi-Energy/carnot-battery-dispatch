"""
SYSTEM OPERATOR OPTIMIZATION WITH PHYSICAL PTES MODEL INTEGRATION

This module performs system operator dispatch optimization with PTES integration:
1. Dispatch all generators and PTES using MILP optimization
2. Apply physical PTES model to get accurate performance
3. Adjust conventional generators to handle power mismatches
4. Roll forward with updated system state
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import *
import time
import os
from ptes_model_mt import PTES_mt
from constants import *

k_charge = 337.64810590080845  # Factor for converting mass flow to charge power
k_discharge = 236.87  # Factor for converting mass flow to discharge power
MAX_DURATION = 8.443449993667413  # Maximum storage duration (hours)
MFR_LOWER = 0.1  # Minimum mass flow rate (normalized)
MFR_UPPER = 1.0  # Maximum mass flow rate (normalized)

def load_all_data(data_dir="./"):
    """
    Load all required data from files

    Args:
        data_dir: Directory containing data files

    Returns:
        data_dict: Dictionary with all loaded data
    """
    data_dict = {}

    # Load demand data
    data_dict['demand'] = pd.read_csv(os.path.join(data_dir, 'demand_2GW_week.csv'), header=0)

    # Load generator fleet data
    data_dict['fleet'] = pd.read_csv(os.path.join(data_dir, 'generator.csv'), header=0, index_col=0)

    # Load capacity factors data
    data_dict['cfs'] = pd.read_csv(os.path.join(data_dir, 'cfs_week.csv'), header=0, index_col=0)
    data_dict['cfs'].index = data_dict['demand'].index

    print(f"Successfully loaded all data files from {data_dir}")

    return data_dict


def build_model(demand_sub, solar_sub, wind_sub, M_0, fleet, is_final_day=False, target_final_mass=MAX_DURATION/2):
    """
    Build a MILP model for system operator optimization

    Args:
        demand_sub: DataFrame with demand data for the window
        solar_sub: Series with solar capacity factors
        wind_sub: Series with wind capacity factors
        M_0: Initial mass in the storage system
        fleet: DataFrame with generator parameters

    Returns:
        model: Pyomo concrete model
    """

    model = ConcreteModel()

    # Sets
    model.generators = Set(initialize=fleet.index.values)
    model.conv_generators = Set(initialize=[g for g in fleet.index.values if g not in ['PV', 'Wind', 'PTES']])
    model.times = Set(initialize=demand_sub.index.values)

    # PARAMETERS
    # Generator fleet parameters
    model.pCap = Param(model.generators, initialize=fleet['max'].to_dict())  # max power
    model.pMin = Param(model.generators, initialize=fleet['min'].to_dict())  # min power
    model.pAlpha = Param(model.generators, initialize=fleet['alpha'].to_dict())  # squared term cost coeff
    model.pBeta = Param(model.generators, initialize=fleet['beta'].to_dict())  # linear term cost coeff

    # PTES parameters
    model.pMaxDuration = Param(initialize=MAX_DURATION)  # Mbar
    model.pMFRUpper = Param(initialize=MFR_UPPER)  # mbar
    model.pMFRLower = Param(initialize=MFR_LOWER)  # m underbar

    # Parameters that need to be changed between runs
    model.pDemand = Param(model.times, initialize=demand_sub.to_dict())  # demand
    solar_dict = {t: solar_sub.loc[t] for t in model.times}
    wind_dict = {t: wind_sub.loc[t] for t in model.times}
    model.pSolarCF = Param(model.times, initialize=solar_dict)
    model.pWindCF = Param(model.times, initialize=wind_dict)
    # model.pSolarCF = Param(model.times, initialize=solar_sub.to_dict())  # solar capacity factor
    # model.pWindCF = Param(model.times, initialize=wind_sub.to_dict())  # wind capacity factor
    model.pM0 = Param(initialize=M_0, domain=Any)  # carry over previous iteration storage state M

    # VARIABLES
    # Generation fleet
    model.vPower = Var(model.generators, model.times, within=NonNegativeReals)
    model.vCommit = Var(model.conv_generators, model.times, domain=Binary)
    # PTES
    model.vCharge = Var(model.times, within=NonNegativeReals)  # P_t^op,c
    model.vDischarge = Var(model.times, within=NonNegativeReals)  # P_t^op,d

    model.vZcop = Var(model.times, domain=Binary)  # z_t^op,c
    model.vZdop = Var(model.times, domain=Binary)  # z_t^op,d

    model.vMFRopc = Var(model.times, within=NonNegativeReals)  # m_t^op,c
    model.vMFRopd = Var(model.times, within=NonNegativeReals)  # m_t^op,d
    model.vM = Var(model.times, within=NonNegativeReals)  # M_t

    # Objective function - minimize total generation cost
    def objFunc(model):
        return sum((model.pAlpha[gen] * model.vPower[gen, t] ** 2 + model.pBeta[gen] * model.vPower[gen, t])
                   for gen in model.generators for t in model.times)

    model.cost = Objective(rule=objFunc, sense=minimize)

    # Constraints
    # Supply-demand balance
    def supplyDemandBalanceConstraint(model, t):
        return sum(model.vPower[gen, t] for gen in model.generators) - model.vCharge[t] + model.vDischarge[t] == \
            model.pDemand[t]

    model.sd = Constraint(model.times, rule=supplyDemandBalanceConstraint)

    # Generator capacity constraints
    def genMaxCapConstraint(model, gen, t):
        if gen in model.conv_generators:
            return model.vPower[gen, t] <= model.pCap[gen] * model.vCommit[gen, t]
        else:  # For renewables
            return model.vPower[gen, t] <= model.pCap[gen]

    model.cap = Constraint(model.generators, model.times, rule=genMaxCapConstraint)

    def genMinConstraint(model, gen, t):
        if gen in model.conv_generators:
            return model.vPower[gen, t] >= model.pMin[gen] * model.vCommit[gen, t]
        else:  # For renewables
            return model.vPower[gen, t] >= model.pMin[gen]

    model.min = Constraint(model.generators, model.times, rule=genMinConstraint)

    # Renewables capacity factor constraints
    def solarCFLimit(model, t):
        return model.vPower['PV', t] <= model.pSolarCF[t] * model.pCap['PV']

    model.scf = Constraint(model.times, rule=solarCFLimit)

    def windCFLimit(model, t):
        return model.vPower['Wind', t] <= model.pWindCF[t] * model.pCap['Wind']

    model.wcf = Constraint(model.times, rule=windCFLimit)

    # PTES CONSTRAINTS
    # Power/mass flow rate constraints
    def ptesCCop(model, t):
        return model.vCharge[t] == k_charge * model.vMFRopc[t]

    model.cop = Constraint(model.times, rule=ptesCCop)

    def ptesDCop(model, t):
        return model.vDischarge[t] == k_discharge * model.vMFRopd[t]

    model.dop = Constraint(model.times, rule=ptesDCop)

    # Stored mass/mass flow constraints
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

    # Binary constraints
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


def convert_milp_to_ptes_inputs(model, time_index):
    """
    Convert MILP solution to inputs for the PTES physical model

    Args:
        model: Solved MILP model
        time_index: Time index to extract values for

    Returns:
        phase: Operation phase ('char', 'dis', or 'idle')
        m_norm: Normalized mass flow rate
    """
    if model.vZcop[time_index].value > 0.5:
        phase = "char"
        m_norm = model.vMFRopc[time_index].value
    elif model.vZdop[time_index].value > 0.5:
        phase = "dis"
        m_norm = model.vMFRopd[time_index].value
    else:
        phase = "idle"
        m_norm = 0.0

    return phase, m_norm


def adjust_generation(gen_values, delta_power, fleet, committed_gens):
    """
    Adjust conventional generation to handle power mismatches

    Args:
        gen_values: Dictionary with current generator power values
        delta_power: Power mismatch to be corrected (positive means increase generation)
        fleet: DataFrame with generator parameters
        committed_gens: List of generators that are committed

    Returns:
        adjusted_gen: Dictionary with adjusted generator power values
    """
    # Make a copy of the current generation values
    adjusted_gen = gen_values.copy()

    # Sort generators by marginal cost (most expensive first if decreasing, cheapest first if increasing)
    if delta_power < 0:  # Decrease generation
        # Calculate marginal costs for each generator
        sorted_gens = []
        for g in committed_gens:
            if g in ['PV', 'Wind', 'PTES']:
                continue
            p = adjusted_gen[g]
            mc = 2 * fleet.loc[g, 'alpha'] * p + fleet.loc[g, 'beta']
            sorted_gens.append((g, p, mc))

        # Sort by decreasing marginal cost (most expensive first)
        sorted_gens.sort(key=lambda x: x[2], reverse=True)

        # Decrease generation starting with the most expensive generator
        remaining_decrease = -delta_power
        for g, p, mc in sorted_gens:
            min_power = fleet.loc[g, 'min']
            available_decrease = p - min_power

            if available_decrease > 0:
                actual_decrease = min(available_decrease, remaining_decrease)
                adjusted_gen[g] = p - actual_decrease
                remaining_decrease -= actual_decrease

                if remaining_decrease <= 1e-6:  # Small tolerance to handle floating point errors
                    break

        # If we couldn't decrease enough, print a warning
        if remaining_decrease > 1e-6:
            print(f"Warning: Could not fully decrease generation. {remaining_decrease:.2f} MW remaining.")

    else:  # Increase generation
        # Calculate available headroom and marginal costs
        sorted_gens = []
        for g in committed_gens:
            if g in ['PV', 'Wind', 'PTES']:
                continue
            p = adjusted_gen[g]
            max_power = fleet.loc[g, 'max']
            headroom = max_power - p

            if headroom > 0:
                mc = 2 * fleet.loc[g, 'alpha'] * p + fleet.loc[g, 'beta']
                sorted_gens.append((g, p, mc, headroom))

        # Sort by increasing marginal cost (cheapest first)
        sorted_gens.sort(key=lambda x: x[2])

        # Increase generation starting with the cheapest generator
        remaining_increase = delta_power
        for g, p, mc, headroom in sorted_gens:
            actual_increase = min(headroom, remaining_increase)
            adjusted_gen[g] = p + actual_increase
            remaining_increase -= actual_increase

            if remaining_increase <= 1e-6:  # Small tolerance
                break

        # If we couldn't increase enough, print a warning
        if remaining_increase > 1e-6:
            print(f"Warning: Could not fully increase generation. {remaining_increase:.2f} MW remaining.")

    return adjusted_gen


def calc_cost(gen_values, fleet):
    """
    Calculate generation cost based on generator outputs

    Args:
        gen_values: Dictionary with generator power values
        fleet: DataFrame with generator parameters

    Returns:
        total_cost: Total generation cost
        costs: Dictionary with individual generator costs
    """
    costs = {}
    for generator, power in gen_values.items():
        if generator == 'PTES':
            costs[generator] = 0  # PTES has no direct fuel cost
            continue

        # Apply quadratic cost function
        try:
            alpha = fleet.loc[generator, "alpha"]
            beta = fleet.loc[generator, "beta"]
            costs[generator] = alpha * power ** 2 + beta * power
        except Exception as e:
            print(f"Warning: Error calculating cost for {generator}: {e}")
            costs[generator] = 0

    total_cost = sum(costs.values())
    return total_cost, costs


def calculate_marginal_price(gen_values, fleet):
    """
    Calculate the system marginal price for each hour

    Args:
        gen_values: Dictionary with generator power values
        fleet: DataFrame with generator parameters

    Returns:
        marginal_price: Marginal price value
    """
    # Calculate marginal cost for each running generator
    mc_values = {}
    for g, p in gen_values.items():
        if g == 'PTES' or p < 1e-6:  # Skip PTES and non-running generators
            continue

        try:
            alpha = fleet.loc[g, 'alpha']
            beta = fleet.loc[g, 'beta']
            mc = 2 * alpha * p + beta
            mc_values[g] = mc
        except:
            continue

    # Find the highest marginal cost
    if mc_values:
        highest_mc_gen = max(mc_values, key=mc_values.get)
        return mc_values[highest_mc_gen]
    else:
        return 0


def perform_system_operator_optimization(data_dict, ptes_design_file, days=7,
                                         output_dir="./results_milp_systemoperator/"):
    """
    Perform system operator optimization with physical PTES model integration
    with result saving format matching MINLP_admm
    """
    # Extract data from dictionary
    demand = data_dict['demand']
    fleet = data_dict['fleet']
    cfs = data_dict['cfs']

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the PTES model
    ptes = PTES_mt(design_file=ptes_design_file)

    # Initialize store temperatures
    current_store_temps = {
        "t_hh": ptes.t_hh_0,
        "t_lh": ptes.t_lh_0,
        "t_hc": ptes.t_hc_0,
        "t_lc": ptes.t_lc_0
    }

    # Initialize mass
    current_mass_norm = MAX_DURATION/2

    # Create DataFrames for results according to MINLP_admm format

    # 1. hourly_dispatch: Demand, generator and PTES dispatch power
    hourly_dispatch_columns = ['hour', 'day', 'demand']
    for g in fleet.index.values:
        hourly_dispatch_columns.append(f'power_{g}')
    hourly_dispatch_columns.extend(['ptes_charge', 'ptes_discharge', 'ptes_power'])
    hourly_dispatch = pd.DataFrame(columns=hourly_dispatch_columns)

    # 2. hourly_cost: Generator costs, system cost, and marginal price
    hourly_cost_columns = ['hour', 'day', 'total_cost', 'marginal_price']
    for g in fleet.index.values:
        if g != 'PTES':  # PTES doesn't have direct costs
            hourly_cost_columns.append(f'cost_{g}')
    hourly_cost = pd.DataFrame(columns=hourly_cost_columns)

    # 3. hourly_ptes: Detailed PTES operation parameters
    hourly_ptes_columns = [
        'hour', 'day',
        'charge_power', 'discharge_power', 'profit',
        'mass_norm', 'charge_m_norm', 'discharge_m_norm',
        't_hh', 't_lh', 't_hc', 't_lc',
        'mode', 'eff_com', 'eff_exp',
        'power_milp', 'power_physical', 'power_mismatch'
    ]
    hourly_ptes = pd.DataFrame(columns=hourly_ptes_columns)

    # 4. summary: Daily and total summary statistics
    summary_columns = [
        'day', 'date',
        'total_demand', 'total_cost',
        'ptes_charge_energy', 'ptes_discharge_energy',
        'ptes_charge_hours', 'ptes_discharge_hours', 'ptes_idle_hours',
        'avg_charge_price', 'avg_discharge_price',
        'ptes_final_mass', 'ptes_final_t_hh', 'ptes_final_t_lh', 'ptes_final_t_hc', 'ptes_final_t_lc',
        'optimization_time'
    ]
    summary = pd.DataFrame(columns=summary_columns)

    # Additional storage for intermediate results
    total_summary = {
        'total_demand': 0,
        'total_cost': 0,
        'ptes_charge_energy': 0,
        'ptes_discharge_energy': 0,
        'charge_price_sum': 0,  # For weighted average calculation
        'discharge_price_sum': 0,  # For weighted average calculation
        'total_optimization_time': 0
    }

    # Optimize each day
    for d in range(int(days)):
        day_start_time = time.time()
        print(f"\nOptimizing day {d + 1}/{days}")

        is_final_day = (d == days - 1)

        # Define time window indices
        start = (d * 24)
        end_day = start + 24  # End of first day
        end_window = start + 48  # End of 2-day window

        # Check if we have enough data for a 2-day window
        if end_window > len(demand):
            end_window = len(demand)
            print(f"Warning: Not enough data for a full 2-day window on day {d + 1}. Using {end_window - start} hours.")

        # Get 2-day subsets of demand, solar, wind
        subset_demand = demand['Load'].iloc[start:end_window]
        subset_solarCF = cfs['solar'].iloc[start:end_window]
        subset_windCF = cfs['wind'].iloc[start:end_window]

        # Create and solve the MILP model
        print(f"Creating and solving MILP model with {len(subset_demand)} hour window...")
        model = build_model(subset_demand, subset_solarCF, subset_windCF, current_mass_norm, fleet,
                            is_final_day=is_final_day, target_final_mass=MAX_DURATION/2)

        solver = SolverFactory('gurobi')
        results = solver.solve(model)

        daily_total_cost = 0
        daily_charge_energy = 0
        daily_discharge_energy = 0
        daily_charge_price_weighted = 0
        daily_discharge_price_weighted = 0
        charge_hours = 0
        discharge_hours = 0
        idle_hours = 0

        if results.solver.termination_condition == TerminationCondition.optimal:
            print(f"Optimal solution found. Total cost: ${model.cost():.2f}")

            # Process the first day's results with physical model integration
            for t_idx in range(start, min(end_day, end_window)):
                hour_in_window = t_idx - start

                # Get the actual time value from the subset_demand index
                t = subset_demand.index[hour_in_window]

                # Extract MILP solution for generators at this hour
                current_gen = {}
                committed_gens = []
                for g in model.generators:
                    current_gen[g] = model.vPower[g, t].value
                    if g in model.conv_generators and model.vCommit[g, t].value > 0.5:
                        committed_gens.append(g)

                # Extract PTES operation from MILP solution
                phase, m_norm = convert_milp_to_ptes_inputs(model, t)

                # Calculate MILP predicted power
                milp_power = 0
                charge_m_norm = 0
                discharge_m_norm = 0

                if phase == "char":
                    milp_power = -model.vCharge[t].value
                    charge_m_norm = m_norm
                    charge_hours += 1
                elif phase == "dis":
                    milp_power = model.vDischarge[t].value
                    discharge_m_norm = m_norm
                    discharge_hours += 1
                else:  # idle
                    idle_hours += 1

                # Get accurate power from physical model if not idle
                physical_power = 0
                if phase != "idle" and m_norm > 0:
                    perf = ptes.calculate_performance(
                        phase=phase,
                        m_norm=m_norm,
                        current_temps=current_store_temps
                    )

                    # Extract results
                    physical_power = perf['power']
                    temperatures = perf['temperatures']
                    eff_com = perf.get('eff_com', np.nan)
                    eff_exp = perf.get('eff_exp', np.nan)

                    # Update storage state based on operation
                    if phase == "char":
                        # Update temperatures for charging
                        mass_new = current_mass_norm + m_norm
                        if 't13' in temperatures:
                            t_hot = temperatures['t13']
                        else:
                            t_hot = temperatures.get('t2', current_store_temps['t_hh'])

                        if 't8' in temperatures:
                            t_cold = temperatures['t8']
                        else:
                            t_cold = temperatures.get('t6', current_store_temps['t_lc'])

                        current_store_temps['t_hh'] = (current_store_temps['t_hh'] * current_mass_norm +
                                                       t_hot * m_norm) / mass_new
                        current_store_temps['t_lc'] = (current_store_temps['t_lc'] * current_mass_norm +
                                                       t_cold * m_norm) / mass_new
                        current_mass_norm = mass_new

                    else:  # discharge
                        # Update temperatures for discharging
                        mass_new = current_mass_norm - m_norm
                        if mass_new < 1e-6:  # Avoid division by zero
                            mass_new = 0
                        else:
                            if 't12' in temperatures:
                                t_hot = temperatures['t12']
                            else:
                                t_hot = temperatures.get('t3', current_store_temps['t_lh'])

                            if 't9' in temperatures:
                                t_cold = temperatures['t9']
                            else:
                                t_cold = temperatures.get('t7', current_store_temps['t_hc'])

                            current_store_temps['t_lh'] = (current_store_temps['t_lh'] * (MAX_DURATION - current_mass_norm) +
                                                           t_hot * m_norm) / (MAX_DURATION - mass_new)
                            current_store_temps['t_hc'] = (current_store_temps['t_hc'] * (MAX_DURATION - current_mass_norm) +
                                                           t_cold * m_norm) / (MAX_DURATION - mass_new)
                        current_mass_norm = mass_new
                else:
                    eff_com = np.nan
                    eff_exp = np.nan

                # Apply heat losses to all thermal storage components
                current_store_temps['t_hh'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_hh']
                current_store_temps['t_lh'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_lh']
                current_store_temps['t_hc'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_hc']
                current_store_temps['t_lc'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_lc']

                # Adjust conventional generation for power balance with actual PTES performance
                delta_power = milp_power - physical_power  # How much generation to add/subtract

                if abs(delta_power) > 1e-6:  # Only adjust if mismatch is significant
                    current_gen = adjust_generation(current_gen, delta_power, fleet, committed_gens)

                # Update PTES power in generation mix
                current_gen['PTES'] = physical_power

                # Extract PTES charge and discharge power
                charge_power = -physical_power if physical_power < 0 else 0
                discharge_power = physical_power if physical_power > 0 else 0

                # Update daily totals
                daily_charge_energy += charge_power
                daily_discharge_energy += discharge_power

                # Calculate generation cost
                hour_costs = {}
                hour_total_cost = 0

                for g in fleet.index:
                    if g != 'PTES':  # PTES has no direct fuel cost
                        p = current_gen[g]
                        if p > 0:
                            alpha = fleet.loc[g, 'alpha']
                            beta = fleet.loc[g, 'beta']
                            cost_g = alpha * p ** 2 + beta * p
                            hour_costs[f'cost_{g}'] = cost_g
                            hour_total_cost += cost_g
                        else:
                            hour_costs[f'cost_{g}'] = 0

                daily_total_cost += hour_total_cost

                # Calculate marginal price
                marginal_price = calculate_marginal_price(current_gen, fleet)

                # Update price-weighted values for average calculation
                if charge_power > 0:
                    daily_charge_price_weighted += charge_power * marginal_price
                if discharge_power > 0:
                    daily_discharge_price_weighted += discharge_power * marginal_price

                # Calculate PTES profit based on marginal price
                ptes_profit = (discharge_power - charge_power) * marginal_price

                # 1. Store hourly dispatch data
                dispatch_row = {'hour': t_idx, 'day': d + 1, 'demand': subset_demand[t]}
                for g in fleet.index.values:
                    dispatch_row[f'power_{g}'] = current_gen[g]

                dispatch_row['ptes_charge'] = charge_power
                dispatch_row['ptes_discharge'] = discharge_power
                dispatch_row['ptes_power'] = physical_power

                hourly_dispatch = pd.concat([hourly_dispatch, pd.DataFrame([dispatch_row])], ignore_index=True)

                # 2. Store hourly cost data
                cost_row = {'hour': t_idx, 'day': d + 1, 'total_cost': hour_total_cost,
                            'marginal_price': marginal_price}
                cost_row.update(hour_costs)

                hourly_cost = pd.concat([hourly_cost, pd.DataFrame([cost_row])], ignore_index=True)

                # 3. Store hourly PTES data
                ptes_row = {
                    'hour': t_idx,
                    'day': d + 1,
                    'charge_power': charge_power,
                    'discharge_power': discharge_power,
                    'profit': ptes_profit,
                    'mass_norm': current_mass_norm,
                    'charge_m_norm': charge_m_norm,
                    'discharge_m_norm': discharge_m_norm,
                    't_hh': current_store_temps['t_hh'],
                    't_lh': current_store_temps['t_lh'],
                    't_hc': current_store_temps['t_hc'],
                    't_lc': current_store_temps['t_lc'],
                    'mode': phase,
                    'eff_com': eff_com,
                    'eff_exp': eff_exp,
                    'power_milp': milp_power,
                    'power_physical': physical_power,
                    'power_mismatch': physical_power - milp_power
                }

                hourly_ptes = pd.concat([hourly_ptes, pd.DataFrame([ptes_row])], ignore_index=True)

            # Calculate daily statistics
            day_demand = subset_demand[:24].sum()

            # Calculate average prices (weighted by power)
            avg_charge_price = 0
            if daily_charge_energy > 0:
                avg_charge_price = daily_charge_price_weighted / daily_charge_energy

            avg_discharge_price = 0
            if daily_discharge_energy > 0:
                avg_discharge_price = daily_discharge_price_weighted / daily_discharge_energy

            # Record optimization time
            optimization_time = time.time() - day_start_time

            # Store daily summary in new format
            summary_row = {
                'day': d + 1,
                'date': pd.Timestamp.now().date() - pd.Timedelta(days=7 - d),
                'total_demand': day_demand,
                'total_cost': daily_total_cost,
                'ptes_charge_energy': daily_charge_energy,
                'ptes_discharge_energy': daily_discharge_energy,
                'ptes_charge_hours': charge_hours,
                'ptes_discharge_hours': discharge_hours,
                'ptes_idle_hours': idle_hours,
                'avg_charge_price': avg_charge_price,
                'avg_discharge_price': avg_discharge_price,
                'ptes_final_mass': current_mass_norm,
                'ptes_final_t_hh': current_store_temps['t_hh'],
                'ptes_final_t_lh': current_store_temps['t_lh'],
                'ptes_final_t_hc': current_store_temps['t_hc'],
                'ptes_final_t_lc': current_store_temps['t_lc'],
                'optimization_time': optimization_time
            }

            summary = pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True)

            # Update totals for final summary row
            total_summary['total_demand'] += day_demand
            total_summary['total_cost'] += daily_total_cost
            total_summary['ptes_charge_energy'] += daily_charge_energy
            total_summary['ptes_discharge_energy'] += daily_discharge_energy
            total_summary['charge_price_sum'] += daily_charge_price_weighted
            total_summary['discharge_price_sum'] += daily_discharge_price_weighted
            total_summary['total_optimization_time'] += optimization_time

        else:
            print(f"Day {d + 1}: Solver did not find an optimal solution.")
            # Add placeholder row to summary with null values
            summary_row = {
                'day': d + 1,
                'date': pd.Timestamp.now().date() - pd.Timedelta(days=7 - d),
                'total_demand': None,
                'total_cost': None,
                'ptes_charge_energy': None,
                'ptes_discharge_energy': None,
                'ptes_charge_hours': None,
                'ptes_discharge_hours': None,
                'ptes_idle_hours': None,
                'avg_charge_price': None,
                'avg_discharge_price': None,
                'ptes_final_mass': None,
                'ptes_final_t_hh': None,
                'ptes_final_t_lh': None,
                'ptes_final_t_hc': None,
                'ptes_final_t_lc': None,
                'optimization_time': time.time() - day_start_time
            }

            summary = pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True)

    # Add total summary row
    total_row = {
        'day': 'Total',
        'date': None,
        'total_demand': total_summary['total_demand'],
        'total_cost': total_summary['total_cost'],
        'ptes_charge_energy': total_summary['ptes_charge_energy'],
        'ptes_discharge_energy': total_summary['ptes_discharge_energy'],
        'ptes_charge_hours': summary['ptes_charge_hours'].sum(),
        'ptes_discharge_hours': summary['ptes_discharge_hours'].sum(),
        'ptes_idle_hours': summary['ptes_idle_hours'].sum(),
        'avg_charge_price': total_summary['charge_price_sum'] / total_summary['ptes_charge_energy'] if total_summary[
                                                                                                           'ptes_charge_energy'] > 0 else 0,
        'avg_discharge_price': total_summary['discharge_price_sum'] / total_summary['ptes_discharge_energy'] if
        total_summary['ptes_discharge_energy'] > 0 else 0,
        'ptes_final_mass': summary.iloc[-1]['ptes_final_mass'] if pd.notna(
            summary.iloc[-1]['ptes_final_mass']) else None,
        'ptes_final_t_hh': summary.iloc[-1]['ptes_final_t_hh'] if pd.notna(
            summary.iloc[-1]['ptes_final_t_hh']) else None,
        'ptes_final_t_lh': summary.iloc[-1]['ptes_final_t_lh'] if pd.notna(
            summary.iloc[-1]['ptes_final_t_lh']) else None,
        'ptes_final_t_hc': summary.iloc[-1]['ptes_final_t_hc'] if pd.notna(
            summary.iloc[-1]['ptes_final_t_hc']) else None,
        'ptes_final_t_lc': summary.iloc[-1]['ptes_final_t_lc'] if pd.notna(
            summary.iloc[-1]['ptes_final_t_lc']) else None,
        'optimization_time': total_summary['total_optimization_time']
    }

    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    # Save all results to CSV files in the MINLP_admm format
    hourly_dispatch.to_csv(f"{output_dir}/hourly_dispatch.csv", index=False)
    hourly_cost.to_csv(f"{output_dir}/hourly_cost.csv", index=False)
    hourly_ptes.to_csv(f"{output_dir}/hourly_ptes.csv", index=False)
    summary.to_csv(f"{output_dir}/summary.csv", index=False)

    # Make sure to explicitly return the dataframes for plotting and analysis
    return hourly_dispatch, hourly_cost, hourly_ptes, summary


def plot_results_milp(hourly_dispatch, hourly_cost, hourly_ptes, demand_data, fleet,
                      output_dir="./results_milp_systemoperator/"):
    """
    Plot optimization results using the same format as MINLP_admm

    Args:
        hourly_dispatch: DataFrame with hourly dispatch data
        hourly_cost: DataFrame with hourly cost data
        hourly_ptes: DataFrame with detailed PTES operation data
        demand_data: DataFrame with demand data
        fleet: DataFrame with generator parameters
        output_dir: Directory to save plots
    """
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

    # Create ticks for days
    hours = hourly_dispatch['hour'].values
    days = np.unique(hourly_dispatch['day'].values)
    max_hour = max(hours)

    new_ticks = np.arange(0, max_hour, 24)
    new_labels = [i // 24 + 1 for i in new_ticks]

    # Define colors for plotting
    colors = [
        '#7FB3D5',  # darker soft blue
        '#BB8FA9',  # deeper pastel pink
        '#E3B04B',  # deeper yellow
        '#8AC7B0',  # deeper mint green
        '#9CBB9C',  # deeper light sage
        '#A58AD0',  # deeper lavender
        '#E6A4A8',  # deeper light coral
        '#C5D86D'  # deeper pale lime
    ]

    # Plot 1: PTES state of charge (mass_norm)
    plt.figure(figsize=(14, 6))
    plt.bar(hourly_ptes['hour'], hourly_ptes['mass_norm'], width=0.8, color='lightgreen')
    plt.ylabel('State of Charge (normalized mass)')
    plt.xlabel('Hour')
    plt.xticks(new_ticks, new_labels)
    plt.title('PTES State of Charge')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ptes_soc.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Generation stack with PTES power and marginal price
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Create stacked generation data
    stack_data = pd.DataFrame(index=hourly_dispatch['hour'])

    # Add conventional generators
    for g in fleet.index.values:
        if g != 'PTES':
            stack_data[g] = hourly_dispatch[f'power_{g}']

    # Add PTES discharge as a separate component (positive values)
    stack_data['PTES_Discharge'] = hourly_ptes['discharge_power']

    # Add PTES charge as a separate component with negative values
    stack_data['PTES_Charge'] = -hourly_ptes['charge_power']

    # Plot stacked generation on left y-axis
    stack_data.plot(kind='bar', stacked=True, width=0.9, color=colors, ax=ax1)

    # Add demand line
    ax1.plot(range(len(hourly_dispatch)), hourly_dispatch['demand'],
             color="black", linewidth=2.5, label="Demand")

    # Set left y-axis properties
    ax1.set_ylabel('Generation (MWh)', color='black')
    ax1.set_xlabel('Hour')
    ax1.set_xticks(new_ticks)
    ax1.set_xticklabels(new_labels)

    # Create right y-axis for marginal price
    ax2 = ax1.twinx()

    # Plot marginal price on right y-axis
    ax2.plot(range(len(hourly_cost)), hourly_cost['marginal_price'], 'r-', linewidth=2, label='Marginal Price')
    ax2.set_ylabel('Marginal Price ($/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title('System Operator Dispatch with PTES Integration')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/generation_stack.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: PTES temperatures
    plt.figure(figsize=(14, 6))
    plt.plot(hourly_ptes['hour'], hourly_ptes['t_hh'], 'r-', label='t_hh (Hot High)', linewidth=2)
    plt.plot(hourly_ptes['hour'], hourly_ptes['t_lh'], 'r--', label='t_lh (Low High)', linewidth=2)
    plt.plot(hourly_ptes['hour'], hourly_ptes['t_hc'], 'b-', label='t_hc (High Cold)', linewidth=2)
    plt.plot(hourly_ptes['hour'], hourly_ptes['t_lc'], 'b--', label='t_lc (Low Cold)', linewidth=2)
    plt.ylabel('Temperature (K)')
    plt.xlabel('Hour')
    plt.xticks(new_ticks, new_labels)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('PTES Storage Temperatures')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ptes_temps.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: PTES charge and discharge power
    plt.figure(figsize=(14, 6))
    plt.bar(hourly_ptes['hour'], hourly_ptes['charge_power'], width=0.4, align='edge',
            color='blue', alpha=0.7, label='Charge Power')
    plt.bar(hourly_ptes['hour'] + 0.4, hourly_ptes['discharge_power'], width=0.4, align='edge',
            color='green', alpha=0.7, label='Discharge Power')
    plt.ylabel('Power (MW)')
    plt.xlabel('Hour')
    plt.xticks(new_ticks, new_labels)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title('PTES Charge and Discharge Power')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ptes_power.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 5: System Cost Breakdown
    plt.figure(figsize=(14, 6))
    plt.plot(hourly_cost['hour'], hourly_cost['total_cost'], 'b-', linewidth=2)
    plt.ylabel('Cost ($)')
    plt.xlabel('Hour')
    plt.xticks(new_ticks, new_labels)
    plt.title('Hourly System Cost')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/system_cost.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 6: PTES Profit
    plt.figure(figsize=(14, 6))
    plt.plot(hourly_ptes['hour'], hourly_ptes['profit'], 'g-', linewidth=2)
    plt.ylabel('Profit ($)')
    plt.xlabel('Hour')
    plt.xticks(new_ticks, new_labels)
    plt.title('PTES Hourly Profit')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ptes_profit.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Add day separators to all plots by adding vertical lines at 24-hour intervals
    for day in range(len(days)):
        for ax in plt.gcf().get_axes():
            ax.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)


def main():
    """
    Main function to run the optimization with matching result format as MINLP_admm
    """
    # Create results directory
    results_dir = "./results_milp_systemoperator_236.87MW"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Define paths
    data_dir = "./"
    ptes_design_file = 'ptes_design_236.87MW_8.443449993667413h.csv'

    # Load all data in one place
    print("Loading all data files...")
    data_dict = load_all_data(data_dir)

    print("Starting system operator optimization with physical PTES model integration...")
    start_time = time.time()

    # Run the optimization with loaded data using new format
    hourly_dispatch, hourly_cost, hourly_ptes, summary = perform_system_operator_optimization(
        data_dict=data_dict,
        ptes_design_file=ptes_design_file,
        days=7,
        output_dir=results_dir
    )

    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Generate plots using the updated plotting function
    plot_results_milp(hourly_dispatch, hourly_cost, hourly_ptes, data_dict['demand'], data_dict['fleet'], results_dir)

    # Print summary of results
    print("\nOptimization Summary:")
    print(f"Total system operation cost: ${summary.iloc[-1]['total_cost']:.2f}")
    print(f"Total demand served: {summary.iloc[-1]['total_demand']:.2f} MWh")
    print(f"Total PTES charge energy: {summary.iloc[-1]['ptes_charge_energy']:.2f} MWh")
    print(f"Total PTES discharge energy: {summary.iloc[-1]['ptes_discharge_energy']:.2f} MWh")

    print("\nComplete! Results saved to:", results_dir)


if __name__ == "__main__":
    main()