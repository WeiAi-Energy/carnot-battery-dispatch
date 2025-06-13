import pandas as pd
import numpy as np
from pyomo.environ import *
from cmaes import CMA
import matplotlib.pyplot as plt
from ptes_model_our import PTES_our
from constants import *
import multiprocessing as mp
import time
from scipy.optimize import minimize

# Constants for the optimization
LEN_HORIZON = 48  # 2-day lookahead window
k_charge = 300.8927795374502  # Factor for converting mass flow to charge power
k_discharge = 203.25815763463828  # Factor for converting mass flow to discharge power
MAX_DURATION = 10  # Maximum storage duration (hours)
MFR_LOWER = 0.08  # Minimum mass flow rate (normalized)
MFR_UPPER = 1.15  # Maximum mass flow rate (normalized)
# 369961.55325183785

def build_milp_model(price_data, M_0=0, is_final_day=False, target_final_mass=5.0):

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


def milp_to_bbo_solution(milp_model, horizon_length):
    """
    Convert MILP solution to BBO variables format

    Args:
        milp_model: Solved MILP model
        horizon_length: Length of the horizon (hours)

    Returns:
        modes: List of modes for BBO (-1: charge, 0: idle, 1: discharge)
        m_norms: List of mass flow rates for BBO
        beta_norms: List of beta values for BBO
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
            beta_norms.append(1.05)  # Default beta for charging
        elif modes[-1] == 1:  # Discharge
            m_norms.append(milp_model.vMFRopd[t].value)
            beta_norms.append(1.07)  # Default beta for discharging
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
    # Initialize data structure with the additional parameters
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
        'eff_ex':[],
        'eff_com': [],
        'eff_exp': [],
        'mass_norm': [],
        'vio_choke_com': [],
        'vio_choke_exp': [],
        'vio_surge': [],
        'vio_beta': [],
        'vio_t13': [],
        'vio_error': [],
        # Add new fields for additional parameters
        'n': [],
        'p_exp': [],
        'alpha_com': [],
        'alpha_exp': []
    }

    # Initialize state variables
    current_mass_norm = mass_norm
    current_store_temps = store_temps.copy()
    old_exergy = ptes.exergy(current_mass_norm, current_store_temps)

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
        vio_choke_com = 0
        vio_choke_exp = 0
        vio_surge = 0
        vio_beta = 0
        vio_t13 = 0
        vio_error = 0
        hourly_profit = 0
        # Initialize new parameters
        n_val = np.nan
        p_exp_val = np.nan
        alpha_com_val = np.nan
        alpha_exp_val = np.nan

        # Calculate power and update state using PTES model
        if phase == "char" or phase == "dis":

            # Call PTES model to calculate performance
            perform_results = ptes.calculate_performance(
                phase=phase,
                m_norm=m_norm,
                beta_norm=beta_norm,
                current_temps=current_store_temps
            )

            # Extract results
            power = perform_results['power']
            beta_actual = perform_results['beta']
            beta_norm = beta_actual / ptes.beta_char_0 if phase == "char" else beta_actual / ptes.beta_dis_0
            eff_com = perform_results['eff_com']
            eff_exp = perform_results['eff_exp']
            vio_choke_com = perform_results.get('vio_choke_com', 0)
            vio_choke_exp = perform_results.get('vio_choke_exp', 0)
            vio_surge = perform_results.get('vio_surge', 0)
            vio_beta = perform_results.get('vio_beta', 0)
            vio_t13 = perform_results.get('vio_t13', 0)
            vio_error = perform_results.get('vio_error', 0)

            # Extract additional parameters
            n_val = perform_results['n']
            p_exp_val = perform_results['p_exp']
            alpha_com_val = perform_results['alpha_com']
            alpha_exp_val = perform_results['alpha_exp']

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
                penalty = 10 * penalty_factor * (vio_choke_com + vio_choke_exp + vio_surge + vio_beta + vio_t13 + vio_error)
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
                penalty = 10 * penalty_factor * (vio_choke_com + vio_choke_exp + vio_surge + vio_beta + vio_t13 + vio_error)
                hourly_profit = power * lmp

                # Update total profit and penalty
                total_profit += hourly_profit
                total_penalty += penalty

        # Apply heat losses to all thermal storage components
        current_store_temps['t_hh'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_hh']
        current_store_temps['t_lh'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_lh']
        current_store_temps['t_hc'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_hc']
        current_store_temps['t_lc'] = heat_loss * t_amb + (1 - heat_loss) * current_store_temps['t_lc']

        new_exergy = ptes.exergy(current_mass_norm, current_store_temps)
        if phase == "char":
            eff_ex = (new_exergy - old_exergy) / (-power * 3.6e9)
        elif phase == "dis":
            eff_ex = (power * 3.6e9) / (old_exergy - new_exergy)
        else:
            eff_ex = new_exergy / old_exergy
        old_exergy = new_exergy

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
        data['eff_ex'].append(eff_ex)
        data['eff_com'].append(eff_com)
        data['eff_exp'].append(eff_exp)
        data['vio_choke_com'].append(vio_choke_com)
        data['vio_choke_exp'].append(vio_choke_exp)
        data['vio_surge'].append(vio_surge)
        data['vio_beta'].append(vio_beta)
        data['vio_t13'].append(vio_t13)
        data['vio_error'].append(vio_error)
        # Store the new parameters
        data['n'].append(n_val)
        data['p_exp'].append(p_exp_val)
        data['alpha_com'].append(alpha_com_val)
        data['alpha_exp'].append(alpha_exp_val)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Return dataframe, updated state, and profit/penalty
    return df, current_mass_norm, current_store_temps, total_profit, total_penalty


def evaluate_candidate(args):
    """
    Evaluate a single candidate solution, suitable for multiprocessing

    Args:
        args: Tuple of (y, params) where y is the candidate solution and params is an OptimizationParameters object

    Returns:
        Tuple of (y, obj, profit, penalty)
    """
    y, params = args

    # Map unbounded variables to bounded variables
    x_compact = sigmoid_map(y, params)

    # Expand parameters to full vectors
    m_norms_day1, beta_norms_day1 = expand_parameters(x_compact, params)

    # Use optimized parameters for first day, keep MILP solution for second day
    m_norms_full = np.copy(params.MILP_m_norms)
    beta_norms_full = np.copy(params.MILP_beta_norms)

    # Update first day parameters (0-23)
    m_norms_full[:24] = m_norms_day1[:24]
    beta_norms_full[:24] = beta_norms_day1[:24]

    # Compute operation for the entire horizon and get profit/penalty directly
    df, _, _, total_profit, operation_penalty = compute_one_horizon(
        params.MILP_modes, m_norms_full, beta_norms_full,
        params.mass_norm, params.store_temps,
        params.lmps, params.ptes
    )
    if params.is_final_day:
        mass_penalty = penalty_factor * (abs(df['mass_norm'][23] - 5) + (df['mass_norm'][23] - 5) ** 2)
    else:
        mass_penalty = 0

    # Total objective is negative (profit - penalty) since we're minimizing
    obj = -(total_profit - operation_penalty - mass_penalty)
    return (y, obj, total_profit, operation_penalty)


# Class for passing data between processes
class OptimizationParameters:
    """
    Class to store optimization parameters for inter-process communication
    """

    def __init__(self, MILP_modes, non_idle_indices, num_non_idle, lower_bounds, upper_bounds,
                 mass_norm, store_temps, lmps, ptes, MILP_m_norms, MILP_beta_norms, is_final_day):
        self.MILP_modes = MILP_modes
        self.non_idle_indices = non_idle_indices
        self.num_non_idle = num_non_idle
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.mass_norm = mass_norm
        self.store_temps = store_temps
        self.lmps = lmps
        self.ptes = ptes
        self.MILP_m_norms = MILP_m_norms
        self.MILP_beta_norms = MILP_beta_norms
        self.is_final_day = is_final_day


def sigmoid_map(y, params):
    """
    Map unbounded variables to bounded variables using sigmoid function

    Args:
        y: Unbounded vector of variables
        params: OptimizationParameters object with bounds

    Returns:
        Bounded vector of variables
    """
    bounded_0_1 = np.zeros_like(y)
    pos_mask = y >= 0
    bounded_0_1[pos_mask] = 1.0 / (1.0 + np.exp(-y[pos_mask]))
    neg_mask = ~pos_mask
    exp_y = np.exp(y[neg_mask])
    bounded_0_1[neg_mask] = exp_y / (1.0 + exp_y)
    return params.lower_bounds + bounded_0_1 * (params.upper_bounds - params.lower_bounds)


def expand_parameters(x_compact, params):
    """
    Expand compact parameters to full parameter vectors

    Args:
        x_compact: Compact vector of parameters (only non-idle time steps)
        params: OptimizationParameters object with mapping info

    Returns:
        m_norms_full: Full vector of mass flow rates
        beta_norms_full: Full vector of beta values
    """
    num_hours = len(params.MILP_modes)
    m_norms_full = np.zeros(num_hours)
    beta_norms_full = np.zeros(num_hours)

    m_norms_compact = x_compact[:params.num_non_idle]
    beta_norms_compact = x_compact[params.num_non_idle:]

    for i, idx in enumerate(params.non_idle_indices):
        m_norms_full[idx] = m_norms_compact[i]
        beta_norms_full[idx] = beta_norms_compact[i]

    return m_norms_full, beta_norms_full


def parallel_powell_polish(initial_y, opt_params, pool):
    """
    Parallel Powell polishing method with adaptive step sizes and convergence criteria,
    without momentum mechanism - using a single delta for all dimensions

    Args:
        initial_y: Initial solution (unbounded representation)
        opt_params: Optimization parameters object
        pool: Multiprocessing pool

    Returns:
        Tuple of (polished_y, polished_value)
    """
    current_y = initial_y.copy()

    # First evaluate the current solution
    _, current_value, _, _ = evaluate_candidate((current_y, opt_params))

    # Optimization parameters
    max_iterations = 500  # Reduced number of iterations
    delta_threshold = 0.002  # Minimum step size to continue

    dim = len(current_y)
    # Single delta for all dimensions (replaced individual deltas)
    delta = 0.01

    # Adaptive parameters
    expansion = 2.0  # Step size expansion factor on success
    contraction = 0.5  # Step size contraction factor on failure

    # Main optimization loop
    for iteration in range(max_iterations):
        if iteration % 10 == 0:
            print(f"Powell iteration {iteration + 1}/{max_iterations}")
        previous_value = current_value

        # Create search directions (coordinate directions only)
        search_directions = []

        # Add coordinate directions
        for i in range(dim):
            # Create point with positive perturbation
            y_plus = current_y.copy()
            y_plus[i] += delta
            search_directions.append((y_plus, opt_params, i, 1, "coordinate"))

            # Create point with negative perturbation
            y_minus = current_y.copy()
            y_minus[i] -= delta
            search_directions.append((y_minus, opt_params, i, -1, "coordinate"))

        # Sample directions if there are too many
        max_parallel = pool._processes
        if len(search_directions) > max_parallel:
            # Randomly sample coordinate directions
            import random
            search_directions = random.sample(search_directions, max_parallel)

        # Extract evaluation points
        eval_points = [(y, params) for y, params, _, _, _ in search_directions]

        # Evaluate points in parallel
        evaluation_results = pool.map(evaluate_candidate, eval_points)

        # Combine results with search directions
        full_results = []
        for dir_info, (_, obj, profit, penalty) in zip(search_directions, evaluation_results):
            y_perturbed, _, dim_idx, multiplier, dir_type = dir_info
            full_results.append((y_perturbed, obj, profit, penalty, dim_idx, multiplier, dir_type))

        # Find best improvement
        best_perturbed_y = None
        best_perturbed_value = current_value
        best_dim = None
        best_multiplier = None

        for (y_perturbed, obj, profit, penalty, dim_idx, multiplier, dir_type) in full_results:
            if obj < best_perturbed_value:
                best_perturbed_value = obj
                best_perturbed_y = y_perturbed
                best_dim = dim_idx
                best_multiplier = multiplier

        # Update current solution if improvement found
        if best_perturbed_value < current_value - 0.01:
            improvement = current_value - best_perturbed_value

            # Update current solution
            current_y = best_perturbed_y
            current_value = best_perturbed_value
            if iteration % 10 == 0:
                print(f"  Improved solution by ${improvement:.4f} to ${-current_value:.2f} using coordinate direction")

            # Expand delta for all dimensions
            delta *= expansion
            if iteration % 10 == 0:
                print(f"  Increasing step size to {delta:.4f}")
        else:
            # No improvement found, reduce step size
            delta *= contraction
            if iteration % 10 == 0:
                print(f"  No improvement found. Reducing step size to {delta:.4f}")

            # Check if step size is below threshold
            if delta < delta_threshold:
                if iteration % 10 == 0:
                    print(f"  Powell converged after {iteration + 1} iterations: step size below threshold")
                break

    return current_y, current_value


def optimize_operation_hybrid(mass_norm, store_temps, lmps, ptes, previous_solutions=None, is_final_day=False):
    """
    Improved hybrid optimization for one horizon using MILP + BBO with warm starting

    Args:
        mass_norm: Initial mass normalized (0-10)
        store_temps: Initial store temperatures
        lmps: Locational marginal prices for the horizon
        ptes: PTES model instance
        previous_solutions: Dictionary with previous solutions for warm start (optional)

    Returns:
        df: DataFrame with operation results
        optimal_modes: List of optimal modes
        optimal_m_norms: List of optimal mass flow rates
        optimal_beta_norms: List of optimal beta values
    """
    print("Starting improved hybrid optimization (MILP + BBO)...")
    start_time = time.time()

    # Step 1: Create price data DataFrame
    price_data = pd.DataFrame({
        'LMP ($/MWh)': lmps
    })

    # Step 2: Solve the MILP model for the entire 48-hour horizon
    print("Solving MILP model for initial solution...")
    milp_model = build_milp_model(price_data, mass_norm, is_final_day=is_final_day, target_final_mass=5.0)

    solver = SolverFactory('gurobi')
    results = solver.solve(milp_model)

    print(f"MILP solution found. Objective value: ${milp_model.profit():.2f}")

    # Convert MILP solution to BBO format
    MILP_modes, MILP_m_norms, MILP_beta_norms = milp_to_bbo_solution(milp_model, len(lmps))

    # Identify non-idle time steps for optimization, but only consider the first 24 hours
    non_idle_indices = [i for i, mode in enumerate(MILP_modes[:24]) if mode != 0]
    num_non_idle = len(non_idle_indices)

    print(f"Optimizing {num_non_idle} non-idle time steps out of 24 (first day only)")

    # Define bounds for optimization with more flexibility for m_norm
    m_norm_lower = 0.07
    m_norm_upper = 1.3  # Increased upper bound for more flexibility
    beta_norm_lower = 0.9  # Expanded lower bound
    beta_norm_upper = 1.2  # Expanded upper bound

    # Create lower and upper bounds only for non-idle time steps of the first day
    lower_bounds = np.array([m_norm_lower] * num_non_idle + [beta_norm_lower] * num_non_idle)
    upper_bounds = np.array([m_norm_upper] * num_non_idle + [beta_norm_upper] * num_non_idle)

    # Create optimization parameters object
    opt_params = OptimizationParameters(
        MILP_modes=MILP_modes,
        non_idle_indices=non_idle_indices,
        num_non_idle=num_non_idle,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        mass_norm=mass_norm,
        store_temps=store_temps,
        lmps=lmps,
        ptes=ptes,
        MILP_m_norms=MILP_m_norms,
        MILP_beta_norms=MILP_beta_norms,
        is_final_day=is_final_day
    )

    # Define inverse sigmoid mapping function
    def inverse_sigmoid_map(x, params):
        bounded_0_1 = (x - params.lower_bounds) / (params.upper_bounds - params.lower_bounds)
        bounded_0_1 = np.clip(bounded_0_1, 0.0001, 0.9999)
        return -np.log(1.0 / bounded_0_1 - 1.0)

    # Prepare initial solution (compact version - only non-idle time steps for the first day)
    initial_m_norms_compact = np.array([MILP_m_norms[i] for i in non_idle_indices])
    initial_beta_norms_compact = np.array([MILP_beta_norms[i] for i in non_idle_indices])
    initial_x_compact = np.concatenate([initial_m_norms_compact, initial_beta_norms_compact])

    # Apply warm starting if available
    if previous_solutions is not None and 'y' in previous_solutions:
        # Check if previous solution has compatible dimensions
        if len(previous_solutions['y']) == 2 * num_non_idle:
            print("Using warm start from previous solution")
            initial_y = previous_solutions['y']
        else:
            print("Previous solution has incompatible dimensions, using MILP solution")
            initial_y = inverse_sigmoid_map(initial_x_compact, opt_params)
    else:
        initial_y = inverse_sigmoid_map(initial_x_compact, opt_params)

    # Initialize the CMA optimizer with compact representation
    dim = initial_y.shape[0]

    # Adaptive population size based on dimension
    popsize = 32  # Round to nearest multiple of 10, min 30

    optimizer = CMA(
        mean=initial_y,
        sigma=0.03,
        lr_adapt=False,
        population_size=popsize
    )

    # Store the best solution
    best_solution_compact = None
    best_value = float('inf')
    best_profit = float('-inf')
    best_penalty = float('inf')
    best_y = None  # Store the best unbounded solution for Powell

    # Set number of processes for parallel execution
    num_processes = 32
    print(f"Using {num_processes} processes for parallel CMA-ES evaluation")

    # Create process pool
    pool = mp.Pool(processes=num_processes)

    # Adaptive termination criteria
    max_generations = 1000
    stagnation_limit = 1000  # Stop if no improvement for this many generations
    stagnation_counter = 0

    # Run with parallel processing
    for generation in range(max_generations):
        # Generate candidates for this generation
        candidates = [optimizer.ask() for _ in range(optimizer.population_size)]

        # Prepare argument list
        args_list = [(y, opt_params) for y in candidates]

        # Evaluate candidates in parallel using the process pool
        detailed_solutions = pool.map(evaluate_candidate, args_list)

        # Extract (y, value) tuples for the optimizer
        optimizer_solutions = [(y, obj) for y, obj, _, _ in detailed_solutions]

        # Track best solution
        gen_best_value = best_value
        for y, obj, profit, penalty in detailed_solutions:
            if obj < best_value:
                best_value = obj
                best_profit = profit
                best_penalty = penalty
                best_solution_compact = sigmoid_map(y, opt_params)
                best_y = y.copy()  # Store for Powell optimization
                stagnation_counter = 0  # Reset stagnation counter

        # Check for stagnation
        if gen_best_value >= best_value - 0.1:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        # Update the optimizer
        optimizer.tell(optimizer_solutions)

        # Print progress with detailed breakdown
        if generation % 10 == 0:
            print(f"Generation {generation}, total obj: ${-best_value:.2f}, profit: ${best_profit:.2f}, "
                  f"penalty: ${best_penalty:.2f}, sigma: {optimizer._sigma:.4f}")
        # optimizer._sigma = 0.03
        # Early termination if stagnation detected
        if stagnation_counter >= stagnation_limit or optimizer._sigma < 1e-3:
            print(
                f"Early termination after {generation + 1} generations: no improvement for {stagnation_limit} generations")
            break

    # Now implement improved parallel Powell polishing
    print("Polishing solution with improved Parallel Powell method...")
    polished_y, polished_value = parallel_powell_polish(best_y, opt_params, pool)

    # Check if Powell improved the solution
    if polished_value < best_value:
        print(f"Powell optimization improved the solution from ${-best_value:.2f} to ${-polished_value:.2f}")
        best_y = polished_y
        best_solution_compact = sigmoid_map(best_y, opt_params)
        best_value = polished_value

        # Re-evaluate to get updated profit and penalty
        _, best_value, best_profit, best_penalty = evaluate_candidate((best_y, opt_params))
    else:
        print("Powell optimization did not improve the solution")

    # Close and join the process pool
    pool.close()
    pool.join()

    # Create full parameter vectors for the entire horizon
    optimal_m_norms = np.copy(MILP_m_norms)
    optimal_beta_norms = np.copy(MILP_beta_norms)

    # Update only the first day parameters (0-23)
    m_norms_day1, beta_norms_day1 = expand_parameters(best_solution_compact, opt_params)
    optimal_m_norms[:24] = m_norms_day1[:24]
    optimal_beta_norms[:24] = beta_norms_day1[:24]

    # Compute final operation with optimal variables
    df, _, _, total_profit, total_penalty = compute_one_horizon(
        MILP_modes, optimal_m_norms, optimal_beta_norms, mass_norm, store_temps, lmps, ptes
    )

    optimized_reward = total_profit - total_penalty

    print(f"Optimization complete. Time elapsed: {time.time() - start_time:.2f} seconds")
    print(
        f"Optimized breakdown: Profit: ${total_profit:.2f}, Penalty: ${total_penalty:.2f}, Net: ${optimized_reward:.2f}")


    return df, MILP_modes, optimal_m_norms, optimal_beta_norms


def rolling_optimization_hybrid(price_data, initial_mass_norm=5, days=7, output_dir="./results/"):
    """
    Perform rolling optimization for specified number of days using hybrid approach

    Args:
        price_data: DataFrame with price data
        initial_mass_norm: Initial mass normalized (0-1)
        days: Number of days to optimize
        output_dir: Directory to save detailed results

    Returns:
        results_df: DataFrame with combined results
    """
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize state variables
    current_mass_norm = initial_mass_norm

    # Initialize PTES model
    ptes = PTES_our(design_file='ptes_design_200MW_10h.csv')

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

        # Optimize operation for this window using hybrid approach
        df, opt_modes, opt_m_norms, opt_beta_norms = optimize_operation_hybrid(
            current_mass_norm,
            current_store_temps,
            lmps,
            ptes,
            is_final_day = is_final_day
        )

        # Update state variables using the results of the first 24 hours
        modes_first_day = opt_modes[:24]
        m_norms_first_day = opt_m_norms[:24]
        beta_norms_first_day = opt_beta_norms[:24]

        # Get the actual operation results for the first day
        df, current_mass_norm, current_store_temps, _, _ = compute_one_horizon(
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
        daily_penalty = 10 * penalty_factor * (
                day_data['vio_choke_com'].sum() +
                day_data['vio_choke_exp'].sum() +
                day_data['vio_surge'].sum() +
                day_data['vio_beta'].sum() +
                day_data['vio_t13'].sum() +
                day_data['vio_error'].sum()
        )
        daily_net_profit = daily_profit + daily_penalty  # Since penalty is negative in the formula

        max_violations = max(
            day_data['vio_choke_com'].max(),
            day_data['vio_choke_exp'].max(),
            day_data['vio_surge'].max(),
            day_data['vio_beta'].max(),
            day_data['vio_t13'].max(),
            day_data['vio_error'].max()
        )

        avg_eff_com = day_data[day_data['eff_com'].notna()]['eff_com'].mean()
        avg_eff_exp = day_data[day_data['eff_exp'].notna()]['eff_exp'].mean()

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
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(f"{output_dir}/complete_hourly_details.csv", index=False)

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


def plot_results(days, results, price_data=None, output_dir="./results_minlp_pricetaker1"):
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
    plt.savefig(f"{output_dir}/minlp_pricetaker.png", dpi=600, bbox_inches='tight')
    plt.show()


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

    # Calculate violations statistics
    total_violations = (results["vio_choke_com"] + results["vio_choke_exp"] + results["vio_surge"] + results["vio_beta"] + results["vio_t13"] + results["vio_error"]).sum()
    violation_penalty = 10 * penalty_factor * total_violations
    profit_without_penalty = total_reward + violation_penalty

    # Create analysis dictionary
    analysis = {
        "total_reward": total_reward,
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

def main():
    """
    Main function to run the optimization
    """
    # Create results directory
    import os
    results_dir = "results_minlp_pricetaker"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Load price data
    price_data = load_price_data("representative_week.csv")

    if price_data is None:
        print("Failed to load price data. Exiting.")
        return

    # Run rolling optimization
    print("\nStarting rolling optimization...")
    start_time = time.time()

    days = 7
    # Use updated function with results directory
    results, daily_summary = rolling_optimization_hybrid(
        price_data,
        initial_mass_norm=5,
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
            'total_reward', 'avg_daily_reward', 'charge_hours', 'discharge_hours',
            'idle_hours',
            'avg_charge_power', 'avg_discharge_power', 'avg_charge_power_per_mflow', 'avg_discharge_power_per_mflow',
            'violation_penalty', 'profit_without_penalty', 'total_violations',
            'max_t_hh', 'min_t_lh', 'max_t_hc', 'min_t_lc'
        ],
        'value': [
            analysis['total_reward'], analysis['avg_daily_reward'],
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
    # Set multiprocessing start method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')  # More stable for Windows
        except RuntimeError:
            # Method already set
            pass

    main()