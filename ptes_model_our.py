"""
Our method
"""
import os
from scipy.optimize import minimize, Bounds, differential_evolution
from constants import *
import pandas as pd
import warnings


class PTES_our:
    """
    Using chloride molten salt for heat storage and methanol for cold storage.
    """

    def __init__(self, power_0=None, duration=None, design_file=None):
        """
        Initialize PTES system with a design file that contains pre-calculated parameters
        """
        potential_paths = [
            # design_file,
            # os.path.abspath(design_file),
            # os.path.join(os.getcwd(), design_file),
            os.path.join(os.path.dirname(__file__), os.path.basename(design_file))
        ]

        success = False
        for path in potential_paths:
            try:
                self._load_design_from_csv(path)
                success = True
                break
            except:
                continue

        if not success:
            raise FileNotFoundError(f"Could not find or upload design file: {design_file}")
        self.t_hh = None
        self.t_lh = None
        self.t_hc = None
        self.t_lc = None
        self.p_com = None
        self.p_exp = None
        self.m = None
        self.beta = None
        self.mass = None

    def _load_design_from_csv(self, csv_file):
        """
        Load design parameters from a CSV file
        """
        try:
            design_df = pd.read_csv(csv_file)

            # Check if the dataframe has data
            if design_df.empty:
                raise ValueError(f"The CSV file {csv_file} is empty")

            # Extract the first row as a dictionary
            design_params = design_df.iloc[0].to_dict()

            # List of original parameters to load
            params = [
                "power_char_0", "power_dis_0", "duration", "beta_char_0", "beta_dis_0",
                "t_com_char_0", "t_exp_char_0", "t_com_dis_0", "t_exp_dis_0",
                "p_com_char_0", "p_exp_char_0", "p_com_dis_0", "p_exp_dis_0",
                "t_hh_0", "t_lh_0", "t_hc_0", "t_lc_0",
                "m_0", "mass_max", "n_0", "rte_0"
            ]

            # Only set attributes for original parameters
            for key in params:
                if key in design_params:
                    setattr(self, key, design_params[key])
                else:
                    print(f"Warning: Parameter '{key}' not found in CSV file")
            # print(f"Successfully loaded design parameters from {csv_file}")
        except Exception as e:
            print(f"Error loading design parameters: {e}")
            raise

    @staticmethod
    def eff_heat_exchanger(name: str, m: float = 1.0) -> float:
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

    @staticmethod
    def temp_ratio(name: str, beta: float, eff: float) -> float:
        if name == "com":
            ratio = 1 + (beta ** kappa - 1) / eff
            return ratio
        elif name == "exp":
            ratio = 1 - (1 - beta ** -kappa) * eff
            return ratio
        else:
            raise ValueError(f"Invalid turbomachinery type: {name}. Use 'com' for compressor or 'exp' for expander.")

    def exergy(self, mass_norm, temps):

        t_hh = temps['t_hh']
        t_lh = temps['t_lh']
        t_hc = temps['t_hc']
        t_lc = temps['t_lc']
        exergy_hh = self.m_0 * 3600 * mass_norm * cp * (t_hh - t_amb - t_amb * np.log(t_hh / t_amb))
        exergy_lh = self.m_0 * 3600 * (self.duration - mass_norm) * cp * (t_lh - t_amb - t_amb * np.log(t_lh / t_amb))
        exergy_hc = self.m_0 * 3600 * (self.duration - mass_norm) * cp * (t_hc - t_amb - t_amb * np.log(t_hc / t_amb))
        exergy_lc = self.m_0 * 3600 * mass_norm * cp * (t_lc - t_amb - t_amb * np.log(t_lc / t_amb))

        return exergy_hh + exergy_lh + exergy_hc + exergy_lc

    def _calculate_common_params(self, phase, m, beta, t_com, t_exp, p_exp, n):
        """
        Calculate common parameters used by multiple methods to avoid redundant calculations.
        """
        # Calculate base parameters
        if phase == "char":
            # Charging phase parameter initialization
            n_com_bar_0 = self.n_0 / np.sqrt(self.t_com_char_0)
            n_exp_bar_0 = self.n_0 / np.sqrt(self.t_exp_char_0)
            G_com_bar_0 = self.m_0 * np.sqrt(self.t_com_char_0) / self.p_com_char_0
            G_exp_bar_0 = self.m_0 * np.sqrt(self.t_exp_char_0) / self.p_exp_char_0
            beta_0 = self.beta_char_0
            beta_exp_factor = self.beta_char_0 ** 2 - 1
            t_exp_ref = self.t_exp_char_0
        else:  # discharge
            # Discharging phase parameter initialization
            n_com_bar_0 = self.n_0 / np.sqrt(self.t_com_dis_0)
            n_exp_bar_0 = self.n_0 / np.sqrt(self.t_exp_dis_0)
            G_com_bar_0 = self.m_0 * np.sqrt(self.t_com_dis_0) / self.p_com_dis_0
            G_exp_bar_0 = self.m_0 * np.sqrt(self.t_exp_dis_0) / self.p_exp_dis_0
            beta_0 = self.beta_dis_0
            beta_exp_factor = self.beta_dis_0 ** 2 - 1
            t_exp_ref = self.t_exp_dis_0

        # Calculate compressor and expander normalized parameters
        p_com = p_exp / beta

        # Compressor parameters
        n_com_bar = n / np.sqrt(t_com)
        n_com_dot = n_com_bar / n_com_bar_0
        G_com_bar = m * np.sqrt(t_com) / p_com
        G_com_dot = G_com_bar / G_com_bar_0

        # Expander parameters
        n_exp_bar = n / np.sqrt(t_exp)
        n_exp_dot = n_exp_bar / n_exp_bar_0
        G_exp_bar = m * np.sqrt(t_exp) / p_exp
        G_exp_dot = G_exp_bar / G_exp_bar_0

        # Calculate compressor coefficients
        c1 = n_com_dot / (d1 * (1 - d2 / n_com_dot) + n_com_dot * (n_com_dot - d2) ** 2)
        c2 = (d1 - 2 * d2 * n_com_dot ** 2) / (d1 * (1 - d2 / n_com_dot) + n_com_dot * (n_com_dot - d2) ** 2)
        c3 = -(d1 * d2 * n_com_dot - d2 ** 2 * n_com_dot ** 3) / (
                d1 * (1 - d2 / n_com_dot) + n_com_dot * (n_com_dot - d2) ** 2)

        # Expander common parameters
        speed_factor = np.sqrt(1.4 - 0.4 * n / self.n_0)
        temp_factor = np.sqrt(t_exp_ref / t_exp)

        return {
            'n_com_bar_0': n_com_bar_0,
            'n_exp_bar_0': n_exp_bar_0,
            'G_com_bar_0': G_com_bar_0,
            'G_exp_bar_0': G_exp_bar_0,
            'beta_0': beta_0,
            'beta_exp_factor': beta_exp_factor,
            't_exp_ref': t_exp_ref,
            'p_com': p_com,
            'n_com_bar': n_com_bar,
            'n_exp_bar': n_exp_bar,
            'n_com_dot': n_com_dot,
            'n_exp_dot': n_exp_dot,
            'G_com_bar': G_com_bar,
            'G_exp_bar': G_exp_bar,
            'G_com_dot': G_com_dot,
            'G_exp_dot': G_exp_dot,
            'c1': c1,
            'c2': c2,
            'c3': c3,
            'speed_factor': speed_factor,
            'temp_factor': temp_factor
        }

    def find_alpha_values(self, phase, m, beta, t_com, t_exp, p_exp, n, common_params=None):
        """
        Find the optimal alpha_com and alpha_exp values

        Optionally accepts pre-calculated common_params to reduce redundant calculations
        """
        # Calculate parameters if not provided
        if common_params is None:
            common_params = self._calculate_common_params(phase, m, beta, t_com, t_exp, p_exp, n)

        # Extract required parameters
        G_com_dot = common_params['G_com_dot']
        c1 = common_params['c1']
        c2 = common_params['c2']
        c3 = common_params['c3']
        beta_0 = common_params['beta_0']
        G_exp_dot = common_params['G_exp_dot']
        speed_factor = common_params['speed_factor']
        temp_factor = common_params['temp_factor']
        beta_exp_factor = common_params['beta_exp_factor']

        # Calculate alpha_com (maintaining original function logic)
        def calculate_beta_com(alpha_com_val):
            G_com_dot_map = G_com_dot / (1 + alpha_com_val / 100)
            beta_com_map = (c1 * G_com_dot_map ** 2 + c2 * G_com_dot_map + c3) * beta_0
            return (beta_com_map - 1) * (1 + 0.01 * alpha_com_val) + 1

        # Solve for alpha_com
        temp_a = c3 * beta_0 - 1
        temp_b = c2 * beta_0 * G_com_dot - beta + 1
        temp_c = c1 * beta_0 * G_com_dot ** 2
        discriminant = temp_b ** 2 - 4 * temp_a * temp_c

        # Compressor alpha_com solution logic (maintain original)
        if discriminant >= 0:
            temp_x1 = (-temp_b + np.sqrt(discriminant)) / (2 * temp_a)
            temp_x2 = (-temp_b - np.sqrt(discriminant)) / (2 * temp_a)
            alpha_com1 = (temp_x1 - 1) * 100
            alpha_com2 = (temp_x2 - 1) * 100

            # Check if only one solution is in valid range [-20, 60]
            valid1 = -20 <= alpha_com1 <= 60
            valid2 = -20 <= alpha_com2 <= 60

            if valid1 and not valid2:
                alpha_com = alpha_com1
            elif valid2 and not valid1:
                alpha_com = alpha_com2
            elif valid1 and valid2:
                alpha_com = alpha_com1 if abs(alpha_com1) <= abs(alpha_com2) else alpha_com2
            else:
                # If both or neither are valid, find which boundary value gives closest beta_com
                candidates = [-20, 60]
                best_alpha = None
                min_diff = float('inf')

                for candidate in candidates:
                    beta_com_candidate = calculate_beta_com(candidate)
                    diff = abs(beta_com_candidate - beta)
                    if diff < min_diff:
                        min_diff = diff
                        best_alpha = candidate

                alpha_com = best_alpha
        else:
            # When discriminant < 0, calculate from formula and compare with bounds
            alpha_com_calc = (-temp_b / (2 * temp_a) - 1) * 100

            # Find which value among alpha_com_calc, -20, 60 gives closest beta_com
            if -20 <= alpha_com_calc <= 60:
                candidates = [alpha_com_calc, -20, 60]
            else:
                candidates = [-20, 60]
            best_alpha = None
            min_diff = float('inf')

            for candidate in candidates:
                beta_com_candidate = calculate_beta_com(candidate)
                diff = abs(beta_com_candidate - beta)
                if diff < min_diff:
                    min_diff = diff
                    best_alpha = candidate

            alpha_com = best_alpha

        # Define calculate_beta_exp function
        def calculate_beta_exp(alpha_exp_val):
            G_exp_dot_map = G_exp_dot / (1.1875 - 3 / 10000 * (alpha_exp_val - 25) ** 2)
            return np.sqrt((G_exp_dot_map / speed_factor / temp_factor) ** 2 * beta_exp_factor + 1)

        # Calculate required G_exp_dot_map that would give beta_exp = beta
        required_G_exp_dot_map = speed_factor * temp_factor * np.sqrt((beta ** 2 - 1) / beta_exp_factor)

        # Solve for alpha_exp
        exp_discriminant = 10000 * (1.1875 * required_G_exp_dot_map - G_exp_dot) / (3 * required_G_exp_dot_map)

        # Expander alpha_exp solution logic (maintain original)
        if exp_discriminant >= 0:
            # Two possible solutions for alpha_exp
            alpha_exp1 = 25 - np.sqrt(exp_discriminant)

            # Check if solution is in valid range [-30, 25]
            valid1 = -30 <= alpha_exp1 <= 25

            if valid1:
                alpha_exp = alpha_exp1
            else:
                # If not valid, find which boundary value gives closest beta_exp
                candidates = [-30, 25]
                best_alpha = None
                min_diff = float('inf')

                for candidate in candidates:
                    beta_exp_candidate = calculate_beta_exp(candidate)
                    diff = abs(beta_exp_candidate - beta)
                    if diff < min_diff:
                        min_diff = diff
                        best_alpha = candidate

                alpha_exp = best_alpha
        else:
            # When discriminant < 0, compare -30 and 25 to see which gives closer beta_exp
            candidates = [-30, 25]
            best_alpha = None
            min_diff = float('inf')

            for candidate in candidates:
                beta_exp_candidate = calculate_beta_exp(candidate)
                diff = abs(beta_exp_candidate - beta)
                if diff < min_diff:
                    min_diff = diff
                    best_alpha = candidate

            alpha_exp = best_alpha

        # Calculate corresponding beta_com and beta_exp, which will be used in exloss
        G_com_dot_map = G_com_dot / (1 + alpha_com / 100)
        beta_com_map = (c1 * G_com_dot_map ** 2 + c2 * G_com_dot_map + c3) * beta_0
        beta_com = (beta_com_map - 1) * (1 + 0.01 * alpha_com) + 1

        G_exp_dot_map = G_exp_dot / (1.1875 - 3 / 10000 * (alpha_exp - 25) ** 2)
        beta_exp = np.sqrt((G_exp_dot_map / speed_factor / temp_factor) ** 2 * beta_exp_factor + 1)

        # Add additional return values for use in exloss
        return alpha_com, alpha_exp, {
            'G_com_dot_map': G_com_dot_map,
            'G_exp_dot_map': G_exp_dot_map,
            'beta_com': beta_com,
            'beta_exp': beta_exp
        }

    def exloss(self, phase, m, beta, t_com, t_exp, p_exp, n):
        """
        Calculate system's exergy loss using pre-calculated parameters to reduce redundant calculations
        """
        # Calculate common parameters
        common_params = self._calculate_common_params(phase, m, beta, t_com, t_exp, p_exp, n)

        # Extract parameters
        n_com_dot = common_params['n_com_dot']
        n_exp_dot = common_params['n_exp_dot']
        G_com_dot = common_params['G_com_dot']
        G_exp_dot = common_params['G_exp_dot']

        # Calculate alpha values and get additional pre-calculated results
        alpha_com, alpha_exp, alpha_results = self.find_alpha_values(
            phase, m, beta, t_com, t_exp, p_exp, n, common_params
        )

        # Extract pre-calculated mapping values
        G_com_dot_map = alpha_results['G_com_dot_map']
        G_exp_dot_map = alpha_results['G_exp_dot_map']
        beta_com = alpha_results['beta_com']
        beta_exp = alpha_results['beta_exp']

        # Calculate compressor efficiency
        eff_com_map = (1 - 0.3 * (1 - n_com_dot) ** 2) * (n_com_dot / G_com_dot_map) * (
                2 - n_com_dot / G_com_dot_map) * eff_com_0

        # Check for surge conditions
        vio_surge = 0
        eff_com_map_surge = 0
        c1 = common_params['c1']
        c2 = common_params['c2']
        c3 = common_params['c3']

        discriminant = (c2 - 1.18) ** 2 - 4 * c1 * c3
        if discriminant >= 0:
            G_s_dot_map_n1 = ((1.18 - c2) - np.sqrt((c2 - 1.18) ** 2 - 4 * c1 * c3)) / 2 / c1
            eff_s_map_n1 = (1 - 0.3 * (1 - n_com_dot) ** 2) * (n_com_dot / G_s_dot_map_n1) * (
                    2 - n_com_dot / G_s_dot_map_n1) * eff_com_0
            eff_s_map = (1 - 0.7 * (1 - n_com_dot)) ** 2 * eff_s_map_n1
            if G_com_dot_map < n_com_dot:
                if eff_s_map > eff_com_map:
                    vio_surge = eff_s_map - eff_com_map
                    eff_com_map_surge = eff_s_map

        # Calculate expander efficiency
        eff_exp_map = (1 - 0.3 * (1 - n_exp_dot) ** 2) * (n_exp_dot / G_exp_dot_map) * (
                2 - n_exp_dot / G_exp_dot_map) * eff_exp_0

        # Check for choke conditions
        eff_com_map_choke = 0
        vio_choke_com = 0
        if G_com_dot_map > n_com_dot:
            if eff_com_map < 0.85 * eff_com_0:
                vio_choke_com = 0.85 * eff_com_0 - eff_com_map
                eff_com_map_choke = 0.85 * eff_com_0

        eff_com_map = max(eff_com_map, eff_com_map_surge, eff_com_map_choke)
        eff_com = eff_com_map * (1 - alpha_com ** 2 / 10000)

        vio_choke_exp = 0
        if G_exp_dot_map > n_exp_dot:
            if eff_exp_map < 0.85 * eff_exp_0:
                vio_choke_exp = 0.85 * eff_exp_0 - eff_exp_map
                eff_exp_map = 0.85 * eff_exp_0
        eff_exp = eff_exp_map * (1 - 0.9 * alpha_exp ** 2 / 10000)

        # Calculate exergy loss
        exloss_norm = np.log((1 + (beta ** kappa - 1) / eff_com) * (1 - (1 - beta ** -kappa) * eff_exp))

        vio_beta = abs(beta_com - beta) + abs(beta_exp - beta)

        return exloss_norm, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta

    def opt_control(self, phase, m_actual, beta, t_com, t_exp, x_pre=None):
        """
        Optimize control parameters for the system.
        """
        if phase not in ["char", "dis"]:
            raise ValueError(f"Invalid phase: {phase}. Use 'char' for charge or 'dis' for discharge.")

        def objective(vars):
            """
            Objective function for optimization.
            """
            n, p_exp = vars

            # Get exergy efficiency from the exergy calculation function
            exloss_norm, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta = self.exloss(
                phase, m_actual, beta, t_com, t_exp, p_exp, n)

            # Add penalty term to avoid constraint violations
            penalty = 0.1 * penalty_factor * np.sqrt(vio_choke_com ** 2 + vio_choke_exp ** 2 + vio_surge ** 2
                                                     + vio_beta ** 2)
            return exloss_norm + penalty

        # Initial guess for p_exp - use the design value
        if phase == "char":
            m_0 = self.m_0
            t_0 = self.t_exp_char_0
            p_exp_initial = self.p_exp_char_0 * m_actual / m_0 * np.sqrt(t_exp / t_0)
        else:  # discharge
            m_0 = self.m_0
            t_0 = self.t_exp_dis_0
            p_exp_initial = self.p_exp_dis_0 * m_actual / m_0 * np.sqrt(t_exp / t_0)

        # Set optimization bounds
        lb = [0.8, 0.1]
        ub = [1.0, 1.0]
        bounds = Bounds(lb, ub)  # Set bounds for optimization variables

        n_values = [1.0]
        p_exp_values = np.clip([p_exp_initial], [0.1], [1])
        p_exp_values = np.unique(p_exp_values)
        best_result = None
        best_obj_value = float('inf')

        if x_pre == None:
            for n_start in n_values:
                for p_exp_start in p_exp_values:
                    # Run optimization from this starting point
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        warnings.filterwarnings('ignore', message='delta_grad == 0.0')
                        result = minimize(objective, [n_start, p_exp_start],
                                          method='l-bfgs-b',
                                          bounds=bounds,
                                          options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 100, 'disp': False})

                    # Check if this result is better than the current best
                    if result.fun < best_obj_value:
                        best_result = result
                        best_obj_value = result.fun
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                warnings.filterwarnings('ignore', message='delta_grad == 0.0')
                result = minimize(objective, x_pre,
                                  method='l-bfgs-b',
                                  bounds=bounds,
                                  options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 100, 'disp': False})

            best_result = result

        n, p_exp = best_result.x

        # Calculate efficiencies and deviation parameters
        exloss_norm, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta = self.exloss(
            phase, m_actual, beta, t_com, t_exp, p_exp, n)

        vio_choke_com = 0 if vio_choke_com < 1e-4 else vio_choke_com
        vio_choke_exp = 0 if vio_choke_exp < 1e-4 else vio_choke_exp
        vio_surge = 0 if vio_surge < 1e-4 else vio_surge
        vio_beta = 0 if vio_beta < 1e-4 else vio_beta
        return n, p_exp, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta

    def calculate_performance(self, phase: str, m_norm: float, beta_norm: float, current_temps: dict) -> dict:
        """
        Helper function to calculate system performance for given mass flow rate, pressure ratio, and storage temperatures
        Improved version with temperature correction via beta adjustment
        """
        # Unpack current temperatures from storage
        t_hh = current_temps['t_hh']
        t_lh = current_temps['t_lh']
        t_hc = current_temps['t_hc']
        t_lc = current_temps['t_lc']

        # if phase == "char":
        #     current_mass = 0.0 * self.mass_max
        #     # beta_norm = min(beta_norm, 1.1)
        # else:
        #     current_mass = self.mass_max
        #     # beta_norm = min(beta_norm, 1.15)
        #
        # old_exergy = self.exergy(current_mass, current_temps)

        # Calculate heat exchanger efficiencies for current flow rate
        eff_hs = self.eff_heat_exchanger("hs", m_norm)
        eff_re = self.eff_heat_exchanger("re", m_norm)
        eff_hr = self.eff_heat_exchanger("hr", m_norm)
        eff_cs = self.eff_heat_exchanger("cs", m_norm)

        # Set up iterative solution parameters
        tol = 1e-4
        error = np.inf
        max_iter = 30
        iter_count = 0
        coeff = 0.7
        # Initialize temperatures based on operation phase
        if phase == "char":
            t1 = self.t_com_char_0 + (t_lh - self.t_lh_0)
            t5 = self.t_exp_char_0
            t13 = self.t_hh_0
            t8 = self.t_lc_0
            if beta_norm > 1.06:
                beta_norm = 1.06
            beta = beta_norm * self.beta_char_0
        else:  # discharge
            t2 = self.t_exp_dis_0 + (t_hh - self.t_hh_0)
            t6 = self.t_com_dis_0 + (t_lc - self.t_lc_0)
            t12 = self.t_lh_0
            t9 = self.t_hc_0
            beta = beta_norm * self.beta_dis_0

        # Dictionary to store calculated temperatures
        temperatures = {}
        m_actual = m_norm * self.m_0
        beta_old = beta
        # Initialize flags for beta adjustment
        beta_flag = 0  # Start with beta adjustment enabled
        # Iterative solution to find converged temperature values
        while (error > tol or beta_flag == 1) and iter_count < max_iter:
            if phase == "char":
                # Calculate optimal control parameters for charging
                if error < 1:
                    n, p_exp, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta = self.opt_control(
                        phase, m_actual, beta, t1, t5, [n, p_exp])
                else:
                    n, p_exp, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta = self.opt_control(
                        phase, m_actual, beta, t1, t5)
                # Calculate temperature ratios for compressor and expander
                t2_over_t1 = self.temp_ratio("com", beta, eff_com)
                t6_over_t5 = self.temp_ratio("exp", beta, eff_exp)
                # Set up matrix equation A*x = b for charge process
                # This solves for all temperatures simultaneously in the cycle
                # x = [t1, t2, t3, t4, t5, t6, t7, t8, t13]
                A = np.zeros((9, 9))  # Coefficient matrix
                b = np.zeros(9)  # Right-hand side vector
                # Equation 1: Compressor outlet temperature relation (t2 = t1 * t2_over_t1)
                A[0, 0] = -t2_over_t1
                A[0, 1] = 1
                b[0] = 0
                # Equation 2: Expander outlet temperature relation (t6 = t5 * t6_over_t5)
                A[1, 4] = -t6_over_t5
                A[1, 5] = 1
                b[1] = 0
                # Equation 3: Hot store charging temperature relation (t13 = eff_hs * t2 + (1 - eff_hs) * t_lh)
                A[2, 1] = -eff_hs
                A[2, 8] = 1  # t13 is at index 8
                b[2] = (1 - eff_hs) * t_lh
                # Equation 4: Hot store outlet temperature relation (t3 = eff_hs * t_lh + (1 - eff_hs) * t2)
                A[3, 1] = -(1 - eff_hs)
                A[3, 2] = 1
                b[3] = eff_hs * t_lh
                # Equation 5: Recuperator cold side outlet temperature (t4 = eff_re * t7 + (1 - eff_re) * t3)
                A[4, 2] = -(1 - eff_re)
                A[4, 3] = 1
                A[4, 6] = -eff_re
                b[4] = 0
                # Equation 6: Compressor inlet temperature relation (t1 = eff_re * t3 + (1 - eff_re) * t7)
                A[5, 0] = 1
                A[5, 2] = -eff_re
                A[5, 6] = -(1 - eff_re)
                b[5] = 0
                # Equation 7: Expander inlet temperature relation (t5 = eff_hr * t_amb + (1 - eff_hr) * t4)
                A[6, 3] = -(1 - eff_hr)
                A[6, 4] = 1
                b[6] = eff_hr * t_amb
                # Equation 8: Cold store outlet temperature relation (t7 = eff_cs * t_hc + (1 - eff_cs) * t6)
                A[7, 5] = -(1 - eff_cs)
                A[7, 6] = 1
                b[7] = eff_cs * t_hc
                # Equation 9: Cold store charging temperature relation (t8 = eff_cs * t6 + (1 - eff_cs) * t_hc)
                A[8, 5] = -eff_cs
                A[8, 7] = 1  # t8 is at index 7
                b[8] = (1 - eff_cs) * t_hc
                # Solve the linear equation system to find all temperatures
                solution = np.linalg.solve(A, b)
                # Extract temperature solutions
                t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t8_new, t13_new = solution

                # Check for temperature constraint violations and adjust beta if needed
                if t13_new > t_hh_max + tol:
                    beta_flag = 1
                    # Calculate a new beta value to keep t13 within limits
                    temp = (t2_new - t13_new + t_hh_max) / t1_new
                    beta_temp = np.exp((np.log(1 + (temp - 1) * eff_com) / kappa))
                    # Gradual adjustment of beta
                    beta = (1 - coeff) * beta + coeff * beta_temp

                # Continue adjusting beta if we were previously correcting but now overcorrected
                if beta_flag == 1 and t_hh_max - t13_new > tol:
                    temp = (t2_new - t13_new + t_hh_max) / t1_new
                    beta_temp = np.exp((np.log(1 + (temp - 1) * eff_com) / kappa))
                    beta = (1 - coeff) * beta + coeff * beta_temp

                if beta_flag == 1 and abs(t_hh_max - t13_new) < tol:
                    beta_flag = 0

                # Calculate the maximum relative error for convergence check
                error = max(
                    abs(t13_new - t13),
                    abs(t8_new - t8)
                )
                # Update temperatures for next iteration
                t1, t2, t3, t4, t5, t6, t7, t8, t13 = t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t8_new, t13_new

                # Calculate power for charge mode (negative value indicates power input)
                power = -m_actual * cp * (
                        (t2 - t1) - (t5 - t6)) / eff_mg / 1e6  # Convert to MW, negative for charging

            else:  # discharge phase calculations
                # Calculate optimal control parameters for discharging
                if error < 1:
                    n, p_exp, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta = self.opt_control(
                        phase, m_actual, beta, t6, t2, [n, p_exp])
                else:
                    n, p_exp, eff_com, eff_exp, alpha_com, alpha_exp, vio_choke_com, vio_choke_exp, vio_surge, vio_beta = self.opt_control(
                        phase, m_actual, beta, t6, t2)
                # Calculate temperature ratios for expander and compressor
                t1_over_t2 = self.temp_ratio("exp", beta, eff_exp)
                t5_over_t6 = self.temp_ratio("com", beta, eff_com)

                # Set up matrix equation A*x = b for discharge process
                # This solves for all temperatures simultaneously in the cycle
                # x = [t1, t2, t3, t4, t5, t6, t7, t9, t12]
                A = np.zeros((9, 9))  # Coefficient matrix
                b = np.zeros(9)  # Right-hand side vector
                # Equation 1: Expander outlet temperature relation (t1 = t2 * t1_over_t2)
                A[0, 0] = 1
                A[0, 1] = -t1_over_t2
                b[0] = 0
                # Equation 2: Compressor outlet temperature relation (t5 = t6 * t5_over_t6)
                A[1, 4] = 1
                A[1, 5] = -t5_over_t6
                b[1] = 0
                # Equation 3: Recuperator hot side outlet temperature (t7 = eff_re * t4 + (1 - eff_re) * t1)
                A[2, 0] = -(1 - eff_re)
                A[2, 3] = -eff_re
                A[2, 6] = 1
                b[2] = 0
                # Equation 4: Recuperator cold side outlet temperature (t3 = eff_re * t1 + (1 - eff_re) * t4)
                A[3, 0] = -eff_re
                A[3, 2] = 1
                A[3, 3] = -(1 - eff_re)
                b[3] = 0
                # Equation 5: Compressor inlet temperature relation (t6 = eff_cs * t_lc + (1 - eff_cs) * t7)
                A[4, 5] = 1
                A[4, 6] = -(1 - eff_cs)
                b[4] = eff_cs * t_lc
                # Equation 6: Cold store charging temperature relation (t9 = eff_cs * t7 + (1 - eff_cs) * t_lc)
                A[5, 6] = -eff_cs
                A[5, 7] = 1  # t9 is at index 7
                b[5] = (1 - eff_cs) * t_lc
                # Equation 7: Heat recovery temperature relation (t4 = eff_hr * t_amb + (1 - eff_hr) * t5)
                A[6, 3] = 1
                A[6, 4] = -(1 - eff_hr)
                b[6] = eff_hr * t_amb
                # Equation 8: Expander inlet temperature relation (t2 = eff_hs * t_hh + (1 - eff_hs) * t3)
                A[7, 1] = 1
                A[7, 2] = -(1 - eff_hs)
                b[7] = eff_hs * t_hh
                # Equation 9: Hot store charging temperature relation (t12 = eff_hs * t3 + (1 - eff_hs) * t_hh)
                A[8, 2] = -eff_hs
                A[8, 8] = 1  # t12 is at index 8
                b[8] = (1 - eff_hs) * t_hh
                # Solve the linear equation system to find all temperatures
                solution = np.linalg.solve(A, b)
                # Extract temperature solutions
                t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t9_new, t12_new = solution

                # Check for temperature constraint violations and adjust beta if needed
                if t12_new < t_lh_min - tol:
                    beta_flag = 1
                    # Calculate a new beta value to keep t12_new within limits
                    temp = (-t12_new + t1_new + t_lh_min) / t2_new
                    beta_new = np.exp((np.log(1 - (1 - temp) / eff_exp) / -kappa))
                    # Gradual adjustment of beta
                    beta = (1 - coeff) * beta + coeff * beta_new

                # Continue adjusting beta if we were previously correcting but now overcorrected
                if beta_flag == 1 and t12_new - t_lh_min > tol:
                    temp = (-t12_new + t1_new + t_lh_min) / t2_new
                    beta_new = np.exp((np.log(1 - (1 - temp) / eff_exp) / -kappa))
                    beta = (1 - coeff) * beta + coeff * beta_new

                if beta_flag == 1 and abs(t_lh_min - t12_new) < tol:
                    beta_flag = 0

                # print(t12_new,beta,eff_com,eff_exp)
                # Calculate the maximum relative error for convergence check
                error = max(
                    abs(t12_new - t12),
                    abs(t9_new - t9)
                )

                # Update temperatures for next iteration
                t1, t2, t3, t4, t5, t6, t7, t9, t12 = t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t9_new, t12_new

                # Calculate power for discharge mode (positive value indicates power output)
                power = m_actual * cp * (
                        (t2 - t1) - (t5 - t6)) * eff_mg / 1e6  # Convert to MW, positive for discharging

            # Increment iteration counter
            iter_count += 1

        # Store final temperatures with any clipping needed
        if phase == "char":
        #     # Ensure hot store temperature doesn't exceed maximum
        #     t13 = np.clip(t13, t_lh_min, t_hh_max)
        #     # Store converged temperatures
            temperatures = {
                "t1": t1,
                "t2": t2,
                "t3": t3,
                "t4": t4,
                "t5": t5,
                "t6": t6,
                "t7": t7,
                "t8": t8,
                "t13": t13
            }
        else:
        #     # Ensure low temperature hot store doesn't fall below minimum
        #     t12 = np.clip(t12, t_lh_min, t_hh_max)
        #     # Store converged temperatures
            temperatures = {
                "t1": t1,
                "t2": t2,
                "t3": t3,
                "t4": t4,
                "t5": t5,
                "t6": t6,
                "t7": t7,
                "t9": t9,
                "t12": t12
            }
        vio_t13 = max(t13 - t_hh_max - tol, 0) if phase == "char" else max(t_hh - t_hh_max - tol, 0)
        # new_mass = current_mass + m_actual * 3600 if phase == "char" else current_mass - m_actual * 3600
        # t_hh = current_temps['t_hh']
        # t_lh = current_temps['t_lh']
        # t_hc = current_temps['t_hc']
        # t_lc = current_temps['t_lc']
        # new_temps = {
        #     "t_hh": t13 if 't13' in locals() else t_hh,
        #     "t_lh": t12 if 't12' in locals() else t_lh,
        #     "t_hc": t9 if 't9' in locals() else t_hc,
        #     "t_lc": t8 if 't8' in locals() else t_lc,
        # }
        # new_exergy = self.exergy(new_mass , new_temps)
        # eff_ex = -(new_exergy - old_exergy) / (power * 3.6e9) if phase == "char" else -(power * 3.6e9) / (new_exergy - old_exergy)
        # Return performance results as a dictionary
        return {
            'm': m_actual / self.m_0,
            'beta': beta,  # Return the adjusted beta value
            'power': power,
            'n': n,
            'p_exp': p_exp,
            'eff_com': eff_com,
            'eff_exp': eff_exp,
            'alpha_com': alpha_com,
            'alpha_exp': alpha_exp,
            'temperatures': temperatures,
            'vio_choke_com': vio_choke_com,
            'vio_choke_exp': vio_choke_exp,
            'vio_surge': vio_surge,
            'vio_beta': vio_beta,
            'vio_t13': vio_t13,
            'vio_error': max(0, error - tol)
            # 'eff_ex': eff_ex
        }