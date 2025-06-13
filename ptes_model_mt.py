"""
Method proposed by McTigue et al.
"""
import numpy as np
from scipy.optimize import minimize, Bounds
from constants import *
import pandas as pd


class PTES_mt:
    """
    Using chloride molten salt for heat storage and methanol for cold storage.
    """

    def __init__(self, power_0=None, duration=None, design_file=None):
        """
        Initialize PTES system with a design file that contains pre-calculated parameters
        """
        self._load_design_from_csv(design_file)
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

            print(f"Successfully loaded design parameters from {csv_file}")
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

    def exergy(self, mass, temps):

        t_hh = temps['t_hh']
        t_lh = temps['t_lh']
        t_hc = temps['t_hc']
        t_lc = temps['t_lc']
        exergy_hh = mass * cp * (t_hh - t_amb - t_amb * np.log(t_hh / t_amb))
        exergy_lh = (self.mass_max - mass) * cp * (t_lh - t_amb - t_amb * np.log(t_lh / t_amb))
        exergy_hc = (self.mass_max - mass) * cp * (t_hc - t_amb - t_amb * np.log(t_hc / t_amb))
        exergy_lc = mass * cp * (t_lc - t_amb - t_amb * np.log(t_lc / t_amb))

        return exergy_hh + exergy_lh + exergy_hc + exergy_lc

    def com_offdesign(self, phase: str, m: float, t: float, p: float) -> dict:
        # Design point parameters
        if phase == "char":
            n_0 = self.n_0
            m_0 = self.m_0
            t_0 = self.t_com_char_0
            p_0 = self.p_com_char_0
            beta_0 = self.beta_char_0
        elif phase == "dis":
            n_0 = self.n_0
            m_0 = self.m_0
            t_0 = self.t_com_dis_0
            p_0 = self.p_com_dis_0
            beta_0 = self.beta_dis_0
        else:
            raise ValueError(f"Invalid phase: {phase}. Use 'char' for charge or 'dis' for discharge.")

        eff_0 = eff_com_0
        n = self.n_0

        # Calculate reduced parameters
        n_bar = n / np.sqrt(t)
        n_bar_0 = n_0 / np.sqrt(t_0)
        n_dot = n_bar / n_bar_0

        G_bar = m * np.sqrt(t) / p
        G_bar_0 = m_0 * np.sqrt(t_0) / p_0
        G_dot = G_bar / G_bar_0

        c1 = n_dot / (d1 * (1 - d2 / n_dot) + n_dot * (n_dot - d2) ** 2)
        c2 = (d1 - 2 * d2 * n_dot ** 2) / (d1 * (1 - d2 / n_dot) + n_dot * (n_dot - d2) ** 2)
        c3 = -(d1 * d2 * n_dot - d2 ** 2 * n_dot ** 3) / (d1 * (1 - d2 / n_dot) + n_dot * (n_dot - d2) ** 2)

        beta = (c1 * G_dot ** 2 + c2 * G_dot + c3) * beta_0
        eff = (1 - 0.3 * (1 - n_dot) ** 2) * (n_dot / G_dot) * (2 - n_dot / G_dot) * eff_0

        return beta, eff

    def exp_offdesign(self, phase: str, m: float, t: float, p: float) -> dict:
        # Design point parameters
        if phase == "char":
            n_0 = self.n_0
            m_0 = self.m_0
            t_0 = self.t_exp_char_0
            p_0 = self.p_exp_char_0
            beta_0 = self.beta_char_0
        elif phase == "dis":
            n_0 = self.n_0
            m_0 = self.m_0
            t_0 = self.t_exp_dis_0
            p_0 = self.p_exp_dis_0
            beta_0 = self.beta_dis_0
        else:
            raise ValueError(f"Invalid phase: {phase}. Use 'char' for charge or 'dis' for discharge.")

        eff_0 = eff_exp_0
        n = self.n_0

        # Calculate reduced parameters
        n_bar = n / np.sqrt(t)
        n_bar_0 = n_0 / np.sqrt(t_0)
        n_dot = n_bar / n_bar_0

        G_bar = m * np.sqrt(t) / p
        G_bar_0 = m_0 * np.sqrt(t_0) / p_0
        G_dot = G_bar / G_bar_0

        beta = np.sqrt(((G_dot) / np.sqrt(1.4 - 0.4 * n / n_0) / np.sqrt(t_0 / t)) ** 2 * (beta_0 ** 2 - 1) + 1)
        eff = (1 - 0.3 * (1 - n_dot) ** 2) * (n_dot / G_dot) * (2 - n_dot / G_dot) * eff_0

        return beta, eff

    def control(self, name: str, m: float, t_com: float, t_exp: float) -> dict:
        """
        Find p_com and p_exp values that make beta equal for compressor and expander
        while satisfying p_com * beta = p_exp.
        """
        if name not in ["char", "dis"]:
            raise ValueError(f"Invalid phase: {name}. Use 'char' for charge or 'dis' for discharge.")

        def f(p_exp):
            beta_exp, eff_exp = self.exp_offdesign(name, m, t_exp, p_exp)
            p_com = p_exp / beta_exp  # Calculate p_com based on pressure equation
            beta_com, eff_com = self.com_offdesign(name, m, t_com, p_com)
            error = (beta_com - beta_exp) ** 2

            return error

        # Initial guess for p_exp - use the design value
        if name == "char":
            m_0 = self.m_0
            t_0 = self.t_exp_char_0
            p_exp_initial = self.p_exp_char_0 * m / m_0 * np.sqrt(t_exp/t_0)
        else:  # discharge
            m_0 = self.m_0
            t_0 = self.t_exp_dis_0
            p_exp_initial = self.p_exp_dis_0 * m / m_0 * np.sqrt(t_exp/t_0)

        # Find the optimal p_exp that minimizes the difference between beta_com and beta_exp
        p_exp_min = lower_bounds[1]
        p_exp_max = upper_bounds[1]
        bounds = Bounds(p_exp_min , p_exp_max)
        result = minimize(f, p_exp_initial, method='SLSQP',
                          bounds=bounds, tol=1e-8, options={'maxiter': 100})

        if not result.success:
            print(f"Warning: Optimization did not converge: {result.message}")

        # Get the optimal p_exp
        p_exp_opt = result.x[0]

        # Calculate final values
        beta_exp, eff_exp = self.exp_offdesign(name, m, t_exp, p_exp_opt)
        p_com_opt = p_exp_opt / beta_exp
        beta_com, eff_com = self.com_offdesign(name, m, t_com, p_com_opt)

        # Create and return result dictionary
        control_values = {
            "p_com": p_com_opt,
            "p_exp": p_exp_opt,
            "beta": (beta_com + beta_exp) / 2,  # beta_com should be approximately equal to beta_exp
            "eff_com": eff_com,
            "eff_exp": eff_exp
        }

        return control_values

    def system_offdesign(self, phase: str) -> dict:
        """
        Calculate system off-design performance at different mass flow rates.
        """
        m_ratio_range = np.linspace(0.1, 1.0, 10).tolist()

        # Design point parameters
        if phase == "char":
            m_0 = self.m_0
            t_com_0 = self.t_com_char_0
            t_exp_0 = self.t_exp_char_0
            beta_0 = self.beta_char_0
            t_hh_0 = self.t_hh_0
            t_lh_0 = self.t_lh_0
            t_hc_0 = self.t_hc_0
            t_lc_0 = self.t_lc_0
        elif phase == "dis":
            m_0 = self.m_0
            t_com_0 = self.t_com_dis_0
            t_exp_0 = self.t_exp_dis_0
            beta_0 = self.beta_dis_0
            t_hh_0 = self.t_hh_0
            t_lh_0 = self.t_lh_0
            t_hc_0 = self.t_hc_0
            t_lc_0 = self.t_lc_0
        else:
            raise ValueError(f"Invalid phase: {phase}. Use 'char' for charge or 'dis' for discharge.")

        # Initialize results dictionary
        results = {
            "m_ratio": [],
            "m": [],
            "power": [],
            "beta": [],
            "eff_com": [],
            "eff_exp": [],
            "eff_overall": [],
            "t_com": [],
            "t_exp": [],
            "p_com": [],
            "p_exp": [],
            "temperatures": []  # Will store list of dictionaries with all temperatures
        }

        for m_ratio in m_ratio_range:
            # Calculate the actual mass flow rate
            m = m_ratio * m_0

            # First initialization
            if phase == "char":
                t1 = t_com_0  # Compressor inlet temperature
                t5 = t_exp_0  # Expander inlet temperature
            else:  # discharge
                t6 = t_com_0  # Compressor inlet temperature
                t2 = t_exp_0  # Expander inlet temperature

            # Iterative solution
            tol = 1e-4
            max_iter = 100
            error = float('inf')
            iter_count = 0

            # Initialize temperature storage
            temperatures = {}

            while error > tol and iter_count < max_iter:
                # Get control parameters (pressures, beta, efficiencies)
                if phase == "char":
                    control_results = self.control(phase, m, t1, t5)
                else:  # discharge
                    control_results = self.control(phase, m, t6, t2)

                p_com = control_results["p_com"]
                p_exp = control_results["p_exp"]
                beta = control_results["beta"]
                eff_com = control_results["eff_com"]
                eff_exp = control_results["eff_exp"]

                # Calculate heat exchanger efficiencies for this flow rate
                eff_hs = self.eff_heat_exchanger("hs", m / m_0)
                eff_re = self.eff_heat_exchanger("re", m / m_0)
                eff_hr = self.eff_heat_exchanger("hr", m / m_0)
                eff_cs = self.eff_heat_exchanger("cs", m / m_0)

                # Solve the temperature equations using matrix method
                if phase == "char":
                    # Calculate temperature ratios for compressor and expander
                    t2_over_t1 = self.temp_ratio("com", beta, eff_com)
                    t6_over_t5 = self.temp_ratio("exp", beta, eff_exp)

                    # Set up matrix equation A*x = b for charge process
                    # x = [t1, t2, t3, t4, t5, t6, t7, t8, t13]
                    A = np.zeros((9, 9))
                    b = np.zeros(9)

                    # Equation 1: t2 = t1 * t2_over_t1
                    A[0, 0] = -t2_over_t1
                    A[0, 1] = 1
                    b[0] = 0

                    # Equation 2: t6 = t5 * t6_over_t5
                    A[1, 4] = -t6_over_t5
                    A[1, 5] = 1
                    b[1] = 0

                    # Equation 3: t13 = eff_hs * t2 + (1 - eff_hs) * self.t_lh_0
                    A[2, 1] = -eff_hs
                    A[2, 8] = 1  # t13 is at index 8
                    b[2] = (1 - eff_hs) * self.t_lh_0

                    # Equation 4: t3 = eff_hs * self.t_lh_0 + (1 - eff_hs) * t2
                    A[3, 1] = -(1 - eff_hs)
                    A[3, 2] = 1
                    b[3] = eff_hs * self.t_lh_0

                    # Equation 5: t4 = eff_re * t7 + (1 - eff_re) * t3
                    A[4, 2] = -(1 - eff_re)
                    A[4, 3] = 1
                    A[4, 6] = -eff_re
                    b[4] = 0

                    # Equation 6: t1 = eff_re * t3 + (1 - eff_re) * t7
                    A[5, 0] = 1
                    A[5, 2] = -eff_re
                    A[5, 6] = -(1 - eff_re)
                    b[5] = 0

                    # Equation 7: t5 = eff_hr * t_amb + (1 - eff_hr) * t4
                    A[6, 3] = -(1 - eff_hr)
                    A[6, 4] = 1
                    b[6] = eff_hr * t_amb

                    # Equation 8: t7 = eff_cs * self.t_hc_0 + (1 - eff_cs) * t6
                    A[7, 5] = -(1 - eff_cs)
                    A[7, 6] = 1
                    b[7] = eff_cs * self.t_hc_0

                    # Equation 9: t8 = eff_cs * t6 + (1 - eff_cs) * self.t_hc_0
                    A[8, 5] = -eff_cs
                    A[8, 7] = 1  # t8 is at index 7
                    b[8] = (1 - eff_cs) * self.t_hc_0

                    # Solve the linear equation system
                    solution = np.linalg.solve(A, b)

                    # Extract the solutions
                    t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t8_new, t13_new = solution

                    # Calculate the error between iterations
                    error = max(
                        abs(t1_new - t1) / t1 if t1 else 1.0,
                        abs(t5_new - t5) / t5 if t5 else 1.0
                    )

                    # Update temperatures for next iteration
                    t1, t2, t3, t4, t5, t6, t7, t8, t13 = t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t8_new, t13_new

                    # Store temperatures after convergence
                    temperatures = {
                        "t1": t1, "t2": t2, "t3": t3, "t4": t4,
                        "t5": t5, "t6": t6, "t7": t7, "t8": t8, "t13": t13
                    }

                    # Calculate power for charge mode
                    power = m * cp * ((t2 - t1) - (t5 - t6)) / eff_mg / 1e6  # Convert to MW

                else:  # discharge
                    # Calculate temperature ratios for expander and compressor
                    t1_over_t2 = self.temp_ratio("exp", beta, eff_exp)
                    t5_over_t6 = self.temp_ratio("com", beta, eff_com)

                    # Set up matrix equation A*x = b for discharge process
                    # x = [t1, t2, t3, t4, t5, t6, t7, t9, t12]
                    A = np.zeros((9, 9))  # Solve for all 9 variables
                    b = np.zeros(9)

                    # Equation 1: t1 = t2 * t1_over_t2
                    A[0, 0] = 1
                    A[0, 1] = -t1_over_t2
                    b[0] = 0

                    # Equation 2: t5 = t6 * t5_over_t6
                    A[1, 4] = 1
                    A[1, 5] = -t5_over_t6
                    b[1] = 0

                    # Equation 3: t7 = eff_re * t4 + (1 - eff_re) * t1
                    A[2, 0] = -(1 - eff_re)
                    A[2, 3] = -eff_re
                    A[2, 6] = 1
                    b[2] = 0

                    # Equation 4: t3 = eff_re * t1 + (1 - eff_re) * t4
                    A[3, 0] = -eff_re
                    A[3, 2] = 1
                    A[3, 3] = -(1 - eff_re)
                    b[3] = 0

                    # Equation 5: t6 = eff_cs * t_lc_0 + (1 - eff_cs) * t7
                    A[4, 5] = 1
                    A[4, 6] = -(1 - eff_cs)
                    b[4] = eff_cs * self.t_lc_0

                    # Equation 6: t9 = eff_cs * t7 + (1 - eff_cs) * t_lc_0
                    A[5, 6] = -eff_cs
                    A[5, 7] = 1  # t9 is at index 7
                    b[5] = (1 - eff_cs) * self.t_lc_0

                    # Equation 7: t4 = eff_hr * t_amb + (1 - eff_hr) * t5
                    A[6, 3] = 1
                    A[6, 4] = -(1 - eff_hr)
                    b[6] = eff_hr * t_amb

                    # Equation 8: t2 = eff_hs * t13 + (1 - eff_hs) * t3
                    A[7, 1] = 1
                    A[7, 2] = -(1 - eff_hs)
                    b[7] = eff_hs * self.t_hh_0

                    # Equation 9: t12 = eff_hs * t3 + (1 - eff_hs) * t13
                    A[8, 2] = -eff_hs
                    A[8, 8] = 1  # t12 is at index 8
                    b[8] = (1 - eff_hs) * self.t_hh_0

                    # Solve the linear equation system
                    solution = np.linalg.solve(A, b)

                    # Extract the solutions
                    t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t9_new, t12_new = solution

                    # Calculate the error between iterations
                    error = max(
                        abs(t2_new - t2) / t2,
                        abs(t6_new - t6) / t6
                    )

                    # Update temperatures for next iteration
                    t1, t2, t3, t4, t5, t6, t7, t9, t12 = t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t9_new, t12_new

                    # Store temperatures after convergence
                    temperatures = {
                        "t1": t1, "t2": t2, "t3": t3, "t4": t4,
                        "t5": t5, "t6": t6, "t7": t7, "t9": t9, "t12": t12
                    }

                    # Calculate power for discharge mode
                    power = m * cp * ((t2 - t1) - (t5 - t6)) * eff_mg / 1e6  # Convert to MW

                iter_count += 1

            if iter_count == max_iter and error > tol:
                print(
                    f"Warning: Maximum iterations reached without convergence for m_ratio={m_ratio}. Final error: {error}")

            # Calculate the overall efficiency for this operating point
            if phase == "char":
                t_com_in = t1
                t_com_out = t2
                t_exp_in = t5
                t_exp_out = t6
            else:  # discharge
                t_exp_in = t2
                t_exp_out = t1
                t_com_in = t6
                t_com_out = t5

            # Store results for this m_ratio
            results["m_ratio"].append(m_ratio)
            results["m"].append(m)
            results["power"].append(power)
            results["beta"].append(beta)
            results["eff_com"].append(eff_com)
            results["eff_exp"].append(eff_exp)
            results["t_com"].append(t_com_in)
            results["t_exp"].append(t_exp_in)
            results["p_com"].append(p_com)
            results["p_exp"].append(p_exp)
            results["temperatures"].append(temperatures)

        return results

    def calculate_performance(self, phase, m_norm, current_temps):
        """
        Helper function to calculate system performance for a given mass flow rate.

        Parameters:
        -----------
        phase : str
            Operating phase ('char' or 'dis')
        m_norm : float
            Norminal mass flow rate
        current_temps : dict
            Dictionary with current storage temperatures

        Returns:
        --------
        dict
            Dictionary with calculated power and temperatures
        """
        # Unpack current temperatures
        t_hh = current_temps['t_hh']
        t_lh = current_temps['t_lh']
        t_hc = current_temps['t_hc']
        t_lc = current_temps['t_lc']

        if phase == "char":
            current_mass = 0.0 * self.mass_max
        else:
            current_mass = self.mass_max

        old_exergy = self.exergy(current_mass, current_temps)

        m_actual = m_norm * self.m_0
        # Iterative solution
        tol = 1e-3
        max_iter = 100
        error = float('inf')
        iter_count = 0

        # Initialize temperatures based on phase
        if phase == "char":
            t1 = self.t_com_char_0  # Compressor inlet temperature
            t5 = self.t_exp_char_0  # Expander inlet temperature
        else:  # discharge
            t2 = self.t_exp_dis_0  # Expander inlet temperature
            t6 = self.t_com_dis_0  # Compressor inlet temperature

        # Storage for temperature variables
        temperatures = {}

        while error > tol and iter_count < max_iter:
            # Get control parameters (pressures, beta, efficiencies)
            if phase == "char":
                control_results = self.control(phase, m_actual, t1, t5)
            else:  # discharge
                control_results = self.control(phase, m_actual, t6, t2)

            p_com = control_results["p_com"]
            p_exp = control_results["p_exp"]
            beta = control_results["beta"]
            eff_com = control_results["eff_com"]
            eff_exp = control_results["eff_exp"]

            # Calculate heat exchanger efficiencies for this flow rate
            m_norm = m_actual / self.m_0
            eff_hs = self.eff_heat_exchanger("hs", m_norm)
            eff_re = self.eff_heat_exchanger("re", m_norm)
            eff_hr = self.eff_heat_exchanger("hr", m_norm)
            eff_cs = self.eff_heat_exchanger("cs", m_norm)

            if phase == "char":
                # Temperature ratios for compressor and expander
                t2_over_t1 = self.temp_ratio("com", beta, eff_com)
                t6_over_t5 = self.temp_ratio("exp", beta, eff_exp)

                # Set up matrix equation A*x = b for charge process
                # Standard matrix setup (solving for temperatures)
                # x = [t1, t2, t3, t4, t5, t6, t7, t8, t13]
                A = np.zeros((9, 9))
                b = np.zeros(9)

                # Equation 1: t2 = t1 * t2_over_t1
                A[0, 0] = -t2_over_t1
                A[0, 1] = 1
                b[0] = 0

                # Equation 2: t6 = t5 * t6_over_t5
                A[1, 4] = -t6_over_t5
                A[1, 5] = 1
                b[1] = 0

                # Equation 3: t13 = eff_hs * t2 + (1 - eff_hs) * t_lh
                A[2, 1] = -eff_hs
                A[2, 8] = 1  # t13 is at index 8
                b[2] = (1 - eff_hs) * t_lh

                # Equation 4: t3 = eff_hs * t_lh + (1 - eff_hs) * t2
                A[3, 1] = -(1 - eff_hs)
                A[3, 2] = 1
                b[3] = eff_hs * t_lh

                # Equation 5: t4 = eff_re * t7 + (1 - eff_re) * t3
                A[4, 2] = -(1 - eff_re)
                A[4, 3] = 1
                A[4, 6] = -eff_re
                b[4] = 0

                # Equation 6: t1 = eff_re * t3 + (1 - eff_re) * t7
                A[5, 0] = 1
                A[5, 2] = -eff_re
                A[5, 6] = -(1 - eff_re)
                b[5] = 0

                # Equation 7: t5 = eff_hr * t_amb + (1 - eff_hr) * t4
                A[6, 3] = -(1 - eff_hr)
                A[6, 4] = 1
                b[6] = eff_hr * t_amb

                # Equation 8: t7 = eff_cs * t_hc + (1 - eff_cs) * t6
                A[7, 5] = -(1 - eff_cs)
                A[7, 6] = 1
                b[7] = eff_cs * t_hc

                # Equation 9: t8 = eff_cs * t6 + (1 - eff_cs) * t_hc
                A[8, 5] = -eff_cs
                A[8, 7] = 1  # t8 is at index 7
                b[8] = (1 - eff_cs) * t_hc

                # Solve the linear equation system
                solution = np.linalg.solve(A, b)

                # Extract the solutions
                t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t8_new, t13_new = solution

                # Calculate the error between iterations
                error = max(
                    abs(t1_new - t1) / t1 if t1 else 1.0,
                    abs(t5_new - t5) / t5 if t5 else 1.0
                )

                # Update temperatures for next iteration
                t1, t2, t3, t4, t5, t6, t7, t8 = t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t8_new

                # Store temperatures
                temperatures = {
                    "t1": t1, "t2": t2, "t3": t3, "t4": t4,
                    "t5": t5, "t6": t6, "t7": t7, "t8": t8, "t13": t13_new
                }

                # Calculate power for charge mode
                power = -m_actual * cp * (
                        (t2 - t1) - (t5 - t6)) / eff_mg / 1e6  # Convert to MW, negative for charging

            else:  # discharge
                # Temperature ratios for expander and compressor
                t1_over_t2 = self.temp_ratio("exp", beta, eff_exp)
                t5_over_t6 = self.temp_ratio("com", beta, eff_com)

                # Set up matrix equation A*x = b for discharge process
                # x = [t1, t2, t3, t4, t5, t6, t7, t9, t12]
                A = np.zeros((9, 9))
                b = np.zeros(9)

                # Equation 1: t1 = t2 * t1_over_t2
                A[0, 0] = 1
                A[0, 1] = -t1_over_t2
                b[0] = 0

                # Equation 2: t5 = t6 * t5_over_t6
                A[1, 4] = 1
                A[1, 5] = -t5_over_t6
                b[1] = 0

                # Equation 3: t7 = eff_re * t4 + (1 - eff_re) * t1
                A[2, 0] = -(1 - eff_re)
                A[2, 3] = -eff_re
                A[2, 6] = 1
                b[2] = 0

                # Equation 4: t3 = eff_re * t1 + (1 - eff_re) * t4
                A[3, 0] = -eff_re
                A[3, 2] = 1
                A[3, 3] = -(1 - eff_re)
                b[3] = 0

                # Equation 5: t6 = eff_cs * t_lc + (1 - eff_cs) * t7
                A[4, 5] = 1
                A[4, 6] = -(1 - eff_cs)
                b[4] = eff_cs * t_lc

                # Equation 6: t9 = eff_cs * t7 + (1 - eff_cs) * t_lc
                A[5, 6] = -eff_cs
                A[5, 7] = 1  # t9 is at index 7
                b[5] = (1 - eff_cs) * t_lc

                # Equation 7: t4 = eff_hr * t_amb + (1 - eff_hr) * t5
                A[6, 3] = 1
                A[6, 4] = -(1 - eff_hr)
                b[6] = eff_hr * t_amb

                # Equation 8: t2 = eff_hs * t_hh + (1 - eff_hs) * t3
                A[7, 1] = 1
                A[7, 2] = -(1 - eff_hs)
                b[7] = eff_hs * t_hh

                # Equation 9: t12 = eff_hs * t3 + (1 - eff_hs) * t_hh
                A[8, 2] = -eff_hs
                A[8, 8] = 1  # t12 is at index 8
                b[8] = (1 - eff_hs) * t_hh

                # Solve the linear equation system
                solution = np.linalg.solve(A, b)

                # Extract the solutions
                t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t9_new, t12_new = solution

                # Calculate the error between iterations
                error = max(
                    abs(t2_new - t2) / t2 if t2 else 1.0,
                    abs(t6_new - t6) / t6 if t6 else 1.0
                )

                # Update temperatures for next iteration
                t1, t2, t3, t4, t5, t6, t7, t9, t12 = t1_new, t2_new, t3_new, t4_new, t5_new, t6_new, t7_new, t9_new, t12_new

                # Store temperatures
                temperatures = {
                    "t1": t1, "t2": t2, "t3": t3, "t4": t4,
                    "t5": t5, "t6": t6, "t7": t7, "t9": t9, "t12": t12
                }

                # Calculate power for discharge mode
                power = m_actual * cp * (
                        (t2 - t1) - (t5 - t6)) * eff_mg / 1e6  # Convert to MW, positive for discharging

            iter_count += 1

        if iter_count == max_iter and error > tol:
            print(f"Warning: Maximum iterations reached without convergence. Final error: {error}")

        new_mass = current_mass + m_actual * 3600 if phase == "char" else current_mass - m_actual * 3600
        t_hh = current_temps['t_hh']
        t_lh = current_temps['t_lh']
        t_hc = current_temps['t_hc']
        t_lc = current_temps['t_lc']
        new_temps = {
            "t_hh": t13_new if 't13' in locals() else t_hh,
            "t_lh": t12 if 't12' in locals() else t_lh,
            "t_hc": t9 if 't9' in locals() else t_hc,
            "t_lc": t8 if 't8' in locals() else t_lc,
        }
        new_exergy = self.exergy(new_mass, new_temps)
        eff_ex = -(new_exergy - old_exergy) / (power * 3.6e9) if phase == "char" else -(power * 3.6e9) / (
                new_exergy - old_exergy)

        # Return performance results
        return {
            'power': power,
            'beta': beta,
            'eff_com': eff_com,
            'eff_exp': eff_exp,
            'temperatures': temperatures,
            'eff_ex': eff_ex
        }

    def operation(self, m_series, z_char_series, z_dis_series):
        """
        Calculate the power output over a time series based on normalized mass flow rate
        and binary variables for charge/discharge states.
        Additionally calculates:
        - What power would be if charging at m=0.1 for each charging period
        - What power would be if operating at m=1.0 for each idle/discharge period

        Parameters:
        -----------
        m_series : array-like
            Time series of normalized mass flow rates (between 0 and 1)
        z_char_series : array-like
            Binary variable time series indicating charging state (1 for charging, 0 otherwise)
        z_dis_series : array-like
            Binary variable time series indicating discharging state (1 for discharging, 0 otherwise)

        Returns:
        --------
        dict
            Dictionary containing time series of power and other operational parameters
        """
        import numpy as np

        if len(m_series) != len(z_char_series) or len(m_series) != len(z_dis_series):
            raise ValueError("All input time series must have the same length")

        # Initialize storage if not already set
        if self.t_hh is None:
            self.t_hh = self.t_hh_0
        if self.t_lh is None:
            self.t_lh = self.t_lh_0
        if self.t_hc is None:
            self.t_hc = self.t_hc_0
        if self.t_lc is None:
            self.t_lc = self.t_lc_0
        if self.mass is None:
            self.mass = 0.0

        # Initialize result arrays
        n_steps = len(m_series)
        results = {
            'time_step': list(range(n_steps)),
            'power': np.zeros(n_steps),
            'mass': np.zeros(n_steps),
            't_hh': np.zeros(n_steps),
            't_lh': np.zeros(n_steps),
            't_hc': np.zeros(n_steps),
            't_lc': np.zeros(n_steps),
            'beta': np.zeros(n_steps),
            'eff_com': np.zeros(n_steps),
            'eff_exp': np.zeros(n_steps),
            'mode': ['idle'] * n_steps,
            'power_alt': np.zeros(n_steps)  # Alternative power at m=0.1 or m=1.0
        }

        # Initialize storage for alternative power scenarios
        self.potential_power_char = []
        self.potential_power_dis = []
        self.potential_power_idle = []

        # Process each time step
        for i in range(n_steps):
            m_norm = m_series[i]
            z_char = z_char_series[i]
            z_dis = z_dis_series[i]

            # Current temperatures dictionary
            current_temps = {
                't_hh': self.t_hh,
                't_lh': self.t_lh,
                't_hc': self.t_hc,
                't_lc': self.t_lc
            }

            if z_char == 0 and z_dis == 0:
                # System is idle
                results['power'][i] = 0
                results['mass'][i] = self.mass
                self.t_hh = heat_loss * t_amb + (1 - heat_loss) * self.t_hh
                self.t_lh = heat_loss * t_amb + (1 - heat_loss) * self.t_lh
                self.t_hc = heat_loss * t_amb + (1 - heat_loss) * self.t_hc
                self.t_lc = heat_loss * t_amb + (1 - heat_loss) * self.t_lc
                results['t_hh'][i] = self.t_hh
                results['t_lh'][i] = self.t_lh
                results['t_hc'][i] = self.t_hc
                results['t_lc'][i] = self.t_lc
                results['beta'][i] = 0
                results['eff_com'][i] = 0
                results['eff_exp'][i] = 0
                results['mode'][i] = 'idle'

                # Calculate what power would be if discharging at m=1.0
                m_alt = self.m_0  # full rated mass flow
                alt_results = self.calculate_performance("dis", m_alt, current_temps)
                results['power_alt'][i] = alt_results['power']
                self.potential_power_idle.append(alt_results['power'])
                continue

            if z_char == 1 and z_dis == 1:
                raise ValueError(f"System cannot charge and discharge simultaneously at time step {i}")

            # Actual mass flow rate
            m_actual = m_norm * self.m_0

            if z_char == 1:
                # Charging mode
                phase = "char"
                results['mode'][i] = 'charge'

                # Calculate performance at actual mass flow rate
                perf_results = self.calculate_performance(phase, m_actual, current_temps)
                power = perf_results['power']
                beta = perf_results['beta']
                eff_com = perf_results['eff_com']
                eff_exp = perf_results['eff_exp']
                temperatures = perf_results['temperatures']

                # Calculate alternative performance at m=0.1
                m_alt = 0.1 * self.m_0
                alt_results = self.calculate_performance(phase, m_alt, current_temps)
                results['power_alt'][i] = alt_results['power']
                self.potential_power_char.append(alt_results['power'])

                # Update storage mass and temperatures
                mass_new = self.mass + m_actual * 3600  # For 1-hour operation, in kg
                self.t_hh = (self.t_hh * self.mass + temperatures['t13'] * m_actual * 3600) / mass_new
                self.t_lc = (self.t_lc * self.mass + temperatures['t8'] * m_actual * 3600) / mass_new
                self.mass = mass_new

            elif z_dis == 1:
                # Discharging mode
                phase = "dis"
                results['mode'][i] = 'discharge'

                # Calculate performance at actual mass flow rate
                perf_results = self.calculate_performance(phase, m_actual, current_temps)
                power = perf_results['power']
                beta = perf_results['beta']
                eff_com = perf_results['eff_com']
                eff_exp = perf_results['eff_exp']
                temperatures = perf_results['temperatures']

                # Calculate alternative performance at m=1.0
                m_alt = self.m_0  # full rated mass flow
                alt_results = self.calculate_performance(phase, m_alt, current_temps)
                results['power_alt'][i] = alt_results['power']
                self.potential_power_dis.append(alt_results['power'])

                # Update storage mass and temperatures
                mass_new = self.mass - m_actual * 3600  # For 1-hour operation, in kg
                self.t_lh = (self.t_lh * (self.mass_max - self.mass) + temperatures['t12'] * m_actual * 3600) / (
                        self.mass_max - mass_new)
                self.t_hc = (self.t_hc * (self.mass_max - self.mass) + temperatures['t9'] * m_actual * 3600) / (
                        self.mass_max - mass_new)
                self.mass = mass_new

            # Store results for this time step
            results['power'][i] = power
            results['mass'][i] = self.mass
            self.t_hh = heat_loss * t_amb + (1 - heat_loss) * self.t_hh
            self.t_lh = heat_loss * t_amb + (1 - heat_loss) * self.t_lh
            self.t_hc = heat_loss * t_amb + (1 - heat_loss) * self.t_hc
            self.t_lc = heat_loss * t_amb + (1 - heat_loss) * self.t_lc
            results['t_hh'][i] = self.t_hh
            results['t_lh'][i] = self.t_lh
            results['t_hc'][i] = self.t_hc
            results['t_lc'][i] = self.t_lc
            results['beta'][i] = beta
            results['eff_com'][i] = eff_com
            results['eff_exp'][i] = eff_exp

        # Add the alternative power scenarios to the results
        results['potential_power_char'] = self.potential_power_char
        results['potential_power_dis'] = self.potential_power_dis
        results['potential_power_idle'] = self.potential_power_idle

        return results

# Example usage
if __name__ == "__main__":
    # Create a PTES instance by loading from a design file
    ptes = PTES_mt(design_file="ptes_design_300MW_10h.csv")