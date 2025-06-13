"""
SYSTEM OPERATOR OPTIMIZATION WITH PHYSICAL PTES MODEL INTEGRATION USING EXCHANGE-ADMM

This module performs system operator dispatch optimization with PTES integration:
1. Initialize variables with MILP system operator solution (warm start)
2. Update conventional generators using closed-form solutions
3. Update PTES operation using MIQP + BBO
4. Update dual variables
5. Repeat until convergence
6. Adjust conventional generators for any remaining power mismatches
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyomo.environ import *
from pyomo.common.tee import capture_output
import time
import os
import multiprocessing as mp
from cmaes import CMA
from sympy import false

from ptes_model_our import PTES_our
from constants import *
from MILP_system_operator import load_all_data
# Flag to use piecewise linear model (set to False for linear model)
USE_PIECEWISE_LINEAR = False

# Charging mode (m_norm, negative power) pairs
CHARGE_SEGMENTS = [
    (0.08, -287.5 * 0.08),
    (0.4, -292 * 0.4),
    (0.7, -297.5 * 0.7),
    (1.0, -302.5 * 1.0),
    (1.1, -305 * 1.1),
    (1.2, -308 * 1.2),
    (1.24, -312 * 1.24),
]
# Discharging mode (m_norm, positive power) pairs
DISCHARGE_SEGMENTS = [
    (0.08, 210.5 * 0.08),
    (0.4, 209.5 * 0.4),
    (0.7, 208 * 0.7),
    (1.0, 206 * 1.0),
    (1.1, 204 * 1.1),
    (1.2, 202 * 1.2),
    (1.24, 198 * 1.24),
]
# Constants for the optimization
LEN_HORIZON = 48  # 2-day lookahead window
k_charge = 302.28  # Factor for converting mass flow to charge power
k_discharge = 205.15  # Factor for converting mass flow to discharge power
MAX_DURATION = 10  # Maximum storage duration (hours)
MFR_LOWER = 0.08  # Minimum mass flow rate (normalized)
MFR_UPPER = 1.2  # Maximum mass flow rate (normalized)



def closed_form_primal(alpha, beta, min_p, max_p, rho, nu, del_p):
    closed_p = (nu + rho * del_p - beta) / (2 * alpha + rho)
    updated_p = np.maximum(min_p, np.minimum(closed_p, max_p))
    return updated_p


def dual_update(nu, rho, p_mismatch):
    updated_nu = nu + rho * p_mismatch
    return updated_nu


def build_milp_system_operator_model(demand_sub, solar_sub, wind_sub, M_0, fleet, is_final_day=False, target_final_mass=5.0):
    """
    Build a MILP model for initial system operator optimization (warm start)
    Based on MILP_system-operator.py build_model function
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
    model.pSolarCF = Param(model.times, initialize=solar_sub.to_dict())  # solar capacity factor
    model.pWindCF = Param(model.times, initialize=wind_sub.to_dict())  # wind capacity factor
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
    if USE_PIECEWISE_LINEAR:
        # Use piecewise linear relationship between mass flow and power
        # For charging
        model.vChargeSegments = Set(initialize=range(len(CHARGE_SEGMENTS)))
        model.pChargeM = Param(model.vChargeSegments,
                               initialize={i: CHARGE_SEGMENTS[i][0] for i in range(len(CHARGE_SEGMENTS))})
        model.pChargeP = Param(model.vChargeSegments,
                               initialize={i: CHARGE_SEGMENTS[i][1] for i in range(len(CHARGE_SEGMENTS))})

        # For discharging
        model.vDischargeSegments = Set(initialize=range(len(DISCHARGE_SEGMENTS)))
        model.pDischargeM = Param(model.vDischargeSegments,
                                  initialize={i: DISCHARGE_SEGMENTS[i][0] for i in range(len(DISCHARGE_SEGMENTS))})
        model.pDischargeP = Param(model.vDischargeSegments,
                                  initialize={i: DISCHARGE_SEGMENTS[i][1] for i in range(len(DISCHARGE_SEGMENTS))})

        # Piecewise variables for charging
        model.vLambdaCharge = Var(model.times, model.vChargeSegments, bounds=(0, 1))
        model.vSumLambdaCharge = Constraint(model.times,
                                            rule=lambda m, t: sum(m.vLambdaCharge[t, i] for i in m.vChargeSegments) ==
                                                              m.vZcop[t])

        # Piecewise constraints for charging
        def charge_mfr_constraint(m, t):
            return m.vMFRopc[t] == sum(m.vLambdaCharge[t, i] * m.pChargeM[i] for i in m.vChargeSegments)

        model.chargeMfrConstraint = Constraint(model.times, rule=charge_mfr_constraint)

        def charge_power_constraint(m, t):
            return m.vCharge[t] == sum(
                -m.vLambdaCharge[t, i] * m.pChargeP[i] for i in m.vChargeSegments)  # Negate to make positive

        model.chargePowerConstraint = Constraint(model.times, rule=charge_power_constraint)

        # Piecewise variables for discharging
        model.vLambdaDischarge = Var(model.times, model.vDischargeSegments, bounds=(0, 1))
        model.vSumLambdaDischarge = Constraint(model.times, rule=lambda m, t: sum(
            m.vLambdaDischarge[t, i] for i in m.vDischargeSegments) == m.vZdop[t])

        # Piecewise constraints for discharging
        def discharge_mfr_constraint(m, t):
            return m.vMFRopd[t] == sum(m.vLambdaDischarge[t, i] * m.pDischargeM[i] for i in m.vDischargeSegments)

        model.dischargeMfrConstraint = Constraint(model.times, rule=discharge_mfr_constraint)

        def discharge_power_constraint(m, t):
            return m.vDischarge[t] == sum(m.vLambdaDischarge[t, i] * m.pDischargeP[i] for i in m.vDischargeSegments)

        model.dischargePowerConstraint = Constraint(model.times, rule=discharge_power_constraint)

        # Add adjacency constraints for SOS2 type behavior
        if len(CHARGE_SEGMENTS) > 2:
            model.vYCharge = Var(model.times, range(len(CHARGE_SEGMENTS) - 1), domain=Binary)

            def charge_sos2_constraint(m, t):
                return sum(m.vYCharge[t, i] for i in range(len(CHARGE_SEGMENTS) - 1)) <= 1

            model.chargeSOS2Constraint = Constraint(model.times, rule=charge_sos2_constraint)

            def charge_adjacency_constraint(m, t, i):
                if i == 0:
                    return m.vLambdaCharge[t, i] <= m.vYCharge[t, i]
                elif i == len(CHARGE_SEGMENTS) - 1:
                    return m.vLambdaCharge[t, i] <= m.vYCharge[t, i - 1]
                else:
                    return m.vLambdaCharge[t, i] <= m.vYCharge[t, i - 1] + m.vYCharge[t, i]

            model.chargeAdjacencyConstraint = Constraint(model.times, model.vChargeSegments,
                                                         rule=charge_adjacency_constraint)

        if len(DISCHARGE_SEGMENTS) > 2:
            model.vYDischarge = Var(model.times, range(len(DISCHARGE_SEGMENTS) - 1), domain=Binary)

            def discharge_sos2_constraint(m, t):
                return sum(m.vYDischarge[t, i] for i in range(len(DISCHARGE_SEGMENTS) - 1)) <= 1

            model.dischargeSOS2Constraint = Constraint(model.times, rule=discharge_sos2_constraint)

            def discharge_adjacency_constraint(m, t, i):
                if i == 0:
                    return m.vLambdaDischarge[t, i] <= m.vYDischarge[t, i]
                elif i == len(DISCHARGE_SEGMENTS) - 1:
                    return m.vLambdaDischarge[t, i] <= m.vYDischarge[t, i - 1]
                else:
                    return m.vLambdaDischarge[t, i] <= m.vYDischarge[t, i - 1] + m.vYDischarge[t, i]

            model.dischargeAdjacencyConstraint = Constraint(model.times, model.vDischargeSegments,
                                                            rule=discharge_adjacency_constraint)
    else:
        # Use linear relationship p = k*m
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


def get_milp_system_operator_solution(demand_sub, solar_sub, wind_sub, M_0, fleet, is_final_day=False, target_final_mass=5.0):
    """
    Get initial solution from MILP system operator model as warm start for ADMM
    """

    # Build and solve MILP model
    milp_model = build_milp_system_operator_model(
        demand_sub, solar_sub, wind_sub, M_0, fleet, is_final_day=is_final_day, target_final_mass=5.0
    )

    solver = SolverFactory('gurobi')
    with capture_output() as output:
        results = solver.solve(milp_model)

    if results.solver.termination_condition != TerminationCondition.optimal:
        print("Warning: MILP solver did not find an optimal solution for warm start.")
        return None

    # Extract generation and PTES operation from solution
    all_times = list(milp_model.times)

    gen_values = {}
    for g in fleet.index.values:
        if g != 'PTES':  # Skip PTES as we'll handle it separately
            gen_values[g] = np.array([milp_model.vPower[g, t].value for t in all_times])

    # Extract PTES operation - ALL 48 HOURS
    ptes_charge = np.array([milp_model.vCharge[t].value for t in all_times])
    ptes_discharge = np.array([milp_model.vDischarge[t].value for t in all_times])
    ptes_mfr_charge = np.array([milp_model.vMFRopc[t].value for t in all_times])
    ptes_mfr_discharge = np.array([milp_model.vMFRopd[t].value for t in all_times])
    ptes_z_charge = np.array([1 if milp_model.vZcop[t].value > 0.5 else 0 for t in all_times])
    ptes_z_discharge = np.array([1 if milp_model.vZdop[t].value > 0.5 else 0 for t in all_times])

    # Get committed generators
    committed_gens = []
    for g in fleet.index.values:
        if g in ['PV', 'Wind']:
            committed_gens.append(g)
        elif g in milp_model.conv_generators:
            # Consider a generator committed if it's committed in any hour of first day
            if any(milp_model.vCommit[g, t].value > 0.5 for t in all_times):
                committed_gens.append(g)

    # Return solution
    return {
        'gen_values': gen_values,
        'ptes_charge': ptes_charge,
        'ptes_discharge': ptes_discharge,
        'ptes_mfr_charge': ptes_mfr_charge,
        'ptes_mfr_discharge': ptes_mfr_discharge,
        'ptes_z_charge': ptes_z_charge,
        'ptes_z_discharge': ptes_z_discharge,
        'committed_gens': committed_gens
    }


def build_ptes_miqp_model(nu, del_p, rho, M_0, day_length=48):
    """
    Build a MILP model for PTES operation with ADMM penalty term

    Args:
        nu: Dual variable
        del_p
        rho: ADMM penalty parameter
        M_0: Initial mass in the storage system
        day_length: Length of the horizon (hours)

    Returns:
        model: Pyomo concrete model
    """

    # Create model
    model = ConcreteModel()

    # Sets
    model.times = Set(initialize=range(day_length))

    # Parameters
    model.pMaxDuration = Param(initialize=MAX_DURATION)
    model.pMFRUpper = Param(initialize=MFR_UPPER)
    model.pMFRLower = Param(initialize=MFR_LOWER)
    model.pNu = Param(model.times, initialize=dict(enumerate(nu)))
    model.pDelP = Param(model.times, initialize=dict(enumerate(del_p)))
    model.pM0 = Param(initialize=M_0, domain=Any)

    # Variables
    model.vCharge = Var(model.times, within=NonNegativeReals)
    model.vDischarge = Var(model.times, within=NonNegativeReals)
    model.vZcop = Var(model.times, domain=Binary)
    model.vZdop = Var(model.times, domain=Binary)
    model.vMFRopc = Var(model.times, within=NonNegativeReals)
    model.vMFRopd = Var(model.times, within=NonNegativeReals)
    model.vM = Var(model.times, within=NonNegativeReals)

    def objFunc(model):
        return sum((-(model.vDischarge[t] - model.vCharge[t]) * model.pNu[t] +
                   rho / 2 * (model.pDelP[t] - model.vDischarge[t] + model.vCharge[t]) ** 2)
                   for t in model.times)

    model.cost = Objective(rule=objFunc, sense=minimize)

    # PTES relationship constraints
    if USE_PIECEWISE_LINEAR:
        # Use piecewise linear relationship between mass flow and power
        # For charging
        model.vChargeSegments = Set(initialize=range(len(CHARGE_SEGMENTS)))
        model.pChargeM = Param(model.vChargeSegments,
                               initialize={i: CHARGE_SEGMENTS[i][0] for i in range(len(CHARGE_SEGMENTS))})
        model.pChargeP = Param(model.vChargeSegments,
                               initialize={i: CHARGE_SEGMENTS[i][1] for i in range(len(CHARGE_SEGMENTS))})

        # For discharging
        model.vDischargeSegments = Set(initialize=range(len(DISCHARGE_SEGMENTS)))
        model.pDischargeM = Param(model.vDischargeSegments,
                                  initialize={i: DISCHARGE_SEGMENTS[i][0] for i in range(len(DISCHARGE_SEGMENTS))})
        model.pDischargeP = Param(model.vDischargeSegments,
                                  initialize={i: DISCHARGE_SEGMENTS[i][1] for i in range(len(DISCHARGE_SEGMENTS))})

        # Piecewise variables for charging
        model.vLambdaCharge = Var(model.times, model.vChargeSegments, bounds=(0, 1))
        model.vSumLambdaCharge = Constraint(model.times,
                                            rule=lambda m, t: sum(m.vLambdaCharge[t, i] for i in m.vChargeSegments) ==
                                                              m.vZcop[t])

        # Piecewise constraints for charging
        def charge_mfr_constraint(m, t):
            return m.vMFRopc[t] == sum(m.vLambdaCharge[t, i] * m.pChargeM[i] for i in m.vChargeSegments)

        model.chargeMfrConstraint = Constraint(model.times, rule=charge_mfr_constraint)

        def charge_power_constraint(m, t):
            return m.vCharge[t] == sum(
                -m.vLambdaCharge[t, i] * m.pChargeP[i] for i in m.vChargeSegments)  # Negate to make positive

        model.chargePowerConstraint = Constraint(model.times, rule=charge_power_constraint)

        # Piecewise variables for discharging
        model.vLambdaDischarge = Var(model.times, model.vDischargeSegments, bounds=(0, 1))
        model.vSumLambdaDischarge = Constraint(model.times, rule=lambda m, t: sum(
            m.vLambdaDischarge[t, i] for i in m.vDischargeSegments) == m.vZdop[t])

        # Piecewise constraints for discharging
        def discharge_mfr_constraint(m, t):
            return m.vMFRopd[t] == sum(m.vLambdaDischarge[t, i] * m.pDischargeM[i] for i in m.vDischargeSegments)

        model.dischargeMfrConstraint = Constraint(model.times, rule=discharge_mfr_constraint)

        def discharge_power_constraint(m, t):
            return m.vDischarge[t] == sum(m.vLambdaDischarge[t, i] * m.pDischargeP[i] for i in m.vDischargeSegments)

        model.dischargePowerConstraint = Constraint(model.times, rule=discharge_power_constraint)

        # Add adjacency constraints for SOS2 type behavior
        if len(CHARGE_SEGMENTS) > 2:
            model.vYCharge = Var(model.times, range(len(CHARGE_SEGMENTS) - 1), domain=Binary)

            def charge_sos2_constraint(m, t):
                return sum(m.vYCharge[t, i] for i in range(len(CHARGE_SEGMENTS) - 1)) <= 1

            model.chargeSOS2Constraint = Constraint(model.times, rule=charge_sos2_constraint)

            def charge_adjacency_constraint(m, t, i):
                if i == 0:
                    return m.vLambdaCharge[t, i] <= m.vYCharge[t, i]
                elif i == len(CHARGE_SEGMENTS) - 1:
                    return m.vLambdaCharge[t, i] <= m.vYCharge[t, i - 1]
                else:
                    return m.vLambdaCharge[t, i] <= m.vYCharge[t, i - 1] + m.vYCharge[t, i]

            model.chargeAdjacencyConstraint = Constraint(model.times, model.vChargeSegments,
                                                         rule=charge_adjacency_constraint)

        if len(DISCHARGE_SEGMENTS) > 2:
            model.vYDischarge = Var(model.times, range(len(DISCHARGE_SEGMENTS) - 1), domain=Binary)

            def discharge_sos2_constraint(m, t):
                return sum(m.vYDischarge[t, i] for i in range(len(DISCHARGE_SEGMENTS) - 1)) <= 1

            model.dischargeSOS2Constraint = Constraint(model.times, rule=discharge_sos2_constraint)

            def discharge_adjacency_constraint(m, t, i):
                if i == 0:
                    return m.vLambdaDischarge[t, i] <= m.vYDischarge[t, i]
                elif i == len(DISCHARGE_SEGMENTS) - 1:
                    return m.vLambdaDischarge[t, i] <= m.vYDischarge[t, i - 1]
                else:
                    return m.vLambdaDischarge[t, i] <= m.vYDischarge[t, i - 1] + m.vYDischarge[t, i]

            model.dischargeAdjacencyConstraint = Constraint(model.times, model.vDischargeSegments,
                                                            rule=discharge_adjacency_constraint)
    else:
        # Use linear relationship p = k*m
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

    return model


# Class for BBO optimization parameters
class OptimizationParameters:
    def __init__(self, MIQP_modes, charge_indices, discharge_indices, num_charge, num_discharge,
                 lower_bounds, upper_bounds, mass_norm, store_temps, nu, del_p, rho, ptes,
                 MIQP_m_norms, MIQP_beta_norms, is_final_day, fix_charge_beta_norm=True,
                 fixed_charge_beta_norm_value=1.05):
        self.MIQP_modes = MIQP_modes
        self.charge_indices = charge_indices
        self.discharge_indices = discharge_indices
        self.num_charge = num_charge
        self.num_discharge = num_discharge
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.mass_norm = mass_norm
        self.store_temps = store_temps
        self.nu = nu
        self.del_p = del_p
        self.rho = rho
        self.ptes = ptes
        self.MIQP_m_norms = MIQP_m_norms
        self.MIQP_beta_norms = MIQP_beta_norms
        self.is_final_day = is_final_day
        self.fix_charge_beta_norm = fix_charge_beta_norm
        self.fixed_charge_beta_norm_value = fixed_charge_beta_norm_value


def sigmoid_map(y, params):
    """
    Map unbounded variables to bounded variables using sigmoid function
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
    Expand compact parameters to full parameter vectors with fixed or optimized charging beta
    Now supports full 48-hour optimization
    """
    num_hours = len(params.MIQP_modes)
    m_norms_full = np.zeros(num_hours)
    beta_norms_full = np.zeros(num_hours)

    if params.fix_charge_beta_norm:
        # Fixed charging beta_norm, only optimize m_norms for charging and m_norms+beta for discharging
        charge_idx_start = 0
        discharge_idx_start = params.num_charge
        discharge_beta_idx_start = params.num_charge + params.num_discharge

        # Extract values from compact representation
        charge_m_norms = x_compact[charge_idx_start:discharge_idx_start]
        discharge_m_norms = x_compact[discharge_idx_start:discharge_beta_idx_start]
        discharge_beta_norms = x_compact[discharge_beta_idx_start:]

        # Set charging parameters (fixed beta) for ALL hours
        for i, idx in enumerate(params.charge_indices):
            m_norms_full[idx] = charge_m_norms[i]
            beta_norms_full[idx] = params.fixed_charge_beta_norm_value  # Fixed beta for charging
    else:
        # Optimize both charging m_norm and beta_norm
        charge_idx_start = 0
        charge_beta_idx_start = params.num_charge
        discharge_idx_start = params.num_charge + params.num_charge  # +params.num_charge again for charge_beta
        discharge_beta_idx_start = discharge_idx_start + params.num_discharge

        # Extract values from compact representation
        charge_m_norms = x_compact[charge_idx_start:charge_beta_idx_start]
        charge_beta_norms = x_compact[charge_beta_idx_start:discharge_idx_start]
        discharge_m_norms = x_compact[discharge_idx_start:discharge_beta_idx_start]
        discharge_beta_norms = x_compact[discharge_beta_idx_start:]

        # Set charging parameters (with optimized beta) for ALL hours
        for i, idx in enumerate(params.charge_indices):
            m_norms_full[idx] = charge_m_norms[i]
            beta_norms_full[idx] = charge_beta_norms[i]

    # Set discharge parameters for ALL hours (same for both cases)
    for i, idx in enumerate(params.discharge_indices):
        m_norms_full[idx] = discharge_m_norms[i]
        beta_norms_full[idx] = discharge_beta_norms[i]

    return m_norms_full, beta_norms_full


def compute_ptes_operation(modes, m_norms, beta_norms, mass_norm, store_temps, ptes, day_length=48):
    """
    Compute PTES operation for a given set of parameters

    Returns:
        df: DataFrame with operation results
        current_mass_norm: Updated mass normalization
        current_store_temps: Updated store temperatures
        power_values: Array of PTES power values
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
        'eff_ex': [],
        'eff_com': [],
        'eff_exp': [],
        'mass_norm': [],
        'vio_choke_com': [],
        'vio_choke_exp': [],
        'vio_surge': [],
        'vio_beta': [],
        'vio_t13': [],
        'vio_error': [],
        'n': [],
        'p_exp': [],
        'alpha_com': [],
        'alpha_exp': []
    }

    # Initialize state variables
    current_mass_norm = mass_norm
    current_store_temps = store_temps.copy()
    power_values = np.zeros(day_length)
    old_exergy = ptes.exergy(current_mass_norm, current_store_temps)

    # Compute each hour
    for i in range(day_length):
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

            # Extract additional parameters if they exist
            n_value = perform_results.get('n', np.nan)
            p_exp_value = perform_results.get('p_exp', np.nan)
            alpha_com_value = perform_results.get('alpha_com', np.nan)
            alpha_exp_value = perform_results.get('alpha_exp', np.nan)

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

            else:  # discharge
                # Update temperatures for discharging
                t12 = perform_results["temperatures"]["t12"]
                t9 = perform_results["temperatures"]["t9"]

                # Calculate new mass
                mass_norm_new = current_mass_norm - m_norm

                # Update low hot and high cold temperatures
                if mass_norm_new < 1e-6:  # Avoid division by zero
                    mass_norm_new = 0
                else:
                    current_store_temps['t_lh'] = (current_store_temps['t_lh'] * (MAX_DURATION - current_mass_norm) +
                                                   t12 * m_norm) / (MAX_DURATION - mass_norm_new)
                    current_store_temps['t_hc'] = (current_store_temps['t_hc'] * (MAX_DURATION - current_mass_norm) +
                                                   t9 * m_norm) / (MAX_DURATION - mass_norm_new)
                # Update mass
                current_mass_norm = mass_norm_new
        else:
            power = 0
            eff_com = np.nan
            eff_exp = np.nan
            vio_choke_com = 0
            vio_choke_exp = 0
            vio_surge = 0
            vio_beta = 0
            vio_t13 = 0
            vio_error = 0
            n_value = np.nan
            p_exp_value = np.nan
            alpha_com_value = np.nan
            alpha_exp_value = np.nan

        # Record power value
        power_values[i] = power

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
        data['eff_ex'].append(eff_ex)
        data['eff_com'].append(eff_com)
        data['eff_exp'].append(eff_exp)
        data['vio_choke_com'].append(vio_choke_com)
        data['vio_choke_exp'].append(vio_choke_exp)
        data['vio_surge'].append(vio_surge)
        data['vio_beta'].append(vio_beta)
        data['vio_t13'].append(vio_t13)
        data['vio_error'].append(vio_error)
        data['n'].append(n_value)
        data['p_exp'].append(p_exp_value)
        data['alpha_com'].append(alpha_com_value)
        data['alpha_exp'].append(alpha_exp_value)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Return dataframe, updated state, and power values
    return df, power_values


def evaluate_candidate(args):
    """
    Evaluate a single candidate solution for BBO
    Now optimizes only first 24 hours, keeps second day fixed
    """
    y, params = args

    # Map unbounded variables to bounded variables
    x_compact = sigmoid_map(y, params)

    # Expand parameters to full vectors for first 24 hours only
    m_norms_optimized, beta_norms_optimized = expand_parameters(x_compact, params)

    # Create full 48-hour parameter vectors
    # Use optimized parameters for first day, keep MIQP solution for second day
    m_norms_full = np.copy(params.MIQP_m_norms)
    beta_norms_full = np.copy(params.MIQP_beta_norms)

    # Update ONLY first day parameters (0-23)
    for idx in params.charge_indices:
        m_norms_full[idx] = m_norms_optimized[idx]
        beta_norms_full[idx] = beta_norms_optimized[idx]

    for idx in params.discharge_indices:
        m_norms_full[idx] = m_norms_optimized[idx]
        beta_norms_full[idx] = beta_norms_optimized[idx]

    # Fix second day (24-47) beta_norm for discharge to 1.075
    for t in range(24, 48):
        if params.MIQP_modes[t] == 1:  # discharge mode
            beta_norms_full[t] = 1.075

    # Compute operation for the entire horizon and get power values
    df, power_values = compute_ptes_operation(
        params.MIQP_modes, m_norms_full, beta_norms_full,
        params.mass_norm, params.store_temps, params.ptes, day_length=48
    )

    # Calculate ADMM objective components separately
    ptes_profit = 0
    admm_penalty = 0

    # Calculate profit for entire horizon (48 hours)
    for t in range(48):
        # Calculate profit term
        p_ptes = power_values[t] if power_values[t] > 0 else 0
        c_ptes = -power_values[t] if power_values[t] < 0 else 0

        # Use dual prices - nu has 48 elements for full horizon
        if t < len(params.nu):
            ptes_profit += params.nu[t] * (p_ptes - c_ptes)
        else:
            # For hours beyond nu length, use last available price as approximation
            ptes_profit += params.nu[-1] * (p_ptes - c_ptes)

    # Calculate operation violation for FIRST DAY ONLY (0-23)
    vio_total = df['vio_choke_com'][:24].sum() + df['vio_choke_exp'][:24].sum() + \
                df['vio_surge'][:24].sum() + df['vio_beta'][:24].sum() + df['vio_t13'][:24].sum()
    vio_penalty = penalty_factor * vio_total

    error = df['vio_error'][:48].sum()
    error_penalty = penalty_factor * error
    # Calculate admm penalty for FIRST DAY ONLY (0-23)
    for t in range(48):  # Only first day
        p_ptes = power_values[t] if power_values[t] > 0 else 0
        c_ptes = -power_values[t] if power_values[t] < 0 else 0
        if t < len(params.del_p):
            admm_penalty += (params.rho / 2) * (params.del_p[t] - p_ptes + c_ptes) ** 2

    if params.is_final_day:
        mass_penalty = penalty_factor * abs(df['mass_norm'][23] - 5)
    else:
        mass_penalty = 0

    # Total objective (minimize)
    obj = -ptes_profit + admm_penalty + vio_penalty + mass_penalty + error_penalty

    return (y, obj, ptes_profit, admm_penalty, vio_penalty, power_values, error)


def parallel_powell_polish(initial_y, opt_params, pool):
    """Modified Powell polishing that only optimizes first day but evaluates full horizon"""
    current_y = initial_y.copy()

    # First evaluate the current solution
    _, current_value, _, _, _, _, _ = evaluate_candidate((current_y, opt_params))

    # Optimization parameters
    max_iterations = 100
    delta_threshold = 0.002

    dim = len(current_y)
    delta = 0.01

    # Adaptive parameters
    expansion = 2
    contraction = 1 / 2

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


        # Extract evaluation points
        eval_points = [(y, params) for y, params, _, _, _ in search_directions]

        # Evaluate points in parallel
        evaluation_results = pool.map(evaluate_candidate, eval_points)

        # Combine results with search directions
        full_results = []
        for dir_info, (_, obj, _, _, _, _, _) in zip(search_directions, evaluation_results):
            y_perturbed, _, dim_idx, multiplier, dir_type = dir_info
            full_results.append((y_perturbed, obj, dim_idx, multiplier, dir_type))

        # Find best improvement
        best_perturbed_y = None
        best_perturbed_value = current_value
        best_dim = None
        best_multiplier = None

        for (y_perturbed, obj, dim_idx, multiplier, dir_type) in full_results:
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

            delta *= expansion
        else:
            # No improvement found, reduce step size
            delta *= contraction
            if iteration % 10 == 0:
                print(f"  No improvement found. Reducing step size by {contraction}. Delta is now: {delta:.4f}")

            # Check if step size is below threshold
            if delta < delta_threshold:
                if iteration % 10 == 0:
                    print(f"  Powell converged after {iteration + 1} iterations: step size below threshold")
                break

    return current_y, current_value


def optimize_ptes_operation(nu, del_p, rho, mass_norm, store_temps, ptes, prev_opt_results=None,
                            day_length=48, is_final_day=False, fix_charge_beta_norm=True,
                            fixed_charge_beta_norm_value=1.05):
    """
    Optimize PTES operation using MILP + BBO approach

    Args:
        nu: Dual variable
        del_p: Power mismatch
        rho: ADMM penalty parameter
        mass_norm: Initial mass in storage
        store_temps: Initial storage temperatures
        ptes: PTES model object
        prev_opt_results: Optional results from previous optimization
        day_length: Length of optimization horizon
        is_final_day: Flag for final day constraint
        fix_charge_beta_norm: If True, use a fixed value for charging beta norm
        fixed_charge_beta_norm_value: Fixed value to use for charging beta norm
    """
    start_time = time.time()

    # Check if previous optimization results are available
    if prev_opt_results is not None:
        # Use previous optimization results directly as initialization
        prev_modes, prev_m_norms, prev_beta_norms = prev_opt_results
        modes = prev_modes.copy()
        m_norms = prev_m_norms.copy()
        beta_norms = prev_beta_norms.copy()
    else:
        # Step 1: Solve MIQP model for initial solution
        miqp_model = build_ptes_miqp_model(nu, del_p, rho, mass_norm, day_length=day_length)  # Solve for full 48 hours

        solver = SolverFactory('gurobi')
        with capture_output() as output:
            results = solver.solve(miqp_model)

        # Convert MIQP solution to BBO format
        # Extract binary decisions (charge/discharge/idle) for full 48 hours
        modes = np.zeros(day_length, dtype=int)
        m_norms = np.zeros(day_length)
        beta_norms = np.zeros(day_length)

        for t in range(day_length):
            if miqp_model.vZcop[t].value > 0.5:
                modes[t] = -1  # Charge
                m_norms[t] = miqp_model.vMFRopc[t].value
                beta_norms[t] = 1.05  # Default beta for charging
            elif miqp_model.vZdop[t].value > 0.5:
                modes[t] = 1  # Discharge
                m_norms[t] = miqp_model.vMFRopd[t].value
                beta_norms[t] = 1.075  # Default beta for discharging
            else:
                modes[t] = 0  # Idle
                m_norms[t] = 0.0
                beta_norms[t] = 0.0

    # Identify non-idle time steps for BBO for the first 24 hours
    charge_indices = [i for i, mode in enumerate(modes[:24]) if mode == -1]
    discharge_indices = [i for i, mode in enumerate(modes[:24]) if mode == 1]
    num_charge = len(charge_indices)
    num_discharge = len(discharge_indices)

    print(f"Optimizing {num_charge} charging time steps and {num_discharge} discharging time steps across first 24 hours only")
    print(f"Fix charging beta_norm: {fix_charge_beta_norm}, Fixed value: {fixed_charge_beta_norm_value}")

    # Define bounds for optimization
    m_norm_lower = 0.07
    m_norm_upper = 1.3
    beta_norm_lower = 0.9
    beta_norm_upper = 1.2

    # Create bounds arrays for BBO based on whether charging beta_norm is fixed
    if fix_charge_beta_norm:
        # Only need bounds for: charge m_norm, discharge m_norm, discharge beta_norm
        lower_bounds = np.array(
            [m_norm_lower] * num_charge +
            [m_norm_lower] * num_discharge +
            [beta_norm_lower] * num_discharge)
        upper_bounds = np.array(
            [m_norm_upper] * num_charge +
            [m_norm_upper] * num_discharge +
            [beta_norm_upper] * num_discharge)
    else:
        # Need bounds for: charge m_norm, charge beta_norm, discharge m_norm, discharge beta_norm
        lower_bounds = np.array(
            [m_norm_lower] * num_charge +
            [beta_norm_lower] * num_charge +
            [m_norm_lower] * num_discharge +
            [beta_norm_lower] * num_discharge)
        upper_bounds = np.array(
            [m_norm_upper] * num_charge +
            [beta_norm_upper] * num_charge +
            [m_norm_upper] * num_discharge +
            [beta_norm_upper] * num_discharge)

    # Create optimization parameters object
    opt_params = OptimizationParameters(
        MIQP_modes=modes,
        charge_indices=charge_indices,
        discharge_indices=discharge_indices,
        num_charge=num_charge,
        num_discharge=num_discharge,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        mass_norm=mass_norm,
        store_temps=store_temps,
        nu=nu,
        del_p=del_p,
        rho=rho,
        ptes=ptes,
        MIQP_m_norms=m_norms,
        MIQP_beta_norms=beta_norms,
        is_final_day=is_final_day,
        fix_charge_beta_norm=fix_charge_beta_norm,
        fixed_charge_beta_norm_value=fixed_charge_beta_norm_value
     )

    #  Define inverse sigmoid mapping function
    def inverse_sigmoid_map(x, params):
         bounded_0_1 = (x - params.lower_bounds) / (params.upper_bounds - params.lower_bounds)
         bounded_0_1 = np.clip(bounded_0_1, 0.0001, 0.9999)
         return -np.log(1.0 / bounded_0_1 - 1.0)

    # Prepare initial solution (compact version - all non-idle time steps for full 48 hours)
    initial_charge_m_norms = np.array([m_norms[i] for i in charge_indices])
    initial_discharge_m_norms = np.array([m_norms[i] for i in discharge_indices])
    initial_discharge_beta_norms = np.array([beta_norms[i] for i in discharge_indices])

    if fix_charge_beta_norm:
        #  Only include charge m_norm, discharge m_norm, discharge beta_norm in initial solution
         initial_x_compact = np.concatenate(
             [initial_charge_m_norms, initial_discharge_m_norms, initial_discharge_beta_norms])
    else:
         # Include charge m_norm, charge beta_norm, discharge m_norm, discharge beta_norm
         initial_charge_beta_norms = np.full(num_charge, fixed_charge_beta_norm_value)  # Start with fixed value
         initial_x_compact = np.concatenate(
             [initial_charge_m_norms, initial_charge_beta_norms,
              initial_discharge_m_norms, initial_discharge_beta_norms])

    # Apply inverse mapping to get unbounded variables
    initial_y = inverse_sigmoid_map(initial_x_compact, opt_params)

    # Initialize the CMA optimizer with compact representation
    dim = initial_y.shape[0]

    # Adaptive population size based on dimension
    # popsize = max(int(round(dim / 30.0)) * 30, 30)
    popsize = 64
    optimizer = CMA(
        mean=initial_y,
        sigma=0.03,
        lr_adapt=false,
        population_size=popsize
    )

    # Store the best solution
    best_solution_compact = None
    best_value = float('inf')
    best_power_values = None
    best_y = None  # Store the best unbounded solution for Powell

    # Set number of processes for parallel execution
    num_processes = 32

    # Create process pool
    pool = mp.Pool(processes=num_processes)

    # Adaptive termination criteria
    max_generations = 1000
    stagnation_limit = 1000  # Stop if no improvement for this many generations
    stagnation_counter = 0

    # Run with parallel processing using the modified evaluate function
    for generation in range(max_generations):
        # Generate candidates for this generation
        candidates = [optimizer.ask() for _ in range(optimizer.population_size)]

        # Prepare argument list
        args_list = [(y, opt_params) for y in candidates]

        # Evaluate candidates in parallel using the process pool
        detailed_solutions = pool.map(evaluate_candidate, args_list)

        # Extract (y, value) tuples for the optimizer
        optimizer_solutions = [(y, obj) for y, obj, _, _, _, _, _ in detailed_solutions]

        # Track best solution
        gen_best_value = best_value
        for y, obj, profit, admm_penalty, vio_penalty, power_values, error in detailed_solutions:
            if obj < best_value:
                best_value = obj
                best_profit = profit
                best_admm_penalty = admm_penalty
                best_vio_penalty = vio_penalty
                best_power_values = power_values
                best_error = error
                best_solution_compact = sigmoid_map(y, opt_params)
                best_y = y.copy()  # Store for Powell optimization
                stagnation_counter = 0  # Reset stagnation counter

        # Check for stagnation
        if gen_best_value >= best_value:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        # Update the optimizer
        optimizer.tell(optimizer_solutions)

        # Print progress with detailed breakdown
        if generation % 20 == 0:
            print(f"Generation {generation + 1}, best obj: ${-best_value:.2f}, "
              f"ptes profit: ${best_profit:.2f}, admm penalty: ${best_admm_penalty:.2f}, "
              f"vio penalty: ${best_vio_penalty:.2f}, error: {best_error:.4f}, sigma: {optimizer._sigma:.4f}")
        # Early termination if stagnation detected
        if generation > 100 and (stagnation_counter >= stagnation_limit or optimizer._sigma < 1e-3):
            print(
                f"Early termination after {generation + 1} generations.")
            break

    # Now implement parallel Powell polishing with modified function
    print("Polishing solution with improved Parallel Powell method...")
    polished_y, polished_value = parallel_powell_polish(best_y, opt_params, pool)

    # Check if Powell improved the solution
    if polished_value < best_value:
        print(f"Powell optimization improved the solution from ${-best_value:.2f} to ${-polished_value:.2f}")
        best_y = polished_y
        best_solution_compact = sigmoid_map(best_y, opt_params)

        # Re-evaluate to get updated components
        _, best_value, best_profit, best_admm_penalty, best_vio_penalty, best_power_values, best_error = evaluate_candidate((best_y, opt_params))
        print(f"Updated breakdown - ptes profit: ${best_profit:.2f}, admm penalty: ${best_admm_penalty:.2f}, vio penalty: ${best_vio_penalty:.2f}, error: ${best_error:.4f}")
    else:
        print("Powell optimization did not improve the solution")

    # Close and join the process pool
    pool.close()
    pool.join()

    # Create parameter vectors based on the best solution for first day only
    m_norms_best_24h, beta_norms_best_24h = expand_parameters(best_solution_compact, opt_params)

    # Create full 48-hour parameter vectors - only update first day parameters
    m_norms_best = np.copy(m_norms)
    beta_norms_best = np.copy(beta_norms)

    # Update ONLY first day parameters (0-23)
    for idx in charge_indices:
        m_norms_best[idx] = m_norms_best_24h[idx]
        beta_norms_best[idx] = beta_norms_best_24h[idx]

    for idx in discharge_indices:
        m_norms_best[idx] = m_norms_best_24h[idx]
        beta_norms_best[idx] = beta_norms_best_24h[idx]

    # Fix second day (24-47) beta_norm for discharge to 1.075
    for t in range(24, 48):
        if modes[t] == 1:  # discharge mode
            beta_norms_best[t] = 1.075

    # Compute final operation with optimal variables
    df, power_values = compute_ptes_operation(
        modes, m_norms_best, beta_norms_best, mass_norm, store_temps, ptes, day_length=day_length
    )

    # Calculate final metrics for reporting - only first day penalties
    final_ptes_profit = 0
    final_admm_penalty = 0

    # Calculate profit for entire horizon (48 hours)
    for t in range(48):
        # Calculate profit term
        p_ptes = power_values[t] if power_values[t] > 0 else 0
        c_ptes = -power_values[t] if power_values[t] < 0 else 0

        # Calculate profit for entire horizon using 48-hour nu
        if t < len(nu):
            final_ptes_profit += nu[t] * (p_ptes - c_ptes)

    # Calculate operation violation for FIRST DAY ONLY (0-23)
    vio_total = df['vio_choke_com'][:24].sum() + df['vio_choke_exp'][:24].sum() + \
                df['vio_surge'][:24].sum() + df['vio_beta'][:24].sum() + df['vio_t13'][:24].sum()
    final_vio_penalty = penalty_factor * vio_total

    # Calculate penalty for FIRST DAY ONLY (0-23)
    for t in range(48):  # Only first day
        p_ptes = power_values[t] if power_values[t] > 0 else 0
        c_ptes = -power_values[t] if power_values[t] < 0 else 0
        if t < len(del_p):
            final_admm_penalty += (rho / 2) * (del_p[t] - p_ptes + c_ptes) ** 2

    final_obj = -final_ptes_profit + final_admm_penalty + final_vio_penalty
    best_obj_value = -final_obj

    print(f"PTES optimization complete. Time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"Final metrics - best obj: ${best_obj_value:.2f}, "
          f"ptes profit: ${final_ptes_profit:.2f}, admm penalty: ${final_admm_penalty:.2f}, vio penalty: ${final_vio_penalty:.2f}")

    # Return also optimized parameters for potential reuse
    return df, power_values, (modes, m_norms_best, beta_norms_best), best_obj_value


def adjust_generation(gen_values, delta_power, fleet, committed_gens):
    """
    Adjust conventional generation to handle power mismatches
    """
    # Make a copy of the current generation values
    adjusted_gen = gen_values.copy()

    # Sort generators by marginal cost (most expensive first if decreasing, cheapest first if increasing)
    if delta_power < 0:  # Decrease generation
        # Calculate marginal costs for each generator
        sorted_gens = []
        for g in committed_gens:
            if g not in ['PTES']:
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
            if g not in ['PV', 'Wind', 'PTES']:
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


def calc_committed_generators(gen_values, fleet):
    """
    Determine which generators are committed (having non-zero output)
    """
    committed = []
    for g in fleet.index.values:
        # PV and Wind are always considered available
        if g in ['PV', 'Wind']:
            committed.append(g)
        # Consider a generator committed if its output exceeds a small threshold
        elif g in gen_values and np.any(gen_values[g] > 1e-6):  # Check if any element exceeds threshold
            committed.append(g)
    return committed


def perform_system_operator_optimization_admm(data_dict, ptes_design_file, days=7,
                                              output_dir="./results_minlp_admm/",
                                              fix_charge_beta_norm=True,
                                              fixed_charge_beta_norm_value=1.05):
    """
    Perform system operator optimization with PTES integration using exchange-ADMM
    with early stopping based on PTES objective change detection
    """
    # Extract data from dictionary
    demand = data_dict['demand']
    fleet = data_dict['fleet']
    cfs = data_dict['cfs']

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the PTES model
    ptes = PTES_our(design_file=ptes_design_file)

    # Initialize store temperatures
    current_store_temps = {
        "t_hh": ptes.t_hh_0,
        "t_lh": ptes.t_lh_0,
        "t_hc": ptes.t_hc_0,
        "t_lc": ptes.t_lc_0
    }

    # Initialize mass
    current_mass_norm = 5.0

    # ADMM parameters
    rho = 1  # Initial penalty parameter
    ADMM_CONVERGENCE_TOLERANCE = 0.1  # Convergence tolerance
    MAX_ADMM_ITERATIONS = 30  # Maximum iterations (safety net)
    COST_REDUCTION_THRESHOLD = 5  # Cost reduction threshold for convergence

    # Create DataFrames for results according to specifications

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
        'charge_beta_norm', 'discharge_beta_norm',
        'eff_ex', 'eff_com', 'eff_exp',
        't_hh', 't_lh', 't_hc', 't_lc',
        'n', 'p_exp', 'alpha_com', 'alpha_exp',
        'vio_choke_com', 'vio_choke_exp', 'vio_surge', 'vio_beta', 'vio_t13', 'vio_error'
    ]
    hourly_ptes = pd.DataFrame(columns=hourly_ptes_columns)

    # 4. summary: Daily and total summary statistics
    summary_columns = [
        'day', 'date',
        'total_demand', 'total_cost',
        'ptes_charge_energy', 'ptes_discharge_energy',
        'avg_charge_power_per_m', 'avg_discharge_power_per_m',
        'avg_charge_price', 'avg_discharge_price',
        'admm_iterations', 'optimization_time'
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
        'admm_total_iterations': 0,
        'total_optimization_time': 0
    }

    # Optimize each day
    for d in range(int(days)):

        is_final_day = (d == days - 1)

        day_start_time = time.time()
        print(f"\nOptimizing day {d + 1}/{days}")

        # Define time window indices
        start = (d * 24)
        end_day = start + 24  # End of first day
        end_window = start + 48  # End of 2-day window

        # Get 2-day subsets of demand, solar, wind
        subset_demand = demand['Load'].iloc[start:end_window].copy()
        subset_solarCF = cfs['solar'].iloc[start:end_window].copy()
        subset_windCF = cfs['wind'].iloc[start:end_window].copy()

        # Get initial solution from MILP system operator model
        milp_solution = get_milp_system_operator_solution(
            subset_demand, subset_solarCF, subset_windCF, current_mass_norm, fleet, is_final_day=is_final_day,
            target_final_mass=5.0
        )

        gen_values = milp_solution['gen_values']
        committed_gens = milp_solution['committed_gens']

        # Convert MILP solution to PTES physical model inputs
        modes = np.zeros(48)
        m_norms = np.zeros(48)
        beta_norms = np.zeros(48)

        for t in range(48):
            if milp_solution['ptes_z_charge'][t] > 0.5:
                modes[t] = -1  # Charge
                m_norms[t] = milp_solution['ptes_mfr_charge'][t]
                beta_norms[t] = 1.05  # Default beta for charging
            elif milp_solution['ptes_z_discharge'][t] > 0.5:
                modes[t] = 1  # Discharge
                m_norms[t] = milp_solution['ptes_mfr_discharge'][t]
                beta_norms[t] = 1.075  # Default beta for discharging
            else:
                modes[t] = 0  # Idle
                m_norms[t] = 0.0
                beta_norms[t] = 0.0

        # Get accurate power from PTES physical model
        ptes_df, ptes_power_actual = compute_ptes_operation(
            modes, m_norms, beta_norms, current_mass_norm, current_store_temps, ptes, day_length=48
        )

        # Update ptes_charge and ptes_discharge with accurate values
        ptes_charge = np.maximum(0, -ptes_power_actual)
        ptes_discharge = np.maximum(0, ptes_power_actual)

        # Initialize dual variables from marginal generation costs
        nu = np.zeros(48)
        for t in range(48):
            # Find the marginal generator for this hour (highest MC of committed generators)
            mc_values = {}
            for g in committed_gens:
                if g not in ['PV', 'Wind', 'PTES'] and gen_values[g][t] > 1e-6:
                    # Calculate marginal cost: derivative of cost function = 2*alpha*P + beta
                    alpha = fleet.loc[g, 'alpha']
                    beta = fleet.loc[g, 'beta']
                    mc = 2 * alpha * gen_values[g][t] + beta
                    mc_values[g] = mc

            # Use the highest marginal cost as the initial dual variable
            if mc_values:
                highest_mc_gen = max(mc_values, key=mc_values.get)
                nu[t] = mc_values[highest_mc_gen]
            else:
                # Fallback if no marginal generator found
                nu[t] = 0

        # Prepare for ADMM iterations
        iteration = 0
        primal_residual = float('inf')
        dual_residual = float('inf')
        prev_gen_total = np.zeros(48)
        prev_ptes_power = np.zeros(48)

        # Initialize cost tracking for convergence
        prev_day_cost = float('inf')  # Initialize to infinity for first iteration

        # Initialize storage for previous PTES optimization results
        prev_ptes_opt_results = None

        # PTES objective tracking for early stopping
        ptes_obj_history = []  # Store PTES objective history

        # ADMM main loop with updated convergence criteria
        while iteration < MAX_ADMM_ITERATIONS:
            iteration += 1
            print(f"ADMM Iteration {iteration}/{MAX_ADMM_ITERATIONS}")

            # Store previous values for over-relaxation and dual residual calculation
            prev_gen_values = {g: gen_values[g].copy() for g in fleet.index.values if g != 'PTES'}

            # Calculate previous total generation for over-relaxation
            if iteration > 1:
                prev_gen_total = np.zeros(48)
                for g in fleet.index.values:
                    if g != 'PTES':
                        prev_gen_total += prev_gen_values[g]
                prev_ptes_power = ptes_discharge - ptes_charge

            # Sequential ADMM updates for each generator
            for g in committed_gens:
                if g not in ['PV', 'Wind', 'PTES']:
                    # Calculate del_p for this generator
                    other_gen = np.zeros(48)
                    for other_g in fleet.index.values:
                        if other_g != g and other_g != 'PTES':
                            other_gen += gen_values[other_g]

                    # Include PTES in the power balance
                    ptes_power = ptes_discharge - ptes_charge
                    other_power = other_gen + ptes_power
                    del_p_g = subset_demand.values - other_power

                    # Apply closed-form solution to update this generator
                    gen_values[g] = closed_form_primal(
                        fleet.loc[g, 'alpha'],
                        fleet.loc[g, 'beta'],
                        fleet.loc[g, 'min'],
                        fleet.loc[g, 'max'],
                        rho,
                        nu,
                        del_p_g
                    )

            # Update PTES operation (after all generators have been updated)
            total_gen = np.zeros(48)
            for g in fleet.index.values:
                if g != 'PTES':
                    total_gen += gen_values[g]
            del_p_ptes = subset_demand.values - total_gen

            # Update PTES operation and get objective value
            ptes_df, ptes_power_updated, current_ptes_opt_results, current_ptes_obj = optimize_ptes_operation(
                nu, del_p_ptes, rho, current_mass_norm, current_store_temps, ptes,
                prev_opt_results=prev_ptes_opt_results, day_length=48, is_final_day=is_final_day,
                fix_charge_beta_norm=fix_charge_beta_norm,
                fixed_charge_beta_norm_value=fixed_charge_beta_norm_value
            )

            prev_ptes_opt_results = current_ptes_opt_results
            ptes_charge = np.maximum(0, -ptes_power_updated)
            ptes_discharge = np.maximum(0, ptes_power_updated)

            # Track PTES objective for early stopping
            ptes_obj_history.append(current_ptes_obj)

            print(f"  Current PTES obj: ${current_ptes_obj:.2f}")

            # Calculate current total generation and PTES power
            current_gen_total = np.zeros(48)
            for g in fleet.index.values:
                if g != 'PTES':
                    current_gen_total += gen_values[g]

            current_ptes_power = ptes_discharge - ptes_charge

            # Calculate primal and dual residuals
            p_mismatch = subset_demand.values - current_gen_total - current_ptes_power
            primal_residual = np.linalg.norm(p_mismatch) / np.sqrt(len(p_mismatch))

            # Dual residual calculation
            if iteration > 1:
                gen_change = current_gen_total - prev_gen_total
                ptes_change = current_ptes_power - prev_ptes_power
                dual_residual = rho * np.linalg.norm(gen_change + ptes_change) / np.sqrt(len(gen_change))
            else:
                dual_residual = float('inf')

            # Calculate current first day cost for reporting and convergence check
            current_day_cost = 0
            for t in range(24):
                hour_gen = {}
                hour_mismatch = 0

                for g in fleet.index.values:
                    if g != 'PTES':
                        hour_gen[g] = gen_values[g][t]
                        hour_mismatch += gen_values[g][t]

                hour_mismatch += ptes_discharge[t] - ptes_charge[t]
                hour_mismatch = hour_mismatch - subset_demand.values[t]

                if abs(hour_mismatch) > 1e-6:
                    adjusted_gen = adjust_generation(hour_gen, -hour_mismatch, fleet, committed_gens)
                    for g in adjusted_gen:
                        p = adjusted_gen[g]
                        if p > 0:
                            alpha = fleet.loc[g, 'alpha']
                            beta = fleet.loc[g, 'beta']
                            current_day_cost += alpha * p ** 2 + beta * p
                else:
                    for g in hour_gen:
                        p = hour_gen[g]
                        if p > 0:
                            alpha = fleet.loc[g, 'alpha']
                            beta = fleet.loc[g, 'beta']
                            current_day_cost += alpha * p ** 2 + beta * p

            # Calculate cost reduction from previous iteration
            cost_reduction = prev_day_cost - current_day_cost if prev_day_cost != float('inf') else float('inf')

            print(f"  ADMM Iteration {iteration} - First day system cost: ${current_day_cost:.2f}")
            if prev_day_cost != float('inf'):
                print(f"  Cost reduction from previous iteration: ${cost_reduction:.2f}")
            print(f"  Primal residual: {primal_residual:.6f}, Dual residual: {dual_residual:.6f}")

            # Updated convergence condition:
            # 1. Both residuals must be < 0.1
            # 2. Cost reduction must be < 5 (and not the first iteration)
            residuals_converged = (primal_residual < ADMM_CONVERGENCE_TOLERANCE and
                                   dual_residual < ADMM_CONVERGENCE_TOLERANCE)

            cost_converged = (iteration > 1 and cost_reduction < COST_REDUCTION_THRESHOLD)

            if residuals_converged and cost_converged:
                print(f"ADMM converged after {iteration} iterations!")
                print(f"  - Primal residual: {primal_residual:.6f} < {ADMM_CONVERGENCE_TOLERANCE}")
                print(f"  - Dual residual: {dual_residual:.6f} < {ADMM_CONVERGENCE_TOLERANCE}")
                print(f"  - Cost reduction: ${cost_reduction:.2f} < ${COST_REDUCTION_THRESHOLD}")
                break

            # Store current cost for next iteration comparison
            prev_day_cost = current_day_cost

            # Update dual variables using mismatch
            print(f"  Old dual prices: $[{', '.join(f'{x:.2f}' for x in nu[:48])}...]")
            nu = nu + rho * p_mismatch
            print(f"  New dual prices: $[{', '.join(f'{x:.2f}' for x in nu[:48])}...]")

        # END OF WHILE LOOP - Output termination reason
        if iteration >= MAX_ADMM_ITERATIONS:
            print(f"ADMM stopped: Reached maximum iterations ({MAX_ADMM_ITERATIONS})")
            print(f"  Final primal residual: {primal_residual:.6f}")
            print(f"  Final dual residual: {dual_residual:.6f}")
            if iteration > 1:
                print(f"  Final cost reduction: ${cost_reduction:.2f}")
        else:
            print(f"ADMM stopped: Converged within tolerance")

        # Final adjustment to eliminate any remaining mismatch
        if primal_residual > 1e-6:  # If mismatch is significant
            print(f"Adjusting conventional generators to eliminate remaining mismatch "
                  f"(primal residual: {primal_residual:.2f})")

            # Calculate final power mismatch for each hour
            for t in range(24):  # Only adjust first day
                hour_gen = {}
                hour_mismatch = 0

                # Calculate total generation for this hour
                for g in fleet.index.values:
                    if g != 'PTES':
                        hour_gen[g] = gen_values[g][t]
                        hour_mismatch += gen_values[g][t]

                # Add PTES power
                hour_mismatch += ptes_discharge[t] - ptes_charge[t]

                # Final mismatch
                hour_mismatch = hour_mismatch - subset_demand.values[t]

                # Only adjust if mismatch is significant
                if abs(hour_mismatch) > 1e-6:
                    # Adjust generation to eliminate mismatch
                    adjusted_gen = adjust_generation(hour_gen, -hour_mismatch, fleet, committed_gens)

                    # Update generation values
                    for g in adjusted_gen:
                        gen_values[g][t] = adjusted_gen[g]

        # Calculate marginal prices for first day (for PTES profit calculation)
        marginal_prices = np.zeros(24)
        for t in range(24):
            mc_values = {}
            for g in committed_gens:
                if g not in ['PV', 'Wind', 'PTES'] and gen_values[g][t] > 1e-6:
                    alpha = fleet.loc[g, 'alpha']
                    beta = fleet.loc[g, 'beta']
                    mc = 2 * alpha * gen_values[g][t] + beta
                    mc_values[g] = mc

            if mc_values:
                highest_mc_gen = max(mc_values, key=mc_values.get)
                marginal_prices[t] = mc_values[highest_mc_gen]
            else:
                marginal_prices[t] = 0

        # Store results for each hour of the first day
        daily_charge_energy = 0
        daily_discharge_energy = 0
        daily_charge_price_weighted = 0
        daily_discharge_price_weighted = 0

        for t in range(24):
            global_hour = start + t

            # 1. Populate hourly_dispatch DataFrame
            dispatch_row = {'hour': global_hour, 'day': d + 1, 'demand': subset_demand.values[t]}
            for g in fleet.index.values:
                if g != 'PTES':
                    dispatch_row[f'power_{g}'] = gen_values[g][t]

            dispatch_row['ptes_charge'] = ptes_charge[t]
            dispatch_row['ptes_discharge'] = ptes_discharge[t]
            dispatch_row['ptes_power'] = ptes_discharge[t] - ptes_charge[t]

            hourly_dispatch = pd.concat([hourly_dispatch, pd.DataFrame([dispatch_row])], ignore_index=True)

            # 2. Populate hourly_cost DataFrame
            hour_costs = {}
            hour_total_cost = 0

            for g in fleet.index.values:
                if g != 'PTES':
                    p = gen_values[g][t]
                    if p > 0:
                        alpha = fleet.loc[g, 'alpha']
                        beta = fleet.loc[g, 'beta']
                        cost_g = alpha * p ** 2 + beta * p
                        hour_costs[f'cost_{g}'] = cost_g
                        hour_total_cost += cost_g
                    else:
                        hour_costs[f'cost_{g}'] = 0

            cost_row = {'hour': global_hour, 'day': d + 1, 'total_cost': hour_total_cost,
                        'marginal_price': marginal_prices[t]}
            cost_row.update(hour_costs)

            hourly_cost = pd.concat([hourly_cost, pd.DataFrame([cost_row])], ignore_index=True)

            # 3. Populate hourly_ptes DataFrame
            # Calculate PTES profit based on marginal price
            charge_m_norm = 0
            discharge_m_norm = 0
            charge_beta_norm = 0
            discharge_beta_norm = 0

            # Get the m_norm and beta_norm values for this hour
            if ptes_df is not None and t < len(ptes_df):
                if ptes_df.loc[t, 'mode'] == 'char':
                    charge_m_norm = ptes_df.loc[t, 'm_norm']
                    charge_beta_norm = ptes_df.loc[t, 'beta_norm']
                elif ptes_df.loc[t, 'mode'] == 'dis':
                    discharge_m_norm = ptes_df.loc[t, 'm_norm']
                    discharge_beta_norm = ptes_df.loc[t, 'beta_norm']

            # Calculate PTES profit
            ptes_profit = (ptes_discharge[t] - ptes_charge[t]) * marginal_prices[t]

            # Extract additional parameters from the original calculate_performance results if available
            ptes_row = {
                'hour': global_hour,
                'day': d + 1,
                'charge_power': ptes_charge[t],
                'discharge_power': ptes_discharge[t],
                'profit': ptes_profit,
                'mass_norm': ptes_df.loc[t, 'mass_norm'] if ptes_df is not None and t < len(ptes_df) else 0,
                'charge_m_norm': charge_m_norm,
                'discharge_m_norm': discharge_m_norm,
                'charge_beta_norm': charge_beta_norm,
                'discharge_beta_norm': discharge_beta_norm,
                'eff_ex': ptes_df.loc[t, 'eff_ex'] if ptes_df is not None and t < len(
                    ptes_df) and 'eff_ex' in ptes_df.columns else np.nan,
                'eff_com': ptes_df.loc[t, 'eff_com'] if ptes_df is not None and t < len(
                    ptes_df) and 'eff_com' in ptes_df.columns else np.nan,
                'eff_exp': ptes_df.loc[t, 'eff_exp'] if ptes_df is not None and t < len(
                    ptes_df) and 'eff_exp' in ptes_df.columns else np.nan,
                't_hh': ptes_df.loc[t, 't_hh'] if ptes_df is not None and t < len(ptes_df) else 0,
                't_lh': ptes_df.loc[t, 't_lh'] if ptes_df is not None and t < len(ptes_df) else 0,
                't_hc': ptes_df.loc[t, 't_hc'] if ptes_df is not None and t < len(ptes_df) else 0,
                't_lc': ptes_df.loc[t, 't_lc'] if ptes_df is not None and t < len(ptes_df) else 0,
                'n': ptes_df.loc[t, 'n'] if ptes_df is not None and t < len(
                    ptes_df) and 'n' in ptes_df.columns else np.nan,
                'p_exp': ptes_df.loc[t, 'p_exp'] if ptes_df is not None and t < len(
                    ptes_df) and 'p_exp' in ptes_df.columns else np.nan,
                'alpha_com': ptes_df.loc[t, 'alpha_com'] if ptes_df is not None and t < len(
                    ptes_df) and 'alpha_com' in ptes_df.columns else np.nan,
                'alpha_exp': ptes_df.loc[t, 'alpha_exp'] if ptes_df is not None and t < len(
                    ptes_df) and 'alpha_exp' in ptes_df.columns else np.nan,
                'vio_choke_com': ptes_df.loc[t, 'vio_choke_com'] if ptes_df is not None and t < len(
                    ptes_df) and 'vio_choke_com' in ptes_df.columns else 0,
                'vio_choke_exp': ptes_df.loc[t, 'vio_choke_exp'] if ptes_df is not None and t < len(
                    ptes_df) and 'vio_choke_exp' in ptes_df.columns else 0,
                'vio_surge': ptes_df.loc[t, 'vio_surge'] if ptes_df is not None and t < len(
                    ptes_df) and 'vio_surge' in ptes_df.columns else 0,
                'vio_beta': ptes_df.loc[t, 'vio_beta'] if ptes_df is not None and t < len(
                    ptes_df) and 'vio_beta' in ptes_df.columns else 0,
                'vio_t13': ptes_df.loc[t, 'vio_t13'] if ptes_df is not None and t < len(
                    ptes_df) and 'vio_t13' in ptes_df.columns else 0,
                'vio_error': ptes_df.loc[t, 'vio_error'] if ptes_df is not None and t < len(
                    ptes_df) and 'vio_error' in ptes_df.columns else 0,
            }

            hourly_ptes = pd.concat([hourly_ptes, pd.DataFrame([ptes_row])], ignore_index=True)

            # Accumulate daily totals
            daily_charge_energy += ptes_charge[t]
            daily_discharge_energy += ptes_discharge[t]

            # Accumulate price-weighted values for average calculation
            if ptes_charge[t] > 0:
                daily_charge_price_weighted += ptes_charge[t] * marginal_prices[t]

            if ptes_discharge[t] > 0:
                daily_discharge_price_weighted += ptes_discharge[t] * marginal_prices[t]

        # Update state variables for next day
        first_day_mass = ptes_df.loc[23, 'mass_norm']
        first_day_temps = {
            't_hh': ptes_df.loc[23, 't_hh'],
            't_lh': ptes_df.loc[23, 't_lh'],
            't_hc': ptes_df.loc[23, 't_hc'],
            't_lc': ptes_df.loc[23, 't_lc']
        }
        current_mass_norm = first_day_mass
        current_store_temps = first_day_temps

        # Calculate daily statistics
        day_demand = subset_demand[:24].sum()

        # Calculate total daily cost
        daily_cost = 0
        for g in fleet.index.values:
            if g != 'PTES':
                for t in range(24):
                    p = gen_values[g][t]
                    if p > 0:
                        alpha = fleet.loc[g, 'alpha']
                        beta = fleet.loc[g, 'beta']
                        daily_cost += alpha * p ** 2 + beta * p

        # Calculate average prices (weighted by power)
        avg_charge_price = 0
        if daily_charge_energy > 0:
            avg_charge_price = daily_charge_price_weighted / daily_charge_energy

        avg_discharge_price = 0
        if daily_discharge_energy > 0:
            avg_discharge_price = daily_discharge_price_weighted / daily_discharge_energy

        # Calculate average power per m_norm
        day_ptes_data = hourly_ptes.loc[hourly_ptes['day'] == d + 1]

        avg_charge_power_per_m = 0
        charge_rows = day_ptes_data[day_ptes_data['charge_power'] > 0]
        if not charge_rows.empty:
            valid_rows = charge_rows[charge_rows['charge_m_norm'] > 0]
            if not valid_rows.empty:
                power_per_m_values = valid_rows['charge_power'] / valid_rows['charge_m_norm']
                avg_charge_power_per_m = np.average(power_per_m_values, weights=valid_rows['charge_power'])

        avg_discharge_power_per_m = 0
        discharge_rows = day_ptes_data[day_ptes_data['discharge_power'] > 0]
        if not discharge_rows.empty:
            valid_rows = discharge_rows[discharge_rows['discharge_m_norm'] > 0]
            if not valid_rows.empty:
                power_per_m_values = valid_rows['discharge_power'] / valid_rows['discharge_m_norm']
                avg_discharge_power_per_m = np.average(power_per_m_values, weights=valid_rows['discharge_power'])

        # Record optimization time
        optimization_time = time.time() - day_start_time

        # 4. Populate summary DataFrame
        summary_row = {
            'day': d + 1,
            'date': pd.Timestamp.now().date() - pd.Timedelta(days=7 - d),
            'total_demand': day_demand,
            'total_cost': daily_cost,
            'ptes_charge_energy': daily_charge_energy,
            'ptes_discharge_energy': daily_discharge_energy,
            'avg_charge_power_per_m': avg_charge_power_per_m,
            'avg_discharge_power_per_m': avg_discharge_power_per_m,
            'avg_charge_price': avg_charge_price,
            'avg_discharge_price': avg_discharge_price,
            'admm_iterations': iteration,
            'optimization_time': optimization_time
        }

        summary = pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True)

        # Update totals for final summary row
        total_summary['total_demand'] += day_demand
        total_summary['total_cost'] += daily_cost
        total_summary['ptes_charge_energy'] += daily_charge_energy
        total_summary['ptes_discharge_energy'] += daily_discharge_energy
        total_summary['charge_price_sum'] += daily_charge_price_weighted
        total_summary['discharge_price_sum'] += daily_discharge_price_weighted
        total_summary['admm_total_iterations'] += iteration
        total_summary['total_optimization_time'] += optimization_time

        print(f"Day {d + 1} completed. PTES state: mass={current_mass_norm:.2f}, "
              f"temps=[{current_store_temps['t_hh']:.1f}, {current_store_temps['t_lh']:.1f}, "
              f"{current_store_temps['t_hc']:.1f}, {current_store_temps['t_lc']:.1f}]")
        print(f"System cost: ${daily_cost:.2f}, PTES charge: {daily_charge_energy:.2f} MWh, "
              f"discharge: {daily_discharge_energy:.2f} MWh")
        print(f"ADMM converged in {iteration} iterations with final primal residual: {primal_residual:.4f}, "
              f"dual residual: {dual_residual:.4f}")
        print(f"Optimization time: {optimization_time:.2f}s")

    # Add total summary row
    total_row = {
        'day': 'Total',
        'date': None,
        'total_demand': total_summary['total_demand'],
        'total_cost': total_summary['total_cost'],
        'ptes_charge_energy': total_summary['ptes_charge_energy'],
        'ptes_discharge_energy': total_summary['ptes_discharge_energy'],
        'avg_charge_power_per_m': summary['avg_charge_power_per_m'].mean(),
        'avg_discharge_power_per_m': summary['avg_discharge_power_per_m'].mean(),
        'avg_charge_price': total_summary['charge_price_sum'] / total_summary['ptes_charge_energy'] if total_summary[
                                                                                                           'ptes_charge_energy'] > 0 else 0,
        'avg_discharge_price': total_summary['discharge_price_sum'] / total_summary['ptes_discharge_energy'] if
        total_summary['ptes_discharge_energy'] > 0 else 0,
        'admm_iterations': total_summary['admm_total_iterations'] / days,
        'optimization_time': total_summary['total_optimization_time']
    }

    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)

    # Save all results to CSV files
    hourly_dispatch.to_csv(f"{output_dir}/hourly_dispatch.csv", index=False)
    hourly_cost.to_csv(f"{output_dir}/hourly_cost.csv", index=False)
    hourly_ptes.to_csv(f"{output_dir}/hourly_ptes.csv", index=False)
    summary.to_csv(f"{output_dir}/summary.csv", index=False)

    return hourly_dispatch, hourly_cost, hourly_ptes, summary


def plot_results_admm(hourly_dispatch, hourly_cost, hourly_ptes, demand_data, fleet,
                      output_dir="./results_minlp_admm/"):
    """
    Plot optimization results for ADMM-based system operator

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
    total_costs = hourly_cost['total_cost']
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
    Main function to run the MINLP system operator optimization with ADMM
    """
    # Parameter to control whether charging m_norm is fixed
    fix_charge_beta_norm = True # Set to False to optimize charging beta_norm
    fixed_charge_beta_norm_value = 1.06  # Default value for charging beta_norm

    # Create results directory
    if fix_charge_beta_norm == False:
        results_dir = "results_minlp_admm"
    else:
        results_dir = "results_minlp_admm_t_hh_max"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Define paths
    data_dir = "./"
    ptes_design_file = 'ptes_design_200MW_10h.csv'

    # Load all data in one place
    print("Loading all data files...")
    data_dict = load_all_data(data_dir)

    # Define segments for piecewise linear model if using it
    if USE_PIECEWISE_LINEAR:
        print(f"Using piecewise linear PTES model with:")
        print(f"  Charge segments: {CHARGE_SEGMENTS}")
        print(f"  Discharge segments: {DISCHARGE_SEGMENTS}")
    else:

        print(f"Using linear PTES model with:")
        print(f"  k_charge = {k_charge}")
        print(f"  k_discharge = {k_discharge}")

    print("Starting system operator optimization with MINLP PTES model integration using adaptive ADMM...")
    start_time = time.time()

    # Set multiprocessing start method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')  # More stable for Windows
        except RuntimeError:
            # Method already set
            pass

    # Run the optimization with loaded data
    hourly_dispatch, hourly_cost, hourly_ptes, summary = perform_system_operator_optimization_admm(
        data_dict=data_dict,
        ptes_design_file=ptes_design_file,
        days=7,
        output_dir=results_dir,
        fix_charge_beta_norm=fix_charge_beta_norm,
        fixed_charge_beta_norm_value=fixed_charge_beta_norm_value
    )

    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Generate plots using the results with the updated plotting function
    plot_results_admm(hourly_dispatch, hourly_cost, hourly_ptes, data_dict['demand'], data_dict['fleet'], results_dir)

    # Print summary of results
    print("\nOptimization Summary:")
    print(f"Total system operation cost: ${summary.iloc[-1]['total_cost']:.2f}")
    print(f"Total demand served: {summary.iloc[-1]['total_demand']:.2f} MWh")
    print(f"Total PTES charge energy: {summary.iloc[-1]['ptes_charge_energy']:.2f} MWh")
    print(f"Total PTES discharge energy: {summary.iloc[-1]['ptes_discharge_energy']:.2f} MWh")

    print("\nComplete! Results saved to:", results_dir)


if __name__ == "__main__":
    main()