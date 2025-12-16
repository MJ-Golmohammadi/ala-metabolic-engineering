"""
Enhanced Model Utilities for Multi-Environment Metabolic Engineering
Q1 Journal Quality - Supports thermo-constrained models and multiple environments

This module provides utilities for:
- creating engineered strains by applying genetic modifications relative to
  wild-type flux baselines computed under one or more reference objectives,
  with the option to lock a reaction flux to a target value (lb == ub),
- simulating models under specified objectives and environments,
- computing yield metrics and other helpers.

Key change in this version:
- When a modification requests a proportional change (e.g., 'knockdown_80' or
  'overexpress_1.6x') the function will compute a WT baseline flux (max abs
  across provided reference objectives) and then **lock** the reaction flux
  to the requested target by setting both lower_bound and upper_bound to the
  same value (signed according to WT direction). If the WT baseline is
  essentially zero, a safe fallback based on current bounds is used.
"""

import cobra
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_engineered_strain(
    base_model: cobra.Model,
    modifications: Dict,
    reference_objectives: List[str] = None,
    environment: Dict = None,
    objective_to_rxn: Dict[str, str] = None,
    cap_value: float = 6000.0,
    lock_on_modify: bool = True,
    small_eps: float = 1e-9
) -> cobra.Model:
    """
    Create an engineered metabolic model applying modifications relative to
    wild-type fluxes computed under one or more reference objectives.

    Behavior (concise)
    ------------------
    - For each reaction in `modifications`, compute a WT baseline flux as the
      maximum absolute flux observed across the provided reference objectives
      (e.g., 'max_biomass', 'max_ala') under the given environment.
    - For proportional modifications (knockdown/overexpress), compute a target
      flux = abs(WT_flux) * factor.
    - If `lock_on_modify` is True, set both lower_bound and upper_bound to the
      signed target (i.e., lb == ub == target) so the reaction flux is locked
      to the engineered value in subsequent optimizations.
    - If WT flux is essentially zero, fall back to scaling current bounds to
      derive a target; if bounds are also degenerate, log a warning and skip.
    - Knockouts set bounds to (0, 0).
    - The function applies `environment` when computing WT baselines.
    - `objective_to_rxn` maps objective names to reaction IDs (defaults provided).

    Parameters
    ----------
    base_model : cobra.Model
        The wild-type model to copy and modify.
    modifications : Dict
        Mapping reaction_id -> modification tag (e.g., {'PPBNGS': 'knockdown_80'}).
        Supported tags are keys of `modification_factors`.
    reference_objectives : List[str], optional
        List of objective names to use as WT references. Defaults to
        ['max_biomass', 'max_ala'].
    environment : Dict, optional
        Environment bounds to apply when computing WT fluxes.
    objective_to_rxn : Dict[str,str], optional
        Mapping from objective name to reaction id. Defaults provided.
    cap_value : float, optional
        Safety cap for any bound set by this function.
    lock_on_modify : bool, optional
        If True, lock reaction fluxes to the computed target (lb == ub).
        If False, only set upper_bound (for knockdown) or increase upper_bound
        (for overexpression) without locking.
    small_eps : float, optional
        Threshold below which a WT flux is considered effectively zero.

    Returns
    -------
    cobra.Model
        A copy of base_model with modifications applied.
    """

    # Default mapping from objective name to reaction id
    if objective_to_rxn is None:
        objective_to_rxn = {
            'max_biomass': 'BIOMASS_KT2440_WT3',
            'max_ala': 'G1SAT'
        }

    if reference_objectives is None:
        reference_objectives = ['max_biomass', 'max_ala']



# Helper: parse modification tags dynamically into numeric factors
def _parse_modification_factor(change_type: str) -> float:
    """
    Parse a modification tag into a numeric factor.

    Supported dynamic formats:
    - 'knockout' -> 0.0
    - 'knockdown_<p>' -> keep (1 - p/100) of WT flux
        e.g., 'knockdown_80' -> 0.2 (keep 20%), 'knockdown_25' -> 0.75 (keep 75%)
    - 'overexpress_<k>x' -> multiply WT flux by k
        e.g., 'overexpress_1.6x' -> 1.6, 'overexpress_10x' -> 10.0

    Returns
    -------
    float
        The numeric factor to apply.

    Raises
    ------
    ValueError
        If the tag format is not recognized.
    """
    ct = (change_type or "").strip().lower()
    if ct == "knockout":
        return 0.0
    if ct.startswith("knockdown_"):
        try:
            perc_str = ct.split("_", 1)[1]
            perc = float(perc_str)
            return 1.0 - (perc / 100.0)
        except Exception:
            raise ValueError(f"Invalid knockdown tag format: {change_type}")
    if ct.startswith("overexpress_"):
        try:
            val_str = ct.split("_", 1)[1]
            if val_str.endswith("x"):
                val_str = val_str[:-1]
            return float(val_str)
        except Exception:
            raise ValueError(f"Invalid overexpress tag format: {change_type}")
    raise ValueError(f"Unknown modification type: {change_type}")

  
    # Work on a copy to avoid mutating the original model
    engineered = base_model.copy()


# Helper: apply environment bounds to a model (non-destructive)
def _apply_environment(m: cobra.Model, env: Dict):
    if not env:
        return
    for rxn_id, bounds in env.items():
        if rxn_id in m.reactions:
            try:
                m.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
            except Exception:
                # ignore individual failures; caller can inspect logs if needed
                logger.debug(f"Could not set environment bound for {rxn_id}")

# Build set of reactions to check
reactions_to_check = set(modifications.keys())

# Compute WT flux baseline for each reaction across reference objectives
wt_flux_baseline: Dict[str, float] = {r: 0.0 for r in reactions_to_check}

for ref_obj in reference_objectives:
    ref_rxn_id = objective_to_rxn.get(ref_obj)
    if ref_rxn_id is None or ref_rxn_id not in base_model.reactions:
        logger.debug(f"Reference objective {ref_obj} not found in model; skipping")
        continue

    tmp = base_model.copy()
    _apply_environment(tmp, environment)

    # Set objective
    try:
        tmp.objective = tmp.reactions.get_by_id(ref_rxn_id)
    except Exception:
        tmp.objective = ref_rxn_id

    try:
        sol = tmp.optimize()
        if sol.status != 'optimal':
            logger.debug(f"WT solve for {ref_obj} not optimal (status={sol.status}); skipping")
            continue
    except Exception as e:
        logger.warning(f"WT optimization for {ref_obj} failed: {e}")
        continue

    # Record fluxes
    for rxn_id in reactions_to_check:
        flux_val = float(sol.fluxes.get(rxn_id, 0.0))
        prev = wt_flux_baseline.get(rxn_id, 0.0)
        if abs(flux_val) > abs(prev):
            wt_flux_baseline[rxn_id] = flux_val

# Apply modifications using computed baselines (dynamic factors)
for rxn_id, change_type in modifications.items():
    # Parse dynamic factor from tag
    try:
        factor = _parse_modification_factor(change_type)
    except ValueError as e:
        logger.warning(f"{e} - skipping {rxn_id}")
        continue

    # Access reaction in engineered model
    try:
        rxn = engineered.reactions.get_by_id(rxn_id)
    except KeyError:
        logger.warning(f"Reaction {rxn_id} not found in engineered model - skipping")
        continue

    # Knockout handling
    if factor == 0.0:
        prev_bounds = (rxn.lower_bound, rxn.upper_bound)
        rxn.bounds = (0.0, 0.0)
        logger.info(f"Knocked out {rxn_id}: bounds {prev_bounds} -> (0.0, 0.0)")
        continue

    # Get WT baseline flux (signed)
    wt_flux = float(wt_flux_baseline.get(rxn_id, 0.0))
    prev_lower = rxn.lower_bound
    prev_upper = rxn.upper_bound

    # If WT flux is essentially zero, fallback to scaling current bounds
    if abs(wt_flux) <= small_eps:
        logger.info(f"WT flux for {rxn_id} is ~0. Using fallback scaling of current bounds.")
        # Fallback logic for both knockdown and overexpression
        if "knockdown" in change_type:
            # scale current bounds toward zero
            if prev_upper > 0:
                new_ub = max(min(prev_upper * factor, cap_value), 0.0)
            else:
                new_ub = prev_upper
            if prev_lower < 0:
                new_lb = min(max(prev_lower * factor, -cap_value), 0.0)
            else:
                new_lb = prev_lower

            if lock_on_modify:
                # Choose a target based on dominant bound direction
                target = new_ub if abs(new_ub) >= abs(new_lb) else new_lb
                target = max(min(target, cap_value), -cap_value)
                rxn.lower_bound = target
                rxn.upper_bound = target
                logger.info(f"Locked {rxn_id} to fallback target {target:.6f} (bounds {prev_lower}->{rxn.lower_bound}, {prev_upper}->{rxn.upper_bound})")
            else:
                rxn.lower_bound = new_lb
                rxn.upper_bound = new_ub
                logger.info(f"Scaled bounds for {rxn_id}: ({prev_lower:.6f}, {prev_upper:.6f}) -> ({rxn.lower_bound:.6f}, {rxn.upper_bound:.6f})")

        else:
            # overexpression fallback: increase current capacity
            if prev_upper > 0:
                candidate = min(prev_upper * factor, cap_value)
                signed_candidate = candidate
            elif prev_lower < 0:
                candidate = min(abs(prev_lower) * factor, cap_value)
                signed_candidate = -candidate
            else:
                signed_candidate = 0.0

            if lock_on_modify:
                rxn.lower_bound = signed_candidate
                rxn.upper_bound = signed_candidate
                logger.info(f"Locked {rxn_id} to fallback overexpression target {signed_candidate:.6f}")
            else:
                if prev_upper > 0:
                    rxn.upper_bound = candidate
                if prev_lower < 0:
                    rxn.lower_bound = -candidate
                logger.info(f"Expanded bounds for {rxn_id}: ({prev_lower:.6f}, {prev_upper:.6f}) -> ({rxn.lower_bound:.6f}, {rxn.upper_bound:.6f})")
        continue

    # Compute unsigned target flux based on WT baseline and factor
    unsigned_target = min(abs(wt_flux) * factor, cap_value)
    # Determine signed target according to WT direction
    signed_target = unsigned_target if wt_flux >= 0 else -unsigned_target

    if "knockdown" in change_type:
        # Reduce flux to fraction of WT and lock if requested
        if lock_on_modify:
            rxn.lower_bound = signed_target
            rxn.upper_bound = signed_target
            logger.info(f"Locked knockdown {rxn_id}: WT_flux={wt_flux:.6f}, factor={factor:.3f}, bounds -> ({signed_target:.6f}, {signed_target:.6f})")
        else:
            # safer non-locked constraint
            if wt_flux >= 0:
                rxn.upper_bound = signed_target
                if rxn.lower_bound > rxn.upper_bound:
                    rxn.lower_bound = min(0.0, rxn.upper_bound)
            else:
                rxn.lower_bound = signed_target
                if rxn.upper_bound < rxn.lower_bound:
                    rxn.upper_bound = max(0.0, rxn.lower_bound)
            logger.info(f"Applied knockdown (non-locked) to {rxn_id}: bounds -> ({rxn.lower_bound:.6f}, {rxn.upper_bound:.6f})")

    else:
        # Overexpression: increase capacity relative to WT and lock if requested
        if lock_on_modify:
            rxn.lower_bound = signed_target
            rxn.upper_bound = signed_target
            logger.info(f"Locked overexpression {rxn_id}: WT_flux={wt_flux:.6f}, factor={factor:.3f}, bounds -> ({signed_target:.6f}, {signed_target:.6f})")
        else:
            # Conservative expansion (non-locked)
            if prev_upper < unsigned_target:
                rxn.upper_bound = unsigned_target
            if prev_lower > -unsigned_target and prev_lower < 0:
                rxn.lower_bound = -unsigned_target
            logger.info(f"Applied overexpression (non-locked) to {rxn_id}: bounds -> ({rxn.lower_bound:.6f}, {rxn.upper_bound:.6f})")

      return engineered


def simulate_with_objective(model: cobra.Model, objective: str, environment: Dict) -> cobra.Solution:
    """
    Simulate the model under a given objective and environment constraints.

    Behavior:
    - Always apply environment constraints from YAML (carbon source uptake, etc.).
    - When objective is biomass, simply optimize for growth.
    - When objective is ALA (G1SAT), first compute the biomass optimum,
      cap biomass upper bound to that optimum, then re-apply environment
      constraints to ensure substrate uptake is correct, and finally optimize
      for ALA production.
    """

    with model:
        # Get biomass reaction and set permissive bounds initially
        biomass_rxn = model.reactions.get_by_id('BIOMASS_KT2440_WT3')
        biomass_rxn.lower_bound = 0
        biomass_rxn.upper_bound = 6000.0  # Allow growth up to a large cap

        # Step 1: Apply environment constraints from YAML
        for rxn_id, bounds in environment.items():
            if rxn_id in model.reactions:
                try:
                    model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                except Exception as e:
                    logger.warning(f"Could not set bounds for {rxn_id}: {e}")

        # Step 2: Special handling if objective is ALA production
        if objective == 'G1SAT':
            # Compute biomass optimum first
            model.objective = biomass_rxn
            sol_biomass = model.optimize()
            biomass_opt = sol_biomass.objective_value if sol_biomass.status == 'optimal' else 1.0

            # Cap biomass upper bound to realistic maximum
            biomass_rxn.upper_bound = biomass_opt

            # Re-apply environment constraints after biomass optimization
            for rxn_id, bounds in environment.items():
                if rxn_id in model.reactions:
                    try:
                        model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                    except Exception as e:
                        logger.warning(f"Could not re-set bounds for {rxn_id}: {e}")

        # Step 3: Set the desired objective (biomass or ALA)
        try:
            if isinstance(objective, str) and objective in model.reactions:
                model.objective = model.reactions.get_by_id(objective)
            else:
                model.objective = objective
        except Exception as e:
            logger.error(f"Error setting objective {objective}: {e}")
            return cobra.Solution(objective_value=0, status='error', fluxes=pd.Series())

        # Step 4: Perform flux balance analysis
        try:
            solution = model.optimize()
            return solution
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return cobra.Solution(objective_value=0, status='error', fluxes=pd.Series())


def get_substrate_rxn_for_environment(environment: str) -> str:
    """
    Get the appropriate substrate exchange reaction for each environment
    """
    substrate_map = {
        'Glu': 'EX_glc__D_e',
        'Cit': 'EX_cit_e',
        'Ser': 'EX_ser__L_e',
        'Fer': 'EX_fer_e'
    }
    return substrate_map.get(environment, 'EX_glc__D_e')


def get_substrate_properties(environment: str) -> Dict:
    """
    Get molecular weight and carbon content for each substrate
    """
    substrate_properties = {
        'Glu': {'mw': 180.16, 'carbon_atoms': 6},   # Glucose: C6H12O6
        'Cit': {'mw': 192.12, 'carbon_atoms': 6},   # Citrate: C6H8O7
        'Ser': {'mw': 105.09, 'carbon_atoms': 3},   # Serine: C3H7NO3
        'Fer': {'mw': 194.19, 'carbon_atoms': 10}   # Ferulate: C10H10O4
    }
    return substrate_properties.get(environment, {'mw': 180.16, 'carbon_atoms': 6})


def calculate_yield_metrics(solution: cobra.Solution, product_rxn: str,
                            substrate_rxn: str, environment: str = 'Glu') -> Dict:
    """
    Calculate comprehensive yield metrics with net ALA computation.
    """
    try:
        # Calculate gross ALA production from synthase reaction
        gross_ala_flux = solution.fluxes.get(product_rxn, 0)

        # Calculate ALA consumption by downstream heme pathway reactions
        ala_consumer_reactions = ['PPBNGS']  # ALA consumer reactions

        ala_consumption_flux = 0.0
        for consumer_rxn in ala_consumer_reactions:
            if consumer_rxn in solution.fluxes:
                flux_val = solution.fluxes[consumer_rxn]
                if flux_val > 0:
                    ala_consumption_flux += flux_val

        # Net ALA production = Gross production - Consumption by downstream pathways
        net_ala_flux = max(0, gross_ala_flux - ala_consumption_flux)

        # Calculate substrate uptake for yield normalization
        substrate_uptake = abs(solution.fluxes.get(substrate_rxn, 0))

        # Avoid division by zero
        if substrate_uptake == 0:
            substrate_uptake = 1e-9

        # Get substrate properties for yield calculations
        substrate_props = get_substrate_properties(environment)
        substrate_mw = substrate_props['mw']
        substrate_carbon_atoms = substrate_props['carbon_atoms']

        # ALA properties (C5H9NO3)
        ala_mw = 131.13  # g/mol
        ala_carbon_atoms = 5  # C5

        # Calculate yields
        yield_mmol_mmol = net_ala_flux / substrate_uptake
        yield_mmol_g = yield_mmol_mmol / substrate_mw
        carbon_yield = (net_ala_flux * ala_carbon_atoms) / (substrate_uptake * substrate_carbon_atoms)

        return {
            'yield_mmol_mmol': yield_mmol_mmol,
            'yield_mmol_g': yield_mmol_g,
            'carbon_yield': carbon_yield,
            'product_titer': net_ala_flux,
            'substrate_uptake': substrate_uptake,
            'gross_ala_flux': gross_ala_flux,
            'net_ala_flux': net_ala_flux,
            'ala_consumption_flux': ala_consumption_flux,
            'ala_retention_efficiency': (net_ala_flux / gross_ala_flux * 100) if gross_ala_flux > 0 else 0,
            'substrate_mw': substrate_mw,
            'substrate_carbon_atoms': substrate_carbon_atoms
        }

    except Exception as e:
        logger.error(f"Error calculating yield metrics: {e}")
        return {
            'yield_mmol_g': 0,
            'yield_mmol_mmol': 0,
            'carbon_yield': 0,
            'product_titer': 0,
            'substrate_uptake': 0,
            'gross_ala_flux': 0,
            'net_ala_flux': 0,
            'ala_consumption_flux': 0,
            'ala_retention_efficiency': 0,
            'substrate_mw': 0,
            'substrate_carbon_atoms': 0
        }


def load_environment_specific_model(environment: str, models_dir: str = "models/final_constrained_rnaseq_thermo") -> cobra.Model:
    """
    Load environment-specific thermo-constrained model
    """
    model_filename = f"iJN1463_{environment}_ExprThermoConstrainedFile.xml"
    model_path = f"{models_dir}/{model_filename}"

    try:
        model = cobra.io.read_sbml_model(model_path)
        logger.info(f"Successfully loaded model for {environment} environment")
        return model
    except Exception as e:
        logger.error(f"Error loading model for {environment}: {e}")
        raise


def validate_model_growth(model: cobra.Model, environment: Dict, min_growth: float = 0.01) -> bool:
    """
    Validate that model can grow under given conditions.
    """
    try:
        with model:
            for rxn_id, bounds in environment.items():
                if rxn_id in model.reactions:
                    model.reactions.get_by_id(rxn_id).bounds = bounds

            solution = model.optimize()
            return solution.status == 'optimal' and solution.objective_value >= min_growth

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


def calculate_flux_summary(model: cobra.Model, solution: cobra.Solution, key_reactions: List[str]) -> Dict:
    """
    Calculate flux summary for key metabolic reactions.
    """
    flux_summary = {}

    for rxn_id in key_reactions:
        if rxn_id in solution.fluxes:
            flux_summary[rxn_id] = solution.fluxes[rxn_id]
        else:
            flux_summary[rxn_id] = 0.0

    return flux_summary


# Key metabolic reactions for analysis
KEY_METABOLIC_REACTIONS = [
    'G1SAT', 'GLUTRR', 'GLUTRS',      # C5 pathway
    'PPBNGS',                         # ALA consumer
    'ICDHyr', 'PPC', 'G6PDH2r',       # Precursor supply
    'GLUDy', 'GLUSy',                 # Glutamate synthesis
    'BIOMASS_KT2440_WT3',             # Biomass
    'ATPS4r',                         # ATP maintenance
]
