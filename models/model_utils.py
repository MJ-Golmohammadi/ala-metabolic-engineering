"""
Enhanced Model Utilities for Multi-Environment Metabolic Engineering
Q1 Journal Quality - Supports thermo-constrained models and multiple environments
"""

import cobra
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__)

def create_engineered_strain(
    base_model: cobra.Model,
    modifications: Dict,
    reference_objectives: List[str] = None,
    environment: Dict = None,
    objective_to_rxn: Dict[str, str] = None,
    cap_value: float = 6000.0
) -> cobra.Model:
    """
    Create an engineered metabolic model applying modifications relative to
    wild-type fluxes computed under one or more reference objectives.

    Behavior and rationale
    ----------------------
    - For each reaction in `modifications`, this function computes a sensible
      target bound based on the wild-type (WT) flux of that reaction under the
      provided reference objectives (e.g., 'max_biomass', 'max_ala').
    - If multiple reference objectives are provided, the function computes the
      WT flux under each and uses the most conservative reference (the largest
      absolute WT flux) as the baseline for percentage-based changes. This
      implements the "dual-reference" behavior you requested.
    - Knockouts set bounds to (0, 0).
    - Knockdowns set the reaction capacity to a fraction of the WT flux.
      If WT flux is zero (or extremely small), the function falls back to
      scaling the current bound to avoid creating infeasible or meaningless
      constraints.
    - Overexpression increases capacity relative to both the current bound
      and the observed WT flux (whichever implies a larger capacity), capped
      by `cap_value`.
    - If `environment` is provided, it will be applied when computing WT fluxes.
    - `objective_to_rxn` maps objective names (strings) to reaction IDs in the
      model (e.g., 'max_biomass' -> 'BIOMASS_KT2440_WT3', 'max_ala' -> 'G1SAT').
      A sensible default mapping is used when None is provided.

    Parameters
    ----------
    base_model : cobra.Model
        The wild-type model to copy and modify.
    modifications : Dict
        Mapping reaction_id -> modification tag (e.g., 'PPBNGS': 'knockdown_80').
        Supported tags: keys of modification_factors below.
    reference_objectives : List[str], optional
        List of objective names to use as WT references. If None, defaults to
        ['max_biomass', 'max_ala'].
    environment : Dict, optional
        Optional environment bounds to apply when computing WT fluxes (same
        format as used elsewhere in the project).
    objective_to_rxn : Dict[str,str], optional
        Mapping from objective name to reaction id. If None, a default mapping
        is used.
    cap_value : float, optional
        Upper cap for any bound set by this function (safety to avoid huge bounds).

    Returns
    -------
    cobra.Model
        A copy of base_model with modifications applied.
    """

    # Default mapping from objective name to reaction id (adjust if your project uses different names)
    if objective_to_rxn is None:
        objective_to_rxn = {
            'max_biomass': 'BIOMASS_KT2440_WT3',
            'max_ala': 'G1SAT'
        }

    if reference_objectives is None:
        reference_objectives = ['max_biomass', 'max_ala']

    # Shorthand modification factors for tags that multiply capacity
    modification_factors = {
        'knockout': 0.0,
        'knockdown_80': 0.2,    # keep 20% of WT
        'knockdown_40': 0.6,    # keep 60% of WT
        'knockdown_30': 0.7,    # keep 70% of WT
        'overexpress_1.5x': 1.5,
        'overexpress_2x': 2.0,
        'overexpress_3x': 3.0,
        'overexpress_5x': 5.0
    }

    # Work on a copy
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
                    pass

    # Compute WT flux baseline for each reaction under each reference objective.
    # We'll store the maximum absolute WT flux across the provided objectives
    # as the conservative baseline for percentage-based modifications.
    wt_flux_baseline: Dict[str, float] = {}

    # If there are no modifications that require WT flux, we can skip computing.
    # But computing per-request is safer and still reasonably fast for a few objectives.
    # Build a set of reactions we need baselines for
    reactions_to_check = set(modifications.keys())

    # For each reference objective, compute WT solution and record fluxes
    for ref_obj in reference_objectives:
        # Map objective name to reaction id
        ref_rxn_id = objective_to_rxn.get(ref_obj)
        if ref_rxn_id is None or ref_rxn_id not in base_model.reactions:
            # skip unknown objectives
            continue

        # Use a copy to avoid side-effects
        tmp = base_model.copy()
        # Apply environment if provided
        _apply_environment(tmp, environment)

        # Set biomass objective or product objective as reaction object
        try:
            tmp.objective = tmp.reactions.get_by_id(ref_rxn_id)
        except Exception:
            # fallback: try setting by string (less preferred)
            try:
                tmp.objective = ref_rxn_id
            except Exception:
                continue

        # Solve WT under this reference objective
        try:
            sol = tmp.optimize()
            if sol.status != 'optimal':
                # skip non-optimal results
                continue
        except Exception:
            continue

        # Record fluxes for reactions of interest
        for rxn_id in reactions_to_check:
            flux_val = float(sol.fluxes.get(rxn_id, 0.0))
            prev = wt_flux_baseline.get(rxn_id, 0.0)
            # keep the maximum absolute flux across objectives (conservative baseline)
            if abs(flux_val) > abs(prev):
                wt_flux_baseline[rxn_id] = flux_val

    # Now apply modifications using the computed baselines
    for rxn_id, change_type in modifications.items():
        try:
            rxn = engineered.reactions.get_by_id(rxn_id)
        except KeyError:
            logger.warning(f"Reaction {rxn_id} not found in model - skipping modification")
            continue
        except Exception as e:
            logger.error(f"Error accessing reaction {rxn_id}: {e}")
            continue

        # If the tag is not recognized, skip with a warning
        if change_type not in modification_factors:
            logger.warning(f"Unknown modification tag '{change_type}' for {rxn_id} - skipping")
            continue

        factor = modification_factors[change_type]

        # Knockout: set both bounds to zero
        if factor == 0.0:
            prev_bounds = (rxn.lower_bound, rxn.upper_bound)
            rxn.bounds = (0.0, 0.0)
            logger.info(f"Knocked out reaction {rxn_id}: bounds {prev_bounds} -> (0.0, 0.0)")
            continue

        # Determine WT baseline flux for this reaction (may be zero or missing)
        wt_flux = float(wt_flux_baseline.get(rxn_id, 0.0))

        # If WT flux is essentially zero, fall back to scaling current bounds
        small_eps = 1e-9
        if abs(wt_flux) <= small_eps:
            # fallback behavior:
            # - for knockdown: scale current upper/lower by factor
            # - for overexpression: multiply current upper by factor
            prev_upper = rxn.upper_bound
            prev_lower = rxn.lower_bound

            if change_type.startswith('knockdown'):
                # reduce capacity proportionally to current bound magnitude
                if prev_upper > 0:
                    rxn.upper_bound = max(min(prev_upper * factor, cap_value), 0.0)
                if prev_lower < 0:
                    rxn.lower_bound = min(max(prev_lower * factor, -cap_value), 0.0)
                logger.info(
                    "Knockdown fallback for %s: bounds (%.6f, %.6f) -> (%.6f, %.6f)",
                    rxn_id, prev_lower, prev_upper, rxn.lower_bound, rxn.upper_bound
                )
            else:
                # overexpression fallback: increase current capacity
                if prev_upper > 0:
                    rxn.upper_bound = min(prev_upper * factor, cap_value)
                if prev_lower < 0:
                    rxn.lower_bound = max(prev_lower * factor, -cap_value)
                logger.info(
                    "Overexpression fallback for %s: bounds (%.6f, %.6f) -> (%.6f, %.6f)",
                    rxn_id, prev_lower, prev_upper, rxn.lower_bound, rxn.upper_bound
                )
            continue

        # If we have a meaningful WT flux, compute target flux based on factor
        target_flux = abs(wt_flux) * factor  # fraction of WT (for knockdown) or multiplier (for overexp use below)

        prev_upper = rxn.upper_bound
        prev_lower = rxn.lower_bound

        if change_type.startswith('knockdown'):
            # For knockdown tags, factor is the fraction to keep (e.g., 0.2 keeps 20%).
            # We set the upper bound (for forward flux) to target_flux and leave lower bound unchanged
            # unless reaction was previously reversible and WT flux was negative.
            if wt_flux >= 0:
                # forward direction dominated in WT
                rxn.upper_bound = min(max(target_flux, 0.0), cap_value)
                # ensure lower bound is not greater than upper bound
                if rxn.lower_bound > rxn.upper_bound:
                    rxn.lower_bound = min(0.0, rxn.upper_bound)
            else:
                # WT flux negative -> reverse direction dominated
                rxn.lower_bound = max(min(-target_flux, 0.0), -cap_value)
                if rxn.upper_bound < rxn.lower_bound:
                    rxn.upper_bound = max(0.0, rxn.lower_bound)

            logger.info(
                "Applied knockdown to %s: WT_flux=%.6f, factor=%.3f, bounds (%.6f -> %.6f)",
                rxn_id, wt_flux, factor, prev_upper, rxn.upper_bound
            )

        else:
            # Overexpression: increase capacity. We choose a conservative rule:
            # new_upper = max(current_upper * factor, abs(WT_flux) * factor)
            # This ensures that if WT flux was small but current bound is large, we still scale the bound,
            # and if WT flux was large, we allow capacity proportional to observed flux.
            new_upper_candidate = max(prev_upper * factor if prev_upper > 0 else 0.0,
                                      target_flux * factor if target_flux > 0 else 0.0)
            # also ensure at least prev_upper (do not shrink)
            new_upper = min(max(prev_upper, new_upper_candidate), cap_value)

            # For reverse direction, scale lower bound similarly (more negative)
            if prev_lower < 0:
                new_lower_candidate = min(prev_lower * factor, -abs(target_flux) * factor)
                new_lower = max(new_lower_candidate, -cap_value)
            else:
                new_lower = prev_lower

            rxn.upper_bound = new_upper
            rxn.lower_bound = new_lower

            logger.info(
                "Applied overexpression to %s: WT_flux=%.6f, factor=%.3f, bounds (%.6f, %.6f) -> (%.6f, %.6f)",
                rxn_id, wt_flux, factor, prev_lower, prev_upper, rxn.lower_bound, rxn.upper_bound
            )

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

            # ✅ Re-apply environment constraints after biomass optimization
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
        'Fer': {'mw': 194.19, 'carbon_atoms': 10}     # ferulate: "C10H10O4
    }
    return substrate_properties.get(environment, {'mw': 180.16, 'carbon_atoms': 6})


def calculate_yield_metrics(solution: cobra.Solution, product_rxn: str, 
                          substrate_rxn: str, environment: str = 'Glu') -> Dict:
    """
    Calculate comprehensive yield metrics with net ALA computation
    Enhanced for multiple environments and robust error handling
    """
    try:
        # Calculate gross ALA production from synthase reaction
        gross_ala_flux = solution.fluxes.get(product_rxn, 0)
        
        # Calculate ALA consumption by downstream heme pathway reactions
        ala_consumer_reactions = ['PPBNGS']  # ALA consumer reactions
        
        ala_consumption_flux = 0.0
        for consumer_rxn in ala_consumer_reactions:
            if consumer_rxn in solution.fluxes:
                # Positive flux indicates consumption of ALA substrate
                flux_val = solution.fluxes[consumer_rxn]
                if flux_val > 0:  # Only count consumption, not production
                    ala_consumption_flux += flux_val
        
        # Net ALA production = Gross production - Consumption by downstream pathways
        net_ala_flux = max(0, gross_ala_flux - ala_consumption_flux)
        
        # Calculate substrate uptake for yield normalization
        substrate_uptake = abs(solution.fluxes.get(substrate_rxn, 0))
        
        # Avoid division by zero
        if substrate_uptake == 0:
            substrate_uptake = 1e-9  # Small value to avoid division by zero
        
        # Get substrate properties for yield calculations
        substrate_props = get_substrate_properties(environment)
        substrate_mw = substrate_props['mw']
        substrate_carbon_atoms = substrate_props['carbon_atoms']
        
        # ALA properties (C5H9NO3)
        ala_mw = 131.13  # g/mol
        ala_carbon_atoms = 5  # C5
        
        # Calculate yields CORRECTLY
        # mmol-ALA / mmol-substrate (molar yield)
        yield_mmol_mmol = net_ala_flux / substrate_uptake

        # mmol-ALA / g-substrate (mass-normalized yield)
        # Convert mmol-substrate to g-substrate using MW (g/mmol)
        yield_mmol_g = yield_mmol_mmol / substrate_mw


        # Carbon yield: (C-mol ALA) / (C-mol substrate)
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
    Validate that model can grow under given conditions
    """
    try:
        with model:
            # Apply environment
            for rxn_id, bounds in environment.items():
                if rxn_id in model.reactions:
                    model.reactions.get_by_id(rxn_id).bounds = bounds
            
            # Test growth
            solution = model.optimize()
            return solution.status == 'optimal' and solution.objective_value >= min_growth
            
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


def calculate_flux_summary(model: cobra.Model, solution: cobra.Solution, key_reactions: List[str]) -> Dict:
    """
    Calculate flux summary for key metabolic reactions
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
