"""
Enhanced Model Utilities for Multi-Environment Metabolic Engineering
Q1 Journal Quality - Supports thermo-constrained models and multiple environments
"""

import cobra
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Setup logging
logger = logging.getLogger(__name__)

def create_engineered_strain(base_model: cobra.Model, modifications: Dict) -> cobra.Model:
    """
    Create engineered metabolic model with specified genetic modifications
    Enhanced for thermo-constrained models
    """
    engineered = base_model.copy()
    
    # Modification factors
    modification_factors = {
        'knockout': 0.0,
        'knockdown_80': 0.2,    # 80% reduction
        'knockdown_40': 0.6,    # 40% reduction
        'knockdown_30': 0.7,    # 30% reduction
        'overexpress_1.5x': 1.5,
        'overexpress_2x': 2.0,
        'overexpress_3x': 3.0,
        'overexpress_5x': 5.0
    }
    
    for rxn_id, change_type in modifications.items():
        try:
            rxn = engineered.reactions.get_by_id(rxn_id)
            
            if change_type in modification_factors:
                factor = modification_factors[change_type]
                
                # For thermo-constrained models, we need to be careful with bounds
                if factor == 0.0:  # Knockout
                    rxn.bounds = (0, 0)
                    logger.info(f"Knocked out reaction: {rxn_id}")
                else:
                    # For overexpression/knockdown, adjust bounds proportionally
                    # But respect thermodynamic constraints
                    current_upper = rxn.upper_bound
                    current_lower = rxn.lower_bound
                    
                    if current_upper > 0:  # Forward reaction
                        new_upper = current_upper * factor
                        rxn.upper_bound = min(new_upper, 1000)  # Cap at reasonable value
                    
                    if current_lower < 0:  # Reverse reaction  
                        new_lower = current_lower * factor
                        rxn.lower_bound = max(new_lower, -1000)
                    
                    logger.info(f"Modified {rxn_id}: {change_type} ({current_upper:.2f} → {rxn.upper_bound:.2f})")
                    
        except KeyError:
            logger.warning(f"Reaction {rxn_id} not found in model - skipping modification")
        except Exception as e:
            logger.error(f"Error modifying {rxn_id}: {e}")
    
    return engineered

def simulate_with_objective(model: cobra.Model, objective: str, environment: Dict) -> cobra.Solution:
    """
    Simulate the model under a given objective and environment constraints.

    Behavior:
    - Always apply environment constraints from YAML (carbon source uptake, etc.).
    - When objective is ALA (G1SAT), compute biomass optimum on a COPY of the model
      to avoid accidental rollback of environment bounds, then cap biomass upper bound
      on the original model and re-apply environment constraints.
    - Extensive logging is included to help debug why environment bounds might not stick.
    """

    # Ensure objective is a reaction id or a reaction object
    try:
        # If objective is a reaction id string, get the reaction object
        if isinstance(objective, str) and objective in model.reactions:
            objective_rxn = model.reactions.get_by_id(objective)
        else:
            objective_rxn = objective  # could be a reaction object or invalid
    except Exception:
        objective_rxn = objective

    # Work inside a context so changes are reverted on exit; we will re-apply as needed
    with model:
        # Get biomass reaction and set permissive bounds initially
        biomass_rxn = model.reactions.get_by_id('BIOMASS_KT2440_WT3')
        biomass_rxn.lower_bound = 0
        biomass_rxn.upper_bound = 6000.0

        # Step 1: Apply environment constraints from YAML and log what we set
        logger.info("Applying environment constraints before any optimization.")
        for rxn_id, bounds in environment.items():
            if rxn_id in model.reactions:
                try:
                    model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                    logger.info("  Applied bounds for %s: %s", rxn_id, tuple(bounds))
                except Exception as e:
                    logger.warning("  Could not set bounds for %s: %s", rxn_id, e)
            else:
                logger.warning("  Environment reaction %s not found in model.", rxn_id)

        # If objective is ALA (G1SAT), compute biomass optimum on a copy to avoid side-effects
        if isinstance(objective, str) and objective == 'G1SAT':
            logger.info("Objective is G1SAT: computing biomass optimum on a model copy.")
            try:
                # Use a copy to compute biomass optimum so original model's environment remains intact
                tmp = model.copy()
                with tmp:
                    # Ensure environment constraints are applied on the copy as well
                    for rxn_id, bounds in environment.items():
                        if rxn_id in tmp.reactions:
                            try:
                                tmp.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                            except Exception:
                                pass
                    # Set biomass objective on the copy and optimize
                    tmp.objective = biomass_rxn
                    sol_biomass = tmp.optimize()
                    if sol_biomass.status == 'optimal':
                        biomass_opt = float(sol_biomass.fluxes.get('BIOMASS_KT2440_WT3', sol_biomass.objective_value))
                        logger.info("  Biomass optimum on copy: %f", biomass_opt)
                    else:
                        biomass_opt = 1.0
                        logger.warning("  Biomass optimization on copy not optimal; using fallback biomass_opt=1.0")
            except Exception as e:
                biomass_opt = 1.0
                logger.error("  Error computing biomass optimum on copy: %s", e)

            # Cap biomass upper bound on the original model
            prev_upper = biomass_rxn.upper_bound
            biomass_rxn.upper_bound = max(0.0, float(biomass_opt))
            logger.info("  Capped biomass upper bound: %f -> %f", prev_upper, biomass_rxn.upper_bound)

            # Re-apply environment constraints on the original model to ensure they persist
            logger.info("Re-applying environment constraints after biomass capping.")
            for rxn_id, bounds in environment.items():
                if rxn_id in model.reactions:
                    try:
                        model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                        logger.info("  Re-applied bounds for %s: %s", rxn_id, tuple(bounds))
                    except Exception as e:
                        logger.warning("  Could not re-set bounds for %s: %s", rxn_id, e)

        # If objective is not G1SAT, we already applied environment and can proceed

        # Step 3: Set the desired objective (use reaction object if possible)
        try:
            if isinstance(objective, str) and objective in model.reactions:
                model.objective = model.reactions.get_by_id(objective)
            else:
                model.objective = objective
        except Exception as e:
            logger.error("Error setting objective %s: %s", objective, e)
            return cobra.Solution(objective_value=0, status='error', fluxes=pd.Series())

        # Log final bounds for key reactions (debugging snapshot)
        debug_keys = ['BIOMASS_KT2440_WT3']
        # include substrate exchange if present in environment
        for k in environment.keys():
            if k.startswith('EX_'):
                debug_keys.append(k)
        logger.info("Final bounds snapshot before optimization:")
        for rxn_id in debug_keys:
            if rxn_id in model.reactions:
                r = model.reactions.get_by_id(rxn_id)
                logger.info("  %s bounds = (%f, %f)", rxn_id, float(r.lower_bound), float(r.upper_bound))

        # Step 4: Perform flux balance analysis
        try:
            solution = model.optimize()
            logger.info("Optimization status: %s, objective_value: %s", solution.status, solution.objective_value)
            return solution
        except Exception as e:
            logger.error("Optimization failed: %s", e)
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
