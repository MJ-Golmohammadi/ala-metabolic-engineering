# models/model_utils.py
"""
Model utilities: expression scaling, wild-secondary baseline locking,
safe DM creation, and simulation helpers.

Key features implemented:
- read_expression_scaling: read CSV with expression columns (expression_Glu, _Cit, ...)
  and compute per-reaction scaling factors relative to the minimum non-zero value
  in the chosen environment (min -> factor 1.0).
- apply_wild_secondary_fluxes: compute "secondary wild flux" = WT_flux * factor,
  lock reaction bounds to that value (signed) as baseline for wild-type.
- ensure_demand_for_metabolite: create DM reaction for a metabolite safely.
- simulate_with_objective: returns (solution, min_growth_abs) and applies environment.

Create engineered strain with advanced mutation logic:
- Uses wild_secondary baseline (already applied or provided)
- Applies overexpression and knockdown according to 3/5 + 2/5 rule
- Checks precursor sum to decide whether to enable the extra 2/5
- Handles consumer reaction (e.g., PPBNGS) with reversed logic
- Provides safe fallback when WT flux ~ 0
"""

import cobra
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_modification_factor(change_type: str) -> float:
    """
    Parse modification tag into numeric factor.

    Supported:
      'knockout' -> 0.0
      'knockdown_<p>' -> keep (1 - p/100) of WT flux (e.g., 'knockdown_80' -> 0.2)
      'overexpress_<k>x' -> multiply WT flux by k (e.g., 'overexpress_1.6x' -> 1.6)
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



def parse_mod_tag(tag: str):
    """
    Parse tags like 'overexpress_4x' or 'knockdown_40' or 'knockout'
    Returns ('overexpress', factor) or ('knockdown', fraction) or ('knockout', 0.0)
    """
    t = (tag or "").strip().lower()
    if t == "knockout":
        return ("knockout", 0.0)
    if t.startswith("knockdown_"):
        try:
            perc = float(t.split("_", 1)[1])
            return ("knockdown", 1.0 - perc / 100.0)
        except Exception:
            raise ValueError(f"Invalid knockdown tag: {tag}")
    if t.startswith("overexpress_"):
        try:
            val = t.split("_", 1)[1]
            if val.endswith("x"):
                val = val[:-1]
            return ("overexpress", float(val))
        except Exception:
            raise ValueError(f"Invalid overexpress tag: {tag}")
    raise ValueError(f"Unknown modification tag: {tag}")



def hard_lock_with_auto_supply(model, rxn_id, target_flux, big_M=1000.0):
    """
    Hard-lock a reaction to a fixed flux value while automatically ensuring
    that all required substrates/cofactors are supplied via artificial supply
    reactions. This prevents infeasibility when multiple reactions are locked.

    Steps:
        1. Add unlimited supply for all consumed metabolites.
        2. Set reaction bounds to (target_flux, target_flux).
    """
    rxn = model.reactions.get_by_id(rxn_id)

    # Ensure all substrates/cofactors can be produced
    ensure_unlimited_supply_for_reaction(model, rxn, big_M=big_M)

    # Hard lock the reaction
    rxn.bounds = (target_flux, target_flux)



def ensure_demand_for_product(model: cobra.Model, reaction_id: str, dm_prefix: str = "DM") -> str:
    """
    Ensure a demand reaction exists for a product of `reaction_id`.
    Strategy:
      - Find product metabolite(s) of the reaction (stoichiometry > 0).
      - Prefer metabolite in cytosol (compartment 'c') if available.
      - Create a demand reaction named f"{dm_prefix}_{reaction_id}_c" consuming that metabolite (stoich -1).
      - Return the demand reaction id.

    Returns:
      demand_rxn_id (str)
    """
    if reaction_id not in model.reactions:
        raise KeyError(f"Reaction {reaction_id} not found in model")

    rxn = model.reactions.get_by_id(reaction_id)

    # Find product metabolites (positive stoichiometry in reaction object)
    product_mets = []
    for met, coeff in rxn.metabolites.items():
        # In cobra Reaction, stoichiometry is negative for reactants, positive for products
        try:
            stoich = rxn.get_coefficient(met)
        except Exception:
            # fallback: use metabolites dict value
            stoich = rxn.metabolites.get(met, 0.0)
        if stoich > 0:
            product_mets.append(met)

    # If none found, fallback to any metabolite
    if not product_mets:
        product_mets = list(rxn.metabolites.keys())

    # Prefer cytosolic metabolite
    chosen = None
    for m in product_mets:
        if hasattr(m, "compartment") and m.compartment == "c":
            chosen = m
            break
    if chosen is None and product_mets:
        chosen = product_mets[0]

    if chosen is None:
        # As a last resort, create a generic metabolite
        met_id = f"{reaction_id}_product_c"
        if met_id not in model.metabolites:
            m = cobra.Metabolite(met_id, name=f"{reaction_id}_product", compartment="c")
            model.add_metabolites([m])
            chosen = m
        else:
            chosen = model.metabolites.get_by_id(met_id)

    # Build demand reaction id
    dm_id = f"{dm_prefix}_{reaction_id}_c"
    # Ensure uniqueness
    if dm_id in model.reactions:
        return dm_id

    # Create demand reaction consuming the chosen metabolite (i.e., export)
    dm_rxn = cobra.Reaction(dm_id)
    dm_rxn.name = f"Demand for {chosen.id}"
    dm_rxn.lower_bound = 0.0
    dm_rxn.upper_bound = 1000.0
    dm_rxn.add_metabolites({chosen: -1.0})  # consume product -> export/demand
    model.add_reactions([dm_rxn])
    logger.info(f"Created demand reaction {dm_id} for metabolite {chosen.id}")
    return dm_id



def read_expression_scaling(csv_path: str, environment: str, expr_col_prefix: str = "expression_") -> pd.DataFrame:
    """
    Read CSV that contains columns:
      reaction_id, expression_Glu, expression_Cit, expression_Ser, expression_Fer, ...
    Compute scaling factor per reaction for the requested environment:
      - find min positive value in that environment column -> baseline = 1.0
      - factor = value / min_value
    Returns DataFrame with columns: reaction_id, expr_value, factor
    """
    df = pd.read_csv(csv_path, sep=';')
    env_col = f"{expr_col_prefix}{environment}"
    if env_col not in df.columns:
        raise KeyError(f"Expression column {env_col} not found in {csv_path}")

    # Replace zeros/negatives with NaN for min calculation (we treat them as missing)
    vals = df[env_col].replace({0: pd.NA}).dropna().astype(float)
    if vals.empty:
        raise ValueError(f"No positive expression values found in column {env_col}")

    min_val = float(vals.min())
    if min_val <= 0:
        raise ValueError("Minimum expression value must be positive")

    # compute factor
    df = df.copy()
    df["expr_value"] = pd.to_numeric(df[env_col], errors="coerce").fillna(0.0)
    # For zero or missing, set factor = 0 (no capacity)
    df["factor"] = df["expr_value"].apply(lambda v: float(v / min_val) if v > 0 else 0.0)
    return df[["reaction_id", "expr_value", "factor"]]


def ensure_demand_for_metabolite(model: cobra.Model, metabolite_id: str, dm_prefix: str = "DM") -> str:
    """
    Ensure a demand reaction exists for a metabolite id (metabolite_id).
    If metabolite not present, raise KeyError.
    Returns demand reaction id.
    """
    if metabolite_id not in model.metabolites:
        raise KeyError(f"Metabolite {metabolite_id} not found in model")

    met = model.metabolites.get_by_id(metabolite_id)
    dm_id = f"{dm_prefix}_{met.id}"
    if dm_id in model.reactions:
        return dm_id

    dm = cobra.Reaction(dm_id)
    dm.name = f"Demand for {met.id}"
    dm.lower_bound = 0.0
    dm.upper_bound = 1000.0
    dm.add_metabolites({met: -1.0})
    model.add_reactions([dm])
    logger.info(f"Created demand reaction {dm_id} for metabolite {met.id}")
    return dm_id


def apply_wild_secondary_fluxes(model: cobra.Model, reaction_factors: Dict[str, float],
                                objective_rxn: str = "BIOMASS_KT2440_WT3") -> Dict[str, Tuple[float, float]]:
    """
    Compute WT fluxes under the model's current environment and return
    per-reaction (wt_flux, wild_secondary) WITHOUT mutating original reaction bounds.

    - model: model with environment bounds already applied
    - reaction_factors: reaction_id -> factor (from expression scaling)
    - objective_rxn: biomass reaction id used to compute WT reference

    Returns: dict reaction_id -> (wt_flux, wild_secondary)
    Notes:
      - Does NOT lock bounds here. Locking is deferred to create_engineered_strain,
        which will apply modifications relative to wild_secondary.
      - If factor == 0 -> wild_secondary returned as 0.0
    """
    results = {}

    # compute WT fluxes on a copy to avoid side-effects
    tmp = model.copy()
    try:
        if objective_rxn in tmp.reactions:
            tmp.objective = tmp.reactions.get_by_id(objective_rxn)
        else:
            tmp.objective = objective_rxn
    except Exception:
        tmp.objective = objective_rxn

    sol = tmp.optimize()
    if sol.status != "optimal":
        logger.warning("WT optimization not optimal when computing wild_secondary fluxes; proceeding with available fluxes")

    for rxn_id, factor in reaction_factors.items():
        if rxn_id not in model.reactions:
            logger.debug(f"Reaction {rxn_id} not in model; skipping wild_secondary")
            continue

        wt_flux = float(sol.fluxes.get(rxn_id, 0.0))
        wild_secondary = wt_flux * float(factor)

        # If factor is zero, return wild_secondary 0.0 but do not change model
        if factor <= 0:
            results[rxn_id] = (wt_flux, 0.0)
            continue

        # If wild_secondary is near-zero, still return it (0.0) but do not mutate model
        if abs(wild_secondary) < 1e-12:
            results[rxn_id] = (wt_flux, 0.0)
            continue

        # Return computed values; do NOT change model bounds here
        results[rxn_id] = (wt_flux, wild_secondary)

    return results



def simulate_with_objective(model: cobra.Model, objective: str, environment: Dict,
                            min_growth_fraction: float = 0.01, biomass_id: str = "BIOMASS_KT2440_WT3") -> Tuple[cobra.Solution, float]:
    """
    Simulate model under environment and objective. For ALA objective (G1SAT),
    compute WT biomass and apply epsilon constraint (min_growth_fraction * WT).
    Returns (solution, min_growth_abs).
    """
    with model:
        # apply environment bounds
        for rxn_id, bounds in (environment or {}).items():
            if rxn_id in model.reactions:
                lb, ub = bounds
                model.reactions.get_by_id(rxn_id).lower_bound = lb
                model.reactions.get_by_id(rxn_id).upper_bound = ub

        if objective in ["max_biomass", biomass_id]:
            model.objective = model.reactions.get_by_id(biomass_id)
            sol = model.optimize()
            return sol, 0.0

        if objective in ["max_ala", "G1SAT"]:
            # compute WT biomass on a copy
            tmp = model.copy()
            tmp.objective = tmp.reactions.get_by_id(biomass_id)
            sol_ref = tmp.optimize()
            ref_growth = float(sol_ref.fluxes.get(biomass_id, 0.0)) if sol_ref.status == "optimal" else 0.0
            min_growth_abs = ref_growth * min_growth_fraction if ref_growth > 0 else 0.0

            # enforce on working model
            prev_lb = model.reactions.get_by_id(biomass_id).lower_bound
            model.reactions.get_by_id(biomass_id).lower_bound = max(prev_lb, min_growth_abs)

            model.objective = model.reactions.get_by_id("G1SAT")
            sol = model.optimize()

            # restore
            model.reactions.get_by_id(biomass_id).lower_bound = prev_lb
            return sol, min_growth_abs

        # other objectives
        if objective in model.reactions:
            model.objective = model.reactions.get_by_id(objective)
        else:
            model.objective = objective
        sol = model.optimize()
        return sol, 0.0


def create_engineered_strain(
    base_model: cobra.Model,
    modifications: Dict[str, str],
    wild_secondary_map: Optional[Dict[str, float]] = None,
    precursor_map: Optional[Dict[str, List[str]]] = None,
    consumer_map: Optional[Dict[str, str]] = None,
    lock_on_modify: bool = True,
    cap_factor: float = 10.0
) -> cobra.Model:
    """
    Apply modifications relative to WT secondary fluxes and lock bounds on the
    modified values (not on original bounds). Also create DM reactions for any
    product metabolites that are produced in excess (prod > total_consumption).

    - wild_secondary_map: reaction_id -> signed wild_secondary flux (from apply_wild_secondary_fluxes)
      If a reaction is not present or wild_secondary == 0, fallback logic may be used.
    - modifications: reaction_id -> tag (e.g., 'overexpress_4x', 'knockdown_40', 'knockout')
    - precursor_map: reaction_id -> list of precursor reaction_ids
    - consumer_map: mapping producer -> consumer (optional). We'll still detect other consumers.
    """

    engineered = base_model.copy()
    wild_secondary_map = wild_secondary_map or {}
    precursor_map = precursor_map or {}
    consumer_map = consumer_map or {}

    # parse modifications
    parsed = {}
    for rxn_id, tag in modifications.items():
        try:
            parsed[rxn_id] = parse_mod_tag(tag)
        except Exception as e:
            logger.warning(f"Failed to parse modification tag for {rxn_id}: {tag} ({e})")
            parsed[rxn_id] = (None, None)

    # apply modifications relative to wild_secondary_map
    for rxn_id, (mtype, mval) in parsed.items():
        if rxn_id not in engineered.reactions:
            logger.warning(f"Reaction {rxn_id} not found in model; skipping")
            continue

        rxn = engineered.reactions.get_by_id(rxn_id)
        wt_secondary = float(wild_secondary_map.get(rxn_id, 0.0))

        # KNOCKOUT
        if mtype == "knockout":
            hard_lock_with_auto_supply(engineered, rxn_id, 0.0)
            logger.info(f"[KO] {rxn_id} locked to 0")
            continue

        # KNOCKDOWN
        if mtype == "knockdown":
            fraction = float(mval)  # remaining fraction (e.g., 0.6)
            if abs(wt_secondary) >= 1e-12:
                target = wt_secondary * fraction
                hard_lock_with_auto_supply(engineered, rxn_id, target)
                logger.info(f"[KD] {rxn_id}: wt_secondary={wt_secondary:.6g} -> target={target:.6g}")
            else:
                # fallback: scale current bounds conservatively
                prev = (rxn.lower_bound, rxn.upper_bound)
                rxn.lower_bound *= fraction
                rxn.upper_bound *= fraction
                logger.info(f"[KD-fallback] {rxn_id}: {prev} -> ({rxn.lower_bound},{rxn.upper_bound})")
            continue

        # OVEREXPRESSION (3/5 + conditional 2/5)
        if mtype == "overexpress":
            k = min(float(mval), cap_factor)
            base_mult = (3.0 / 5.0) * k
            extra_mult = (2.0 / 5.0) * k

            # compute extra_fraction from precursors (use real_increase = k-1)
            extra_fraction = 0.0
            precs = precursor_map.get(rxn_id, [])
            if precs:
                sum_support = 0.0
                sum_negative = 0.0
                for p in precs:
                    pmod = parsed.get(p, (None, None))
                    if pmod[0] == "overexpress":
                        p_k = min(float(pmod[1]), cap_factor)
                        real_inc = max(p_k - 1.0, 0.0)
                        sum_support += (3.0 / 5.0) * real_inc
                    elif pmod[0] == "knockdown":
                        kd_frac = 1.0 - float(pmod[1])
                        sum_negative += kd_frac
                    elif pmod[0] == "knockout":
                        sum_negative += 1.0
                support_fraction = (sum_support - sum_negative) / 5.0
                extra_fraction = min(max(support_fraction, 0.0), 1.0)
            else:
                extra_fraction = 0.0

            # final multiplier applied to wt_secondary
            if abs(wt_secondary) >= 1e-12:
                base_target = wt_secondary * base_mult
                extra_target = wt_secondary * extra_mult
                final_target = base_target + extra_fraction * extra_target
                hard_lock_with_auto_supply(engineered, rxn_id, final_target)
                logger.info(f"[OE] {rxn_id}: wt_secondary={wt_secondary:.6g}, k={k}, base={base_target:.6g}, extra_frac={extra_fraction:.3f}, final={final_target:.6g}")
            else:
                # fallback capacity when no WT baseline exists: set a reasonable capacity
                fallback = cap_factor * k              
                hard_lock_with_auto_supply(engineered, rxn_id, fallback)
                logger.info(f"[OE-fallback] {rxn_id}: WT_secondary≈0 -> fallback ub={fallback}")
            continue

    # ---------------------------------------------------------
    # After applying modifications, create DM for any produced metabolite
    # that is produced more than consumed (across all consumers).
    # We run a quick optimization to get fluxes and then inspect net production per metabolite.
    # ---------------------------------------------------------
    with engineered:
        # objective: maximize total production of all demand reactions if exist, else biomass
        # but we only need fluxes; use biomass objective if present
        if "BIOMASS_KT2440_WT3" in engineered.reactions:
            engineered.objective = engineered.reactions.get_by_id("BIOMASS_KT2440_WT3")
        sol = engineered.optimize()
        if sol.status != "optimal":
            logger.warning("Post-modification optimization not optimal; DM creation will still attempt using available fluxes")

        # compute net production per metabolite from reaction fluxes
        # net_prod(m) = sum_r flux_r * stoich(m in r) where stoich positive means production
        net_prod = {}
        fluxes = sol.fluxes if sol is not None else {}
        for rxn_id, flux in fluxes.items():
            if rxn_id not in engineered.reactions:
                continue
            r = engineered.reactions.get_by_id(rxn_id)
            f = float(flux)
            if abs(f) < 1e-12:
                continue
            for met, coeff in r.metabolites.items():
                # coeff < 0 means consumed by reaction as written; coeff > 0 means produced
                net_prod[met.id] = net_prod.get(met.id, 0.0) + f * float(coeff)

        # For each metabolite with positive net production, ensure a DM exists to export it
        for met_id, net in net_prod.items():
            if net > 1e-9:
                # prefer cytosolic metabolite object if available
                try:
                    met = engineered.metabolites.get_by_id(met_id)
                except KeyError:
                    continue
                dm_id = f"DM_{met.id}"
                if dm_id not in engineered.reactions:
                    dm_rxn = cobra.Reaction(dm_id)
                    dm_rxn.name = f"Demand for {met.id}"
                    dm_rxn.lower_bound = 0.0
                    dm_rxn.upper_bound = 1000.0
                    dm_rxn.add_metabolites({met: -1.0})
                    engineered.add_reactions([dm_rxn])
                    logger.info(f"[DM] Created {dm_id} to export net production {net:.6g} of {met.id}")
                else:
                    logger.info(f"[DM] {dm_id} exists; net production {net:.6g} will be exported")

    return engineered





def get_substrate_rxn_for_environment(environment: str) -> str:
    substrate_map = {
        'Glu': 'EX_glc__D_e',
        'Cit': 'EX_cit_e',
        'Ser': 'EX_ser__L_e',
        'Fer': 'EX_fer_e'
    }
    return substrate_map.get(environment, 'EX_glc__D_e')


def get_substrate_properties(environment: str) -> Dict:
    substrate_properties = {
        'Glu': {'mw': 180.16, 'carbon_atoms': 6},
        'Cit': {'mw': 192.12, 'carbon_atoms': 6},
        'Ser': {'mw': 105.09, 'carbon_atoms': 3},
        'Fer': {'mw': 194.19, 'carbon_atoms': 10}
    }
    return substrate_properties.get(environment, {'mw': 180.16, 'carbon_atoms': 6})


    """
    Calculate yield metrics and net product flux (ALA) with biological consistency.

    - gross_flux: flux through the product-forming reaction (e.g. G1SAT).
    - consumption: sum of downstream consumption fluxes (e.g. PPBNGS / hemB).
    - net_flux: gross_flux - stoich * consumption, floored at 0.
    - If biomass flux < min_growth_abs → production is treated as invalid (net_flux = 0).
    """

def calculate_yield_metrics(
    solution: cobra.Solution,
    product_rxn: str,
    substrate_rxn: str,
    environment: str,
    min_growth_abs: float,
) -> Dict:

    try:
        gross_flux = float(solution.fluxes.get(product_rxn, 0.0))

        # hemB consumption (PPBNGS)
        consumption = abs(float(solution.fluxes.get("PPBNGS", 0.0)))

        # Subtract ALA consumption using the true stoichiometric coefficient of hemB (PPBNGS).
        # In the iJN1463 model, the PPBNGS reaction consumes two molecules of ALA per
        # porphobilinogen formed (stoichiometric coefficient = -2 for ALA). Therefore,
        # the raw consumption flux must be multiplied by 2 to correctly account for the
        # biochemical loss of ALA through the hemB step. Without applying this factor,
        # net ALA production would be overestimated and biologically inaccurate.
        stoich_ala_in_hemB = 2.0
        
        net_ala_flux = max(0.0, gross_flux - stoich_ala_in_hemB * consumption)

        biomass_flux = float(solution.fluxes.get("BIOMASS_KT2440_WT3", 0.0))

        # epsilon constraint check
        valid = True
        if biomass_flux < min_growth_abs:
            valid = False
            gross_flux = 0.0
            net_ala_flux = 0.0
            consumption = 0.0

        # Calculate substrate uptake for yield normalization
        substrate_uptake = abs(float(solution.fluxes.get(substrate_rxn, 0.0)))
        
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
            "yield_mmol_mmol": yield_mmol_mmol,
            "yield_mmol_g": yield_mmol_g,
            "carbon_yield": carbon_yield,
            "product_titer": net_ala_flux,
            "substrate_uptake": substrate_uptake,
            "gross_ala_flux": gross_flux,
            "net_ala_flux": net_ala_flux,
            "ala_consumption_flux": consumption,
            "biomass_flux": biomass_flux,
            "valid_production": valid,
        }

    except Exception as e:
        logger.error(f"Yield error: {e}")
        return {
            "yield_mmol_mmol": 0,
            "yield_mmol_g": 0,
            "carbon_yield": 0,
            "product_titer": 0,
            "substrate_uptake": 0,
            "gross_ala_flux": 0,
            "net_ala_flux": 0,
            "ala_consumption_flux": 0,
            "biomass_flux": 0,
            "valid_production": False,
        }





def load_environment_specific_model(environment: str, models_dir: str = "models/final_constrained_rnaseq_thermo") -> cobra.Model:
    model_filename = f"iJN1463_{environment}_preprocessed_with_DM.xml"
    model_path = f"{models_dir}/{model_filename}"
    try:
        model = cobra.io.read_sbml_model(model_path)
        logger.info(f"Successfully loaded model for {environment} environment")
        return model
    except Exception as e:
        logger.error(f"Error loading model for {environment}: {e}")
        raise


def validate_model_growth(model: cobra.Model, environment: Dict, min_growth: float = 0.01) -> bool:
    try:
        with model:
            for rxn_id, bounds in (environment or {}).items():
                if rxn_id in model.reactions:
                    model.reactions.get_by_id(rxn_id).bounds = bounds
            solution = model.optimize()
            return solution.status == 'optimal' and solution.objective_value >= min_growth
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False




def calculate_flux_summary(model: cobra.Model, solution: cobra.Solution, key_reactions: List[str]) -> Dict:
    flux_summary = {}
    for rxn_id in key_reactions:
        flux_summary[rxn_id] = float(solution.fluxes.get(rxn_id, 0.0)) if rxn_id in solution.fluxes else 0.0
    return flux_summary



def ensure_unlimited_supply_for_reaction(model, rxn, supply_prefix="SUPPLY_", big_M=1000.0):
    """
    Create artificial supply reactions for all substrates (negative stoichiometry)
    consumed by a reaction. This guarantees that hard-locking the reaction will
    never make the model infeasible due to missing precursors or cofactors.

    Example:
        A + NADPH → B
        This function will create:
            SUPPLY_A:   → A
            SUPPLY_nadph_c: → nadph_c
    """
    for met, coeff in rxn.metabolites.items():
        if coeff < 0:  # substrate or cofactor
            supply_id = f"{supply_prefix}{met.id}"
            if supply_id in model.reactions:
                continue  # already exists

            supply_rxn = cobra.Reaction(supply_id)
            supply_rxn.name = f"Artificial supply for {met.id}"
            supply_rxn.lower_bound = 0.0      # only produce
            supply_rxn.upper_bound = big_M    # unlimited production

            # nothing → metabolite
            supply_rxn.add_metabolites({met: 1.0})

            model.add_reactions([supply_rxn])



# Key metabolic reactions for analysis
KEY_METABOLIC_REACTIONS = [
    'G1SAT', 'GLUTRR', 'GLUTRS',
    'PPBNGS',
    'ICDHyr', 'PPC', 'G6PDH2r',
    'GLUDy', 'GLUSy',
    'BIOMASS_KT2440_WT3',
    'ATPS4r',
]
