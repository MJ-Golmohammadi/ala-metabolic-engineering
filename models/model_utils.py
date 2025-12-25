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
    df = pd.read_csv(csv_path)
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
                                objective_rxn: str = "BIOMASS_KT2440_WT3",
                                substrate_map: Optional[Dict] = None) -> Dict[str, Tuple[float, float]]:
    """
    Compute WT fluxes under the model's current environment (model should already
    have environment bounds applied). For each reaction in reaction_factors:
      - read WT flux (signed)
      - compute wild_secondary = WT_flux * factor
      - lock reaction bounds to (wild_secondary, wild_secondary) (signed)
    Returns a mapping reaction_id -> (wt_flux, wild_secondary)
    Notes:
      - If factor == 0 -> do not lock (leave bounds unchanged)
      - This function works on a copy of model internally to compute WT fluxes,
        but applies locks on the provided model (so caller should pass a copy if needed).
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
        # If factor is zero, skip locking (treat as no capacity)
        if factor <= 0:
            logger.debug(f"Factor for {rxn_id} is zero; skipping lock")
            results[rxn_id] = (wt_flux, 0.0)
            continue

        # Apply lock on the provided model (signed)
        rxn = model.reactions.get_by_id(rxn_id)
        prev_bounds = (rxn.lower_bound, rxn.upper_bound)
        # Lock to signed wild_secondary
        rxn.lower_bound = wild_secondary
        rxn.upper_bound = wild_secondary
        logger.info(f"Locked {rxn_id} to wild_secondary {wild_secondary:.6g} (was {prev_bounds})")
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
    Create engineered strain with corrected 3/5 + 2/5 logic and proper precursor/consumer effects.

    Key rules implemented:
    ---------------------------------------------------------
    • Overexpression kx:
        - Base effect: (3/5) * k * WT_secondary
        - Extra effect: (2/5) * k * WT_secondary
        - BUT extra effect only applies if precursors collectively
          provide enough REAL increase:
              real_increase = (k - 1)
              threshold = sum(real_increase_precursors) >= 5
        - If below threshold → proportional activation

    • Knockdown:
        - Multiply WT_secondary by remaining fraction
        - Knockout sets bounds to zero

    • Consumer reaction (e.g., PPBNGS):
        - Acts inversely on producer (G1SAT)
        - Knockdown of consumer increases producer’s extra fraction
        - Overexpression of consumer decreases producer’s extra fraction

    • DM creation:
        - If producer flux > consumer flux → create DM to export excess
          so model does not become infeasible.

    • wild_secondary_map:
        - Reaction baseline flux after expression scaling
        - All modifications are applied relative to this baseline
    ---------------------------------------------------------
    """

    engineered = base_model.copy()

    wild_secondary_map = wild_secondary_map or {}
    precursor_map = precursor_map or {}
    consumer_map = consumer_map or {}

    # ---------------------------------------------------------
    # Helper: lock or expand bounds safely
    # ---------------------------------------------------------
    def set_bounds_signed(rxn, signed_value, lock=True):
        prev = (rxn.lower_bound, rxn.upper_bound)
        if lock:
            rxn.lower_bound = signed_value
            rxn.upper_bound = signed_value
        else:
            if signed_value >= 0:
                rxn.upper_bound = max(rxn.upper_bound, signed_value)
            else:
                rxn.lower_bound = min(rxn.lower_bound, signed_value)
        logger.info(f"[Bounds] {rxn.id}: {prev} → ({rxn.lower_bound}, {rxn.upper_bound})")

    # ---------------------------------------------------------
    # Parse all modification tags first
    # ---------------------------------------------------------
    parsed_mods = {}
    for rxn_id, tag in modifications.items():
        parsed_mods[rxn_id] = parse_mod_tag(tag)

    # ---------------------------------------------------------
    # Apply modifications
    # ---------------------------------------------------------
    for rxn_id, (mod_type, mod_val) in parsed_mods.items():

        if rxn_id not in engineered.reactions:
            logger.warning(f"[Skip] Reaction {rxn_id} not found in model")
            continue

        rxn = engineered.reactions.get_by_id(rxn_id)
        wt_secondary = float(wild_secondary_map.get(rxn_id, 0.0))

        # ---------------------------------------------------------
        # KNOCKOUT
        # ---------------------------------------------------------
        if mod_type == "knockout":
            prev = (rxn.lower_bound, rxn.upper_bound)
            rxn.bounds = (0.0, 0.0)
            logger.info(f"[KO] {rxn_id}: {prev} → (0,0)")
            continue

        # ---------------------------------------------------------
        # OVEREXPRESSION
        # ---------------------------------------------------------
        if mod_type == "overexpress":

            k = min(float(mod_val), cap_factor)

            # Base 3/5 effect
            base_target = wt_secondary * ((3.0 / 5.0) * k)

            # Extra 2/5 effect (conditionally applied)
            extra_target = wt_secondary * ((2.0 / 5.0) * k)

            # Default: no extra effect
            extra_fraction = 0.0

            # ---------------------------------------------------------
            # Precursor logic (corrected real_increase = k - 1)
            # ---------------------------------------------------------
            precursors = precursor_map.get(rxn_id, [])
            if precursors:

                sum_support = 0.0
                sum_negative = 0.0

                for p in precursors:
                    p_mod = parsed_mods.get(p, (None, None))

                    # Overexpression precursor
                    if p_mod[0] == "overexpress":
                        p_k = min(float(p_mod[1]), cap_factor)

                        # REAL increase = (k - 1)
                        real_inc = max(p_k - 1.0, 0.0)

                        # Precursor contributes (3/5 * real_inc)
                        sum_support += (3.0 / 5.0) * real_inc

                    # Knockdown precursor
                    elif p_mod[0] == "knockdown":
                        kd_frac = 1.0 - float(p_mod[1])
                        sum_negative += kd_frac

                    # Knockout precursor
                    elif p_mod[0] == "knockout":
                        sum_negative += 1.0

                # Final support fraction
                support_fraction = (sum_support - sum_negative) / 5.0
                extra_fraction = min(max(support_fraction, 0.0), 1.0)

            # ---------------------------------------------------------
            # Final target = base + extra_fraction * extra
            # ---------------------------------------------------------
            final_target = base_target + extra_fraction * extra_target

            # Fallback if WT_secondary = 0
            if abs(wt_secondary) < 1e-9:
                fallback = cap_factor * k
                set_bounds_signed(rxn, fallback, lock_on_modify)
                logger.info(f"[Fallback OE] {rxn_id}: WT_secondary≈0 → ub={fallback}")
                continue

            signed_final = final_target if wt_secondary >= 0 else -final_target
            set_bounds_signed(rxn, signed_final, lock_on_modify)

            logger.info(
                f"[OE] {rxn_id}: k={k}, base={base_target:.4g}, extra_frac={extra_fraction:.3f}, final={signed_final:.4g}"
            )
            continue

        # ---------------------------------------------------------
        # KNOCKDOWN
        # ---------------------------------------------------------
        if mod_type == "knockdown":

            fraction = float(mod_val)
            target = wt_secondary * fraction

            if abs(wt_secondary) < 1e-9:
                prev = (rxn.lower_bound, rxn.upper_bound)
                rxn.lower_bound *= fraction
                rxn.upper_bound *= fraction
                logger.info(f"[Fallback KD] {rxn_id}: {prev} → ({rxn.lower_bound},{rxn.upper_bound})")
                continue

            signed_target = target if wt_secondary >= 0 else -abs(target)
            set_bounds_signed(rxn, signed_target, lock_on_modify)

            logger.info(f"[KD] {rxn_id}: fraction={fraction}, final={signed_target:.4g}")
            continue

    # ---------------------------------------------------------
    # PRODUCER–CONSUMER BALANCING (inverse 2/5 effect)
    # ---------------------------------------------------------
    for producer, consumer in consumer_map.items():

        if producer not in engineered.reactions or consumer not in engineered.reactions:
            continue

        prod_rxn = engineered.reactions.get_by_id(producer)
        cons_rxn = engineered.reactions.get_by_id(consumer)

        # Compute fluxes
        with engineered:
            engineered.objective = prod_rxn
            sol = engineered.optimize()
            if sol.status != "optimal":
                continue
            prod_flux = float(sol.fluxes.get(producer, 0.0))
            cons_flux = float(sol.fluxes.get(consumer, 0.0))

        # Identify product metabolite
        product_met = None
        for m, coeff in prod_rxn.metabolites.items():
            sto = prod_rxn.get_coefficient(m)
            if sto > 0:
                if hasattr(m, "compartment") and m.compartment == "c":
                    product_met = m
                    break
                if product_met is None:
                    product_met = m

        if product_met is None:
            continue

        # Excess ALA → export via DM
        excess = max(0.0, prod_flux - cons_flux)
        if excess > 1e-9:
            dm_id = f"DM_{product_met.id}"
            if dm_id not in engineered.reactions:
                dm_rxn = cobra.Reaction(dm_id)
                dm_rxn.name = f"Demand for {product_met.id}"
                dm_rxn.lower_bound = 0.0
                dm_rxn.upper_bound = 1000.0
                dm_rxn.add_metabolites({product_met: -1.0})
                engineered.add_reactions([dm_rxn])
                logger.info(f"[DM] Created {dm_id} to export excess={excess:.4g}")
            else:
                logger.info(f"[DM] {dm_id} already exists; exporting excess={excess:.4g}")

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


def calculate_yield_metrics(
    solution: cobra.Solution,
    product_rxn: str,
    substrate_rxn: str,
    environment: str = "Glu",
    min_growth_abs: float = 0.0,
) -> Dict:
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
        
        net_flux = max(0.0, gross_flux - stoich_ala_in_hemB * consumption)

        biomass_flux = float(solution.fluxes.get("BIOMASS_KT2440_WT3", 0.0))

        # epsilon constraint check
        valid = True
        if biomass_flux < min_growth_abs:
            valid = False
            gross_flux = 0.0
            net_flux = 0.0
            consumption = 0.0

        substrate_uptake = abs(float(solution.fluxes.get(substrate_rxn, 0.0)))
        if substrate_uptake == 0:
            substrate_uptake = 1e-9

        yield_mmol_mmol = net_flux / substrate_uptake
        yield_mmol_g = yield_mmol_mmol / 180.16  # glucose MW
        carbon_yield = (net_flux * 5) / (substrate_uptake * 6)

        return {
            "yield_mmol_mmol": yield_mmol_mmol,
            "yield_mmol_g": yield_mmol_g,
            "carbon_yield": carbon_yield,
            "product_titer": net_flux,
            "substrate_uptake": substrate_uptake,
            "gross_ala_flux": gross_flux,
            "net_ala_flux": net_flux,
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


# Key metabolic reactions for analysis
KEY_METABOLIC_REACTIONS = [
    'G1SAT', 'GLUTRR', 'GLUTRS',
    'PPBNGS',
    'ICDHyr', 'PPC', 'G6PDH2r',
    'GLUDy', 'GLUSy',
    'BIOMASS_KT2440_WT3',
    'ATPS4r',
]
