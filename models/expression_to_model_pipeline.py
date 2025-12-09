# expression_to_model_pipeline_v3.py
"""
Pipeline v3: E-Flux scaling with LB adjustment
- multi-environment CSV input (semicolon separated)
- detect essential reactions via single_reaction_deletion
- apply E-Flux scaling that adjusts both UB and LB for internal reactions
- EX_ reactions are controlled only via YAML environment files (not by RNA-seq)
- optional pyTFA thermodynamic curation
- outputs: one SBML per environment + diagnostics JSON
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cobra
from cobra import Model, Reaction
from cobra.flux_analysis import single_reaction_deletion


# -------------------------
# Normalization helpers
# -------------------------
def median_ratio_normalization(counts_df: pd.DataFrame) -> pd.Series:
    with np.errstate(divide='ignore'):
        geom_means = counts_df.replace(0, np.nan).apply(lambda x: np.exp(np.nanmean(np.log(x))), axis=1)
    geom_means[geom_means == 0] = np.nan
    ratios = counts_df.div(geom_means, axis=0)
    size_factors = ratios.median(axis=0)
    normalized = counts_df.div(size_factors, axis=1)
    if normalized.shape[1] == 1:
        return normalized.iloc[:, 0]
    return normalized

def counts_to_cpm(counts: pd.Series) -> pd.Series:
    total = counts.sum()
    if total == 0:
        return counts * 0.0
    return counts / total * 1e6

def log_transform(series: pd.Series, pseudocount: float = 1.0) -> pd.Series:
    return np.log2(series + pseudocount)

# -------------------------
# GPR mapping helpers
# -------------------------
def aggregate_expression_for_reaction(reaction: Reaction, gene_expr: Dict[str, float]) -> float:
    rule = reaction.gene_reaction_rule.strip()
    genes = [g.id for g in reaction.genes]
    if len(genes) == 0:
        return 0.0
    vals = [float(gene_expr.get(g, 0.0)) for g in genes]
    rule_lower = rule.lower()
    if ' and ' in rule_lower and ' or ' not in rule_lower:
        return float(np.nanmin(vals)) if len(vals) > 0 else 0.0
    elif ' or ' in rule_lower and ' and ' not in rule_lower:
        return float(np.nanmax(vals)) if len(vals) > 0 else 0.0
    else:
        return float(np.nanmax(vals)) if len(vals) > 0 else 0.0

def map_expression_to_reactions(model: Model, gene_expression: pd.Series) -> Dict[str, float]:
    gene_expr_dict = gene_expression.to_dict()
    rxn_scores = {}
    for rxn in model.reactions:
        rxn_scores[rxn.id] = aggregate_expression_for_reaction(rxn, gene_expr_dict)
    return rxn_scores

# -------------------------
# E-Flux scaling with LB adjustment
# -------------------------
def apply_eflux_scaling_with_lb(
    model: Model,
    rxn_scores: Dict[str, float],
    ub_max: float = 1000.0,
    ub_min_fraction: float = 0.01,
    eps: float = 1e-6,
    whitelist_rxns: Optional[List[str]] = None,
    exchange_prefixes: Optional[List[str]] = None,
    preserve_original_bounds: bool = True
) -> Tuple[Model, Dict[str, Tuple[float, float]]]:
    """
    Scale UB and adjust LB for internal reactions based on expression scores.
    - EX_ reactions (prefixes) are excluded from scaling and controlled by environment YAML
    - whitelist_rxns are preserved (not closed); if scaling would close them, they are restored to eps capacity
    Returns modified model and original bounds dict
    """
    if whitelist_rxns is None:
        whitelist_rxns = []
    if exchange_prefixes is None:
        exchange_prefixes = ["EX_"]

    if preserve_original_bounds:
        model = model.copy()

    # prepare score range
    scores = np.array(list(rxn_scores.values()))
    rmin = float(np.nanmin(scores)) if len(scores) > 0 else 0.0
    rmax = float(np.nanmax(scores)) if len(scores) > 0 else 1.0
    if rmax == rmin:
        rmax = rmin + 1.0

    ub_min = ub_max * ub_min_fraction
    original_bounds = {}
    changed_rxns = []

    for rxn in model.reactions:
        original_bounds[rxn.id] = (rxn.lower_bound, rxn.upper_bound)
        # skip exchange reactions
        if any(rxn.id.startswith(pref) for pref in exchange_prefixes):
            continue
        # skip whitelist (do not scale them)
        if rxn.id in whitelist_rxns:
            continue

        score = float(rxn_scores.get(rxn.id, 0.0))
        norm = np.clip((score - rmin) / (rmax - rmin), 0.0, 1.0)
        ub_scaled = ub_min + norm * (ub_max - ub_min)
        if ub_scaled < eps:
            ub_scaled = eps

        # apply rules based on reversibility and original bounds
        if rxn.reversibility:
            rxn.lower_bound = -ub_scaled
            rxn.upper_bound = ub_scaled
        else:
            orig_lb, orig_ub = original_bounds[rxn.id]
            # irreversible forward (orig_lb >= 0)
            if orig_lb >= 0:
                rxn.lower_bound = max(orig_lb, eps)
                rxn.upper_bound = ub_scaled
            # irreversible backward (orig_ub <= 0)
            elif orig_ub <= 0:
                # keep negative direction, set ub to small negative and lb to -ub_scaled
                rxn.upper_bound = min(orig_ub, -eps)
                rxn.lower_bound = -ub_scaled
            else:
                # fallback: keep original lb sign, set ub scaled
                rxn.lower_bound = orig_lb
                rxn.upper_bound = ub_scaled

        changed_rxns.append(rxn.id)

    # ensure whitelist reactions have at least eps capacity
    for rxn_id in whitelist_rxns:
        if rxn_id in model.reactions:
            rxn = model.reactions.get_by_id(rxn_id)
            lb, ub = rxn.lower_bound, rxn.upper_bound
            if rxn.reversibility:
                if abs(lb) < eps and abs(ub) < eps:
                    rxn.lower_bound = -eps
                    rxn.upper_bound = eps
            else:
                # irreversible forward: ensure ub positive
                if ub <= 0:
                    rxn.upper_bound = eps
                if lb >= 0 and lb < eps:
                    rxn.lower_bound = eps

    return model, original_bounds

# -------------------------
# Essential reaction detection using single_reaction_deletion
# -------------------------
def find_essential_reactions(
    model: Model,
    biomass_rxn_id: str = "BIOMASS_KT2440_WT3",
    threshold_fraction: float = 0.05,
    absolute_threshold: float = 1e-6,
    processes: int = 4,
    reactions_to_test: Optional[List[str]] = None
) -> List[str]:
    m = model.copy()
    if biomass_rxn_id not in m.reactions:
        raise KeyError(f"Biomass reaction {biomass_rxn_id} not found in model.")
    with m:
        m.objective = biomass_rxn_id
        ref_sol = m.optimize()
        ref_growth = float(ref_sol.fluxes.get(biomass_rxn_id, 0.0)) if ref_sol.status == 'optimal' else 0.0
    cutoff = threshold_fraction * ref_growth if ref_growth > 0 else absolute_threshold
    if reactions_to_test is None:
        reactions_to_test = [rxn.id for rxn in m.reactions if rxn.id != biomass_rxn_id]
    try:
        deletion_result = single_reaction_deletion(m, reactions=reactions_to_test, processes=processes)
    except Exception:
        deletion_result = single_reaction_deletion(m, reactions=reactions_to_test, processes=1)
    if 'growth' in deletion_result.columns:
        growth_after = deletion_result['growth']
    elif 'fluxes' in deletion_result.columns:
        growth_after = deletion_result['fluxes'].apply(lambda d: d.get(biomass_rxn_id, 0.0) if isinstance(d, dict) else 0.0)
    else:
        growth_after = deletion_result.iloc[:, 0]
    essential = []
    for rxn_id, g in growth_after.items():
        try:
            gval = float(g)
        except Exception:
            gval = 0.0
        if gval < cutoff:
            essential.append(rxn_id)
    return essential

# -------------------------
# Optional pyTFA thermodynamic curation
# -------------------------
def run_pytfa_thermo_curation(model: Model, thermo_db_path: Optional[str] = None) -> Model:
    if not PYTFA_AVAILABLE:
        warnings.warn("pyTFA not available; skipping thermodynamic curation.")
        return model
    if thermo_db_path is None:
        try:
            thermo_db = load_thermoDB()
        except Exception as e:
            warnings.warn(f"Could not load default thermoDB: {e}. Skipping pyTFA.")
            return model
    else:
        thermo_db = load_thermoDB(thermo_db_path)
    from pytfa import ThermoModel
    tmodel = ThermoModel(thermo_db, model)
    tmodel.prepare()
    try:
        sol = tmodel.optimize()
        if sol.status != 'optimal':
            warnings.warn("ThermoModel optimization not optimal; review thermo constraints.")
    except Exception as e:
        warnings.warn(f"pyTFA optimization failed: {e}")
    return model

# -------------------------
# Build context-specific model for one environment
# -------------------------
def build_context_specific_model_from_rnaseq_single_env(
    model: Model,
    rnaseq_counts: pd.Series,
    biomass_rxn_id: str = "BIOMASS_KT2440_WT3",
    ub_max: float = 1000.0,
    ub_min_fraction: float = 0.01,
    eps: float = 1e-6,
    processes_for_deletion: int = 4,
    apply_thermo: bool = False
) -> Tuple[Model, Dict]:
    counts_df = rnaseq_counts.to_frame(name='counts')
    norm = median_ratio_normalization(counts_df)
    if isinstance(norm, pd.Series):
        norm_series = norm
    else:
        norm_series = norm.iloc[:, 0]
    cpm = counts_to_cpm(norm_series)
    expr = log_transform(cpm, pseudocount=1.0)
    rxn_scores = map_expression_to_reactions(model, expr)
    essential_rxns = find_essential_reactions(
        model,
        biomass_rxn_id=biomass_rxn_id,
        threshold_fraction=0.05,
        absolute_threshold=1e-6,
        processes=processes_for_deletion
    )
    whitelist = set(essential_rxns)
    for core in [biomass_rxn_id, 'ATPM']:
        if core in model.reactions:
            whitelist.add(core)
    for rxn in ['EX_glc__D_e', 'EX_nh4_e', 'EX_pi_e', 'EX_so4_e', 'EX_o2_e']:
        if rxn in model.reactions:
            whitelist.add(rxn)
    whitelist = list(whitelist)
    scaled_model, original_bounds = apply_eflux_scaling_with_lb(
        model,
        rxn_scores,
        ub_max=ub_max,
        ub_min_fraction=ub_min_fraction,
        eps=eps,
        whitelist_rxns=whitelist,
        exchange_prefixes=["EX_"],
        preserve_original_bounds=True
    )
    # ensure essential reactions have minimal capacity
    for rxn_id in essential_rxns:
        if rxn_id in scaled_model.reactions:
            rxn = scaled_model.reactions.get_by_id(rxn_id)
            if rxn.reversibility:
                if abs(rxn.lower_bound) < eps and abs(rxn.upper_bound) < eps:
                    rxn.lower_bound = -eps
                    rxn.upper_bound = eps
            else:
                if rxn.upper_bound <= 0:
                    rxn.upper_bound = eps
                if rxn.lower_bound >= 0 and rxn.lower_bound < eps:
                    rxn.lower_bound = eps
    if apply_thermo:
        scaled_model = run_pytfa_thermo_curation(scaled_model)
    diagnostics = {}
    with scaled_model:
        try:
            scaled_model.objective = biomass_rxn_id
            sol = scaled_model.optimize()
            diagnostics['growth_feasible'] = (sol.status == 'optimal' and float(sol.fluxes.get(biomass_rxn_id, 0.0)) > 1e-8)
            diagnostics['growth_rate'] = float(sol.fluxes.get(biomass_rxn_id, 0.0)) if sol.status == 'optimal' else 0.0
        except Exception as e:
            diagnostics['growth_feasible'] = False
            diagnostics['growth_rate'] = 0.0
            diagnostics['growth_error'] = str(e)
    diagnostics.update({
        'essential_rxns_count': len(essential_rxns),
        'essential_rxns_sample': essential_rxns[:20],
        'rxn_scores_summary': {
            'min': float(np.nanmin(list(rxn_scores.values()))) if len(rxn_scores)>0 else 0.0,
            'max': float(np.nanmax(list(rxn_scores.values()))) if len(rxn_scores)>0 else 0.0,
            'median': float(np.nanmedian(list(rxn_scores.values()))) if len(rxn_scores)>0 else 0.0
        },
        'whitelist_rxns_count': len(whitelist)
    })
    return scaled_model, diagnostics

# -------------------------
# Build models from multi-env CSV
# -------------------------
def build_models_from_multi_env_csv(
    model_path: str,
    rnaseq_csv_path: str,
    output_dir: str,
    biomass_rxn_id: str = "BIOMASS_KT2440_WT3",
    env_columns: Dict[str, str] = None,
    sep: str = ';',
    ub_max: float = 1000.0,
    ub_min_fraction: float = 0.01,
    eps: float = 1e-6,
    processes_for_deletion: int = 4,
    apply_thermo: bool = False
) -> Dict[str, Dict]:
    if env_columns is None:
        env_columns = {'Glu':'Expression_Glu', 'Cit':'Expression_Cit', 'Fer':'Expression_Fer', 'Ser':'Expression_Ser'}
    model = cobra.io.read_sbml_model(model_path)
    expr_df = pd.read_csv(rnaseq_csv_path, sep=sep, index_col=0)
    diagnostics_all = {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for env_short, col_name in env_columns.items():
        if col_name not in expr_df.columns:
            warnings.warn(f"Column {col_name} not found; skipping {env_short}.")
            continue
        print(f"Building model for environment: {env_short}")
        series = expr_df[col_name].copy()
        series.index = series.index.astype(str)
        adjusted_model, diag = build_context_specific_model_from_rnaseq_single_env(
            model,
            series,
            biomass_rxn_id=biomass_rxn_id,
            ub_max=ub_max,
            ub_min_fraction=ub_min_fraction,
            eps=eps,
            processes_for_deletion=processes_for_deletion,
            apply_thermo=apply_thermo
        )
        out_model_path = out_dir / f"{Path(model_path).stem}_{env_short}_eflux.xml"
        cobra.io.write_sbml_model(adjusted_model, str(out_model_path))
        diag_path = out_dir / f"{Path(model_path).stem}_{env_short}_diagnostics.json"
        pd.Series(diag).to_json(str(diag_path), orient='index')
        diagnostics_all[env_short] = diag
    return diagnostics_all

# -------------------------
# Example main usage
# -------------------------
if __name__ == "__main__":
    #project_root = Path(__file__).parent
    model_path = "models/iJN1463.xml"
    rnaseq_csv_path = "expression_txt_files/merged_expression.csv"
    output_dir = "models/context_specific"
    diagnostics = build_models_from_multi_env_csv(
        model_path=str(model_path),
        rnaseq_csv_path=str(rnaseq_csv_path),
        output_dir=str(output_dir),
        biomass_rxn_id="BIOMASS_KT2440_WT3",
        env_columns={'Glu':'Expression_Glu', 'Cit':'Expression_Cit', 'Fer':'Expression_Fer', 'Ser':'Expression_Ser'},
        sep=';',
        ub_max=1000.0,
        ub_min_fraction=0.01,
        eps=1e-6,
        processes_for_deletion=4,
        apply_thermo=False
    )
    print("Done. Diagnostics summary:")
    for env, d in diagnostics.items():
        print(env, d)
