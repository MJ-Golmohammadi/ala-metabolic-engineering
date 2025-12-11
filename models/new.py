# expression_to_model_pipeline.py
"""
Pipeline v3 Fixed: E-Flux scaling with LB adjustment, rollback and SBML write fix
- multi-environment CSV input (semicolon separated)
- detect essential reactions via single_reaction_deletion (after applying environment)
- apply E-Flux scaling that adjusts both UB and LB for internal reactions
- EX_ reactions are controlled only via YAML environment files (not by RNA-seq)
- optional pyTFA thermodynamic curation
- outputs: one SBML per environment + diagnostics JSON
Fixes and features:
- ensure median_ratio_normalization handles empty rows without NaN propagation
- ensure reversibility flags are plain Python bool before writing SBML
- numeric and type safety checks for bounds
- incremental rollback to restore feasibility if scaling makes model infeasible
- quick safe fixes (BIOMASS/ATPM/uptakes) applied automatically before rollback
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
import cobra
from cobra import Model, Reaction
from cobra.flux_analysis import single_reaction_deletion

# Optional pyTFA
try:
    from pytfa.io import load_thermoDB  # type: ignore
    PYTFA_AVAILABLE = True
except Exception:
    PYTFA_AVAILABLE = False

# -------------------------
# Normalization helpers
# -------------------------
def median_ratio_normalization(counts_df: pd.DataFrame) -> pd.Series:
    """
    DESeq2-like median ratio normalization with robust handling of zero rows.
    Returns a Series when single column, otherwise DataFrame-like normalized.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        def geom_mean_row(x):
            x_nonzero = x.replace(0, np.nan).astype(float)
            if x_nonzero.isna().all():
                return np.nan
            return float(np.exp(np.nanmean(np.log(x_nonzero))))
        geom_means = counts_df.apply(geom_mean_row, axis=1)
    geom_means = geom_means.fillna(1.0)
    ratios = counts_df.div(geom_means, axis=0)
    size_factors = ratios.median(axis=0)
    size_factors = size_factors.replace(0, np.nan).fillna(1.0)
    normalized = counts_df.div(size_factors, axis=1)
    if normalized.shape[1] == 1:
        return normalized.iloc[:, 0]
    return normalized

def counts_to_cpm(counts: pd.Series) -> pd.Series:
    total = float(counts.sum())
    if total == 0.0:
        return counts * 0.0
    return counts / total * 1e6

def log_transform(series: pd.Series, pseudocount: float = 1.0) -> pd.Series:
    return np.log2(series + pseudocount)

# -------------------------
# GPR mapping helpers
# -------------------------
def aggregate_expression_for_reaction(reaction: Reaction, gene_expr: Dict[str, float]) -> float:
    rule = reaction.gene_reaction_rule.strip() if reaction.gene_reaction_rule is not None else ""
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

    scores = np.array(list(rxn_scores.values()), dtype=float) if len(rxn_scores) > 0 else np.array([0.0])
    rmin = float(np.nanmin(scores)) if scores.size > 0 else 0.0
    rmax = float(np.nanmax(scores)) if scores.size > 0 else 1.0
    if rmax == rmin:
        rmax = rmin + 1.0

    ub_min = float(ub_max) * float(ub_min_fraction)
    original_bounds: Dict[str, Tuple[float, float]] = {}
    changed_rxns: List[str] = []

    for rxn in model.reactions:
        original_bounds[rxn.id] = (float(rxn.lower_bound), float(rxn.upper_bound))
        if any(rxn.id.startswith(pref) for pref in exchange_prefixes):
            continue
        if rxn.id in whitelist_rxns:
            continue

        score = float(rxn_scores.get(rxn.id, 0.0))
        norm = float(np.clip((score - rmin) / (rmax - rmin), 0.0, 1.0))
        ub_scaled = float(ub_min + norm * (ub_max - ub_min))
        if ub_scaled < eps:
            ub_scaled = float(eps)

        if bool(rxn.reversibility):
            rxn.lower_bound = float(-ub_scaled)
            rxn.upper_bound = float(ub_scaled)
        else:
            orig_lb, orig_ub = original_bounds[rxn.id]
            if orig_lb >= 0:
                if orig_ub == 0 or ub_scaled <= eps:
                    rxn.lower_bound = 0.0
                    rxn.upper_bound = float(eps)
                else:
                    rxn.lower_bound = float(max(orig_lb, eps))
                    rxn.upper_bound = float(ub_scaled)
            elif orig_ub <= 0:
                if ub_scaled <= eps:
                    rxn.lower_bound = 0.0
                    rxn.upper_bound = float(eps)
                else:
                    rxn.upper_bound = float(min(orig_ub, -eps))
                    rxn.lower_bound = float(-ub_scaled)
            else:
                rxn.lower_bound = float(orig_lb)
                rxn.upper_bound = float(ub_scaled)

        if float(rxn.lower_bound) > float(rxn.upper_bound):
            rxn.lower_bound = float(rxn.upper_bound)

        changed_rxns.append(rxn.id)

    for rxn_id in whitelist_rxns:
        if rxn_id in model.reactions:
            rxn = model.reactions.get_by_id(rxn_id)
            lb, ub = float(rxn.lower_bound), float(rxn.upper_bound)
            if bool(rxn.reversibility):
                if abs(lb) < eps and abs(ub) < eps:
                    rxn.lower_bound = -float(eps)
                    rxn.upper_bound = float(eps)
            else:
                if ub <= 0:
                    rxn.upper_bound = float(eps)
                if lb >= 0 and lb < eps:
                    rxn.lower_bound = float(eps)

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
    
    # ابتدا بررسی می‌کنیم که مدل قابل حل است
    with m:
        m.objective = biomass_rxn_id
        ref_sol = m.optimize()
        if ref_sol.status != 'optimal':
            raise ValueError(f"Model is infeasible before reaction deletion. Status: {ref_sol.status}")
        
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
    from pytfa import ThermoModel  # type: ignore
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
# Incremental rollback helper
# -------------------------
def incremental_restore_until_feasible(scaled_model: Model,
                                       original_bounds: Dict[str, Tuple[float, float]],
                                       rxn_scores: Dict[str, float],
                                       biomass_rxn_id: str = "BIOMASS_KT2440_WT3",
                                       batch: int = 5) -> Tuple[Model, List[str]]:
    """
    Restore reactions in small batches (highest expression first) until model becomes feasible.
    Returns (model, restored_list).
    """
    scaled_model.objective = biomass_rxn_id
    sol = scaled_model.optimize()
    if sol.status == 'optimal' and float(sol.fluxes.get(biomass_rxn_id, 0.0)) > 1e-8:
        return scaled_model, []

    sorted_rxns = sorted(rxn_scores.items(), key=lambda x: x[1], reverse=True)
    restored: List[str] = []

    for i in range(0, len(sorted_rxns), batch):
        for rxn_id, _ in sorted_rxns[i:i+batch]:
            if rxn_id in scaled_model.reactions and rxn_id in original_bounds:
                lb, ub = original_bounds[rxn_id]
                r = scaled_model.reactions.get_by_id(rxn_id)
                r.lower_bound = float(lb)
                r.upper_bound = float(ub)
                restored.append(rxn_id)
        scaled_model.objective = biomass_rxn_id
        sol = scaled_model.optimize()
        if sol.status == 'optimal' and float(sol.fluxes.get(biomass_rxn_id, 0.0)) > 1e-8:
            return scaled_model, restored

    return scaled_model, restored

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
    apply_thermo: bool = False,
    env_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Tuple[Model, Dict]:
    """
    Revised pipeline:
    1) copy model
    2) apply environment bounds (env_bounds or sensible defaults)
    3) normalize RNA-seq and map to reactions
    4) detect essential reactions on model WITH environment applied
    5) build whitelist and apply eflux scaling
    6) quick fixes, incremental rollback if needed
    7) diagnostics + debug info
    """
    # مرحله 1: کپی کردن مدل اصلی
    m = model.copy()

    if env_bounds is None:
        env_bounds = {
            "EX_glc__D_e": (-10.0, 1000.0),
            "EX_nh4_e": (-10.0, 1000.0),
            "EX_pi_e": (-10.0, 1000.0),
            "EX_so4_e": (-10.0, 1000.0),
            "EX_o2_e": (-20.0, 1000.0),
            "EX_h2o_e": (-1000.0, 1000.0),
            "EX_co2_e": (-1000.0, 1000.0)
        }
    
    # مرحله 2: اعمال محدودیت‌های محیطی
    for rxn_id, (lb, ub) in env_bounds.items():
        if rxn_id in m.reactions:
            r = m.reactions.get_by_id(rxn_id)
            r.lower_bound = float(lb)
            r.upper_bound = float(ub)
    
    # بررسی اینکه مدل پس از اعمال محدودیت‌های محیطی قابل حل است
    diagnostics: Dict = {}
    m.objective = biomass_rxn_id
    sol_env = m.optimize()
    
    if sol_env.status != 'optimal':
        diagnostics['environment_infeasible'] = True
        diagnostics['solver_status_env'] = sol_env.status
        
        # آزاد کردن محدودیت‌ها برای بازیابی feasibility
        print(f"⚠️  مدل پس از اعمال محدودیت‌های محیطی غیرقابل حل است. وضعیت: {sol_env.status}")
        print("🔧 در حال آزاد کردن محدودیت‌های منابع...")
        
        # لیست منابع احتمالی
        potential_sources = [
            "EX_glc__D_e", "EX_glyc_e", "EX_ac_e", "EX_succ_e", "EX_lac__D_e", 
            "EX_etoh_e", "EX_glu__L_e", "EX_ala__L_e", "EX_fru_e", "EX_gal_e",
            "EX_nh4_e", "EX_pi_e", "EX_so4_e", "EX_o2_e", "EX_h2o_e", "EX_co2_e"
        ]
        
        for rxn_id in potential_sources:
            if rxn_id in m.reactions:
                r = m.reactions.get_by_id(rxn_id)
                if r.lower_bound < 0:
                    r.lower_bound = -1000.0
                if r.upper_bound > 0:
                    r.upper_bound = 1000.0
        
        # بررسی دوباره
        sol_env = m.optimize()
        if sol_env.status != 'optimal':
            print(f"❌ مدل همچنان غیرقابل حل باقی ماند. استفاده از مدل اصلی...")
            diagnostics['model_reverted_to_original'] = True
            # برگشت به مدل اصلی با محدودیت‌های پیش‌فرض
            m = model.copy()
            # فقط محدودیت‌های ضروری را اعمال کن
            for rxn_id in ["EX_glc__D_e", "EX_nh4_e", "EX_pi_e", "EX_so4_e", "EX_o2_e"]:
                if rxn_id in m.reactions:
                    r = m.reactions.get_by_id(rxn_id)
                    r.lower_bound = -1000.0 if rxn_id != "EX_o2_e" else -100.0
                    r.upper_bound = 1000.0
        else:
            print(f"✅ feasibility بازیابی شد. نرخ رشد: {sol_env.objective_value}")
            diagnostics['environment_infeasible'] = False
    else:
        diagnostics['environment_infeasible'] = False
        diagnostics['initial_growth_rate'] = float(sol_env.objective_value)
    
    # مرحله 3: نرمال‌سازی RNA-seq
    counts_df = rnaseq_counts.to_frame(name='counts')
    norm = median_ratio_normalization(counts_df)
    if isinstance(norm, pd.Series):
        norm_series = norm
    else:
        norm_series = norm.iloc[:, 0]
    cpm = counts_to_cpm(norm_series)
    expr = log_transform(cpm, pseudocount=1.0)
    rxn_scores = map_expression_to_reactions(m, expr)

    # مرحله 4: شناسایی واکنش‌های ضروری
    try:
        essential_rxns = find_essential_reactions(
            m,
            biomass_rxn_id=biomass_rxn_id,
            threshold_fraction=0.05,
            absolute_threshold=1e-6,
            processes=processes_for_deletion
        )
        diagnostics['essential_detection_success'] = True
    except Exception as e:
        warnings.warn(f"شناسایی واکنش‌های ضروری با خطا مواجه شد: {e}")
        print(f"استفاده از لیست خالی برای واکنش‌های ضروری...")
        essential_rxns = []
        diagnostics['essential_detection_success'] = False
        diagnostics['essential_detection_error'] = str(e)

    # مرحله 5: ساخت whitelist
    whitelist = set(essential_rxns)
    for core in [biomass_rxn_id, 'ATPM']:
        if core in m.reactions:
            whitelist.add(core)
    for rxn in env_bounds.keys():
        if rxn in m.reactions:
            whitelist.add(rxn)
    whitelist = list(whitelist)

    # مرحله 6: اعمال مقیاس‌گذاری E-Flux
    scaled_model, original_bounds = apply_eflux_scaling_with_lb(
        m,
        rxn_scores,
        ub_max=ub_max,
        ub_min_fraction=ub_min_fraction,
        eps=eps,
        whitelist_rxns=whitelist,
        exchange_prefixes=["EX_"],
        preserve_original_bounds=True
    )

    # مرحله 7: رفع سریع خطاهای امن
    if biomass_rxn_id in scaled_model.reactions:
        b = scaled_model.reactions.get_by_id(biomass_rxn_id)
        b.lower_bound = 0.0
        b.upper_bound = 1e6
    
    if "ATPM" in scaled_model.reactions:
        a = scaled_model.reactions.get_by_id("ATPM")
        a.lower_bound = 0.0
        a.upper_bound = 1000.0
    
    # تنظیم محدودیت‌های ضروری برای جذب
    essential_uptakes = [
        ("EX_glc__D_e", -10.0),
        ("EX_o2_e", -20.0),
        ("EX_nh4_e", -10.0),
        ("EX_pi_e", -10.0),
        ("EX_so4_e", -10.0)
    ]
    
    for rxn_id, lb in essential_uptakes:
        if rxn_id in scaled_model.reactions:
            r = scaled_model.reactions.get_by_id(rxn_id)
            # فقط اگر قبلاً تنظیم نشده باشد
            if r.lower_bound > lb:  # یعنی کمتر محدود نباشد
                r.lower_bound = float(lb)
            r.upper_bound = 1000.0

    # مرحله 8: بررسی feasibility پس از مقیاس‌گذاری
    scaled_model.objective = biomass_rxn_id
    sol = scaled_model.optimize()
    diagnostics['post_scaling_feasible'] = (sol.status == 'optimal' and 
                                           float(sol.fluxes.get(biomass_rxn_id, 0.0)) > 1e-8)
    diagnostics['post_scaling_growth'] = float(sol.fluxes.get(biomass_rxn_id, 0.0)) if sol.status == 'optimal' else 0.0
    diagnostics['post_scaling_status'] = sol.status

    # مرحله 9: اگر غیرقابل حل بود، برگرداندن تدریجی
    if not diagnostics['post_scaling_feasible']:
        print(f"⚠️  مدل پس از مقیاس‌گذاری غیرقابل حل است. نرخ رشد: {diagnostics['post_scaling_growth']}")
        print("در حال برگرداندن واکنش‌ها به صورت تدریجی...")
        
        scaled_model, restored = incremental_restore_until_feasible(
            scaled_model, original_bounds, rxn_scores, 
            biomass_rxn_id=biomass_rxn_id, batch=5
        )
        
        diagnostics['restored_count'] = len(restored)
        diagnostics['restored_sample'] = restored[:50]
        
        # ارزیابی مجدد
        scaled_model.objective = biomass_rxn_id
        sol2 = scaled_model.optimize()
        diagnostics['final_feasible'] = (sol2.status == 'optimal' and 
                                        float(sol2.fluxes.get(biomass_rxn_id, 0.0)) > 1e-8)
        diagnostics['final_growth'] = float(sol2.fluxes.get(biomass_rxn_id, 0.0)) if sol2.status == 'optimal' else 0.0
        
        if diagnostics['final_feasible']:
            print(f"✅ feasibility پس از برگرداندن بازیابی شد. نرخ رشد: {diagnostics['final_growth']}")
    else:
        diagnostics['final_feasible'] = diagnostics['post_scaling_feasible']
        diagnostics['final_growth'] = diagnostics['post_scaling_growth']

    # مرحله 10: اطمینان از ظرفیت حداقلی برای واکنش‌های ضروری
    for rxn_id in essential_rxns:
        if rxn_id in scaled_model.reactions:
            rxn = scaled_model.reactions.get_by_id(rxn_id)
            if bool(rxn.reversibility):
                if abs(float(rxn.lower_bound)) < eps and abs(float(rxn.upper_bound)) < eps:
                    rxn.lower_bound = -float(eps)
                    rxn.upper_bound = float(eps)
            else:
                if float(rxn.upper_bound) <= 0:
                    rxn.upper_bound = float(eps)
                if float(rxn.lower_bound) >= 0 and float(rxn.lower_bound) < eps:
                    rxn.lower_bound = float(eps)

    # مرحله 11: اعمال ترمودینامیک اختیاری
    if apply_thermo:
        scaled_model = run_pytfa_thermo_curation(scaled_model)

    # مرحله 12: جمع‌آوری آمار
    closed = [r.id for r in scaled_model.reactions if float(r.lower_bound) == 0.0 and float(r.upper_bound) == 0.0]
    diagnostics.update({
        'closed_reactions_count': len(closed),
        'closed_reactions_sample': closed[:50],
        'essential_rxns_count': len(essential_rxns),
        'essential_rxns_sample': essential_rxns[:20],
        'rxn_scores_summary': {
            'min': float(np.nanmin(list(rxn_scores.values()))) if len(rxn_scores) > 0 else 0.0,
            'max': float(np.nanmax(list(rxn_scores.values()))) if len(rxn_scores) > 0 else 0.0,
            'median': float(np.nanmedian(list(rxn_scores.values()))) if len(rxn_scores) > 0 else 0.0
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
    """
    ساخت مدل‌های خاص بافت از یک CSV چند محیطی
    نکته کلیدی: مدل اصلی یک بار لود می‌شود و برای هر محیط کپی می‌شود
    """
    if env_columns is None:
        env_columns = {'Glu':'Expression_Glu', 'Cit':'Expression_Cit', 'Fer':'Expression_Fer', 'Ser':'Expression_Ser'}
    
    # مرحله 1: لود کردن مدل اصلی (فقط یک بار)
    print(f"📥 در حال لود کردن مدل از: {model_path}")
    model = cobra.io.read_sbml_model(model_path)
    
    # بررسی feasibility مدل اصلی
    model.objective = biomass_rxn_id
    sol_base = model.optimize()
    print(f"✅ مدل اصلی لود شد. نرخ رشد پایه: {sol_base.objective_value if sol_base.status == 'optimal' else 'غیرقابل حل'}")
    
    # مرحله 2: لود کردن داده‌های RNA-seq
    print(f"📊 در حال لود کردن داده‌های RNA-seq از: {rnaseq_csv_path}")
    expr_df = pd.read_csv(rnaseq_csv_path, sep=sep, index_col=0)
    
    # مرحله 3: ساخت دایرکتوری خروجی
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    diagnostics_all = {}
    
    # مرحله 4: پردازش هر محیط
    for env_short, col_name in env_columns.items():
        if col_name not in expr_df.columns:
            warnings.warn(f"ستون {col_name} یافت نشد؛ رد کردن {env_short}.")
            continue
        
        print(f"\n{'='*50}")
        print(f"🔨 ساخت مدل برای محیط: {env_short}")
        print(f"{'='*50}")
        
        # استخراج داده‌های بیان برای این محیط
        series = expr_df[col_name].copy()
        series.index = series.index.astype(str)
        
        # ساخت مدل خاص بافت
        adjusted_model, diag = build_context_specific_model_from_rnaseq_single_env(
            model,  # مدل اصلی پاس داده می‌شود
            series,
            biomass_rxn_id=biomass_rxn_id,
            ub_max=ub_max,
            ub_min_fraction=ub_min_fraction,
            eps=eps,
            processes_for_deletion=processes_for_deletion,
            apply_thermo=apply_thermo
        )
        
        # اطمینان از اینکه پرچم‌های reversibility از نوع bool هستند
        for rxn in adjusted_model.reactions:
            try:
                rxn.reversibility = bool(float(rxn.lower_bound) < 0.0)
            except Exception:
                rxn.reversibility = False

        # ذخیره مدل
        model_stem = Path(model_path).stem
        out_model_path = out_dir / f"{model_stem}_{env_short}_eflux.xml"
        cobra.io.write_sbml_model(adjusted_model, str(out_model_path))
        print(f"💾 مدل ذخیره شد: {out_model_path}")

        # ذخیره تشخیص‌ها
        diag_path = out_dir / f"{model_stem}_{env_short}_diagnostics.json"
        with open(str(diag_path), 'w') as jf:
            json.dump(diag, jf, indent=2)
        print(f"📝 تشخیص‌ها ذخیره شد: {diag_path}")

        diagnostics_all[env_short] = diag
    
    return diagnostics_all

# -------------------------
# تابع اصلی برای دیباگ
# -------------------------
def test_model_loading(model_path: str, biomass_rxn_id: str = "BIOMASS_KT2440_WT3"):
    """تابع کمکی برای تست مدل"""
    import cobra
    print(f"\n🔧 تست مدل: {model_path}")
    model = cobra.io.read_sbml_model(model_path)
    
    print(f"📊 اطلاعات مدل:")
    print(f"  - تعداد واکنش‌ها: {len(model.reactions)}")
    print(f"  - تعداد متابولیت‌ها: {len(model.metabolites)}")
    print(f"  - تعداد ژن‌ها: {len(model.genes)}")
    
    if biomass_rxn_id in model.reactions:
        model.objective = biomass_rxn_id
        sol = model.optimize()
        print(f"  - واکنش زیست‌توده: {biomass_rxn_id} یافت شد")
        print(f"  - نرخ رشد: {sol.objective_value if sol.status == 'optimal' else 'N/A'}")
        print(f"  - وضعیت حل: {sol.status}")
    else:
        print(f"  - ⚠️  واکنش زیست‌توده {biomass_rxn_id} یافت نشد!")
        print(f"  - واکنش‌های زیست‌توده موجود: {[r.id for r in model.reactions if 'BIOMASS' in r.id]}")
    
    return model

# -------------------------
# Example main usage
# -------------------------
if __name__ == "__main__":
    # مسیرهای فایل
    model_path = "models/iJN1463.xml"
    rnaseq_csv_path = "expression_txt_files/merged_expression.csv"
    output_dir = "models/context_specific"
    
    # ابتدا مدل را تست کنید
    try:
        test_model = test_model_loading(model_path, "BIOMASS_KT2440_WT3")
    except Exception as e:
        print(f"❌ خطا در لود کردن مدل: {e}")
        exit(1)
    
    # سپس مدل‌های خاص بافت را بسازید
    print(f"\n{'='*60}")
    print("🚀 شروع ساخت مدل‌های خاص بافت")
    print(f"{'='*60}")
    
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
        processes_for_deletion=1,  # کاهش به 1 برای جلوگیری از خطاهای موازی
        apply_thermo=False
    )
    
    print(f"\n{'='*60}")
    print("✅ انجام شد. خلاصه تشخیص:")
    print(f"{'='*60}")
    
    for env, d in diagnostics.items():
        print(f"\n📋 محیط: {env}")
        print(f"  - قابل حل نهایی: {d.get('final_feasible', 'N/A')}")
        print(f"  - نرخ رشد نهایی: {d.get('final_growth', 'N/A'):.6f}")
        print(f"  - تعداد واکنش‌های بسته: {d.get('closed_reactions_count', 'N/A')}")
        print(f"  - تعداد واکنش‌های ضروری: {d.get('essential_rxns_count', 'N/A')}")
        print(f"  - تعداد واکنش‌های برگردانده شده: {d.get('restored_count', 0)}")
