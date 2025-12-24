#!/usr/bin/env python3
"""
Comprehensive Baseline FBA Simulations for Multi-Environmental Analysis

Performs systematic flux balance analysis across four environmental conditions
(Glu, Cit, Ser, Fer) with sensitivity analysis and robustness testing.

This script uses the environment-specific preprocessed models (one per environment).
It expects models in models/final_constrained_rnaseq_thermo named:
  iJN1463_Glu_ExprThermoConstrainedFile.xml
  iJN1463_Cit_ExprThermoConstrainedFile.xml
  iJN1463_Ser_ExprThermoConstrainedFile.xml
  iJN1463_Fer_ExprThermoConstrainedFile.xml
"""

import cobra
import pandas as pd
import numpy as np
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model_utils import simulate_with_objective, calculate_yield_metrics, get_substrate_rxn_for_environment

# Configure basic logging so debug/info messages from model_utils are visible
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Plotting defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans'
})


def get_config_path() -> Path:
    """Return path to config/engineering_scenarios.yaml"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "engineering_scenarios.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    return config_path


def get_model_path(environment: str) -> Path:
    """Return path to environment-specific model file"""
    project_root = Path(__file__).parent.parent
    model_filename = f"iJN1463_{environment}_preprocessed_with_DM.xml"
    model_path = project_root / "models" / "final_constrained_rnaseq_thermo" / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found for {environment}: {model_path}")
    return model_path


def perform_sensitivity_analysis(model: cobra.Model, environment: Dict, parameter_range: np.ndarray = np.linspace(0.5, 2.0, 10)) -> pd.DataFrame:
    """
    Perform sensitivity analysis on key environmental parameters (carbon uptake).
    Returns a DataFrame with tested variations.
    """
    sensitivity_results = []
    base_conditions = environment.copy()

    carbon_rxns = ['EX_glc__D_e', 'EX_cit_e', 'EX_ser__L_e', 'EX_fer_e']
    for carbon_rxn in carbon_rxns:
        if carbon_rxn in base_conditions:
            for factor in parameter_range:
                with model:
                    modified_environment = base_conditions.copy()
                    modified_environment[carbon_rxn] = [base_conditions[carbon_rxn][0] * factor, 1000.0]
                    for rxn_id, bounds in modified_environment.items():
                        if rxn_id in model.reactions:
                            model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                    # Test both objectives
                    for obj_name in ['max_biomass', 'max_ala']:
                        if obj_name == 'max_biomass':
                            model.objective = model.reactions.get_by_id('BIOMASS_KT2440_WT3')
                        else:
                            model.objective = model.reactions.get_by_id('G1SAT')
                        sol = model.optimize()
                        if sol.status == 'optimal':
                            ala_flux = float(sol.fluxes.get('G1SAT', 0.0))
                            growth_flux = float(sol.fluxes.get('BIOMASS_KT2440_WT3', 0.0))
                            sensitivity_results.append({
                                'parameter': carbon_rxn,
                                'variation_factor': float(factor),
                                'objective': obj_name,
                                'growth_rate': growth_flux,
                                'ala_flux': ala_flux
                            })
    return pd.DataFrame(sensitivity_results)


def analyze_gene_essentiality(model: cobra.Model, environment: Dict, target_genes: List[str] = None) -> pd.DataFrame:
    """
    Perform gene essentiality analysis using COBRApy single_gene_deletion.
    Returns DataFrame with gene deletion growth results.
    """
    from cobra.flux_analysis import single_gene_deletion
    if target_genes is None:
        target_genes = ['PP_0732', 'PP_4784', 'PP_1977']  # example gene IDs

    with model:
        for rxn_id, bounds in (environment or {}).items():
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
        deletion_results = single_gene_deletion(model, target_genes)
        rows = []
        for gene_id, row in deletion_results.iterrows():
            rows.append({
                'gene': gene_id,
                'growth': row['growth'],
                'status': 'essential' if row['growth'] < 0.01 else 'non_essential'
            })
    return pd.DataFrame(rows)


def create_multi_environment_summary(simulation_results: pd.DataFrame) -> plt.Figure:
    """
    Create a 2x2 summary figure for growth, production, yield and efficiency.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    growth_data = simulation_results[simulation_results['objective'] == 'max_biomass']
    environments = growth_data['environment'].unique()
    growth_rates = [growth_data[growth_data['environment'] == env]['growth_rate'].mean() for env in environments]

    bars = ax1.bar(environments, growth_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Growth Rate (h^-1)')
    ax1.set_title('A. Growth Rate Across Environments')
    ax1.grid(True, alpha=0.3)

    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center')

    ala_data = simulation_results[simulation_results['objective'] == 'max_biomass']
    ala_production = [ala_data[ala_data['environment'] == env]['ala_flux_net'].mean() for env in environments]
    ax2.bar(environments, ala_production, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Net ALA Production (mmol/gDW/h)')
    ax2.set_title('B. ALA Production Across Environments')
    ax2.grid(True, alpha=0.3)

    yields = [ala_data[ala_data['environment'] == env]['yield_mmol_g'].mean() for env in environments]
    ax3.bar(environments, yields, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_ylabel('Yield (mmol ALA / mmol substrate)')
    ax3.set_title('C. Production Yield Across Environments')
    ax3.grid(True, alpha=0.3)

    efficiencies = []
    for env in environments:
        env_data = ala_data[ala_data['environment'] == env]
        if len(env_data) > 0:
            ala_mean = env_data['ala_flux_net'].mean()
            growth_mean = env_data['growth_rate'].mean()
            efficiencies.append(ala_mean / growth_mean if growth_mean > 1e-9 else np.nan)
        else:
            efficiencies.append(np.nan)
    ax4.bar(environments, efficiencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax4.set_ylabel('Production Efficiency (ALA / Growth)')
    ax4.set_title('D. Metabolic Efficiency by Environment')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def optimize_max_ala_with_min_growth(
    model: cobra.Model,
    ala_obj_id: str = "G1SAT",
    biomass_id: str = "BIOMASS_KT2440_WT3",
    b_min_fraction: float = 0.01,
    environment: Dict = None
):
    """
    Optimize ALA production while enforcing a minimum biomass growth (epsilon-constraint).
    Uses a copy to compute reference growth and then enforces lower bound on original model.
    """
    tmp = model.copy()
    if environment:
        for rxn_id, bounds in environment.items():
            if rxn_id in tmp.reactions:
                tmp.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
    try:
        tmp.objective = tmp.reactions.get_by_id(biomass_id)
    except Exception:
        tmp.objective = biomass_id
    ref_solution = tmp.optimize()
    ref_growth = max(0.0, float(ref_solution.fluxes.get(biomass_id, 0.0))) if ref_solution.status == 'optimal' else 0.0

    b_min = b_min_fraction * ref_growth

    biomass_rxn = model.reactions.get_by_id(biomass_id)
    prev_lb = biomass_rxn.lower_bound
    biomass_rxn.lower_bound = max(prev_lb, b_min)

    if environment:
        for rxn_id, bounds in environment.items():
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)

    try:
        model.objective = model.reactions.get_by_id(ala_obj_id)
    except Exception:
        model.objective = ala_obj_id
    solution = model.optimize()

    biomass_rxn.lower_bound = prev_lb
    return solution


def main():
    """
    Run baseline simulations for all four environments and save results.
    """
    try:
        config_path = get_config_path()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        environments_to_analyze = ['Glu', 'Cit', 'Ser', 'Fer']
        all_simulation_results = []
        sensitivity_results = []
        essentiality_results = []

        for env_name in environments_to_analyze:
            logger.info(f"Analyzing environment: {env_name}")
            model_path = get_model_path(env_name)
            model = cobra.io.read_sbml_model(str(model_path))
            model.solver = "glpk"

            environment_config = config['environments'].get(env_name, {})

            for obj_name, objective in config['objectives'].items():
                logger.info(f"  Running objective: {obj_name}")
                try:
                    if obj_name == "max_ala":
                        solution = simulate_with_objective(
                            model,
                            'G1SAT',
                            environment_config,
                            min_growth_fraction=0.01
                        )
                    else:
                        solution = simulate_with_objective(model, objective, environment_config)

                    substrate_rxn = get_substrate_rxn_for_environment(env_name)
                    yield_metrics = calculate_yield_metrics(solution, 'G1SAT', substrate_rxn, environment=env_name,
                                       min_growth_abs=0.01 * max(1e-9, float(solution.fluxes.get('BIOMASS_KT2440_WT3', 0.0))))

                    growth_flux = float(solution.fluxes.get('BIOMASS_KT2440_WT3', 0.0))
                    ala_flux_gross = float(solution.fluxes.get('G1SAT', 0.0))
                    ala_flux_net = yield_metrics['net_ala_flux']

                    result = {
                        'environment': env_name,
                        'objective': obj_name,
                        'growth_rate': growth_flux,
                        'ala_flux': ala_flux_gross,
                        'ala_flux_net': ala_flux_net,
                        'ala_consumption': yield_metrics['ala_consumption_flux'],
                        'solution_status': solution.status,
                        'modifications': 'wild_type',
                        'modification_count': 0,
                        'yield_mmol_g': yield_metrics['yield_mmol_g'],
                        'yield_mmol_mmol': yield_metrics['yield_mmol_mmol'],
                        'carbon_yield': yield_metrics['carbon_yield'],
                        'substrate_uptake': yield_metrics['substrate_uptake']
                    }
                    all_simulation_results.append(result)

                    logger.info(f"    {obj_name}: Growth={growth_flux:.6f}, ALA_net={ala_flux_net:.6f}")

                except Exception as e:
                    logger.error(f"    Simulation failed for {obj_name} in {env_name}: {e}")
                    all_simulation_results.append({
                        'environment': env_name,
                        'objective': obj_name,
                        'growth_rate': 0,
                        'ala_flux': 0,
                        'ala_flux_net': 0,
                        'ala_consumption': 0,
                        'solution_status': 'failed',
                        'modifications': 'wild_type',
                        'modification_count': 0,
                        'yield_mmol_g': 0,
                        'yield_mmol_mmol': 0,
                        'carbon_yield': 0,
                        'substrate_uptake': 0
                    })

            if env_name == 'Glu':
                logger.info("  Performing sensitivity analysis (Glu)...")
                sens_df = perform_sensitivity_analysis(model, environment_config)
                sensitivity_results.append(sens_df)

            logger.info("  Performing gene essentiality analysis...")
            ess_df = analyze_gene_essentiality(model, environment_config)
            essentiality_results.append(ess_df)

        results_dir = Path(__file__).parent.parent / "results" / "tables"
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_simulation_results).to_csv(results_dir / "enhanced_baseline_simulations.csv", index=False)

        if sensitivity_results:
            pd.concat(sensitivity_results, ignore_index=True).to_csv(results_dir / "sensitivity_analysis.csv", index=False)
        if essentiality_results:
            pd.concat(essentiality_results, ignore_index=True).to_csv(results_dir / "gene_essentiality.csv", index=False)

        figures_dir = Path(__file__).parent.parent / "results" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig = create_multi_environment_summary(pd.DataFrame(all_simulation_results))
        fig.savefig(figures_dir / "multi_environment_summary.png", dpi=300, bbox_inches='tight')

        logger.info("Baseline simulations completed successfully.")

    except Exception as e:
        logger.exception("Critical error in baseline simulation: %s", e)
        raise


if __name__ == "__main__":
    main()

