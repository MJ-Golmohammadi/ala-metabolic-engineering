# 1_baseline_simulation_enhanced.py
"""
Comprehensive Baseline FBA Simulations for Multi-Environmental Analysis
Q1 Journal Quality - Enhanced for Pseudomonas putida KT2440 5-ALA Production

Performs systematic flux balance analysis across four environmental conditions
(Glu, Cit, Ser, Fer) with sensitivity analysis and robustness testing.

This file is the original baseline script with targeted fixes:
- Robust objective handling: the script validates that objective reaction IDs
  (values from the YAML `objectives` mapping) exist in the loaded model before
  attempting simulation.
- Defensive model copying: per-environment simulations operate on a copy of the
  environment-specific model to avoid accidental mutation.
- Removed an inline non-informative comment and improved logging for clarity.
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

from models.model_utils import simulate_with_objective, calculate_yield_metrics

# Set publication-quality plotting parameters
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def get_config_path() -> Path:
    """Get absolute path to config file with error handling"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "engineering_scenarios.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found at: {config_path}")

    return config_path


def get_model_path(environment: str) -> Path:
    """Get absolute path to environment-specific model file"""
    project_root = Path(__file__).parent.parent
    model_filename = f"iJN1463_{environment}_ExprThermoConstrainedFile.xml"
    model_path = project_root / "models" / "final_constrained_rnaseq_thermo" / model_filename

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found for {environment}: {model_path}")

    return model_path


def perform_sensitivity_analysis(model: cobra.Model, environment: Dict,
                                 parameter_range: np.ndarray = np.linspace(0.5, 2.0, 10)) -> pd.DataFrame:
    """
    Perform sensitivity analysis on key environmental parameters

    Parameters:
    -----------
    model : cobra.Model
        Metabolic model
    environment : Dict
        Base environmental conditions
    parameter_range : np.ndarray
        Range of parameter variations to test

    Returns:
    --------
    pd.DataFrame
        Sensitivity analysis results
    """
    sensitivity_results = []
    base_conditions = environment.copy()

    # Test sensitivity to carbon source uptake
    for carbon_rxn in ['EX_glc__D_e', 'EX_cit_e', 'EX_ser__L_e', 'EX_fer_e']:
        if carbon_rxn in base_conditions:
            for factor in parameter_range:
                with model:
                    # Modify carbon uptake
                    modified_environment = base_conditions.copy()
                    try:
                        modified_environment[carbon_rxn] = [base_conditions[carbon_rxn][0] * factor, 1000]
                    except Exception:
                        # If bounds are not in expected format, skip this variation
                        logging.debug("Skipping sensitivity variation for %s due to unexpected bounds format.", carbon_rxn)
                        continue

                    # Apply environment and simulate
                    for rxn_id, bounds in modified_environment.items():
                        if rxn_id in model.reactions:
                            try:
                                model.reactions.get_by_id(rxn_id).bounds = bounds
                            except Exception:
                                logging.debug("Failed to set bounds for %s during sensitivity test.", rxn_id)

                    # Test both objectives (only biomass here; extend if needed)
                    # Validate objective presence before setting
                    if 'BIOMASS_KT2440_WT3' in model.reactions:
                        model.objective = 'BIOMASS_KT2440_WT3'
                        solution = model.optimize()

                        if solution.status == 'optimal':
                            ala_flux = solution.fluxes.get('G1SAT', 0)
                            sensitivity_results.append({
                                'parameter': carbon_rxn,
                                'variation_factor': factor,
                                'objective': 'max_biomass',
                                'growth_rate': solution.objective_value,
                                'ala_flux': ala_flux,
                                'environment': 'sensitivity_test'
                            })
                    else:
                        logging.warning("BIOMASS_KT2440_WT3 not present in model; skipping sensitivity objective.")

    if sensitivity_results:
        return pd.DataFrame(sensitivity_results)
    else:
        return pd.DataFrame(columns=['parameter', 'variation_factor', 'objective', 'growth_rate', 'ala_flux', 'environment'])


def analyze_gene_essentiality(model: cobra.Model, environment: Dict,
                              target_genes: List[str] = None) -> pd.DataFrame:
    """
    Perform gene essentiality analysis using COBRApy single gene deletion

    Parameters:
    -----------
    model : cobra.Model
        Metabolic model
    environment : Dict
        Environmental conditions
    target_genes : List[str]
        List of gene IDs to test for essentiality

    Returns:
    --------
    pd.DataFrame
        Gene essentiality analysis results
    """
    from cobra.flux_analysis import single_gene_deletion

    if target_genes is None:
        # Focus on C5 pathway and central metabolism genes (example placeholders)
        target_genes = ['PP_0732', 'PP_4784', 'PP_1977']  # hemA, hemL, gltX equivalents

    essentiality_results = []

    with model:
        # Apply environment
        for rxn_id, bounds in environment.items():
            if rxn_id in model.reactions:
                try:
                    model.reactions.get_by_id(rxn_id).bounds = bounds
                except Exception:
                    logging.debug("Failed to apply environment bound for %s during essentiality analysis.", rxn_id)

        # Perform single gene deletion
        deletion_results = single_gene_deletion(model, target_genes)

        # Iterate over DataFrame rows returned by single_gene_deletion
        # COBRApy returns a pandas DataFrame
        for gene_id, row in deletion_results.iterrows():
            essentiality_results.append({
                'gene': gene_id,  # Gene ID tested for essentiality
                'growth_rate': row.get('growth', np.nan),  # Growth rate after deletion
                'status': 'essential' if row.get('growth', 0) < 0.01 else 'non_essential',
                'environment': 'Glu'  # Base environment for essentiality test
            })

    return pd.DataFrame(essentiality_results)


def create_multi_environment_summary(simulation_results: pd.DataFrame) -> plt.Figure:
    """
    Create comprehensive multi-environment summary visualization

    Parameters:
    -----------
    simulation_results : pd.DataFrame
        Results from multi-environment simulations

    Returns:
    --------
    plt.Figure
        Multi-panel summary figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Growth rate comparison across environments
    growth_data = simulation_results[simulation_results['objective'] == 'max_biomass']
    environments = growth_data['environment'].unique()
    growth_rates = [growth_data[growth_data['environment'] == env]['growth_rate'].mean() for env in environments]

    bars = ax1.bar(environments, growth_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax1.set_ylabel('Growth Rate (h$^{-1}$)', fontweight='bold')
    ax1.set_title('A. Growth Rate Across Environmental Conditions', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. ALA production comparison
    ala_data = simulation_results[simulation_results['objective'] == 'max_biomass']
    ala_production = [ala_data[ala_data['environment'] == env]['ala_flux_net'].mean() for env in environments]

    ax2.bar(environments, ala_production, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax2.set_ylabel('Net ALA Production (mmol/gDW/h)', fontweight='bold')
    ax2.set_title('B. ALA Production Across Environmental Conditions', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Yield comparison
    yields = [ala_data[ala_data['environment'] == env]['yield_mmol_g'].mean() for env in environments]

    ax3.bar(environments, yields, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax3.set_ylabel('Yield (mmol ALA/mmol Substrate)', fontweight='bold')
    ax3.set_title('C. Production Yield Across Environments', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Environmental condition efficiency
    efficiencies = []
    for env in environments:
        env_data = ala_data[ala_data['environment'] == env]
        if len(env_data) > 0:
            ala_mean = env_data['ala_flux_net'].mean()
            growth_mean = env_data['growth_rate'].mean()

            if growth_mean is not None and growth_mean > 1e-9:
                # Efficiency defined as ALA flux per unit growth.
                # Guard against division by zero when growth_mean ≈ 0 to prevent invalid values.
                efficiency = ala_mean / growth_mean
            else:
                efficiency = float('nan')  # or 0.0 if you prefer to treat no-growth as zero efficiency
            efficiencies.append(efficiency)
        else:
            efficiencies.append(0.0)

    ax4.bar(environments, efficiencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax4.set_ylabel('Production Efficiency (ALA/Growth)', fontweight='bold')
    ax4.set_title('D. Metabolic Efficiency by Environment', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def main():
    """
    Execute comprehensive multi-environment baseline simulations with advanced analyses
    """

    try:
        config_path = get_config_path()

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logging.info("🚀 Starting comprehensive multi-environment analysis...")

        # Initialize results storage
        all_simulation_results = []
        sensitivity_results = []
        essentiality_results = []

        # Analyze all four environments
        environments_to_analyze = ['Glu', 'Cit', 'Ser', 'Fer']

        # Validate objectives structure
        objectives_config = config.get('objectives', {})
        if not isinstance(objectives_config, dict):
            logging.error("Invalid 'objectives' in YAML: expected mapping of labels to reaction ids.")
            objectives_config = {}

        for env_name in environments_to_analyze:
            logging.info("\n📊 Analyzing environment: %s", env_name)

            try:
                # Load environment-specific model
                model_path = get_model_path(env_name)
                model = cobra.io.read_sbml_model(str(model_path))
                # Optionally set solver if desired; leave default otherwise
                # model.solver = "glpk"

                # Get environment configuration (fall back to Glu if missing)
                environment_config = config['environments'].get(env_name, config['environments'].get('Glu', {}))

                # Work on a copy to avoid mutating the loaded model across analyses
                model_for_sim = model.copy()

                # Apply environment bounds to the model copy
                if isinstance(environment_config, dict):
                    for rxn_id, bounds in environment_config.items():
                        try:
                            if rxn_id in model_for_sim.reactions:
                                model_for_sim.reactions.get_by_id(rxn_id).bounds = bounds
                            else:
                                logging.debug("Environment reaction %s not found in model; skipping bound assignment.", rxn_id)
                        except Exception:
                            logging.debug("Failed to apply bounds for %s in environment %s.", rxn_id, env_name)
                else:
                    logging.warning("Environment configuration for %s is not a mapping; skipping bound application.", env_name)

                # Perform baseline simulations for each objective defined in YAML
                for obj_name, objective in objectives_config.items():
                    logging.info("  Running objective: %s", obj_name)

                    # Validate that the objective value is a reaction id present in the model
                    if not isinstance(objective, str):
                        logging.error("    ❌ Objective %s has invalid reaction id: %s", obj_name, str(objective))
                        # Record failed simulation entry
                        all_simulation_results.append({
                            'environment': env_name,
                            'objective': obj_name,
                            'growth_rate': 0,
                            'ala_flux': 0,
                            'ala_flux_net': 0,
                            'ala_consumption': 0,
                            'solution_status': 'invalid_objective',
                            'modifications': 'wild_type',
                            'modification_count': 0,
                            'yield_mmol_g': 0,
                            'yield_mmol_mmol': 0,
                            'carbon_yield': 0,
                            'substrate_uptake': 0
                        })
                        continue

                    if objective not in model_for_sim.reactions:
                        logging.error("    ❌ Objective %s refers to %s, which is not present in the model for environment %s.", obj_name, objective, env_name)
                        # Record failed simulation entry
                        all_simulation_results.append({
                            'environment': env_name,
                            'objective': obj_name,
                            'growth_rate': 0,
                            'ala_flux': 0,
                            'ala_flux_net': 0,
                            'ala_consumption': 0,
                            'solution_status': 'missing_objective',
                            'modifications': 'wild_type',
                            'modification_count': 0,
                            'yield_mmol_g': 0,
                            'yield_mmol_mmol': 0,
                            'carbon_yield': 0,
                            'substrate_uptake': 0
                        })
                        continue

                    try:
                        # Use simulate_with_objective helper; ensure it receives a model copy to avoid side effects
                        sim_model = model_for_sim.copy()
                        solution = simulate_with_objective(sim_model, objective, environment_config)

                        # Calculate yield metrics
                        if env_name == 'Glu':
                            substrate_rxn = 'EX_glc__D_e'
                        elif env_name == 'Cit':
                            substrate_rxn = 'EX_cit_e'
                        elif env_name == 'Ser':
                            substrate_rxn = 'EX_ser__L_e'
                        elif env_name == 'Fer':
                            substrate_rxn = 'EX_fer_e'
                        else:
                            substrate_rxn = 'EX_glc__D_e'

                        yield_metrics = calculate_yield_metrics(solution, 'G1SAT', substrate_rxn)

                        result = {
                            'environment': env_name,
                            'objective': obj_name,
                            'growth_rate': solution.objective_value,
                            'ala_flux': solution.fluxes.get('G1SAT', 0),
                            'ala_flux_net': yield_metrics.get('net_ala_flux', 0),
                            'ala_consumption': yield_metrics.get('ala_consumption_flux', 0),
                            'solution_status': solution.status,
                            'modifications': 'wild_type',
                            'modification_count': 0,
                            'yield_mmol_g': yield_metrics.get('yield_mmol_g', 0),
                            'yield_mmol_mmol': yield_metrics.get('yield_mmol_mmol', 0),
                            'carbon_yield': yield_metrics.get('carbon_yield', 0),
                            'substrate_uptake': yield_metrics.get('substrate_uptake', 0)
                        }

                        all_simulation_results.append(result)
                        logging.info("    ✅ %s: Growth = %.4f, ALA = %.4f", obj_name, solution.objective_value, yield_metrics.get('net_ala_flux', 0))

                    except Exception as e:
                        logging.error("    ❌ Simulation failed for objective %s in environment %s: %s", obj_name, env_name, e)
                        # Store failed simulation
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

                # Perform sensitivity analysis for glucose environment
                if env_name == 'Glu':
                    logging.info("  🔍 Performing sensitivity analysis...")
                    try:
                        sens_results = perform_sensitivity_analysis(model_for_sim, environment_config)
                        if not sens_results.empty:
                            sensitivity_results.append(sens_results)
                    except Exception as e:
                        logging.warning("  Sensitivity analysis failed for %s: %s", env_name, e)

                # Perform gene essentiality analysis
                logging.info("  🧬 Performing gene essentiality analysis...")
                try:
                    ess_results = analyze_gene_essentiality(model_for_sim, environment_config)
                    if not ess_results.empty:
                        essentiality_results.append(ess_results)
                except Exception as e:
                    logging.warning("  Gene essentiality analysis failed for %s: %s", env_name, e)

            except Exception as e:
                logging.error("❌ Error analyzing environment %s: %s", env_name, e)
                continue

        # Save all results
        results_dir = Path(__file__).parent.parent / "results" / "tables"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save simulation results
        results_df = pd.DataFrame(all_simulation_results)
        results_path = results_dir / "enhanced_baseline_simulations.csv"
        results_df.to_csv(results_path, index=False)

        # Save sensitivity results
        if sensitivity_results:
            sens_df = pd.concat(sensitivity_results, ignore_index=True)
            sens_df.to_csv(results_dir / "sensitivity_analysis.csv", index=False)

        # Save essentiality results
        if essentiality_results:
            ess_df = pd.concat(essentiality_results, ignore_index=True)
            ess_df.to_csv(results_dir / "gene_essentiality.csv", index=False)

        # Create and save summary visualization
        logging.info("\n📈 Creating comprehensive visualizations...")
        try:
            summary_fig = create_multi_environment_summary(results_df)
            figures_dir = Path(__file__).parent.parent / "results" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            summary_fig.savefig(figures_dir / "multi_environment_summary.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            logging.warning("Failed to create/save summary visualization: %s", e)

        # Print comprehensive summary
        logging.info("\n" + "=" * 80)
        logging.info("COMPREHENSIVE MULTI-ENVIRONMENT ANALYSIS SUMMARY")
        logging.info("=" * 80)

        # Performance summary by environment
        production_data = results_df[results_df['objective'] == 'max_biomass']
        if not production_data.empty:
            summary_stats = production_data.groupby('environment').agg({
                'growth_rate': ['mean', 'std'],
                'ala_flux_net': ['mean', 'std'],
                'yield_mmol_g': ['mean', 'std']
            }).round(4)

            logging.info("\n📊 Performance Summary by Environment:")
            logging.info("\n%s", summary_stats)

            # Identify optimal environment (guard against empty frames)
            try:
                best_ala_env = production_data.loc[production_data['ala_flux_net'].idxmax()]
                best_growth_env = production_data.loc[production_data['growth_rate'].idxmax()]
                best_yield_env = production_data.loc[production_data['yield_mmol_g'].idxmax()]

                logging.info("\n🏆 Optimal Conditions:")
                logging.info("  Highest ALA Production: %s (%.4f mmol/gDW/h)", best_ala_env['environment'], best_ala_env['ala_flux_net'])
                logging.info("  Highest Growth: %s (%.4f h⁻¹)", best_growth_env['environment'], best_growth_env['growth_rate'])
                logging.info("  Highest Yield: %s (%.4f mmol/mmol)", best_yield_env['environment'], best_yield_env['yield_mmol_g'])
            except Exception:
                logging.info("Could not determine best environment due to missing or invalid data.")
        else:
            logging.info("No production data available for summary statistics.")

        logging.info("\n✅ Enhanced baseline analysis completed successfully!")

    except Exception as e:
        logging.error("❌ Critical error in main execution: %s", e)
        raise


if __name__ == "__main__":
    main()
