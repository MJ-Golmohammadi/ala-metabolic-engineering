# 1_baseline_simulation_enhanced.py
"""
Comprehensive Baseline FBA Simulations for Multi-Environmental Analysis
Q1 Journal Quality - Enhanced for Pseudomonas putida KT2440 5-ALA Production

Performs systematic flux balance analysis across four environmental conditions
(Glu, Cit, Ser, Fer) with sensitivity analysis and robustness testing.
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
                    modified_environment[carbon_rxn] = [base_conditions[carbon_rxn][0] * factor, 1000]
                    
                    # Apply environment and simulate
                    for rxn_id, bounds in modified_environment.items():
                        if rxn_id in model.reactions:
                            model.reactions.get_by_id(rxn_id).bounds = bounds
                    
                    # Test both objectives
                    for obj_name in ['max_biomass']:
                        model.objective = obj_name
                        solution = model.optimize()
                        
                        if solution.status == 'optimal':
                            ala_flux = solution.fluxes.get('G1SAT', 0)
                            sensitivity_results.append({
                                'parameter': carbon_rxn,
                                'variation_factor': factor,
                                'objective': obj_name,
                                'growth_rate': solution.objective_value,
                                'ala_flux': ala_flux,
                                'environment': 'sensitivity_test'
                            })
    
    return pd.DataFrame(sensitivity_results)

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
        # Focus on C5 pathway and central metabolism genes
        target_genes = ['PP_0732', 'PP_4784', 'PP_1977']  # hemA, hemL, gltX equivalents
    
    essentiality_results = []
    
    with model:
        # Apply environment
        for rxn_id, bounds in environment.items():
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).bounds = bounds
        
        # Perform single gene deletion
        deletion_results = single_gene_deletion(model, target_genes)
        
        # Iterate over DataFrame rows returned by single_gene_deletion
        # Professional comment: COBRApy returns a pandas DataFrame, not a list of dicts
        for gene_id, row in deletion_results.iterrows():
            essentiality_results.append({
                'gene': gene_id,  # Gene ID tested for essentiality
                'growth_rate': row['growth'],  # Growth rate after deletion
                'status': 'essential' if row['growth'] < 0.01 else 'non_essential',
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
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
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

def optimize_max_ala_with_min_growth(
    model,
    ala_obj_id: str = "G1SAT",                # Reaction ID for ALA production
    biomass_id: str = "BIOMASS_KT2440_WT3",   # Reaction ID for biomass growth
    b_min_fraction: float = 5.0              # Fraction of reference growth to enforce
):
    """
    Optimize ALA production while enforcing a minimum biomass growth (ε-constraint).

    Parameters
    ----------
    model : cobra.Model
        The metabolic model to optimize.
    ala_obj_id : str
        Reaction ID for ALA production (default: "G1SAT").
    biomass_id : str
        Reaction ID for biomass growth (default: "BIOMASS_KT2440_WT3").
    b_min_fraction : float
        Fraction of reference biomass growth to enforce as a minimum (default: 0.01 = 1%).

    Returns
    -------
    cobra.Solution
        Solution object with ALA maximized subject to minimal growth constraint.
    """

    # Step 1: Compute reference growth under biomass objective
    with model:
        model.objective = biomass_id
        ref_solution = model.optimize()
        ref_growth = max(0.0, float(ref_solution.fluxes.get(biomass_id, 0.0)))

    # Step 2: Define minimum growth requirement (ε-constraint)
    b_min = b_min_fraction * ref_growth

    # Step 3: Apply growth lower bound
    biomass_rxn = model.reactions.get_by_id(biomass_id)
    prev_lb = biomass_rxn.lower_bound
    biomass_rxn.lower_bound = max(prev_lb, b_min)

    # Step 4: Optimize for ALA production
    model.objective = ala_obj_id
    solution = model.optimize()

    # Step 5: Restore original biomass lower bound
    biomass_rxn.lower_bound = prev_lb

    return solution


def main():
    """
    Execute comprehensive multi-environment baseline simulations with advanced analyses
    """

    try:
        # Load configuration file
        config_path = get_config_path()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("🚀 Starting comprehensive multi-environment analysis...")

        # Initialize result storage
        all_simulation_results = []
        sensitivity_results = []
        essentiality_results = []

        # Define environments to analyze
        environments_to_analyze = ['Glu', 'Cit', 'Ser', 'Fer']

        for env_name in environments_to_analyze:
            print(f"\n📊 Analyzing environment: {env_name}")
            try:
                # Load environment-specific model
                model_path = get_model_path(env_name)
                model = cobra.io.read_sbml_model(str(model_path))
                model.solver = "glpk"

                # Get environment configuration
                environment_config = config['environments'].get(env_name, config['environments']['Glu'])

                # Run baseline simulations for each objective
                for obj_name, objective in config['objectives'].items():
                    print(f"  Running objective: {obj_name}")
                    try:
                        # Special handling for max_ala: enforce minimal growth (ε-constraint)
                        if obj_name == "max_ala":
                            solution = optimize_max_ala_with_min_growth(
                                model,
                                ala_obj_id='G1SAT',
                                biomass_id='BIOMASS_KT2440_WT3',
                                b_min_fraction=0.01  # enforce at least 1% of reference growth
                            )
                        else:
                            solution = simulate_with_objective(model, objective, environment_config)

                        # Select substrate exchange reaction depending on environment
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

                        # Calculate yield metrics (includes net ALA flux)
                        yield_metrics = calculate_yield_metrics(solution, 'G1SAT', substrate_rxn)

                        # Growth always from biomass flux, not objective_value
                        growth_flux = float(solution.fluxes.get('BIOMASS_KT2440_WT3', 0.0))
                        ala_flux_gross = float(solution.fluxes.get('G1SAT', 0.0))
                        ala_flux_net = yield_metrics['net_ala_flux']

                        # Build result dictionary
                        result = {
                            'environment': env_name,
                            'objective': obj_name,
                            'growth_rate': growth_flux,              # biomass flux
                            'ala_flux': ala_flux_gross,              # gross ALA flux
                            'ala_flux_net': ala_flux_net,            # net ALA flux
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

                        # Print results clearly
                        if obj_name == "max_biomass":
                            print(f"    ✅ {obj_name}: Growth = {growth_flux:.4f}, ALA_net = {ala_flux_net:.4f}")
                        elif obj_name == "max_ala":
                            print(f"    ✅ {obj_name}: Growth = {growth_flux:.4f}, ALA_net = {ala_flux_net:.4f} (ALA_gross = {ala_flux_gross:.4f})")

                    except Exception as e:
                        print(f"    ❌ Simulation failed: {e}")
                        # Store failed simulation with zeros
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

                # Sensitivity analysis only for glucose environment
                if env_name == 'Glu':
                    print("  🔍 Performing sensitivity analysis...")
                    sens_results = perform_sensitivity_analysis(model, environment_config)
                    sensitivity_results.append(sens_results)

                # Gene essentiality analysis
                print("  🧬 Performing gene essentiality analysis...")
                ess_results = analyze_gene_essentiality(model, environment_config)
                essentiality_results.append(ess_results)

            except Exception as e:
                print(f"❌ Error analyzing environment {env_name}: {e}")
                continue

        # Save results to disk
        results_dir = Path(__file__).parent.parent / "results" / "tables"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(all_simulation_results)
        results_df.to_csv(results_dir / "enhanced_baseline_simulations.csv", index=False)

        if sensitivity_results:
            sens_df = pd.concat(sensitivity_results, ignore_index=True)
            sens_df.to_csv(results_dir / "sensitivity_analysis.csv", index=False)

        if essentiality_results:
            ess_df = pd.concat(essentiality_results, ignore_index=True)
            ess_df.to_csv(results_dir / "gene_essentiality.csv", index=False)

        # Create and save summary visualization
        print("\n📈 Creating comprehensive visualizations...")
        summary_fig = create_multi_environment_summary(results_df)
        figures_dir = Path(__file__).parent.parent / "results" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        summary_fig.savefig(figures_dir / "multi_environment_summary.png", dpi=300, bbox_inches='tight')

        # Print summary statistics
        print("\n" + "="*80)
        print("COMPREHENSIVE MULTI-ENVIRONMENT ANALYSIS SUMMARY")
        print("="*80)

        production_data = results_df[results_df['objective'] == 'max_biomass']
        summary_stats = production_data.groupby('environment').agg({
            'growth_rate': 'mean',
            'ala_flux_net': 'mean',
            'yield_mmol_g': 'mean',
        }).round(4)

        print("\n📊 Performance Summary by Environment:")
        print(summary_stats)

        # Identify optimal environments
        best_ala_env = production_data.loc[production_data['ala_flux_net'].idxmax()]
        best_growth_env = production_data.loc[production_data['growth_rate'].idxmax()]
        best_yield_env = production_data.loc[production_data['yield_mmol_g'].idxmax()]

        print(f"\n🏆 Optimal Conditions:")
        print(f"  Highest ALA Production: {best_ala_env['environment']} ({best_ala_env['ala_flux_net']:.4f} mmol/gDW/h)")
        print(f"  Highest Growth: {best_growth_env['environment']} ({best_growth_env['growth_rate']:.4f} h⁻¹)")
        print(f"  Highest Yield: {best_yield_env['environment']} ({best_yield_env['yield_mmol_g']:.4f} mmol/mmol)")

        print("\n✅ Enhanced baseline analysis completed successfully!")

    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        raise



if __name__ == "__main__":
    main()
