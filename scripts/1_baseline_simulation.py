# 1_baseline_simulation.py
"""
Comprehensive Baseline FBA Simulations for Multi-Environmental Analysis
Q1 Journal Quality - Enhanced for Pseudomonas putida KT2440 5-ALA Production

Performs systematic flux balance analysis across four environmental conditions
(Glu, Cit, Ser, Fer) with robust error handling and validation.
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

from models.model_utils import *

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
    'font.family': 'Arial'
})

def get_config_path() -> Path:
    """Get absolute path to config file with comprehensive error handling"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "engineering_scenarios.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"❌ Config file not found at: {config_path}")
    
    return config_path

def get_model_path(environment: str) -> Path:
    """Get absolute path to environment-specific model file with validation"""
    project_root = Path(__file__).parent.parent
    model_filename = f"iJN1463_{environment}_ExprThermoConstrainedFile.xml"
    model_path = project_root / "models" / "final_constrained_rnaseq_thermo" / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found for {environment}: {model_path}")
    
    return model_path

def validate_environment_config(environment_config: Dict) -> bool:
    """
    Validate environment configuration format and bounds
    
    Parameters:
    -----------
    environment_config : Dict
        Environment configuration with reaction bounds
        
    Returns:
    --------
    bool
        True if configuration is valid
    """
    if not isinstance(environment_config, dict):
        print(f"❌ Environment config must be a dictionary")
        return False
    
    required_reactions = ['EX_glc__D_e', 'EX_o2_e', 'EX_nh4_e']
    for rxn_id in required_reactions:
        if rxn_id not in environment_config:
            print(f"⚠️ Required reaction {rxn_id} not in environment config")
    
    # Validate bounds format
    for rxn_id, bounds in environment_config.items():
        if not isinstance(bounds, (list, tuple)):
            print(f"❌ Bounds for {rxn_id} must be list/tuple, got {type(bounds)}")
            return False
        if len(bounds) != 2:
            print(f"❌ Bounds for {rxn_id} must have 2 elements, got {len(bounds)}")
            return False
        if not all(isinstance(x, (int, float)) for x in bounds):
            print(f"❌ Bounds for {rxn_id} must be numeric")
            return False
    
    return True

def robust_simulate_with_objective(model: cobra.Model, objective: str, environment: Dict) -> cobra.Solution:
    """
    Enhanced FBA simulation with comprehensive error handling and validation
    
    Parameters:
    -----------
    model : cobra.Model
        Metabolic model
    objective : str
        Objective reaction ID
    environment : Dict
        Environmental conditions with reaction bounds
        
    Returns:
    --------
    cobra.Solution
        FBA solution with status information
    """
    
    with model:
        try:
            # Ensure biomass reaction is properly configured
            if 'BIOMASS_KT2440_WT3' in model.reactions:
                biomass_rxn = model.reactions.get_by_id('BIOMASS_KT2440_WT3')
                biomass_rxn.lower_bound = 0
                biomass_rxn.upper_bound = 1000
            
            # Apply environmental constraints with validation
            for rxn_id, bounds in environment.items():
                if rxn_id in model.reactions:
                    try:
                        # Validate bounds format
                        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                            lb, ub = float(bounds[0]), float(bounds[1])
                            model.reactions.get_by_id(rxn_id).bounds = (lb, ub)
                        else:
                            print(f"⚠️ Invalid bounds format for {rxn_id}: {bounds}")
                            continue
                    except (ValueError, TypeError) as e:
                        print(f"⚠️ Could not parse bounds for {rxn_id}: {e}")
                        continue
                else:
                    print(f"⚠️ Reaction {rxn_id} not found in model")
            
            # Set objective function with fallback
            if objective in model.reactions:
                model.objective = objective
                print(f"    Set objective: {objective}")
            else:
                print(f"⚠️ Objective reaction {objective} not found, using biomass as fallback")
                model.objective = 'BIOMASS_KT2440_WT3'
            
            # Perform flux balance analysis
            solution = model.optimize()
            
            # Validate solution
            if solution.status != 'optimal':
                print(f"⚠️ Optimization failed with status: {solution.status}")
            
            return solution
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            # Return empty solution with error status
            return cobra.Solution(objective_value=0, status='error', fluxes=pd.Series())

def perform_sensitivity_analysis(model: cobra.Model, environment: Dict, 
                               parameter_range: np.ndarray = np.linspace(0.5, 2.0, 5)) -> pd.DataFrame:
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
    
    print("  🔍 Performing sensitivity analysis...")
    
    # Test sensitivity to carbon source uptake
    carbon_rxns = ['EX_glc__D_e', 'EX_cit_e', 'EX_ser__L_e', 'EX_fer_e']
    
    for carbon_rxn in carbon_rxns:
        if carbon_rxn in base_conditions:
            original_bounds = base_conditions[carbon_rxn]
            
            for factor in parameter_range:
                with model:
                    # Modify carbon uptake
                    modified_bounds = [original_bounds[0] * factor, original_bounds[1]]
                    
                    # Apply modified environment
                    for rxn_id, bounds in base_conditions.items():
                        if rxn_id in model.reactions:
                            if rxn_id == carbon_rxn:
                                model.reactions.get_by_id(rxn_id).bounds = modified_bounds
                            else:
                                model.reactions.get_by_id(rxn_id).bounds = bounds
                    
                    # Test biomass objective
                    try:
                        model.objective = 'BIOMASS_KT2440_WT3'
                        solution = model.optimize()
                        
                        if solution.status == 'optimal':
                            ala_flux = solution.fluxes.get('G1SAT', 0)
                            sensitivity_results.append({
                                'parameter': carbon_rxn,
                                'variation_factor': factor,
                                'growth_rate': solution.objective_value,
                                'ala_flux': ala_flux,
                                'environment': 'sensitivity_test'
                            })
                    except Exception as e:
                        print(f"    ⚠️ Sensitivity test failed for {carbon_rxn} (factor {factor}): {e}")
    
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
    
    print("  🧬 Performing gene essentiality analysis...")
    
    if target_genes is None:
        # Focus on C5 pathway and central metabolism genes
        target_genes = ['PP_0732', 'PP_4784', 'PP_1977']  # hemA, hemL, gltX equivalents
    
    essentiality_results = []
    
    with model:
        # Apply environment with error handling
        for rxn_id, bounds in environment.items():
            if rxn_id in model.reactions:
                try:
                    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                        model.reactions.get_by_id(rxn_id).bounds = tuple(bounds)
                except Exception as e:
                    print(f"    ⚠️ Could not set bounds for {rxn_id}: {e}")
        
        # Check baseline growth
        try:
            model.objective = 'BIOMASS_KT2440_WT3'
            baseline_solution = model.optimize()
            baseline_growth = baseline_solution.objective_value if baseline_solution.status == 'optimal' else 0
            
            if baseline_growth < 0.01:
                print(f"    ⚠️ Skipping essentiality analysis - low baseline growth: {baseline_growth:.4f}")
                return pd.DataFrame(essentiality_results)
                
        except Exception as e:
            print(f"    ❌ Baseline growth check failed: {e}")
            return pd.DataFrame(essentiality_results)
        
        # Filter genes that exist in the model
        valid_genes = [gene for gene in target_genes if gene in model.genes]
        
        if not valid_genes:
            print("    ⚠️ No valid genes found for essentiality analysis")
            return pd.DataFrame(essentiality_results)
        
        # Perform single gene deletion
        try:
            deletion_results = single_gene_deletion(model, valid_genes)
            
            for result in deletion_results:
                if result['ids']:  # Ensure valid result
                    gene_id = result['ids'][0]
                    growth_rate = result['growth']
                    status = 'essential' if growth_rate < 0.01 else 'non_essential'
                    
                    essentiality_results.append({
                        'gene': gene_id,
                        'growth_rate': growth_rate,
                        'status': status,
                        'environment': 'Glu',  # Base environment for essentiality test
                        'baseline_growth': baseline_growth,
                        'growth_ratio': growth_rate / baseline_growth if baseline_growth > 0 else 0
                    })
                    
            print(f"    ✅ Analyzed {len(essentiality_results)} genes")
            
        except Exception as e:
            print(f"    ❌ Gene essentiality analysis failed: {e}")
    
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
    
    # Filter successful simulations
    successful_data = simulation_results[
        (simulation_results['solution_status'] == 'optimal') & 
        (simulation_results['objective'] == 'max_biomass')
    ]
    
    if successful_data.empty:
        # Create placeholder figure if no successful data
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'No successful simulations', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        return fig
    
    environments = successful_data['environment'].unique()
    
    # 1. Growth rate comparison across environments
    growth_rates = [successful_data[successful_data['environment'] == env]['growth_rate'].mean() 
                   for env in environments]
    
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
    ala_production = [successful_data[successful_data['environment'] == env]['ala_flux_net'].mean() 
                     for env in environments]
    
    ax2.bar(environments, ala_production, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax2.set_ylabel('Net ALA Production (mmol/gDW/h)', fontweight='bold')
    ax2.set_title('B. ALA Production Across Environmental Conditions', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Yield comparison
    yields = [successful_data[successful_data['environment'] == env]['yield_mmol_g'].mean() 
             for env in environments]
    
    ax3.bar(environments, yields, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax3.set_ylabel('Yield (mmol ALA/g Substrate)', fontweight='bold')
    ax3.set_title('C. Production Yield Across Environments', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Environmental condition efficiency
    efficiencies = []
    for env in environments:
        env_data = successful_data[successful_data['environment'] == env]
        if len(env_data) > 0 and env_data['growth_rate'].mean() > 0:
            efficiency = (env_data['ala_flux_net'].mean() / env_data['growth_rate'].mean()) 
            efficiencies.append(efficiency)
        else:
            efficiencies.append(0)
    
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
        
        print("🚀 Starting comprehensive multi-environment analysis...")
        print("=" * 60)
        
        # Initialize results storage
        all_simulation_results = []
        sensitivity_results = []
        essentiality_results = []
        
        # Analyze all four environments
        environments_to_analyze = ['Glu', 'Cit', 'Ser', 'Fer']
        
        for env_name in environments_to_analyze:
            print(f"\n📊 Analyzing environment: {env_name}")
            print("-" * 40)
            
            try:
                # Load environment-specific model
                model_path = get_model_path(env_name)
                model = cobra.io.read_sbml_model(str(model_path))
                print(f"✅ Model loaded successfully: {len(model.reactions)} reactions, {len(model.metabolites)} metabolites")
                
                # Get environment configuration with validation
                environment_config = config['environments'].get(env_name, config['environments']['Glu'])
                
                if not validate_environment_config(environment_config):
                    print(f"⚠️ Environment config validation failed for {env_name}, using Glu config")
                    environment_config = config['environments']['Glu']
                
                # Validate objectives
                valid_objectives = {}
                for obj_name, objective in config['objectives'].items():
                    if objective in model.reactions:
                        valid_objectives[obj_name] = objective
                    else:
                        print(f"⚠️ Objective {objective} not found in model, skipping {obj_name}")
                
                if not valid_objectives:
                    print("⚠️ No valid objectives found, using biomass only")
                    valid_objectives = {'max_biomass': 'BIOMASS_KT2440_WT3'}
                
                # Perform baseline simulations
                simulation_count = 0
                successful_count = 0
                
                for obj_name, objective in valid_objectives.items():
                    print(f"  Running objective: {obj_name}")
                    simulation_count += 1
                    
                    try:
                        solution = robust_simulate_with_objective(model, objective, environment_config)
                        
                        # Calculate yield metrics only for successful simulations
                        if solution.status == 'optimal' and solution.objective_value > 1e-6:
                            substrate_rxn = get_substrate_rxn_for_environment(env_name)
                            yield_metrics = calculate_yield_metrics(solution, 'G1SAT', substrate_rxn, env_name)
                            
                            result = {
                                'environment': env_name,
                                'objective': obj_name,
                                'growth_rate': solution.objective_value,
                                'ala_flux': solution.fluxes.get('G1SAT', 0),
                                'ala_flux_net': yield_metrics['net_ala_flux'],
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
                            successful_count += 1
                            print(f"    ✅ {obj_name}: Growth = {solution.objective_value:.4f}, ALA = {yield_metrics['net_ala_flux']:.4f}")
                            
                        else:
                            print(f"    ⚠️ {obj_name}: Failed - Status: {solution.status}, Growth: {solution.objective_value:.6f}")
                            # Store failed simulation record
                            all_simulation_results.append({
                                'environment': env_name,
                                'objective': obj_name,
                                'growth_rate': 0,
                                'ala_flux': 0,
                                'ala_flux_net': 0,
                                'ala_consumption': 0,
                                'solution_status': solution.status,
                                'modifications': 'wild_type',
                                'modification_count': 0,
                                'yield_mmol_g': 0,
                                'yield_mmol_mmol': 0,
                                'carbon_yield': 0,
                                'substrate_uptake': 0
                            })
                            
                    except Exception as e:
                        print(f"    ❌ Simulation failed: {e}")
                        # Store error record
                        all_simulation_results.append({
                            'environment': env_name,
                            'objective': obj_name,
                            'growth_rate': 0,
                            'ala_flux': 0,
                            'ala_flux_net': 0,
                            'ala_consumption': 0,
                            'solution_status': 'error',
                            'modifications': 'wild_type',
                            'modification_count': 0,
                            'yield_mmol_g': 0,
                            'yield_mmol_mmol': 0,
                            'carbon_yield': 0,
                            'substrate_uptake': 0
                        })
                
                print(f"  📈 {successful_count}/{simulation_count} simulations successful")
                
                # Perform sensitivity analysis for glucose environment only (to save time)
                if env_name == 'Glu' and successful_count > 0:
                    sens_results = perform_sensitivity_analysis(model, environment_config)
                    sensitivity_results.append(sens_results)
                    print(f"  🔍 Sensitivity analysis completed: {len(sens_results)} data points")
                
                # Perform gene essentiality analysis for glucose environment
                if env_name == 'Glu':
                    ess_results = analyze_gene_essentiality(model, environment_config)
                    essentiality_results.append(ess_results)
                    if not ess_results.empty:
                        print(f"  🧬 Gene essentiality analysis completed: {len(ess_results)} genes analyzed")
                
            except Exception as e:
                print(f"❌ Error analyzing environment {env_name}: {e}")
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
        print("\n📈 Creating comprehensive visualizations...")
        summary_fig = create_multi_environment_summary(results_df)
        figures_dir = Path(__file__).parent.parent / "results" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        summary_fig.savefig(figures_dir / "multi_environment_summary.png", dpi=300, bbox_inches='tight')
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("COMPREHENSIVE MULTI-ENVIRONMENT ANALYSIS SUMMARY")
        print("="*80)
        
        # Performance summary by environment
        production_data = results_df[results_df['objective'] == 'max_biomass']
        successful_production = production_data[production_data['solution_status'] == 'optimal']
        
        print(f"\n📊 Performance Summary:")
        print(f"  Total simulations: {len(results_df)}")
        print(f"  Successful simulations: {len(successful_production)}")
        print(f"  Success rate: {len(successful_production)/len(results_df)*100:.1f}%")
        
        if not successful_production.empty:
            summary_stats = successful_production.groupby('environment').agg({
                'growth_rate': ['mean', 'std'],
                'ala_flux_net': ['mean', 'std'],
                'yield_mmol_g': ['mean', 'std']
            }).round(4)
            
            print("\n📈 Performance by Environment (Successful Simulations Only):")
            print(summary_stats)
            
            # Identify optimal environment
            best_ala_env = successful_production.loc[successful_production['ala_flux_net'].idxmax()]
            best_growth_env = successful_production.loc[successful_production['growth_rate'].idxmax()]
            best_yield_env = successful_production.loc[successful_production['yield_mmol_g'].idxmax()]
            
            print(f"\n🏆 Optimal Conditions:")
            print(f"  Highest ALA Production: {best_ala_env['environment']} "
                  f"({best_ala_env['ala_flux_net']:.4f} mmol/gDW/h)")
            print(f"  Highest Growth: {best_growth_env['environment']} "
                  f"({best_growth_env['growth_rate']:.4f} h⁻¹)")
            print(f"  Highest Yield: {best_yield_env['environment']} "
                  f"({best_yield_env['yield_mmol_g']:.4f} mmol/g)")
        
        print(f"\n✅ Enhanced baseline analysis completed successfully!")
        print(f"📁 Results saved to:")
        print(f"   - {results_path}")
        if sensitivity_results:
            print(f"   - results/tables/sensitivity_analysis.csv")
        if essentiality_results:
            print(f"   - results/tables/gene_essentiality.csv")
        print(f"   - results/figures/multi_environment_summary.png")
        
    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
