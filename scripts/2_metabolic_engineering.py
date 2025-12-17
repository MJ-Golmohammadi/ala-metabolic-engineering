# 2_metabolic_engineering_enhanced.py
"""
Systematic Multi-Environment Metabolic Engineering Evaluation for 5-ALA Production
Q1 Journal Quality - Enhanced with Sensitivity and Robustness Analysis
"""

import cobra
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import logging
import highspy
logging.getLogger('optlang').setLevel(logging.WARNING)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.stdout = open(os.devnull, 'w')
# load/initialize model here
sys.stdout = sys.__stdout__
from models.model_utils import create_engineered_strain, simulate_with_objective, calculate_yield_metrics

# Configure logging so create_engineered_strain messages are visible
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_multi_environment_engineering(base_model_paths: Dict, scenarios: Dict, 
                                         environments: Dict, objectives: Dict) -> pd.DataFrame:
    """
    Comprehensive evaluation of engineering scenarios across multiple environments
    """
    all_engineering_results = []
    
    for env_name, model_path in base_model_paths.items():
        print(f"\n🔧 Evaluating engineering scenarios in {env_name} environment...")
        
        try:            
          
            
            
            # Load environment-specific model
            model = cobra.io.read_sbml_model(str(model_path))
            
            # Build a HighsModel object
            hm = highspy.HighsModel()
            
            # Number of variables (columns) = number of reactions
            num_col = len(model.reactions)
            # Number of constraints (rows) = number of metabolites
            num_row = len(model.metabolites)
            
            hm.lp.num_col = num_col
            hm.lp.num_row = num_row
            
            # Objective coefficients: 1 for objective reaction, 0 otherwise
            obj = np.zeros(num_col)
            for i, rxn in enumerate(model.reactions):
                if rxn == model.objective.expression.keys()[0]:
                    obj[i] = 1.0
            hm.lp.col_cost = obj.tolist()
            
            # Bounds for each flux
            hm.lp.col_lower = [rxn.lower_bound for rxn in model.reactions]
            hm.lp.col_upper = [rxn.upper_bound for rxn in model.reactions]
            
            # Row bounds: all metabolite balances = 0
            hm.lp.row_lower = [0.0 for _ in model.metabolites]
            hm.lp.row_upper = [0.0 for _ in model.metabolites]
            
            # Constraint matrix (stoichiometric matrix S in CSR format)
            S = cobra.util.array.create_stoichiometric_matrix(model)
            # Convert to CSR
            csr = S.tocsr()
            hm.lp.a_matrix.start = csr.indptr.tolist()
            hm.lp.a_matrix.index = csr.indices.tolist()
            hm.lp.a_matrix.value = csr.data.tolist()
            
            # Initialize HiGHS solver
            highs = highspy.Highs()
            highs.setOptionValue("output_flag", True)
            
            # Pass the HighsModel object
            highs.passModel(hm)
            
            # Run the solver
            highs.run()
            
            # Get the solution
            solution = highs.getSolution()
            print("Objective value:", solution.objective_value)
            print("Fluxes:", solution.col_value)

            
            # Continue with environment-specific configuration
            environment_config = environments.get(env_name)

            
            if not environment_config:
                print(f"⚠️ No environment config for {env_name}, skipping...")
                continue
            
            # Evaluate all engineering scenarios
            for scenario_name, modifications in scenarios.items():
                print(f"  Testing scenario: {scenario_name}")
                
                try:
                    # Create engineered strain
                    # Pass reference objectives and environment so baselines are computed correctly
                    engineered_model = create_engineered_strain(
                        model,
                        modifications,
                        reference_objectives=['max_biomass', 'max_ala'],
                        environment=environment_config,
                        objective_to_rxn={'max_biomass': 'BIOMASS_KT2440_WT3', 'max_ala': 'G1SAT'},
                        cap_value=6000.0
                    )
                    
                    # Test all objective functions
                    for obj_name, objective in objectives.items():
                        # simulate_with_objective expects objective (reaction id or object) and environment
                        solution = simulate_with_objective(engineered_model, objective, environment_config)
                        
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
                        
                        # Use biomass flux for growth_rate (objective may be ALA)
                        growth_flux = float(solution.fluxes.get('BIOMASS_KT2440_WT3', 0.0))
                        ala_flux_gross = float(solution.fluxes.get('G1SAT', 0.0))
                        ala_flux_net = yield_metrics['net_ala_flux']
                        
                        # Store comprehensive results
                        result = {
                            'environment': env_name,
                            'scenario': scenario_name,
                            'objective': obj_name,
                            'growth_rate': growth_flux,
                            'ala_flux': ala_flux_gross,
                            'ala_flux_net': ala_flux_net,
                            'ala_consumption': yield_metrics['ala_consumption_flux'],
                            'glucose_uptake': abs(solution.fluxes.get('EX_glc__D_e', 0)),
                            'solution_status': solution.status,
                            'modifications': str(modifications),
                            'modification_count': len(modifications),
                            'yield_mmol_g': yield_metrics['yield_mmol_g'],
                            'yield_mmol_mmol': yield_metrics['yield_mmol_mmol'],
                            'carbon_yield': yield_metrics['carbon_yield'],
                            'substrate_uptake': yield_metrics['substrate_uptake']
                        }
                        
                        all_engineering_results.append(result)
                        
                        print(f"    ✅ {obj_name}: Growth = {growth_flux:.6f}, ALA_net = {ala_flux_net:.6f}")
                            
                except Exception as e:
                    print(f"    ❌ Scenario {scenario_name} failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing environment {env_name}: {e}")
            continue
    
    return pd.DataFrame(all_engineering_results)

def calculate_robustness_metrics(engineering_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate robustness metrics for engineering scenarios across environments
    """
    robustness_results = []
    
    for scenario in engineering_df['scenario'].unique():
        scenario_data = engineering_df[engineering_df['scenario'] == scenario]
        
        # Calculate performance metrics across environments
        env_performance = []
        for env in scenario_data['environment'].unique():
            env_data = scenario_data[scenario_data['environment'] == env]
            production_data = env_data[env_data['objective'] == 'max_biomass']
            
            if len(production_data) > 0:
                env_performance.append({
                    'environment': env,
                    'ala_production': production_data['ala_flux_net'].iloc[0],
                    'growth_rate': production_data['growth_rate'].iloc[0],
                    'yield': production_data['yield_mmol_g'].iloc[0]
                })
        
        if len(env_performance) > 0:
            ala_values = [p['ala_production'] for p in env_performance]
            growth_values = [p['growth_rate'] for p in env_performance]
            yield_values = [p['yield'] for p in env_performance]
            
            robustness_results.append({
                'scenario': scenario,
                'ala_mean': np.mean(ala_values),
                'ala_std': np.std(ala_values),
                'ala_cv': np.std(ala_values) / np.mean(ala_values) if np.mean(ala_values) > 0 else 0,
                'growth_mean': np.mean(growth_values),
                'growth_std': np.std(growth_values),
                'growth_cv': np.std(growth_values) / np.mean(growth_values) if np.mean(growth_values) > 0 else 0,
                'yield_mean': np.mean(yield_values),
                'yield_std': np.std(yield_values),
                'robustness_score': 1 / (1 + np.std(ala_values) / np.mean(ala_values)) if np.mean(ala_values) > 0 else 0,
                'environments_tested': len(env_performance)
            })
    
    return pd.DataFrame(robustness_results)

def create_engineering_summary_visualization(engineering_df: pd.DataFrame, 
                                           robustness_df: pd.DataFrame) -> plt.Figure:
    """
    Create comprehensive engineering summary visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter for production objective
    production_data = engineering_df[engineering_df['objective'] == 'max_biomass']
    
    # 1. ALA production by scenario and environment
    scenario_env_performance = production_data.groupby(['scenario', 'environment'])['ala_flux_net'].mean().unstack()
    
    # Select top scenarios for clarity
    top_scenarios = production_data.groupby('scenario')['ala_flux_net'].mean().nlargest(10).index
    plot_data = scenario_env_performance.loc[top_scenarios]
    
    plot_data.plot(kind='bar', ax=ax1, width=0.8, alpha=0.8)
    ax1.set_ylabel('Net ALA Production (mmol/gDW/h)', fontweight='bold')
    ax1.set_title('A. ALA Production by Engineering Scenario and Environment', fontweight='bold', pad=20)
    ax1.legend(title='Environment', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Growth vs Production trade-off
    scenarios_to_plot = production_data['scenario'].unique()[:15]  # Limit for clarity
    for scenario in scenarios_to_plot:
        scenario_data = production_data[production_data['scenario'] == scenario]
        ax2.scatter(scenario_data['growth_rate'], scenario_data['ala_flux_net'], 
                   s=80, alpha=0.7, label=scenario)
    
    ax2.set_xlabel('Growth Rate (h$^{-1}$)', fontweight='bold')
    ax2.set_ylabel('Net ALA Production (mmol/gDW/h)', fontweight='bold')
    ax2.set_title('B. Growth-Production Trade-off by Scenario', fontweight='bold', pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Robustness analysis
    top_robust = robustness_df.nlargest(10, 'robustness_score')
    x_pos = np.arange(len(top_robust))
    width = 0.35
    
    ax3.bar(x_pos - width/2, top_robust['ala_mean'], width, label='Mean ALA', alpha=0.8)
    ax3.bar(x_pos + width/2, top_robust['ala_std'], width, label='Std Dev', alpha=0.8)
    
    ax3.set_xlabel('Engineering Scenario', fontweight='bold')
    ax3.set_ylabel('ALA Production (mmol/gDW/h)', fontweight='bold')
    ax3.set_title('C. Production Robustness Across Environments', fontweight='bold', pad=20)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(top_robust['scenario'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Modification count vs improvement
    improvement_data = []
    for scenario in production_data['scenario'].unique():
        scenario_data = production_data[production_data['scenario'] == scenario]
        if len(scenario_data) > 0:
            avg_ala = scenario_data['ala_flux_net'].mean()
            mod_count = scenario_data['modification_count'].iloc[0]
            improvement_data.append({
                'scenario': scenario,
                'modification_count': mod_count,
                'ala_production': avg_ala
            })
    
    improvement_df = pd.DataFrame(improvement_data)
    ax4.scatter(improvement_df['modification_count'], improvement_df['ala_production'], 
               s=100, alpha=0.7, color='purple', edgecolor='black')
    
    # Add trend line
    if len(improvement_df) > 1:
        z = np.polyfit(improvement_df['modification_count'], improvement_df['ala_production'], 1)
        p = np.poly1d(z)
        ax4.plot(improvement_df['modification_count'], p(improvement_df['modification_count']), 
                "r--", alpha=0.8, label='Trend')
    
    ax4.set_xlabel('Number of Genetic Modifications', fontweight='bold')
    ax4.set_ylabel('Average ALA Production (mmol/gDW/h)', fontweight='bold')
    ax4.set_title('D. Engineering Complexity vs Production Benefit', fontweight='bold', pad=20)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """
    Execute comprehensive multi-environment metabolic engineering analysis
    """
    
    # Load configuration
    with open('config/engineering_scenarios.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🚀 Starting comprehensive multi-environment metabolic engineering analysis...")
    
    # Define model paths for all environments
    base_model_paths = {
        'Glu': 'models/final_constrained_rnaseq_thermo/iJN1463_Glu_ExprThermoConstrainedFile.xml',
        'Cit': 'models/final_constrained_rnaseq_thermo/iJN1463_Cit_ExprThermoConstrainedFile.xml',
        'Ser': 'models/final_constrained_rnaseq_thermo/iJN1463_Ser_ExprThermoConstrainedFile.xml',
        'Fer': 'models/final_constrained_rnaseq_thermo/iJN1463_Fer_ExprThermoConstrainedFile.xml'
    }
    
    # Perform comprehensive engineering evaluation
    engineering_df = evaluate_multi_environment_engineering(
        base_model_paths,
        config['engineering_scenarios'],
        config['environments'],
        config['objectives']
    )
    
    # Calculate robustness metrics
    robustness_df = calculate_robustness_metrics(engineering_df)
    
    # Save results
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    engineering_df.to_csv(results_dir / "enhanced_engineering_results.csv", index=False)
    robustness_df.to_csv(results_dir / "engineering_robustness.csv", index=False)
    
    # Create and save visualizations
    print("\n📈 Creating engineering summary visualizations...")
    summary_fig = create_engineering_summary_visualization(engineering_df, robustness_df)
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_fig.savefig(figures_dir / "multi_environment_engineering_summary.png", 
                       dpi=300, bbox_inches='tight')
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("MULTI-ENVIRONMENT METABOLIC ENGINEERING SUMMARY")
    print("="*80)
    
    # Top performing scenarios
    production_data = engineering_df[engineering_df['objective'] == 'max_biomass']
    scenario_performance = production_data.groupby('scenario').agg({
        'ala_flux_net': ['mean', 'std', 'count'],
        'growth_rate': 'mean',
        'yield_mmol_g': 'mean',
        'modification_count': 'first'
    }).round(4)
    
    scenario_performance.columns = ['ala_mean', 'ala_std', 'env_count', 'growth_mean', 'yield_mean', 'mod_count']
    scenario_performance = scenario_performance.sort_values('ala_mean', ascending=False)
    
    print("\n🏆 Top 5 Engineering Scenarios (Average ALA Production):")
    print(scenario_performance.head(5))
    
    # Robustness analysis summary
    robust_summary = robustness_df.sort_values('robustness_score', ascending=False)
    print(f"\n🛡️ Top 5 Most Robust Scenarios:")
    for _, row in robust_summary.head(5).iterrows():
        print(f"  {row['scenario']:30} | Robustness: {row['robustness_score']:.3f} | "
              f"ALA: {row['ala_mean']:.4f} ± {row['ala_std']:.4f}")
    
    # Environment-specific recommendations
    print(f"\n🌍 Environment-Specific Recommendations:")
    for env in ['Glu', 'Cit', 'Ser', 'Fer']:
        env_data = production_data[production_data['environment'] == env]
        if len(env_data) > 0:
            best_env_scenario = env_data.loc[env_data['ala_flux_net'].idxmax()]
            print(f"  {env}: {best_env_scenario['scenario']} "
                  f"(ALA: {best_env_scenario['ala_flux_net']:.4f})")
    
    print(f"\n✅ Enhanced metabolic engineering analysis completed successfully!")

if __name__ == "__main__":
    main()
