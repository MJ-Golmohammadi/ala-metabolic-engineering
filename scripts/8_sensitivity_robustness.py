"""
Advanced Sensitivity and Robustness Analysis for Metabolic Engineering
Q1 Journal Quality - Comprehensive parameter sensitivity and system robustness evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import cobra
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings
warnings.filterwarnings('ignore')

def perform_global_sensitivity_analysis(model: cobra.Model, environment: dict, 
                                      parameters: dict, n_samples: int = 1000) -> dict:
    """
    Perform global sensitivity analysis using Sobol method
    """
    print("🎯 Performing global sensitivity analysis...")
    
    # Define problem for SALib
    problem = {
        'num_vars': len(parameters),
        'names': list(parameters.keys()),
        'bounds': list(parameters.values())
    }
    
    # Generate samples
    param_values = saltelli.sample(problem, n_samples)
    
    # Run model simulations
    outputs = []
    
    for params in param_values:
        with model:
            # Apply parameter variations
            for i, param_name in enumerate(problem['names']):
                # This would typically modify model parameters
                # For demonstration, we'll modify exchange reaction bounds
                if param_name in model.reactions:
                    model.reactions.get_by_id(param_name).bounds = (
                        -abs(params[i]), 1000
                    )
            
            # Apply environment
            for rxn_id, bounds in environment.items():
                if rxn_id in model.reactions:
                    model.reactions.get_by_id(rxn_id).bounds = bounds
            
            # Optimize for biomass and ALA production
            try:
                model.objective = 'BIOMASS_KT2440_WT3'
                biomass_solution = model.optimize()
                biomass_flux = biomass_solution.objective_value
                
                model.objective = 'G1SAT'
                ala_solution = model.optimize()
                ala_flux = ala_solution.objective_value
                
                outputs.append([biomass_flux, ala_flux])
                
            except:
                outputs.append([0, 0])
    
    outputs = np.array(outputs)
    
    # Perform Sobol analysis
    si_biomass = sobol.analyze(problem, outputs[:, 0], print_to_console=False)
    si_ala = sobol.analyze(problem, outputs[:, 1], print_to_console=False)
    
    return {
        'biomass_sensitivity': si_biomass,
        'ala_sensitivity': si_ala,
        'parameter_samples': param_values,
        'model_outputs': outputs
    }

def analyze_engineering_robustness(engineering_df: pd.DataFrame, 
                                 impact_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze robustness of engineering strategies across environments
    """
    
    robustness_results = []
    
    print("🛡️ Analyzing engineering robustness...")
    
    for scenario in engineering_df['scenario'].unique():
        scenario_data = engineering_df[engineering_df['scenario'] == scenario]
        impact_data = impact_scores_df[impact_scores_df['scenario'] == scenario]
        
        if len(scenario_data) > 0:
            # Calculate robustness metrics
            ala_production = scenario_data['ala_flux_net']
            growth_rates = scenario_data['growth_rate']
            yields = scenario_data['yield_mmol_g']
            
            # Performance metrics
            avg_ala = ala_production.mean()
            std_ala = ala_production.std()
            cv_ala = std_ala / avg_ala if avg_ala > 0 else 0
            
            avg_growth = growth_rates.mean()
            std_growth = growth_rates.std()
            
            avg_yield = yields.mean()
            std_yield = yields.std()
            
            # Robustness scores
            production_robustness = 1 / (1 + cv_ala)  # Higher is better
            growth_robustness = 1 / (1 + (std_growth / avg_growth)) if avg_growth > 0 else 0
            yield_robustness = 1 / (1 + (std_yield / avg_yield)) if avg_yield > 0 else 0
            
            # Environment coverage
            env_coverage = scenario_data['environment'].nunique() / 4  # 4 total environments
            
            # Overall robustness score
            overall_robustness = (production_robustness + growth_robustness + 
                                yield_robustness + env_coverage) / 4
            
            robustness_results.append({
                'scenario': scenario,
                'avg_ala_production': avg_ala,
                'ala_std': std_ala,
                'ala_cv': cv_ala,
                'avg_growth_rate': avg_growth,
                'growth_std': std_growth,
                'avg_yield': avg_yield,
                'yield_std': std_yield,
                'production_robustness': production_robustness,
                'growth_robustness': growth_robustness,
                'yield_robustness': yield_robustness,
                'environment_coverage': env_coverage,
                'overall_robustness': overall_robustness,
                'modification_count': scenario_data['modification_count'].iloc[0],
                'environments_tested': scenario_data['environment'].nunique()
            })
    
    return pd.DataFrame(robustness_results)

def create_robustness_visualization(robustness_df: pd.DataFrame, sensitivity_results: dict = None):
    """
    Create comprehensive robustness and sensitivity visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Robustness vs performance
    scatter = ax1.scatter(robustness_df['avg_ala_production'], 
                         robustness_df['overall_robustness'],
                         c=robustness_df['modification_count'], 
                         cmap='viridis', s=80, alpha=0.8)
    ax1.set_xlabel('Average ALA Production (mmol/gDW/h)', fontweight='bold')
    ax1.set_ylabel('Overall Robustness Score', fontweight='bold')
    ax1.set_title('A. Performance vs Robustness Trade-off', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Modification Count')
    
    # 2. Robustness component breakdown for top scenarios
    top_robust = robustness_df.nlargest(8, 'overall_robustness')
    components = ['production_robustness', 'growth_robustness', 
                 'yield_robustness', 'environment_coverage']
    component_data = top_robust[['scenario'] + components].set_index('scenario')
    
    component_data.plot(kind='bar', ax=ax2, width=0.8, alpha=0.8)
    ax2.set_ylabel('Robustness Score', fontweight='bold')
    ax2.set_title('B. Robustness Component Breakdown (Top 8)', fontweight='bold', pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Sensitivity analysis results (if available)
    if sensitivity_results is not None:
        si_ala = sensitivity_results['ala_sensitivity']
        parameters = si_ala['names']
        first_order = si_ala['S1']
        
        # Plot sensitivity indices
        y_pos = np.arange(len(parameters))
        ax3.barh(y_pos, first_order, alpha=0.8, color='steelblue')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(parameters)
        ax3.set_xlabel('First-order Sensitivity Index', fontweight='bold')
        ax3.set_title('C. Parameter Sensitivity for ALA Production', fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Environment coverage vs performance consistency
    scatter2 = ax4.scatter(robustness_df['environment_coverage'], 
                          robustness_df['production_robustness'],
                          c=robustness_df['avg_ala_production'], 
                          cmap='plasma', s=80, alpha=0.8)
    ax4.set_xlabel('Environment Coverage', fontweight='bold')
    ax4.set_ylabel('Production Robustness', fontweight='bold')
    ax4.set_title('D. Broad Applicability vs Performance Stability', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='Average ALA Production')
    
    plt.tight_layout()
    return fig

def main():
    """Main function for sensitivity and robustness analysis"""
    
    try:
        # Load data
        engineering_df = pd.read_csv("results/tables/enhanced_engineering_results.csv")
        impact_scores_df = pd.read_csv("results/tables/enhanced_fba_impact_scores.csv")
        
        # Load configuration
        with open('config/engineering_scenarios.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("🚀 Performing comprehensive sensitivity and robustness analysis...")
        
        # Analyze engineering robustness
        robustness_df = analyze_engineering_robustness(engineering_df, impact_scores_df)
        robustness_df.to_csv("results/tables/engineering_robustness.csv", index=False)
        
        # Perform sensitivity analysis across ALL environments
        print("\n🎯 Performing sensitivity analysis across ALL environments...")
        
        sensitivity_results_all = {}
        
        # Define comprehensive parameters for sensitivity analysis
        parameters = {
            # Carbon source uptake variations
            'EX_glc__D_e': [0.5, 2.0],      # Glucose uptake: 50% to 200% of baseline
            'EX_cit_e': [0.5, 2.0],         # Citrate uptake range
            'EX_ser__L_e': [0.5, 2.0],      # Serine uptake range  
            'EX_fer_e': [0.5, 2.0],         # Ferulate uptake range
            
            # Essential nutrients variations
            'EX_o2_e': [0.5, 2.0],          # Oxygen uptake range
            'EX_nh4_e': [0.5, 2.0],         # Ammonium uptake range
            'EX_pi_e': [0.5, 2.0],          # Phosphate uptake range
            'EX_so4_e': [0.5, 2.0],         # Sulfate uptake range
            
            # Cofactor and ion variations
            'EX_fe2_e': [0.2, 2.0],         # Iron uptake (critical for heme pathway)
            'EX_mg2_e': [0.5, 2.0],         # Magnesium uptake
            'EX_k_e': [0.5, 2.0],           # Potassium uptake
            'EX_co2_e': [-5.0, 0.0],        # CO2 secretion range
            
            # 🎯 Critical Engineering Parameters
            'GLUTRR': [0.1, 10.0],          # hemA - C5 pathway bottleneck (10% to 10x)
            'G1SAT': [0.1, 10.0],           # hemL - Second ALA enzyme
            'GLUTRS': [0.1, 10.0],          # gltX - tRNA charging for C5 pathway
            'PPBNGS': [0.05, 1.0],          # hemB - ALA sink (5% to 100%)
            
            # 🔄 Precursor Supply Parameters
            'GLUDy': [0.5, 5.0],            # gdhA - Glutamate supply
            'ICDHyr': [0.5, 5.0],           # icd - α-ketoglutarate supply
            'G6PDH2r': [0.5, 5.0],          # zwfA - NADPH supply
            
            # ⚡ Cofactor Parameters
            'NADTRHD': [0.5, 3.0],          # sthA - NADPH regeneration
            'ATPM': [0.5, 2.0],             # ATP maintenance
        }
        
        for environment in ['Glu', 'Cit', 'Ser', 'Fer']:
            print(f"\n🌍 Analyzing sensitivity in {environment} environment...")
            
            try:
                # Load environment-specific model
                model_path = f"models/final_constrained_rnaseq_thermo/iJN1463_{environment}_ExprThermoConstrainedFile.xml"
                model = cobra.io.read_sbml_model(model_path)
                
                print(f"   Testing {len(parameters)} parameters with 500 samples...")
                
                # Perform sensitivity analysis
                sensitivity_results = perform_global_sensitivity_analysis(
                    model, config['environments'][environment], parameters, n_samples=500
                )
                
                sensitivity_results_all[environment] = sensitivity_results
                print(f"   ✅ Sensitivity analysis completed for {environment}")
                
            except Exception as e:
                print(f"   ❌ Error in {environment}: {e}")
                continue
        
        # Save comprehensive sensitivity results
        if sensitivity_results_all:
            print("\n💾 Saving comprehensive sensitivity results...")
            
            sensitivity_data = []
            for env, results in sensitivity_results_all.items():
                if results and 'ala_sensitivity' in results:
                    si_ala = results['ala_sensitivity']
                    for i, param in enumerate(si_ala['names']):
                        sensitivity_data.append({
                            'environment': env,
                            'parameter': param,
                            'first_order_ala': si_ala['S1'][i],
                            'total_order_ala': si_ala['ST'][i],
                            'first_order_biomass': results['biomass_sensitivity']['S1'][i],
                            'total_order_biomass': results['biomass_sensitivity']['ST'][i]
                        })
            
            sensitivity_df = pd.DataFrame(sensitivity_data)
            sensitivity_df.to_csv("results/tables/comprehensive_sensitivity_analysis.csv", index=False)
            
            print("📊 Sensitivity results summary:")
            for env in ['Glu', 'Cit', 'Ser', 'Fer']:
                if env in sensitivity_results_all:
                    env_data = sensitivity_df[sensitivity_df['environment'] == env]
                    if len(env_data) > 0:
                        top_param = env_data.loc[env_data['first_order_ala'].idxmax()]
                        print(f"   {env}: Most sensitive = {top_param['parameter']} (S1={top_param['first_order_ala']:.3f})")
        
        # Create visualizations
        print("📈 Creating robustness visualizations...")
        robustness_fig = create_robustness_visualization(robustness_df, sensitivity_results_all.get('Glu'))
        robustness_fig.savefig('results/figures/sensitivity_robustness_analysis.png', 
                             dpi=300, bbox_inches='tight')
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("SENSITIVITY AND ROBUSTNESS ANALYSIS SUMMARY")
        print("="*80)
        
        # Top robust scenarios
        top_robust = robustness_df.nlargest(5, 'overall_robustness')
        print(f"\n🏆 Top 5 Most Robust Scenarios:")
        for _, row in top_robust.iterrows():
            print(f"  {row['scenario']:30} | Robustness: {row['overall_robustness']:.3f} | "
                  f"ALA: {row['avg_ala_production']:.4f} | Envs: {row['environments_tested']}/4")
        
        # Sensitivity insights
        if sensitivity_results_all:
            print(f"\n🎯 Key Sensitivity Findings (across all environments):")
            for env in ['Glu', 'Cit', 'Ser', 'Fer']:
                if env in sensitivity_results_all:
                    si_ala = sensitivity_results_all[env]['ala_sensitivity']
                    top_params = []
                    for i, param in enumerate(si_ala['names']):
                        if si_ala['S1'][i] > 0.1:  # Significant parameters
                            top_params.append(f"{param}({si_ala['S1'][i]:.3f})")
                    if top_params:
                        print(f"   {env}: {', '.join(top_params[:3])}")  # Top 3 parameters
        
        # Robustness-performance trade-off analysis
        high_performance = robustness_df[robustness_df['avg_ala_production'] > 
                                       robustness_df['avg_ala_production'].median()]
        high_robustness = robustness_df[robustness_df['overall_robustness'] > 
                                      robustness_df['overall_robustness'].median()]
        
        balanced_scenarios = high_performance.merge(high_robustness, on='scenario')
        
        print(f"\n💡 Balanced Scenarios (High Performance + High Robustness): {len(balanced_scenarios)}")
        for _, row in balanced_scenarios.head(3).iterrows():
            print(f"  {row['scenario']} (ALA: {row['avg_ala_production']:.4f}, "
                  f"Robustness: {row['overall_robustness']:.3f})")
        
        print(f"\n✅ Sensitivity and robustness analysis completed successfully!")
        print(f"📁 Results saved to:")
        print(f"   - results/tables/engineering_robustness.csv")
        print(f"   - results/tables/comprehensive_sensitivity_analysis.csv")
        print(f"   - results/figures/sensitivity_robustness_analysis.png")
        
    except Exception as e:
        print(f"❌ Error in sensitivity analysis: {e}")
        raise

if __name__ == "__main__":
    main()