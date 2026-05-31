"""
Flux Variability Analysis (FVA) for identification of metabolic bottlenecks
in 5-ALA biosynthesis pathway. Identifies key flux constraints and potential
targets for further metabolic engineering.
"""

import cobra
import pandas as pd
import numpy as np
from cobra.flux_analysis import flux_variability_analysis
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model_utils import create_engineered_strain


def perform_flux_variability_analysis(model: cobra.Model, 
                                    target_reaction: str = 'G1SAT',
                                    fraction_of_optimum: float = 0.9) -> pd.DataFrame:
    """
    Perform Flux Variability Analysis to identify flux constraints
    
    Parameters:
    -----------
    model : cobra.Model
        Metabolic model for analysis
    target_reaction : str
        Primary target reaction for production
    fraction_of_optimum : float
        Fraction of optimal growth for FVA constraint
        
    Returns:
    --------
    pd.DataFrame
        FVA results with flux ranges for key reactions
    """
    
    # Define key reactions in 5-ALA biosynthesis and central metabolism
    # These reactions represent the core C5 ALA biosynthesis pathway,
    # growth objective, and essential environmental exchanges.
                                      
    key_reactions = [
        "GLUTRR",                # Glutamyl-tRNA reductase (HemA) – committed step of C5 pathway, NADPH-dependent
        "G1SAT",                 # Glutamate-1-semialdehyde aminotransferase (HemL) – converts GSA to ALA, PLP-dependent
        "GLUTRS",                # Glutamyl-tRNA synthetase (GltX) – initiates C5 pathway by charging tRNA^Glu
        "BIOMASS_KT2440_WT3",    # Biomass objective – growth benchmark for constraint-based analysis
        "EX_glc__D_e",           # Glucose exchange – representative carbon source uptake (based on the medium)
        "EX_cit_e",              # Citrate exchange – representative carbon source uptake
        "EX_ser__L_e",           # Serine exchange – representative carbon source uptake
        "EX_fer_e",              # Ferulate exchange – representative carbon source uptake
        "EX_nh4_e",              # Ammonium exchange – nitrogen source for glutamate synthesis
        "EX_o2_e",               # Oxygen exchange – essential for respiration and redox balance
        "EX_pi_e",               # Phosphate exchange – required for nucleotide and cofactor biosynthesis
        "EX_so4_e",              # Sulfate exchange – sulfur assimilation for amino acids/cofactors
        "EX_co2_e",              # CO2 exchange – secretion only, maintains realistic carbon balance
        "EX_h2o_e",              # H2O exchange – secretion only, ensures thermodynamic feasibility
        "EX_h_e"                 # Proton exchange – secretion only, maintains charge balance
    ]

    
    # Filter for reactions actually present in the model
    available_reactions = [rxn for rxn in key_reactions if rxn in model.reactions]
    
    print(f"Performing FVA on {len(available_reactions)} key reactions...")
    
    # Perform Flux Variability Analysis
    fva_results = flux_variability_analysis(
        model, 
        reaction_list=available_reactions,
        fraction_of_optimum=fraction_of_optimum
    )
    
    return fva_results


def identify_metabolic_bottlenecks(fva_results: pd.DataFrame, 
                                 threshold: float = 0.1) -> pd.DataFrame:
    """
    Identify metabolic bottlenecks based on flux variability
    
    Parameters:
    -----------
    fva_results : pd.DataFrame
        Results from flux variability analysis
    threshold : float
        Flux range threshold for bottleneck classification
        
    Returns:
    --------
    pd.DataFrame
        Identified bottlenecks with severity classification
    """
    
    bottlenecks = []
    
    for rxn_id, row in fva_results.iterrows():
        flux_min = row['minimum']
        flux_max = row['maximum']
        flux_range = flux_max - flux_min
        
        # Classify bottleneck severity
        if flux_range < threshold:
            if flux_range < 0.01:
                severity = 'High'
                bottleneck_type = 'Absolute constraint'
            elif flux_range < 0.05:
                severity = 'Medium'
                bottleneck_type = 'Significant constraint'
            else:
                severity = 'Low'
                bottleneck_type = 'Moderate constraint'
            
            bottlenecks.append({
                'reaction': rxn_id,
                'flux_min': flux_min,
                'flux_max': flux_max,
                'flux_range': flux_range,
                'bottleneck_severity': severity,
                'bottleneck_type': bottleneck_type,
                'potential_target': True if severity in ['High', 'Medium'] else False
            })
    
    return pd.DataFrame(bottlenecks)

def calculate_bottleneck_impact(model: cobra.Model, 
                              bottleneck_reactions: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate potential production improvement from relieving bottlenecks
    
    Parameters
    ----------
    model : cobra.Model
        Metabolic model
    bottleneck_reactions : pd.DataFrame
        Identified bottleneck reactions
        
    Returns
    -------
    pd.DataFrame
        Impact analysis of bottleneck relief
    """
    
    impact_results = []
    
    # Get baseline production
    with model:
        model.objective = 'G1SAT'
        baseline_solution = model.optimize()
        baseline_production = baseline_solution.fluxes.get('G1SAT', 0)
    
    for _, bottleneck in bottleneck_reactions.iterrows():
        if bottleneck.get('potential_target', False):
            rxn_id = bottleneck['reaction']
            
            with model:
                reaction = model.reactions.get_by_id(rxn_id)
                original_upper = reaction.upper_bound
                
                # Relax constraint
                reaction.upper_bound = original_upper * 2 if original_upper < 1e5 else original_upper
                
                # Re-optimize
                model.objective = 'G1SAT'
                improved_solution = model.optimize()
                improved_production = improved_solution.fluxes.get('G1SAT', 0)
                
                # Calculate improvement
                production_increase = improved_production - baseline_production
                percent_increase = (production_increase / baseline_production * 100) if baseline_production > 0 else 0
                
                impact_results.append({
                    'reaction': rxn_id,
                    'bottleneck_severity': bottleneck['bottleneck_severity'],
                    'original_upper_bound': original_upper,
                    'relaxed_upper_bound': reaction.upper_bound,
                    'baseline_production': baseline_production,
                    'improved_production': improved_production,
                    'production_increase': production_increase,
                    'percent_increase': percent_increase,
                    'potential_rank': percent_increase
                })
    
    impact_df = pd.DataFrame(impact_results, columns=[
        'reaction','bottleneck_severity','original_upper_bound',
        'relaxed_upper_bound','baseline_production','improved_production',
        'production_increase','percent_increase','potential_rank'
    ])
    
    if impact_df.empty:
        print("⚠️ No bottleneck impacts identified, returning empty DataFrame with predefined columns")
        return impact_df
    else:
        return impact_df.sort_values('percent_increase', ascending=False)


def main():
    """Enhanced main function for multi-model bottleneck identification"""
    
    # Load configuration
    with open('config/engineering_scenarios.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Define all environments to analyze
    environments = ['Glu', 'Cit', 'Ser', 'Fer']
    
    # Define engineering scenarios to analyze (all of them)
    engineering_scenarios = list(config['engineering_scenarios'].keys())
    
    print("🔍 Performing comprehensive bottleneck analysis across all environments and scenarios...")
    
    all_bottlenecks = {}
    all_impacts = {}
    
    for environment in environments:
        print(f"\n🌍 Analyzing environment: {environment}")
        
        try:
            # Load environment-specific constrained
            model_path = f"models/final_constrained_rnaseq_thermo/iJN1463_{environment}_preprocessed_with_DM.xml"
            base_model = cobra.io.read_sbml_model(model_path)
            
            # Analyze wild-type for this environment
            wt_model = base_model.copy()
            wt_scenario_name = f"wild_type_{environment}"
            
            print(f"  🧬 Wild-type analysis for {environment}...")
            fva_results_wt = perform_flux_variability_analysis(wt_model)
            bottlenecks_wt = identify_metabolic_bottlenecks(fva_results_wt)
            impacts_wt = calculate_bottleneck_impact(wt_model, bottlenecks_wt)
            
            all_bottlenecks[wt_scenario_name] = bottlenecks_wt
            all_impacts[wt_scenario_name] = impacts_wt
            
            # Analyze all engineering scenarios for this environment
            for scenario_name in engineering_scenarios:
                if scenario_name == 'wild_type':
                    continue  # Skip, already analyzed
                    
                print(f"  🔧 Engineering scenario: {scenario_name} in {environment}...")
                
                try:
                    # Create engineered strain from environment-specific model
                    engineered_model = create_engineered_strain(
                        base_model, 
                        config['engineering_scenarios'][scenario_name]
                    )
                    
                    scenario_full_name = f"{scenario_name}_{environment}"
                    
                    # Perform bottleneck analysis
                    fva_results = perform_flux_variability_analysis(engineered_model)
                    bottlenecks = identify_metabolic_bottlenecks(fva_results)
                    impacts = calculate_bottleneck_impact(engineered_model, bottlenecks)
                    
                    all_bottlenecks[scenario_full_name] = bottlenecks
                    all_impacts[scenario_full_name] = impacts
                    
                    print(f"    ✅ Identified {len(bottlenecks)} bottlenecks")
                    
                except Exception as e:
                    print(f"    ❌ Error analyzing {scenario_name} in {environment}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error processing environment {environment}: {e}")
            continue
    
    # Save all results
    print("\n💾 Saving bottleneck analysis results...")
    
    # Save bottlenecks
    if all_bottlenecks:
        bottlenecks_combined = pd.concat(
            [df.assign(scenario=name) for name, df in all_bottlenecks.items() if not df.empty]
        )
        bottlenecks_combined.to_csv("results/tables/comprehensive_bottlenecks.csv", index=False)
    
    # Save impacts  
    if all_impacts:
        impacts_combined = pd.concat(
            [df.assign(scenario=name) for name, df in all_impacts.items() if not df.empty]
        )
        impacts_combined.to_csv("results/tables/comprehensive_bottleneck_impacts.csv", index=False)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE BOTTLENECK ANALYSIS SUMMARY")
    print("="*80)
    
    for environment in environments:
        print(f"\n📊 {environment} Environment:")
        
        # Wild-type bottlenecks
        wt_key = f"wild_type_{environment}"
        if wt_key in all_bottlenecks:
            wt_bottlenecks = all_bottlenecks[wt_key]
            severe_wt = wt_bottlenecks[wt_bottlenecks['bottleneck_severity'].isin(['High', 'Medium'])]
            print(f"  🧬 Wild-type: {len(severe_wt)} severe bottlenecks")
        
        # Engineering scenario bottlenecks
        for scenario in engineering_scenarios:
            if scenario == 'wild_type':
                continue
                
            scenario_key = f"{scenario}_{environment}"
            if scenario_key in all_bottlenecks:
                scenario_bottlenecks = all_bottlenecks[scenario_key]
                severe_scenario = scenario_bottlenecks[scenario_bottlenecks['bottleneck_severity'].isin(['High', 'Medium'])]
                
                if len(severe_scenario) > 0:
                    # Compare with wild-type
                    if wt_key in all_bottlenecks:
                        wt_severe_count = len(all_bottlenecks[wt_key][all_bottlenecks[wt_key]['bottleneck_severity'].isin(['High', 'Medium'])])
                        improvement = "✅" if len(severe_scenario) < wt_severe_count else "⚠️"
                    else:
                        improvement = "🔍"
                    
                    print(f"  {improvement} {scenario}: {len(severe_scenario)} severe bottlenecks")
    
    print(f"\n✅ Comprehensive bottleneck analysis completed successfully!")
    print(f"📁 Results saved to:")
    print(f"   - results/tables/comprehensive_bottlenecks.csv")
    print(f"   - results/tables/comprehensive_bottleneck_impacts.csv")


if __name__ == "__main__":
    main()
