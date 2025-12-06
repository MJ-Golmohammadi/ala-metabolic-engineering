"""
Flux Coupling Analysis for Metabolic Network Structure Investigation
Q1 Journal Quality - Identifies coupled reaction sets and pathway dependencies
"""

import cobra
import pandas as pd
import numpy as np
from cobra.flux_analysis import find_blocked_reactions, flux_variability_analysis
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

def perform_flux_coupling_analysis(model: cobra.Model, environment: dict, 
                                 reaction_subset: list = None) -> pd.DataFrame:
    """
    Perform flux coupling analysis to identify reaction dependencies
    """
    
    print("🔗 Performing flux coupling analysis...")
    
    with model:
        # Apply environment
        for rxn_id, bounds in environment.items():
            if rxn_id in model.reactions:
                model.reactions.get_by_id(rxn_id).bounds = bounds
        
        # Find blocked reactions
        blocked_reactions = find_blocked_reactions(model)
        print(f"   Found {len(blocked_reactions)} blocked reactions")
        
        # Define reactions to analyze
        if reaction_subset is None:
            # Focus on central metabolism and ALA-related reactions
            reaction_subset = [
                'G1SAT', 'GLUTRR', 'GLUTRS', 'PPBNGS',
                'ICDHyr', 'PPC', 'G6PDH2r', 'GLUDy', 'GLUSy'
            ]
        
        # Perform FVA for coupling analysis
        fva_result = flux_variability_analysis(model, reaction_list=reaction_subset, 
                                              fraction_of_optimum=0.9)
        
        # Identify coupled reaction pairs
        coupling_results = []
        
        for i, rxn1 in enumerate(reaction_subset):
            for j, rxn2 in enumerate(reaction_subset[i+1:], i+1):
                if rxn1 in fva_result.index and rxn2 in fva_result.index:
                    flux_min1, flux_max1 = fva_result.loc[rxn1, ['minimum', 'maximum']]
                    flux_min2, flux_max2 = fva_result.loc[rxn2, ['minimum', 'maximum']]
                    
                    flux_range1 = flux_max1 - flux_min1
                    flux_range2 = flux_max2 - flux_min2
                    
                    # Calculate coupling metrics
                    if flux_range1 > 0 and flux_range2 > 0:
                        # Flux correlation strength
                        coupling_strength = min(flux_range1, flux_range2) / max(flux_range1, flux_range2)
                        
                        # Direction coupling
                        if flux_min1 * flux_min2 >= 0 and flux_max1 * flux_max2 >= 0:
                            direction = 'same'
                        else:
                            direction = 'opposite'
                        
                        # Coupling type classification
                        if coupling_strength > 0.9:
                            coupling_type = 'strong'
                        elif coupling_strength > 0.6:
                            coupling_type = 'moderate'
                        else:
                            coupling_type = 'weak'
                        
                        coupling_results.append({
                            'reaction_1': rxn1,
                            'reaction_2': rxn2,
                            'coupling_strength': coupling_strength,
                            'coupling_type': coupling_type,
                            'direction': direction,
                            'flux_range_1': flux_range1,
                            'flux_range_2': flux_range2,
                            'reaction_1_min': flux_min1,
                            'reaction_1_max': flux_max1,
                            'reaction_2_min': flux_min2,
                            'reaction_2_max': flux_max2
                        })
    
    return pd.DataFrame(coupling_results)

def analyze_pathway_coupling(coupling_df: pd.DataFrame) -> dict:
    """
    Analyze pathway-level coupling patterns
    """
    
    pathway_groups = {
        'C5_Pathway': ['GLUTRS', 'GLUTRR', 'G1SAT'],
        'Precursor_Supply': ['GLUDy', 'GLUSy', 'ICDHyr', 'PPC'],
        'Competitive_Pathways': ['PPBNGS'],
        'Central_Metabolism': ['G6PDH2r', 'PDH', 'CS']
    }
    
    pathway_coupling = {}
    
    for pathway1, reactions1 in pathway_groups.items():
        for pathway2, reactions2 in pathway_groups.items():
            if pathway1 != pathway2:
                # Find couplings between pathways
                pathway_couplings = coupling_df[
                    (coupling_df['reaction_1'].isin(reactions1) & coupling_df['reaction_2'].isin(reactions2)) |
                    (coupling_df['reaction_1'].isin(reactions2) & coupling_df['reaction_2'].isin(reactions1))
                ]
                
                if len(pathway_couplings) > 0:
                    avg_strength = pathway_couplings['coupling_strength'].mean()
                    strong_couplings = len(pathway_couplings[pathway_couplings['coupling_type'] == 'strong'])
                    
                    pathway_coupling[f"{pathway1}_{pathway2}"] = {
                        'pathway_1': pathway1,
                        'pathway_2': pathway2,
                        'avg_coupling_strength': avg_strength,
                        'strong_couplings': strong_couplings,
                        'total_couplings': len(pathway_couplings)
                    }
    
    return pathway_coupling

def create_coupling_visualization(coupling_df: pd.DataFrame, environment: str):
    """
    Create flux coupling visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Coupling strength distribution
    coupling_types = coupling_df['coupling_type'].value_counts()
    ax1.pie(coupling_types.values, labels=coupling_types.index, autopct='%1.1f%%', 
            colors=['#ff9999', '#66b3ff', '#99ff99'])
    ax1.set_title(f'A. Coupling Type Distribution - {environment}', fontweight='bold', pad=20)
    
    # 2. Strong coupling network (simplified)
    strong_couplings = coupling_df[coupling_df['coupling_type'] == 'strong']
    
    if len(strong_couplings) > 0:
        # Create coupling matrix for heatmap
        reactions = list(set(strong_couplings['reaction_1'].tolist() + strong_couplings['reaction_2'].tolist()))
        coupling_matrix = pd.DataFrame(0.0, index=reactions, columns=reactions)
        
        for _, row in strong_couplings.iterrows():
            coupling_matrix.loc[row['reaction_1'], row['reaction_2']] = row['coupling_strength']
            coupling_matrix.loc[row['reaction_2'], row['reaction_1']] = row['coupling_strength']
        
        sns.heatmap(coupling_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('B. Strong Flux Coupling Network', fontweight='bold', pad=20)
    
    # 3. Coupling strength by reaction type
    coupling_df['reaction_pair'] = coupling_df['reaction_1'] + ' - ' + coupling_df['reaction_2']
    top_couplings = coupling_df.nlargest(10, 'coupling_strength')
    
    ax3.barh(range(len(top_couplings)), top_couplings['coupling_strength'])
    ax3.set_yticks(range(len(top_couplings)))
    ax3.set_yticklabels(top_couplings['reaction_pair'], fontsize=8)
    ax3.set_xlabel('Coupling Strength', fontweight='bold')
    ax3.set_title('C. Top 10 Strongest Reaction Couplings', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Direction analysis
    direction_counts = coupling_df['direction'].value_counts()
    ax4.bar(direction_counts.index, direction_counts.values, color=['#1f77b4', '#ff7f0e'])
    ax4.set_ylabel('Number of Couplings', fontweight='bold')
    ax4.set_title('D. Coupling Direction Analysis', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def main():
    """Main function for flux coupling analysis"""
    
    # Load configuration
    with open('config/engineering_scenarios.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Analyze all environments
    environments = ['Glu', 'Cit', 'Ser', 'Fer']
    all_coupling_results = []
    all_pathway_coupling = []
    
    for env in environments:
        print(f"\n🌍 Analyzing flux coupling in {env} environment...")
        
        try:
            # Load environment-specific model
            model_path = f"models/final_constrained_rnaseq_thermo/iJN1463_{env}_ExprThermoConstrainedFile.xml"
            model = cobra.io.read_sbml_model(model_path)
            
            # Perform coupling analysis
            coupling_df = perform_flux_coupling_analysis(model, config['environments'][env])
            coupling_df['environment'] = env
            
            all_coupling_results.append(coupling_df)
            
            # Analyze pathway coupling
            pathway_coupling = analyze_pathway_coupling(coupling_df)
            for key, value in pathway_coupling.items():
                value['environment'] = env
                all_pathway_coupling.append(value)
            
            print(f"✅ Found {len(coupling_df)} coupled reaction pairs in {env}")
            
            # Create environment-specific visualization
            coupling_fig = create_coupling_visualization(coupling_df, env)
            coupling_fig.savefig(f'results/figures/flux_coupling_{env}.png', 
                               dpi=300, bbox_inches='tight')
            plt.close(coupling_fig)
            
        except Exception as e:
            print(f"❌ Error analyzing {env}: {e}")
    
    # Combine and save results
    if all_coupling_results:
        combined_results = pd.concat(all_coupling_results, ignore_index=True)
        combined_results.to_csv("results/tables/flux_coupling_analysis.csv", index=False)
        
        pathway_coupling_df = pd.DataFrame(all_pathway_coupling)
        pathway_coupling_df.to_csv("results/tables/pathway_coupling_analysis.csv", index=False)
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print("FLUX COUPLING ANALYSIS SUMMARY")
        print("="*60)
        
        for env in environments:
            env_data = combined_results[combined_results['environment'] == env]
            strong_couplings = env_data[env_data['coupling_type'] == 'strong']
            moderate_couplings = env_data[env_data['coupling_type'] == 'moderate']
            
            print(f"\n{env} Environment:")
            print(f"  Total couplings: {len(env_data)}")
            print(f"  Strong couplings: {len(strong_couplings)}")
            print(f"  Moderate couplings: {len(moderate_couplings)}")
            
            if len(strong_couplings) > 0:
                strongest = strong_couplings.loc[strong_couplings['coupling_strength'].idxmax()]
                print(f"  Strongest coupling: {strongest['reaction_1']} ↔ {strongest['reaction_2']} "
                      f"({strongest['coupling_strength']:.3f})")
        
        # Pathway coupling summary
        print(f"\n🔄 Pathway-Level Coupling Insights:")
        pathway_summary = pathway_coupling_df.groupby(['pathway_1', 'pathway_2']).agg({
            'avg_coupling_strength': 'mean',
            'strong_couplings': 'sum'
        }).round(3)
        
        print(pathway_summary.sort_values('avg_coupling_strength', ascending=False).head(10))
        
        print(f"\n✅ Flux coupling analysis completed successfully!")

if __name__ == "__main__":
    main()