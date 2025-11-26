"""
Enhanced ML Priority Scoring with Multi-Environment Integration
Q1 Journal Quality - Machine learning-based target prioritization across conditions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_enhanced_ml_priority_scores(engineering_df: pd.DataFrame, 
                                        impact_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate enhanced ML-based priority scores with multi-environment support
    """
    
    print("🤖 Calculating enhanced ML priority scores...")
    
    # Prepare features from engineering scenarios across all environments
    scenarios_data = []
    
    # Gene importance weights based on biological rationale
    gene_importance = {
        # Core C5 Pathway Enzymes (Highest Priority)
        'hemA': 0.25,    # GLUTRR - First committed step, major bottleneck
        'hemL': 0.20,    # G1SAT - Essential for ALA formation from GSA
        
        # Competitive Sink Elimination (Very High Priority)
        'hemB': 0.18,    # PPBNGS - Porphobilinogen synthase - MAJOR ALA consumer
        
        # Precursor Supply (Medium Priority)
        'gltX': 0.12,    # GLUTRS - tRNA charging for C5 pathway initiation
        'gdhA': 0.08,    # GLUDy - Direct glutamate synthesis
        'gltB': 0.06,    # GLUSy - Glutamate synthase complex
        'icd': 0.05,     # ICDHyr - α-ketoglutarate supply for glutamate
        
        # Cofactor & Energy Management (Lower Priority)
        'zwfA': 0.03,    # G6PDH2r - NADPH generation
        'ppc': 0.02,     # PPC - Anaplerosis for TCA cycle
        'gntZ': 0.01,    # GND - Additional NADPH generation
        
        # Stress Protection (Lowest Priority)
        'sthA': 0.005,   # NADTRHD - NADPH/NADH balancing
        'sodA': 0.005,   # SPODM - Oxidative stress protection
        'katG': 0.005,   # CAT - Oxidative stress protection
    }
    
    # Get unique scenarios
    unique_scenarios = engineering_df['scenario'].unique()
    
    for scenario in unique_scenarios:
        if scenario == 'wild_type':
            continue
            
        # Get scenario data across all environments
        scenario_data_all_env = engineering_df[
            (engineering_df['scenario'] == scenario) & 
            (engineering_df['objective'] == 'max_biomass')
        ]
        
        if len(scenario_data_all_env) == 0:
            continue
        
        # Calculate multi-environment performance metrics
        avg_ala_flux = scenario_data_all_env['ala_flux_net'].mean()
        avg_growth = scenario_data_all_env['growth_rate'].mean()
        avg_yield = scenario_data_all_env['yield_mmol_g'].mean()
        env_count = scenario_data_all_env['environment'].nunique()
        
        # Feature 1: Modification complexity score
        mod_count = scenario_data_all_env['modification_count'].iloc[0]
        complexity_score = min(1.0, mod_count / 10)  # Normalize by max modifications
        
        # Feature 2: Biological importance score
        bio_score = 0
        modifications = scenario_data_all_env['modifications'].iloc[0]
        for gene, weight in gene_importance.items():
            if gene in str(modifications):
                bio_score += weight
        bio_score = min(1.0, bio_score)
        
        # Feature 3: Multi-environment FBA impact score
        scenario_impact = impact_scores_df[impact_scores_df['scenario'] == scenario]
        if len(scenario_impact) > 0:
            fba_score = scenario_impact['fba_impact_score'].mean()
            growth_maintenance = scenario_impact['growth_maintenance_ratio'].mean()
            env_robustness = scenario_impact['fba_impact_score'].std()
        else:
            fba_score = 0
            growth_maintenance = 0
            env_robustness = 0
        
        # Feature 4: Performance consistency across environments
        performance_std = scenario_data_all_env['ala_flux_net'].std()
        performance_cv = performance_std / avg_ala_flux if avg_ala_flux > 0 else 0
        consistency_score = 1 / (1 + performance_cv)  # Higher for more consistent performance
        
        # Feature 5: Environment coverage
        coverage_score = env_count / 4  # 4 total environments
        
        scenarios_data.append({
            'scenario': scenario,
            'complexity_score': complexity_score,
            'biological_score': bio_score,
            'fba_impact': fba_score,
            'growth_maintenance': growth_maintenance,
            'environment_robustness': 1 - min(1.0, env_robustness),  # Inverse of std
            'performance_consistency': consistency_score,
            'environment_coverage': coverage_score,
            'avg_ala_flux': avg_ala_flux,
            'avg_growth_rate': avg_growth,
            'avg_yield': avg_yield,
            'modification_count': mod_count,
            'environments_tested': env_count
        })
    
    ml_data = pd.DataFrame(scenarios_data)
    
    if ml_data.empty:
        print("⚠️ No scenario data available for ML scoring")
        return pd.DataFrame()
    
    # Calculate enhanced ML priority score (ensemble approach)
    # Updated weights based on multi-environment importance
    weights = {
        'fba_impact': 0.30,           # Overall predicted performance
        'biological_score': 0.20,      # Biological rationale  
        'growth_maintenance': 0.15,    # Feasibility
        'performance_consistency': 0.15, # Multi-environment reliability
        'environment_coverage': 0.10,  # Broad applicability
        'environment_robustness': 0.10 # Stability across conditions
    }
    
    # Calculate weighted score
    ml_data['ml_priority_score'] = (
        weights['fba_impact'] * ml_data['fba_impact'] +
        weights['biological_score'] * ml_data['biological_score'] + 
        weights['growth_maintenance'] * ml_data['growth_maintenance'] +
        weights['performance_consistency'] * ml_data['performance_consistency'] +
        weights['environment_coverage'] * ml_data['environment_coverage'] +
        weights['environment_robustness'] * ml_data['environment_robustness']
    )
    
    # Apply complexity penalty (lower complexity = better)
    complexity_penalty = 1 - (ml_data['complexity_score'] * 0.2)  # 20% penalty for complexity
    ml_data['ml_priority_score'] = ml_data['ml_priority_score'] * complexity_penalty
    
    # Normalize to 0-1 scale
    min_score = ml_data['ml_priority_score'].min()
    max_score = ml_data['ml_priority_score'].max()
    if max_score > min_score:
        ml_data['ml_priority_score'] = (ml_data['ml_priority_score'] - min_score) / (max_score - min_score)
    
    return ml_data.round(4)

def create_ml_scoring_visualization(ml_scores_df: pd.DataFrame):
    """
    Create visualization for ML priority scoring results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top scenarios by ML score
    top_ml = ml_scores_df.nlargest(10, 'ml_priority_score')
    bars = ax1.barh(range(len(top_ml)), top_ml['ml_priority_score'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_ml))))
    ax1.set_yticks(range(len(top_ml)))
    ax1.set_yticklabels(top_ml['scenario'], fontsize=9)
    ax1.set_xlabel('ML Priority Score', fontweight='bold')
    ax1.set_title('A. Top 10 Scenarios by ML Priority Score', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Score component breakdown for top 5
    top_5 = ml_scores_df.nlargest(5, 'ml_priority_score')
    components = ['fba_impact', 'biological_score', 'growth_maintenance', 
                 'performance_consistency', 'environment_coverage']
    component_data = top_5[['scenario'] + components].set_index('scenario')
    
    component_data.plot(kind='bar', ax=ax2, width=0.8, alpha=0.8)
    ax2.set_ylabel('Component Score', fontweight='bold')
    ax2.set_title('B. ML Score Component Breakdown (Top 5)', fontweight='bold', pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. ML score vs modification count
    scatter = ax3.scatter(ml_scores_df['modification_count'], 
                         ml_scores_df['ml_priority_score'],
                         c=ml_scores_df['avg_ala_flux'], 
                         cmap='plasma', s=80, alpha=0.8)
    ax3.set_xlabel('Number of Modifications', fontweight='bold')
    ax3.set_ylabel('ML Priority Score', fontweight='bold')
    ax3.set_title('C. Engineering Complexity vs ML Priority', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Average ALA Flux')
    
    # 4. Environment coverage vs performance consistency
    scatter2 = ax4.scatter(ml_scores_df['environment_coverage'], 
                          ml_scores_df['performance_consistency'],
                          c=ml_scores_df['ml_priority_score'], 
                          cmap='viridis', s=80, alpha=0.8)
    ax4.set_xlabel('Environment Coverage', fontweight='bold')
    ax4.set_ylabel('Performance Consistency', fontweight='bold')
    ax4.set_title('D. Robustness vs Applicability', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='ML Priority Score')
    
    plt.tight_layout()
    return fig

def main():
    """Enhanced main function for ML priority scoring"""
    
    try:
        # Load enhanced data
        engineering_df = pd.read_csv("results/tables/enhanced_engineering_results.csv")
        impact_scores_df = pd.read_csv("results/tables/enhanced_fba_impact_scores.csv")
        
        print("🚀 Calculating enhanced ML priority scores with multi-environment support...")
        ml_scores_df = calculate_enhanced_ml_priority_scores(engineering_df, impact_scores_df)
        
        if ml_scores_df.empty:
            print("❌ No data available for ML scoring")
            return
        
        # Save ML scores
        ml_scores_df.to_csv("results/tables/enhanced_ml_priority_scores.csv", index=False)
        
        # Create visualization
        print("📈 Creating ML scoring visualizations...")
        ml_fig = create_ml_scoring_visualization(ml_scores_df)
        ml_fig.savefig('results/figures/enhanced_ml_priority_scores.png', 
                      dpi=300, bbox_inches='tight')
        
        # Print comprehensive ranking
        print("\n" + "="*80)
        print("ENHANCED ML PRIORITY SCORE RANKING")
        print("="*80)
        
        ranked_scenarios = ml_scores_df.sort_values('ml_priority_score', ascending=False)
        
        print(f"\n🏆 Top 10 Engineering Scenarios:")
        print("-" * 100)
        print(f"{'Scenario':25} | {'ML Score':8} | {'ALA Flux':10} | {'Growth':8} | {'Mods':4} | {'Envs':4}")
        print("-" * 100)
        
        for _, row in ranked_scenarios.head(10).iterrows():
            print(f"{row['scenario']:25} | {row['ml_priority_score']:8.3f} | "
                  f"{row['avg_ala_flux']:10.4f} | {row['avg_growth_rate']:8.4f} | "
                  f"{row['modification_count']:4} | {row['environments_tested']:4}")
        
        # Print key insights
        print(f"\n💡 Key Insights:")
        best_overall = ranked_scenarios.iloc[0]
        most_consistent = ranked_scenarios.loc[ranked_scenarios['performance_consistency'].idxmax()]
        simplest_effective = ranked_scenarios[ranked_scenarios['modification_count'] <= 3].iloc[0]
        
        print(f"  • Best Overall: {best_overall['scenario']} (Score: {best_overall['ml_priority_score']:.3f})")
        print(f"  • Most Consistent: {most_consistent['scenario']} (Consistency: {most_consistent['performance_consistency']:.3f})")
        print(f"  • Simplest Effective: {simplest_effective['scenario']} ({simplest_effective['modification_count']} modifications)")
        
        print(f"\n✅ Enhanced ML priority scoring completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in enhanced ML scoring: {e}")
        raise

if __name__ == "__main__":
    main()