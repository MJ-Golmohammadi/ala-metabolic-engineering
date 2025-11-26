"""
Enhanced FBA Impact Scoring with Multi-Environment Support
Q1 Journal Quality - Comprehensive impact assessment across all environmental conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_comprehensive_impact_scores(engineering_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate comprehensive FBA impact scores across all environments
    with enhanced statistical analysis
    """
    
    environments = engineering_df['environment'].unique()
    impact_scores = []
    
    print("📊 Calculating multi-environment impact scores...")
    
    for env in environments:
        print(f"  Processing {env} environment...")
        
        # Get environment-specific wild-type baseline
        wt_data = engineering_df[
            (engineering_df['scenario'] == 'wild_type') & 
            (engineering_df['environment'] == env) &
            (engineering_df['objective'] == 'max_biomass')
        ]
        
        if len(wt_data) == 0:
            print(f"    ⚠️ No wild-type data for {env}")
            continue
            
        wt_ala = wt_data['ala_flux_net'].values[0]
        wt_growth = wt_data['growth_rate'].values[0]
        wt_yield = wt_data['yield_mmol_g'].values[0]
        
        print(f"    Wild-type baseline - ALA: {wt_ala:.4f}, Growth: {wt_growth:.4f}")
        
        # Calculate scores for all scenarios in this environment
        scenarios = engineering_df[engineering_df['environment'] == env]['scenario'].unique()
        
        for scenario in scenarios:
            if scenario == 'wild_type':
                continue
                
            scenario_data = engineering_df[
                (engineering_df['scenario'] == scenario) & 
                (engineering_df['environment'] == env) &
                (engineering_df['objective'] == 'max_biomass')
            ]
            
            if len(scenario_data) == 0:
                continue
                
            # Extract performance metrics
            ala_flux = scenario_data['ala_flux_net'].values[0]
            growth_flux = scenario_data['growth_rate'].values[0]
            yield_value = scenario_data['yield_mmol_g'].values[0]
            modification_count = scenario_data['modification_count'].values[0]
            
            # Calculate Δflux and improvements
            delta_flux_ala = ala_flux - wt_ala
            delta_flux_ala_percent = (delta_flux_ala / wt_ala * 100) if wt_ala > 0 else 0
            
            # Multi-criteria scoring with enhanced metrics
            ala_improvement = ala_flux / wt_ala if wt_ala > 0 else 0
            growth_maintenance = growth_flux / wt_growth if wt_growth > 0 else 0
            yield_improvement = yield_value / wt_yield if wt_yield > 0 else 0
            
            # Normalized impact scores (0-1) with saturation
            ala_impact = min(1.0, max(0, (ala_flux - wt_ala) / max(wt_ala * 2, 0.1)))
            growth_impact = min(1.0, growth_flux / max(wt_growth, 0.1))
            yield_impact = min(1.0, max(0, (yield_value - wt_yield) / max(wt_yield * 2, 0.01)))
            
            # Engineering efficiency penalty for complexity
            complexity_penalty = max(0, 1 - (modification_count / 15))  # Penalize over-engineering
            
            # Combined FBA impact score with complexity adjustment
            fba_score = (0.6 * ala_impact + 0.25 * growth_impact + 0.15 * yield_impact) * complexity_penalty
            
            impact_scores.append({
                'environment': env,
                'scenario': scenario,
                'fba_impact_score': round(fba_score, 4),
                'delta_flux_ala': round(delta_flux_ala, 4),
                'delta_flux_ala_percent': round(delta_flux_ala_percent, 1),
                'ala_flux_absolute': round(ala_flux, 4),
                'growth_rate_absolute': round(growth_flux, 4),
                'yield_absolute': round(yield_value, 4),
                'ala_improvement_ratio': round(ala_improvement, 4),
                'growth_maintenance_ratio': round(growth_maintenance, 4),
                'yield_improvement_ratio': round(yield_improvement, 4),
                'modification_count': modification_count,
                'engineering_efficiency': round(complexity_penalty, 4),
                'wt_ala_flux': round(wt_ala, 4),
                'wt_growth_rate': round(wt_growth, 4),
                'wt_yield': round(wt_yield, 4)
            })
    
    return pd.DataFrame(impact_scores)

def create_impact_visualization(impact_scores_df: pd.DataFrame):
    """
    Create comprehensive impact score visualization
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Impact scores by environment
    env_scores = impact_scores_df.groupby('environment')['fba_impact_score'].mean().sort_values(ascending=False)
    bars = ax1.bar(env_scores.index, env_scores.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Average FBA Impact Score', fontweight='bold')
    ax1.set_title('A. Average Impact Score by Environment', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. ΔALA flux across environments
    scenario_avg_delta = impact_scores_df.groupby('scenario')['delta_flux_ala'].mean().nlargest(10)
    ax2.barh(range(len(scenario_avg_delta)), scenario_avg_delta.values)
    ax2.set_yticks(range(len(scenario_avg_delta)))
    ax2.set_yticklabels(scenario_avg_delta.index, fontsize=9)
    ax2.set_xlabel('ΔALA Flux (mmol/gDW/h)', fontweight='bold')
    ax2.set_title('B. Top 10 Scenarios by Average ΔALA Flux', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Improvement ratios heatmap
    pivot_data = impact_scores_df.pivot_table(
        index='scenario', 
        columns='environment', 
        values='ala_improvement_ratio',
        aggfunc='mean'
    ).fillna(0)
    
    # Select top scenarios for clarity
    top_scenarios = impact_scores_df.groupby('scenario')['ala_improvement_ratio'].mean().nlargest(15).index
    plot_data = pivot_data.loc[top_scenarios]
    
    sns.heatmap(plot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=1, ax=ax3)
    ax3.set_title('C. ALA Improvement Ratio Heatmap', fontweight='bold', pad=20)
    
    # 4. Modification count vs impact
    scatter = ax4.scatter(impact_scores_df['modification_count'], 
                         impact_scores_df['fba_impact_score'],
                         c=impact_scores_df['ala_improvement_ratio'], 
                         cmap='viridis', alpha=0.7, s=60)
    ax4.set_xlabel('Number of Modifications', fontweight='bold')
    ax4.set_ylabel('FBA Impact Score', fontweight='bold')
    ax4.set_title('D. Engineering Complexity vs Impact', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='ALA Improvement Ratio')
    
    plt.tight_layout()
    return fig

def main():
    """Enhanced main function for multi-environment impact scoring"""
    try:
        # Load enhanced engineering results
        engineering_df = pd.read_csv("results/tables/enhanced_engineering_results.csv")
        
        print("🚀 Calculating comprehensive FBA impact scores across all environments...")
        impact_scores_df = calculate_comprehensive_impact_scores(engineering_df)
        
        # Save results
        impact_scores_df.to_csv("results/tables/enhanced_fba_impact_scores.csv", index=False)
        
        # Create visualization
        print("📈 Creating impact score visualizations...")
        impact_fig = create_impact_visualization(impact_scores_df)
        impact_fig.savefig('results/figures/multi_environment_impact_scores.png', 
                          dpi=300, bbox_inches='tight')
        
        # Print comprehensive summary
        print("\n" + "="*80)
        print("COMPREHENSIVE FBA IMPACT SCORE SUMMARY")
        print("="*80)
        
        # Top performers by environment
        for env in ['Glu', 'Cit', 'Ser', 'Fer']:
            env_scores = impact_scores_df[impact_scores_df['environment'] == env]
            if len(env_scores) > 0:
                best_scenario = env_scores.loc[env_scores['fba_impact_score'].idxmax()]
                worst_scenario = env_scores.loc[env_scores['fba_impact_score'].idxmin()]
                
                print(f"\n🏆 {env} Environment:")
                print(f"  Best: {best_scenario['scenario']:25} | Score: {best_scenario['fba_impact_score']:.3f}")
                print(f"        ΔALA: {best_scenario['delta_flux_ala']:+.4f} ({best_scenario['delta_flux_ala_percent']:+.1f}%)")
                print(f"  Worst: {worst_scenario['scenario']:24} | Score: {worst_scenario['fba_impact_score']:.3f}")
        
        # Overall top performers
        overall_best = impact_scores_df.groupby('scenario')['fba_impact_score'].mean().nlargest(5)
        print(f"\n🎯 Overall Top 5 Scenarios:")
        for scenario, score in overall_best.items():
            avg_delta = impact_scores_df[impact_scores_df['scenario'] == scenario]['delta_flux_ala'].mean()
            print(f"  {scenario:30} | Avg Score: {score:.3f} | Avg ΔALA: {avg_delta:+.4f}")
            
        print(f"\n✅ Enhanced FBA impact scoring completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in enhanced impact scoring: {e}")
        raise

if __name__ == "__main__":
    main()