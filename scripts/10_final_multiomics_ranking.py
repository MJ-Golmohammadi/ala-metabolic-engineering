# scripts/10_final_multiomics_ranking.py
"""
Final Multi-omics Ranking for Metabolic Engineering
Q1 Journal Quality - Integrates FBA, ML, and Expression data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_multiomics_ranking():
    """Create final ranking using all data sources"""
    
    print("🏆 Creating final multi-omics ranking...")
    
    # Load all data sources
    engineering_df = pd.read_csv("results/tables/enhanced_engineering_results.csv")
    impact_scores = pd.read_csv("results/tables/enhanced_fba_impact_scores.csv")
    ml_scores = pd.read_csv("results/tables/ml_priority_scores.csv")
    robustness_df = pd.read_csv("results/tables/engineering_robustness.csv")
    
    # Merge all data
    final_ranking = pd.merge(impact_scores, ml_scores, on='scenario', how='left')
    final_ranking = pd.merge(final_ranking, robustness_df, on='scenario', how='left')
    
    # Calculate integrated multi-omics score
    final_ranking['multiomics_score'] = (
        0.40 * final_ranking['fba_impact_score'] +           # FBA performance
        0.25 * final_ranking['ml_priority_score'] +          # ML prediction
        0.20 * final_ranking['overall_robustness'] +         # Robustness
        0.15 * final_ranking['ala_improvement_ratio']        # Improvement ratio
    )
    
    # Add modification complexity penalty
    complexity_penalty = np.exp(-final_ranking['modification_count'] / 8)
    final_ranking['multiomics_score'] = final_ranking['multiomics_score'] * complexity_penalty
    
    # Final ranking
    final_ranking = final_ranking.sort_values('multiomics_score', ascending=False)
    final_ranking['rank'] = range(1, len(final_ranking) + 1)
    
    # Select publication columns
    publication_columns = [
        'rank', 'scenario', 'multiomics_score', 
        'fba_impact_score', 'ml_priority_score', 'overall_robustness',
        'delta_flux_ala', 'delta_flux_ala_percent', 'ala_improvement_ratio',
        'growth_maintenance_ratio', 'modification_count', 'environments_tested'
    ]
    
    final_ranking = final_ranking[publication_columns]
    
    # Save final table
    final_ranking.to_csv("results/tables/final_multiomics_ranking.csv", index=False)
    
    # Create ranking visualization
    create_ranking_visualization(final_ranking)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("FINAL MULTI-OMICS RANKING - TOP 10 SCENARIOS")
    print("="*80)
    
    top_10 = final_ranking.head(10)
    for _, row in top_10.iterrows():
        print(f"{row['rank']:2d}. {row['scenario']:30} | "
              f"Score: {row['multiomics_score']:.3f} | "
              f"ΔALA: {row['delta_flux_ala']:+.4f} | "
              f"Mods: {row['modification_count']}")
    
    return final_ranking

def create_ranking_visualization(final_ranking):
    """Create publication-ready ranking visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Top scenarios by multi-omics score
    top_10 = final_ranking.head(10)
    bars = ax1.barh(range(len(top_10)), top_10['multiomics_score'])
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels(top_10['scenario'])
    ax1.set_xlabel('Multi-omics Score', fontweight='bold')
    ax1.set_title('A. Top 10 Engineering Scenarios', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Score component breakdown
    components = ['fba_impact_score', 'ml_priority_score', 'overall_robustness']
    component_data = top_10[['scenario'] + components].set_index('scenario')
    component_data.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_ylabel('Component Score', fontweight='bold')
    ax2.set_title('B. Score Component Breakdown', fontweight='bold', pad=20)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. ΔALA flux vs modification count
    scatter = ax3.scatter(final_ranking['modification_count'], 
                         final_ranking['delta_flux_ala'],
                         c=final_ranking['multiomics_score'], 
                         cmap='viridis', s=80, alpha=0.8)
    ax3.set_xlabel('Number of Modifications', fontweight='bold')
    ax3.set_ylabel('ΔALA Flux (mmol/gDW/h)', fontweight='bold')
    ax3.set_title('C. Engineering Complexity vs Benefit', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Multi-omics Score')
    
    # 4. Environment coverage vs performance
    scatter2 = ax4.scatter(final_ranking['environments_tested'], 
                          final_ranking['multiomics_score'],
                          c=final_ranking['delta_flux_ala'], 
                          cmap='plasma', s=80, alpha=0.8)
    ax4.set_xlabel('Environments Tested', fontweight='bold')
    ax4.set_ylabel('Multi-omics Score', fontweight='bold')
    ax4.set_title('D. Robustness vs Overall Performance', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax4, label='ΔALA Flux')
    
    plt.tight_layout()
    plt.savefig('results/figures/final_multiomics_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Execute final multi-omics ranking"""
    final_ranking = create_multiomics_ranking()
    
    print(f"\n✅ Final multi-omics ranking completed!")
    print(f"📊 Results saved to:")
    print(f"   - results/tables/final_multiomics_ranking.csv")
    print(f"   - results/figures/final_multiomics_ranking.png")

if __name__ == "__main__":
    main()