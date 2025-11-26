"""
Final Engineering Target Ranking combining FBA impact and ML priority scores
Includes Δflux to 5-ALA for main publication table
"""

import pandas as pd

def create_final_ranking():
    """Create final ranked table of engineering targets including Δflux"""
    
    # Load all score tables
    engineering_df = pd.read_csv("results/tables/engineering_results.csv")
    impact_scores_df = pd.read_csv("results/tables/fba_impact_scores.csv") 
    ml_scores_df = pd.read_csv("results/tables/ml_priority_scores.csv")
    
    # Merge all scores
    final_ranking = pd.merge(impact_scores_df, ml_scores_df, on='scenario', how='left')
    
    # Calculate overall score (combined FBA + ML)
    final_ranking['overall_score'] = (
        0.6 * final_ranking['fba_impact_score'] + 
        0.4 * final_ranking['ml_priority_score']
    )
    
    # Add engineering details
    scenario_details = engineering_df[
        ['scenario', 'modifications', 'modification_count']
    ].drop_duplicates()
    
    final_ranking = pd.merge(final_ranking, scenario_details, on='scenario')
    
    # Final ranking
    final_ranking = final_ranking.sort_values('overall_score', ascending=False)
    final_ranking['rank'] = range(1, len(final_ranking) + 1)
    
    # 🔥 SELECT COLUMNS FOR MAIN PUBLICATION TABLE
    publication_columns = [
        'rank', 
        'scenario', 
        'delta_flux_ala',   
        'delta_flux_ala_percent',   
        'overall_score', 
        'fba_impact_score', 
        'ml_priority_score',
        'ala_improvement_ratio',    
        'growth_maintenance_ratio', 
        'modification_count',
        'modifications'
    ]
    
    final_ranking = final_ranking[publication_columns]
    
    # Save final table
    final_ranking.to_csv("results/tables/final_engineering_ranking.csv", index=False)
    
    print("🎯 FINAL ENGINEERING TARGET RANKING (with Δflux to 5-ALA)")
    print("="*90)
    
    # Print compact version for main table
    main_table = final_ranking[['rank', 'scenario', 'delta_flux_ala', 'delta_flux_ala_percent', 'overall_score']]
    print(main_table.head(10).to_string(index=False, float_format='%.4f'))
    
    return final_ranking

if __name__ == "__main__":
    final_results = create_final_ranking()
    print(f"\n✅ Final ranking with Δflux to 5-ALA completed!")
