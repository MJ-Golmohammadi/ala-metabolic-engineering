"""
Growth-production trade-off analysis for metabolic engineering scenarios
Identifies optimal engineering strategies using Pareto front analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial import ConvexHull

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


def calculate_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Calculate Pareto-optimal points for multi-objective optimization
    
    Parameters:
    -----------
    points : np.ndarray
        Array of [growth_rate, production] points
        
    Returns:
    --------
    np.ndarray
        Pareto-optimal points
    """
    if len(points) < 3:
        return points
    
    # For maximization of both objectives
    pareto_points = []
    for point in points:
        is_pareto = True
        for other_point in points:
            if (other_point[0] >= point[0] and other_point[1] >= point[1] and 
                not np.array_equal(other_point, point)):
                is_pareto = False
                break
        if is_pareto:
            pareto_points.append(point)
    
    return np.array(pareto_points)


def create_tradeoff_analysis(engineering_results: pd.DataFrame) -> dict:
    """
    Comprehensive trade-off analysis between growth and production
    
    Parameters:
    -----------
    engineering_results : pd.DataFrame
        Results from metabolic engineering simulations
        
    Returns:
    --------
    dict
        Analysis results and generated figures
    """
    # Filter for ALA production objective
    production_data = engineering_results[engineering_results['objective'] == 'max_biomass']
    
    # Create comprehensive trade-off visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Main trade-off scatter plot
    scatter = ax1.scatter(production_data['growth_rate'], 
                         production_data['ala_flux'],
                         c=production_data['yield_mmol_g'],
                         s=120, alpha=0.8, cmap='viridis', edgecolor='black', linewidth=0.5)
    
    # Add scenario labels
    for _, row in production_data.iterrows():
        ax1.annotate(row['scenario'], 
                    (row['growth_rate'], row['ala_flux']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_xlabel('Growth Rate (h$^{-1}$)', fontweight='bold')
    ax1.set_ylabel('5-ALA Production (mmol/gDW/h)', fontweight='bold')
    ax1.set_title('Growth-Production Trade-off Analysis', fontweight='bold', pad=20)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Yield (mmol ALA/mmol Glucose)', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add Pareto front
    points = production_data[['growth_rate', 'ala_flux']].values
    pareto_points = calculate_pareto_front(points)
    if len(pareto_points) > 1:
        pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
        ax1.plot(pareto_points[:, 0], pareto_points[:, 1], 'r--', linewidth=2, 
                label='Pareto Front', alpha=0.8)
        ax1.legend()
    
    # 2. Yield comparison across scenarios
    scenarios = production_data['scenario']
    yields = production_data['yield_mmol_g']
    
    bars = ax2.barh(range(len(scenarios)), yields, color='steelblue', alpha=0.8)
    ax2.set_yticks(range(len(scenarios)))
    ax2.set_yticklabels(scenarios, fontsize=9)
    ax2.set_xlabel('Yield (mmol ALA/mmol Glucose)', fontweight='bold')
    ax2.set_title('Production Yield by Engineering Scenario', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=8)
    
    # 3. Improvement over wild-type
    wt_flux = production_data[production_data['scenario'] == 'wild_type']['ala_flux'].values[0]
    improvements = (production_data['ala_flux'] / wt_flux - 1) * 100
    
    ax3.bar(range(len(scenarios)), improvements, color='green', alpha=0.7)
    ax3.set_xticks(range(len(scenarios)))
    ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Improvement Over Wild-type (%)', fontweight='bold')
    ax3.set_title('Production Improvement Percentage', fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 4. Modification count vs improvement
    ax4.scatter(production_data['modification_count'], 
               production_data['ala_flux'], 
               s=100, alpha=0.8, color='purple', edgecolor='black')
    
    for _, row in production_data.iterrows():
        ax4.annotate(row['scenario'], 
                    (row['modification_count'], row['ala_flux']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Number of Genetic Modifications', fontweight='bold')
    ax4.set_ylabel('5-ALA Production (mmol/gDW/h)', fontweight='bold')
    ax4.set_title('Engineering Complexity vs Production', fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/figures/comprehensive_tradeoff_analysis.png', dpi=300)
    plt.show()
    
    return {
        'production_data': production_data,
        'pareto_points': pareto_points,
        'max_improvement': improvements.max()
    }


def main():
    """Execute comprehensive trade-off analysis"""
    
    # Load engineering results
    engineering_df = pd.read_csv("results/tables/engineering_results.csv")
    
    print("Performing growth-production trade-off analysis...")
    analysis_results = create_tradeoff_analysis(engineering_df)
    
    # Save analysis results
    analysis_results['production_data'].to_csv("results/tables/tradeoff_analysis.csv", index=False)
    
    # Print key findings
    production_data = analysis_results['production_data']
    best_scenario = production_data.loc[production_data['ala_flux'].idxmax()]
    
    print("\n" + "="*60)
    print("TRADE-OFF ANALYSIS KEY FINDINGS")
    print("="*60)
    print(f"Best performing scenario: {best_scenario['scenario']}")
    print(f"Maximum ALA production: {best_scenario['ala_flux_net']:.3f} mmol/gDW/h")
    print(f"Associated growth rate: {best_scenario['growth_rate']:.3f} h⁻¹")
    print(f"Yield: {best_scenario['yield_mmol_g']:.3f} mmol ALA/mmol glucose")
    print(f"Improvement over wild-type: {best_scenario.get('fold_improvement_net', best_scenario.get('ala_improvement_ratio', 1)):.1f}x")
    
    print("✅ Trade-off analysis completed successfully!")


if __name__ == "__main__":
    main()
