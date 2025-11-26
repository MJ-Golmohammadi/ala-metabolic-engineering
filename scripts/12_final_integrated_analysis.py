"""
Final Integrated Analysis and Executive Summary
Q1 Journal Quality - Comprehensive integration of all analyses for final recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

class MetabolicEngineeringRecommender:
    """
    Integrated recommender system for metabolic engineering strategies
    """
    
    def __init__(self):
        self.results = {}
        self.recommendations = {}
        
    def load_all_data(self):
        """Load all analysis results"""
        print("📁 Loading all analysis results...")
        
        data_files = {
            'engineering': 'results/tables/enhanced_engineering_results.csv',
            'fba_impact': 'results/tables/enhanced_fba_impact_scores.csv',
            'ml_scores': 'results/tables/enhanced_ml_priority_scores.csv',
            'robustness': 'results/tables/engineering_robustness.csv',
            'pathway': 'results/tables/pathway_analysis.csv',
            'bottlenecks': 'results/tables/pathway_bottlenecks.csv',
            'coupling': 'results/tables/flux_coupling_analysis.csv'
        }
        
        for name, path in data_files.items():
            try:
                self.results[name] = pd.read_csv(path)
                print(f"   ✅ Loaded {name}: {len(self.results[name])} records")
            except FileNotFoundError:
                print(f"   ⚠️  File not found: {path}")
                self.results[name] = pd.DataFrame()
        
        return self
    
    def calculate_integrated_scores(self):
        """Calculate integrated scores combining all analyses"""
        
        print("🔢 Calculating integrated scores...")
        
        # Get unique scenarios across all analyses
        all_scenarios = set()
        for df in self.results.values():
            if 'scenario' in df.columns:
                all_scenarios.update(df['scenario'].unique())
        
        integrated_scores = []
        
        for scenario in all_scenarios:
            if scenario == 'wild_type':
                continue
                
            # Collect scores from different analyses
            scores = {'scenario': scenario}
            
            # FBA impact score
            fba_data = self.results['fba_impact']
            if not fba_data.empty and scenario in fba_data['scenario'].values:
                fba_scores = fba_data[fba_data['scenario'] == scenario]['fba_impact_score']
                scores['fba_score'] = fba_scores.mean()
                scores['fba_consistency'] = 1 - fba_scores.std()  # Higher = more consistent
            else:
                scores['fba_score'] = 0
                scores['fba_consistency'] = 0
            
            # ML priority score
            ml_data = self.results['ml_scores']
            if not ml_data.empty and scenario in ml_data['scenario'].values:
                scores['ml_score'] = ml_data[ml_data['scenario'] == scenario]['ml_priority_score'].iloc[0]
            else:
                scores['ml_score'] = 0
            
            # Robustness score
            robustness_data = self.results['robustness']
            if not robustness_data.empty and scenario in robustness_data['scenario'].values:
                scores['robustness_score'] = robustness_data[robustness_data['scenario'] == scenario]['overall_robustness'].iloc[0]
            else:
                scores['robustness_score'] = 0
            
            # Pathway efficiency
            pathway_data = self.results['pathway']
            if not pathway_data.empty and scenario in pathway_data['scenario'].values:
                pathway_scores = pathway_data[pathway_data['scenario'] == scenario]
                scores['pathway_efficiency'] = pathway_scores['metabolic_efficiency'].mean()
                scores['carbon_efficiency'] = pathway_scores['carbon_yield'].mean()
            else:
                scores['pathway_efficiency'] = 0
                scores['carbon_efficiency'] = 0
            
            # Bottleneck severity (inverse)
            bottleneck_data = self.results['bottlenecks']
            if not bottleneck_data.empty and scenario in bottleneck_data['scenario'].values:
                bottleneck_severity = bottleneck_data[bottleneck_data['scenario'] == scenario]['bottleneck_severity'].iloc[0]
                severity_map = {'Severe': 0.2, 'Moderate': 0.5, 'Mild': 0.8}
                scores['bottleneck_score'] = severity_map.get(bottleneck_severity, 0.5)
            else:
                scores['bottleneck_score'] = 0.5  # Default neutral
            
            # Calculate integrated score
            weights = {
                'fba_score': 0.25,
                'ml_score': 0.20,
                'robustness_score': 0.20,
                'pathway_efficiency': 0.15,
                'carbon_efficiency': 0.10,
                'bottleneck_score': 0.10
            }
            
            integrated_score = sum(scores[key] * weight for key, weight in weights.items())
            scores['integrated_score'] = integrated_score
            
            # Get modification count
            eng_data = self.results['engineering']
            if not eng_data.empty and scenario in eng_data['scenario'].values:
                scores['modification_count'] = eng_data[eng_data['scenario'] == scenario]['modification_count'].iloc[0]
            else:
                scores['modification_count'] = 0
            
            integrated_scores.append(scores)
        
        self.integrated_scores = pd.DataFrame(integrated_scores)
        return self
    
    def generate_recommendations(self):
        """Generate comprehensive engineering recommendations"""
        
        print("💡 Generating engineering recommendations...")
        
        if not hasattr(self, 'integrated_scores') or self.integrated_scores.empty:
            print("⚠️ No integrated scores available")
            return self
        
        recommendations = []
        
        # Top overall recommendations
        top_overall = self.integrated_scores.nlargest(5, 'integrated_score')
        
        for _, scenario in top_overall.iterrows():
            rec = {
                'scenario': scenario['scenario'],
                'recommendation_type': 'Top_Performer',
                'integrated_score': scenario['integrated_score'],
                'rationale': self._generate_rationale(scenario),
                'priority': 'High',
                'implementation_complexity': self._assess_complexity(scenario['modification_count']),
                'expected_improvement': f"{scenario.get('fba_score', 0):.1f}x ALA production",
                'risk_level': self._assess_risk(scenario)
            }
            recommendations.append(rec)
        
        # Robustness-focused recommendations
        robust_scenarios = self.integrated_scores.nlargest(3, 'robustness_score')
        for _, scenario in robust_scenarios.iterrows():
            if scenario['scenario'] not in top_overall['scenario'].values:
                rec = {
                    'scenario': scenario['scenario'],
                    'recommendation_type': 'Robust_Performer',
                    'integrated_score': scenario['integrated_score'],
                    'rationale': "High performance consistency across environments",
                    'priority': 'Medium',
                    'implementation_complexity': self._assess_complexity(scenario['modification_count']),
                    'expected_improvement': f"Stable {scenario.get('fba_score', 0):.1f}x improvement",
                    'risk_level': 'Low'
                }
                recommendations.append(rec)
        
        # Simple effective recommendations (low modification count)
        simple_scenarios = self.integrated_scores[
            self.integrated_scores['modification_count'] <= 3
        ].nlargest(3, 'integrated_score')
        
        for _, scenario in simple_scenarios.iterrows():
            if scenario['scenario'] not in [r['scenario'] for r in recommendations]:
                rec = {
                    'scenario': scenario['scenario'],
                    'recommendation_type': 'Simple_Effective',
                    'integrated_score': scenario['integrated_score'],
                    'rationale': "High impact with minimal genetic modifications",
                    'priority': 'High',
                    'implementation_complexity': 'Low',
                    'expected_improvement': f"{scenario.get('fba_score', 0):.1f}x ALA production",
                    'risk_level': 'Low'
                }
                recommendations.append(rec)
        
        self.recommendations = pd.DataFrame(recommendations)
        return self
    
    def _generate_rationale(self, scenario):
        """Generate rationale for recommendation"""
        strengths = []
        
        if scenario.get('fba_score', 0) > 0.7:
            strengths.append("exceptional production improvement")
        elif scenario.get('fba_score', 0) > 0.5:
            strengths.append("strong production improvement")
        
        if scenario.get('robustness_score', 0) > 0.7:
            strengths.append("excellent environmental robustness")
        elif scenario.get('robustness_score', 0) > 0.5:
            strengths.append("good performance consistency")
        
        if scenario.get('ml_score', 0) > 0.7:
            strengths.append("high machine learning confidence")
        
        if scenario.get('bottleneck_score', 0) > 0.7:
            strengths.append("effective bottleneck resolution")
        
        if strengths:
            return f"Recommended due to {', '.join(strengths)}."
        else:
            return "Balanced performance across multiple metrics."
    
    def _assess_complexity(self, modification_count):
        """Assess implementation complexity"""
        if modification_count <= 3:
            return 'Low'
        elif modification_count <= 6:
            return 'Medium'
        else:
            return 'High'
    
    def _assess_risk(self, scenario):
        """Assess implementation risk"""
        risk_factors = 0
        
        if scenario['modification_count'] > 8:
            risk_factors += 1
        if scenario.get('bottleneck_score', 0.5) < 0.4:
            risk_factors += 1
        if scenario.get('robustness_score', 0) < 0.4:
            risk_factors += 1
        
        if risk_factors == 0:
            return 'Low'
        elif risk_factors == 1:
            return 'Medium'
        else:
            return 'High'
    
    def create_executive_summary(self):
        """Create executive summary visualization"""
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3)
        
        # 1. Top recommendations
        ax1 = fig.add_subplot(gs[0, :])
        if not self.recommendations.empty:
            top_recs = self.recommendations.head(8)
            y_pos = np.arange(len(top_recs))
            
            bars = ax1.barh(y_pos, top_recs['integrated_score'], 
                           color=['#2ecc71', '#3498db', '#e74c3c'][:len(top_recs)])
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(top_recs['scenario'], fontsize=10)
            ax1.set_xlabel('Integrated Score', fontweight='bold')
            ax1.set_title('A. Top Engineering Recommendations', fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add score labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 2. Score component radar chart
        ax2 = fig.add_subplot(gs[1, 0])
        if hasattr(self, 'integrated_scores') and not self.integrated_scores.empty:
            top_scenario = self.integrated_scores.iloc[0]
            components = ['fba_score', 'ml_score', 'robustness_score', 
                         'pathway_efficiency', 'carbon_efficiency', 'bottleneck_score']
            values = [top_scenario.get(comp, 0) for comp in components]
            labels = ['FBA Impact', 'ML Priority', 'Robustness', 
                     'Pathway Eff.', 'Carbon Eff.', 'Bottleneck']
            
            # Create radar chart
            angles = np.linspace(0, 2*np.pi, len(components), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, label='Top Scenario')
            ax2.fill(angles, values, alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(labels, fontsize=9)
            ax2.set_ylim(0, 1)
            ax2.set_title('B. Performance Profile - Top Scenario', fontweight='bold', pad=20)
            ax2.grid(True)
        
        # 3. Implementation complexity vs benefit
        ax3 = fig.add_subplot(gs[1, 1])
        if hasattr(self, 'integrated_scores') and not self.integrated_scores.empty:
            scatter = ax3.scatter(self.integrated_scores['modification_count'],
                                self.integrated_scores['integrated_score'],
                                c=self.integrated_scores['robustness_score'],
                                cmap='RdYlGn', s=60, alpha=0.7)
            ax3.set_xlabel('Number of Modifications', fontweight='bold')
            ax3.set_ylabel('Integrated Score', fontweight='bold')
            ax3.set_title('C. Implementation Complexity vs Benefit', fontweight='bold', pad=20)
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Robustness Score')
        
        # 4. Risk-benefit analysis
        ax4 = fig.add_subplot(gs[1, 2])
        if not self.recommendations.empty:
            risk_data = self.recommendations.copy()
            risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
            risk_data['risk_numeric'] = risk_data['risk_level'].map(risk_map)
            
            scatter2 = ax4.scatter(risk_data['risk_numeric'],
                                 risk_data['integrated_score'],
                                 c=risk_data['integrated_score'],
                                 cmap='viridis', s=80, alpha=0.7)
            ax4.set_xlabel('Risk Level', fontweight='bold')
            ax4.set_ylabel('Integrated Score', fontweight='bold')
            ax4.set_xticks([0, 1, 2])
            ax4.set_xticklabels(['Low', 'Medium', 'High'])
            ax4.set_title('D. Risk-Benefit Analysis', fontweight='bold', pad=20)
            ax4.grid(True, alpha=0.3)
        
        # 5. Summary statistics
        ax5 = fig.add_subplot(gs[2, :])
        if hasattr(self, 'integrated_scores') and not self.integrated_scores.empty:
            stats_data = {
                'Total Scenarios Analyzed': len(self.integrated_scores),
                'High Priority Recommendations': len(self.recommendations[self.recommendations['priority'] == 'High']),
                'Average Modification Count': self.integrated_scores['modification_count'].mean(),
                'Top Integrated Score': self.integrated_scores['integrated_score'].max(),
                'Robust Scenarios (>0.7)': len(self.integrated_scores[self.integrated_scores['robustness_score'] > 0.7])
            }
            
            ax5.axis('off')
            ax5.set_title('E. Analysis Summary Statistics', fontweight='bold', pad=20)
            
            # Create table
            table_data = [[k, v] for k, v in stats_data.items()]
            table = ax5.table(cellText=table_data,
                             colLabels=['Metric', 'Value'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0.1, 0.1, 0.8, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
        
        plt.tight_layout()
        return fig

def main():
    """Main function for integrated analysis"""
    
    print("🚀 Starting final integrated analysis...")
    
    # Initialize recommender
    recommender = MetabolicEngineeringRecommender()
    
    # Load data and generate recommendations
    recommender.load_all_data()\
              .calculate_integrated_scores()\
              .generate_recommendations()
    
    # Save results
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recommender.integrated_scores.to_csv(output_dir / "final_integrated_scores.csv", index=False)
    recommender.recommendations.to_csv(output_dir / "engineering_recommendations.csv", index=False)
    
    # Create executive summary
    print("📊 Creating executive summary...")
    summary_fig = recommender.create_executive_summary()
    summary_fig.savefig('results/figures/executive_summary.png', dpi=300, bbox_inches='tight')
    
    # Print final recommendations
    print("\n" + "="*80)
    print("FINAL ENGINEERING RECOMMENDATIONS")
    print("="*80)
    
    if not recommender.recommendations.empty:
        print(f"\n🏆 TOP RECOMMENDATIONS FOR IMPLEMENTATION:")
        print("-" * 100)
        for _, rec in recommender.recommendations.iterrows():
            print(f"📍 {rec['scenario']}")
            print(f"   Type: {rec['recommendation_type']:20} | Priority: {rec['priority']:6} | "
                  f"Score: {rec['integrated_score']:.3f}")
            print(f"   Risk: {rec['risk_level']:6} | Complexity: {rec['implementation_complexity']:6}")
            print(f"   Expected: {rec['expected_improvement']}")
            print(f"   Rationale: {rec['rationale']}")
            print()
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    if hasattr(recommender, 'integrated_scores') and not recommender.integrated_scores.empty:
        top_scenario = recommender.integrated_scores.iloc[0]
        robust_count = len(recommender.integrated_scores[recommender.integrated_scores['robustness_score'] > 0.7])
        simple_effective = len(recommender.integrated_scores[
            (recommender.integrated_scores['modification_count'] <= 3) &
            (recommender.integrated_scores['integrated_score'] > 0.6)
        ])
        
        print(f"  • Best overall scenario: {top_scenario['scenario']} "
              f"(Score: {top_scenario['integrated_score']:.3f})")
        print(f"  • Robust scenarios identified: {robust_count}")
        print(f"  • Simple yet effective strategies: {simple_effective}")
        print(f"  • Average modifications per scenario: {recommender.integrated_scores['modification_count'].mean():.1f}")
    
    print(f"\n✅ Final integrated analysis completed successfully!")
    print(f"📁 Results saved in:")
    print(f"   - results/tables/final_integrated_scores.csv")
    print(f"   - results/tables/engineering_recommendations.csv")
    print(f"   - results/figures/executive_summary.png")

if __name__ == "__main__":
    main()