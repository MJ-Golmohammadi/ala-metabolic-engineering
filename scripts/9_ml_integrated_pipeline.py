# scripts/9_ml_integrated_pipeline.py
"""
Integrated Machine Learning Pipeline for Metabolic Engineering
Q1 Journal Quality - Combines DESeq2, FBA, and ML for predictive modeling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

class IntegratedMLPipeline:
    """
    Integrated ML pipeline combining DESeq2 expression data with FBA results
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.shap_explainers = {}
        
    def load_and_integrate_data(self):
        """Load and integrate all data sources"""
        print("📁 Loading and integrating multi-omics data...")
        
        # 1. Load DESeq2 data
        deseq_data = self._load_deseq2_data()
        
        # 2. Load FBA engineering results
        fba_results = pd.read_csv("results/tables/enhanced_engineering_results.csv")
        
        # 3. Load gene-reaction mapping
        gene_reaction_map = {
            'PP_0732': 'GLUTRR', 'PP_4784': 'G1SAT', 'PP_1977': 'GLUTRS', 'PP_0223': 'PPBNGS',
            'PP_1014': 'GLUDy', 'PP_1015': 'GLUSy', 'PP_2425': 'ICDHyr', 'PP_1406': 'PPC',
            'PP_1444': 'G6PDH2r', 'PP_1446': 'GND', 'PP_1443': 'NADTRHD'
        }
        
        # 4. Integrate data
        integrated_data = self._integrate_expression_with_flux(
            deseq_data, fba_results, gene_reaction_map
        )
        
        return integrated_data, fba_results
    
    def _load_deseq2_data(self):
        """Load and preprocess DESeq2 data with comprehensive validation"""
        comparisons = {
            'cit_vs_glu': 'data/deseq2/cit_vs_glu.csv',
            'fer_vs_glu': 'data/deseq2/fer_vs_glu.csv', 
            'ser_vs_glu': 'data/deseq2/ser_vs_glu.csv'
        }
        
        all_deseq = []
        for comp_name, file_path in comparisons.items():
            try:
                # Check if file exists
                if not Path(file_path).exists():
                    print(f"  ⚠️ DESeq2 file not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path)
                
                # Validate required columns
                required_columns = ['GeneID', 'Base mean', 'log2(FC)', 'P-adj']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"  ⚠️ Missing columns in {comp_name}: {missing_columns}")
                    continue
                
                df['comparison'] = comp_name
                df['environment'] = comp_name.split('_')[0].capitalize()
                
                # Filter significant genes (P-adj < 0.05)
                df_significant = df[df['P-adj'] < 0.05].copy()
                
                if df_significant.empty:
                    print(f"  ⚠️ No significant genes found in {comp_name}")
                else:
                    all_deseq.append(df_significant)
                    print(f"  ✅ {comp_name}: {len(df_significant)} significant genes")
                    
            except Exception as e:
                print(f"  ❌ Error loading {comp_name}: {e}")
                continue
        
        if not all_deseq:
            print("  ⚠️ No DESeq2 data loaded - ML will use only FBA features")
            return pd.DataFrame()
        
        return pd.concat(all_deseq, ignore_index=True)
    
    def _integrate_expression_with_flux(self, deseq_data, fba_results, gene_map):
        """Integrate expression data with flux results"""
        integrated_results = []
        
        # Create expression features for each scenario
        for scenario in fba_results['scenario'].unique():
            if scenario == 'wild_type':
                continue
                
            scenario_data = fba_results[fba_results['scenario'] == scenario]
            modifications = scenario_data['modifications'].iloc[0]
            
            # Extract modified genes from scenario
            modified_genes = self._extract_modified_genes(modifications, gene_map)
            
            for environment in ['Cit', 'Fer', 'Ser']:
                env_data = scenario_data[scenario_data['environment'] == environment]
                
                if len(env_data) > 0:
                    # Get expression features for modified genes in this environment
                    expression_features = self._calculate_expression_features(
                        deseq_data, modified_genes, environment
                    )
                    
                    # Combine with performance metrics
                    integrated_row = {
                        'scenario': scenario,
                        'environment': environment,
                        'ala_flux': env_data['ala_flux_net'].iloc[0],
                        'growth_rate': env_data['growth_rate'].iloc[0],
                        'yield_mmol_g': env_data['yield_mmol_g'].iloc[0],
                        'modification_count': env_data['modification_count'].iloc[0],
                        **expression_features
                    }
                    
                    integrated_results.append(integrated_row)
        
        return pd.DataFrame(integrated_results)
    
    def _extract_modified_genes(self, modifications_str, gene_map):
        """Extract modified genes from modifications string"""
        modified_genes = []
        reverse_gene_map = {v: k for k, v in gene_map.items()}
        
        for reaction in reverse_gene_map.keys():
            if reaction in str(modifications_str):
                modified_genes.append(reverse_gene_map[reaction])
        
        return modified_genes
    
    def _calculate_expression_features(self, deseq_data, genes, environment):
        """Calculate expression-based features for ML"""
        features = {}
        
        for gene in genes:
            gene_expr = deseq_data[
                (deseq_data['GeneID'] == gene) & 
                (deseq_data['environment'] == environment)
            ]
            
            if len(gene_expr) > 0:
                expr_row = gene_expr.iloc[0]
                features[f'{gene}_log2FC'] = expr_row['log2(FC)']
                features[f'{gene}_base_mean'] = expr_row['Base mean']
                features[f'{gene}_significant'] = 1 if expr_row['P-adj'] < 0.05 else 0
                features[f'{gene}_wald_stat'] = expr_row['Wald-Stats']
            else:
                # Default values for genes without expression data
                features.update({
                    f'{gene}_log2FC': 0,
                    f'{gene}_base_mean': 0,
                    f'{gene}_significant': 0,
                    f'{gene}_wald_stat': 0
                })
        
        # Aggregate features
        if genes:
            features['avg_log2FC'] = np.mean([features.get(f'{g}_log2FC', 0) for g in genes])
            features['significant_ratio'] = np.mean([features.get(f'{g}_significant', 0) for g in genes])
            features['total_wald_stats'] = np.sum([features.get(f'{g}_wald_stat', 0) for g in genes])
        else:
            features.update({'avg_log2FC': 0, 'significant_ratio': 0, 'total_wald_stats': 0})
        
        return features
    
    def prepare_ml_features(self, integrated_data):
        """Prepare features and targets for ML"""
        print("🛠️ Preparing ML features from integrated data...")
        
        # Separate features and target
        feature_columns = [col for col in integrated_data.columns 
                         if col not in ['scenario', 'environment', 'ala_flux', 'growth_rate', 'yield_mmol_g']]
        
        X = integrated_data[feature_columns].fillna(0)
        y = integrated_data['ala_flux']
        
        # Remove constant columns
        X = X.loc[:, X.std() > 0]
        
        print(f"  Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y, integrated_data['scenario']
    
    def train_ensemble_models(self, X, y, test_size: float = 0.2):
        """Train ensemble of ML models"""
        print("🤖 Training ensemble ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=self.random_state, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=150, max_depth=8, random_state=self.random_state
            ),
            'BayesianRidge': BayesianRidge(),
            'NeuralNetwork': MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state
            )
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=5, scoring='r2', n_jobs=-1)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            self.models[name] = model
            
            # Predictions and evaluation
            y_pred = model.predict(X_test_scaled)
            
            results[name] = {
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'test_r2': r2_score(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'test_mae': mean_absolute_error(y_test, y_pred),
                'predictions': y_pred,
                'actuals': y_test
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        # Ensemble prediction
        ensemble_pred = np.mean([results[name]['predictions'] for name in models.keys()], axis=0)
        results['Ensemble'] = {
            'test_r2': r2_score(y_test, ensemble_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'test_mae': mean_absolute_error(y_test, ensemble_pred)
        }
        
        return results, X_test, y_test
    
    def perform_shap_analysis(self, X, model_name='RandomForest'):
        """Perform SHAP analysis for model interpretability"""
        print("📊 Performing SHAP analysis...")
        
        model = self.models.get(model_name)
        if model is None:
            print(f"  ⚠️ Model {model_name} not found")
            return None
        
        # Use scaled features
        X_scaled = self.scalers['standard'].transform(X)
        
        # SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        self.shap_explainers[model_name] = explainer
        
        return shap_values, X.columns
    
    def predict_scenario_performance(self, integrated_data, X):
        """Predict performance for all scenarios"""
        print("🎯 Predicting scenario performance...")
        
        # Use ensemble of models for prediction
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        # Create prediction dataframe
        prediction_df = integrated_data[['scenario', 'environment']].copy()
        prediction_df['actual_ala_flux'] = integrated_data['ala_flux']
        prediction_df['predicted_ala_flux'] = ensemble_pred
        prediction_df['prediction_error'] = prediction_df['predicted_ala_flux'] - prediction_df['actual_ala_flux']
        prediction_df['absolute_error'] = np.abs(prediction_df['prediction_error'])
        
        return prediction_df
    
    def calculate_ml_priority_scores(self, prediction_df):
        """Calculate ML-based priority scores for scenarios"""
        print("🏆 Calculating ML priority scores...")
        
        # Calculate scores based on prediction performance and consistency
        scenario_scores = []
        
        for scenario in prediction_df['scenario'].unique():
            scenario_data = prediction_df[prediction_df['scenario'] == scenario]
            
            # Performance metrics
            avg_predicted = scenario_data['predicted_ala_flux'].mean()
            avg_actual = scenario_data['actual_ala_flux'].mean()
            prediction_accuracy = 1 - (scenario_data['absolute_error'].mean() / max(1, avg_actual))
            consistency = 1 - scenario_data['predicted_ala_flux'].std() / max(1, scenario_data['predicted_ala_flux'].mean())
            
            # ML priority score
            ml_score = (0.6 * (avg_predicted / max(1, avg_actual)) + 
                      0.2 * prediction_accuracy + 
                      0.2 * consistency)
            
            scenario_scores.append({
                'scenario': scenario,
                'ml_priority_score': min(1.0, max(0, ml_score)),
                'avg_predicted_ala': avg_predicted,
                'avg_actual_ala': avg_actual,
                'prediction_accuracy': prediction_accuracy,
                'performance_consistency': consistency,
                'environments_tested': len(scenario_data)
            })
        
        return pd.DataFrame(scenario_scores).sort_values('ml_priority_score', ascending=False)
    
    def create_comprehensive_visualizations(self, results, prediction_df, shap_values, feature_names):
        """Create comprehensive ML visualizations"""
        print("📈 Creating comprehensive visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Model performance comparison
        models = list(results.keys())
        r2_scores = [results[model].get('test_r2', 0) for model in models]
        
        bars = ax1.bar(models, r2_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax1.set_ylabel('R² Score', fontweight='bold')
        ax1.set_title('A. Model Performance Comparison', fontweight='bold', pad=20)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. SHAP summary plot
        if shap_values is not None:
            shap.summary_plot(shap_values, feature_names=feature_names, 
                             show=False, plot_size=None, max_display=10)
            ax2.set_title('B. SHAP Feature Importance', fontweight='bold', pad=20)
        
        # 3. Prediction vs Actual
        ax3.scatter(prediction_df['actual_ala_flux'], prediction_df['predicted_ala_flux'], 
                   alpha=0.6, s=50)
        ax3.plot([prediction_df['actual_ala_flux'].min(), prediction_df['actual_ala_flux'].max()],
                 [prediction_df['actual_ala_flux'].min(), prediction_df['actual_ala_flux'].max()], 
                 'r--', linewidth=2)
        ax3.set_xlabel('Actual ALA Flux', fontweight='bold')
        ax3.set_ylabel('Predicted ALA Flux', fontweight='bold')
        ax3.set_title(f'C. Prediction vs Actual (Ensemble)', fontweight='bold', pad=20)
        ax3.grid(True, alpha=0.3)
        
        # 4. Top feature importance
        if 'RandomForest' in self.feature_importance:
            top_features = self.feature_importance['RandomForest'].head(10)
            ax4.barh(range(len(top_features)), top_features['importance'])
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features['feature'])
            ax4.set_xlabel('Feature Importance', fontweight='bold')
            ax4.set_title('D. Top 10 Most Important Features', fontweight='bold', pad=20)
            ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig

def main():
    """Execute integrated ML pipeline"""
    print("🚀 Starting Integrated ML Pipeline for Metabolic Engineering")
    
    # Initialize pipeline
    ml_pipeline = IntegratedMLPipeline(random_state=42)
    
    # 1. Load and integrate data
    integrated_data, fba_results = ml_pipeline.load_and_integrate_data()
    
    if integrated_data.empty:
        print("❌ No integrated data available. Check DESeq2 files.")
        return
    
    # 2. Prepare ML features
    X, y, scenarios = ml_pipeline.prepare_ml_features(integrated_data)
    
    # 3. Train models
    results, X_test, y_test = ml_pipeline.train_ensemble_models(X, y)
    
    # 4. SHAP analysis
    shap_values, feature_names = ml_pipeline.perform_shap_analysis(X_test)
    
    # 5. Predict scenario performance
    prediction_df = ml_pipeline.predict_scenario_performance(
        integrated_data[integrated_data.index.isin(X_test.index)], X_test
    )
    
    # 6. Calculate ML priority scores
    ml_scores = ml_pipeline.calculate_ml_priority_scores(prediction_df)
    
    # 7. Create visualizations
    viz_fig = ml_pipeline.create_comprehensive_visualizations(
        results, prediction_df, shap_values, feature_names
    )
    
    # 8. Save results
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    integrated_data.to_csv(results_dir / "ml_integrated_data.csv", index=False)
    prediction_df.to_csv(results_dir / "ml_predictions.csv", index=False)
    ml_scores.to_csv(results_dir / "ml_priority_scores.csv", index=False)
    
    # Save feature importance
    for model_name, importance_df in ml_pipeline.feature_importance.items():
        importance_df.to_csv(results_dir / f"feature_importance_{model_name}.csv", index=False)
    
    # Save visualization
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    viz_fig.savefig(figures_dir / "integrated_ml_analysis.png", dpi=300, bbox_inches='tight')
    
    # Print comprehensive results
    print("\n" + "="*80)
    print("INTEGRATED ML PIPELINE RESULTS")
    print("="*80)
    
    print(f"\n📊 Model Performance:")
    for model_name, result in results.items():
        if 'test_r2' in result:
            print(f"  {model_name:15} | R² = {result['test_r2']:.3f} | "
                  f"RMSE = {result.get('test_rmse', 0):.3f}")
    
    print(f"\n🎯 Top 5 Scenarios by ML Priority Score:")
    for _, row in ml_scores.head(5).iterrows():
        print(f"  {row['scenario']:30} | Score: {row['ml_priority_score']:.3f} | "
              f"Predicted ALA: {row['avg_predicted_ala']:.4f}")
    
    print(f"\n🔍 Top 5 Most Important Features:")
    if 'RandomForest' in ml_pipeline.feature_importance:
        top_features = ml_pipeline.feature_importance['RandomForest'].head(5)
        for _, row in top_features.iterrows():
            print(f"  {row['feature']:25} | Importance: {row['importance']:.4f}")
    
    print(f"\n✅ Integrated ML pipeline completed successfully!")

if __name__ == "__main__":
    main()