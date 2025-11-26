#!/usr/bin/env python3
"""
Main execution script for comprehensive 5-ALA production analysis
Orchestrates the complete workflow from baseline simulation to result generation
"""

import subprocess
import sys
import time
from datetime import datetime

def run_script(script_path: str, description: str) -> bool:
    """
    Execute individual analysis script with error handling
    
    Parameters:
    -----------
    script_path : str
        Path to the Python script to execute
    description : str
        Description of the script's purpose
        
    Returns:
    --------
    bool
        True if execution successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"SCRIPT: {script_path}")
    print(f"START TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    try:
        start_time = time.time()
        
        # Execute script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        execution_time = time.time() - start_time
        
        print(f"✅ SUCCESS: {description}")
        print(f"⏱️  Execution time: {execution_time:.1f} seconds")
        
        # Print any output from the script
        if result.stdout:
            print(f"📋 Output:\n{result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {description}")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error details:\n{e.stderr}")
        return False


def main():
    """Main execution function"""
    
    print("🚀 INITIATING COMPREHENSIVE 5-ALA PRODUCTION ANALYSIS")
    print("=" * 70)
    
    # Define analysis workflow
    workflow = [
        # Phase 1: Core Simulations
        {
            'script': 'scripts/1_baseline_simulation.py',
            'description': 'Multi-environment baseline simulations (Glu, Cit, Ser, Fer)'
        },
        {
            'script': 'scripts/2_metabolic_engineering.py', 
            'description': 'Comprehensive engineering scenario evaluation across environments'
        },
        
        # Phase 2: Core Analysis
        {
            'script': 'scripts/3_fba_impact_scoring.py',
            'description': 'FBA impact scoring with Δflux metrics for main table'
        },
        {
            'script': 'scripts/4_tradeoff_analysis.py',
            'description': 'Growth-production trade-off and Pareto optimality analysis'
        },
        {
            'script': 'scripts/5_bottleneck_identification.py',
            'description': 'Metabolic bottleneck identification via flux variability analysis'
        },
        
        # Phase 3: Advanced Analysis
        {
            'script': 'scripts/6_flux_coupling_analysis.py',
            'description': 'Flux coupling analysis for network structure insights'
        },
        {
            'script': 'scripts/7_pathway_analysis.py',
            'description': 'Pathway-centric analysis and alternative route identification'
        },
        {
            'script': 'scripts/8_sensitivity_robustness.py',
            'description': 'Parameter sensitivity and engineering robustness evaluation'
        },
        
        # Phase 4: ML Integration
        {
            'script': 'scripts/9_ml_integration.py',
            'description': 'Machine learning integration with ensemble models and SHAP'
        },
        {
            'script': 'scripts/10_ml_priority_scoring.py',
            'description': 'ML-based priority scoring for target ranking'
        },
        
        # Phase 5: Final Integration
        {
            'script': 'scripts/11_final_ranking.py',
            'description': 'Final target ranking combining FBA and ML scores'
        },
        {
            'script': 'scripts/12_final_integrated_analysis.py',
            'description': 'Integrated analysis and engineering recommendations'
        },
        
        # Phase 6: Visualization
        {
            'script': 'scripts/13_publication_figures.py',
            'description': 'Publication-ready figure generation for manuscript'
        }
    ]
    
    # Execute workflow
    successful_scripts = 0
    total_scripts = len(workflow)
    
    for step in workflow:
        success = run_script(step['script'], step['description'])
        if success:
            successful_scripts += 1
    
    # Generate final report
    print("\n" + "=" * 70)
    print("ANALYSIS WORKFLOW COMPLETED")
    print("=" * 70)
    print(f"Successful scripts: {successful_scripts}/{total_scripts}")
    
    if successful_scripts == total_scripts:
        print("🎉 ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("\n📊 Results available in:")
        print("   - ../results/tables/  : Data tables for supplementary materials")
        print("   - ../results/figures/ : Publication-ready figures")
        print("   - ../results/flux_maps/: Metabolic flux distributions")
    else:
        print("⚠️  Some analyses failed. Check error messages above.")
    
    print(f"\n🏁 Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
