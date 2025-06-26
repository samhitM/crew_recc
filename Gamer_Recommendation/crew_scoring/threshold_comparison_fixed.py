#!/usr/bin/env python3
"""
Threshold Method Comparison Script
Compares percentile-based vs KNN-based threshold calculation for crew level assignment.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

# Add the parent directory to sys.path to import the calculator
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from standalone_level_calculator import StandaloneCrewLevelCalculator

class ThresholdComparisonAnalyzer:
    """Compare different threshold calculation methods for crew level assignment."""
    
    def __init__(self):
        self.results = {}
        
    def run_comparison(self) -> Dict:
        """Run both threshold methods and compare results."""
        print("=" * 60)
        print("THRESHOLD METHOD COMPARISON")
        print("=" * 60)
        
        results = {}
        
        # Method 1: Percentile-based thresholds
        print("\\n1. Running PERCENTILE-based threshold calculation...")
        print("-" * 50)
        
        calculator_percentile = StandaloneCrewLevelCalculator()
        calculator_percentile.threshold_method = 'percentile'
        
        try:
            df_percentile = calculator_percentile.calculate_crew_levels()
            results['percentile'] = {
                'dataframe': df_percentile,
                'method': 'Percentile-based',
                'calculator': calculator_percentile
            }
            print(f"✓ Percentile method completed: {len(df_percentile)} users processed")
        except Exception as e:
            print(f"✗ Percentile method failed: {e}")
            results['percentile'] = None
        
        # Method 2: KNN-based thresholds
        print("\\n2. Running KNN-based threshold calculation...")
        print("-" * 50)
        
        calculator_knn = StandaloneCrewLevelCalculator()
        calculator_knn.threshold_method = 'knn'
        
        try:
            df_knn = calculator_knn.calculate_crew_levels()
            results['knn'] = {
                'dataframe': df_knn,
                'method': 'KNN-based',
                'calculator': calculator_knn
            }
            print(f"✓ KNN method completed: {len(df_knn)} users processed")
        except Exception as e:
            print(f"✗ KNN method failed: {e}")
            results['knn'] = None
        
        self.results = results
        return results
    
    def analyze_differences(self) -> pd.DataFrame:
        """Analyze the differences between threshold methods."""
        if not self.results or not all(self.results.values()):
            print("Cannot analyze differences - one or both methods failed")
            return None
        
        df_percentile = self.results['percentile']['dataframe']
        df_knn = self.results['knn']['dataframe']
        
        print("\\n" + "=" * 60)
        print("THRESHOLD METHOD COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Compare level distributions
        print("\\n1. LEVEL DISTRIBUTION COMPARISON")
        print("-" * 40)
        
        level_dist_percentile = df_percentile['crew_level'].value_counts().sort_index()
        level_dist_knn = df_knn['crew_level'].value_counts().sort_index()
        
        print(f"{'Level':<8} {'Percentile':<12} {'KNN':<12} {'Difference':<12}")
        print("-" * 45)
        
        all_levels = sorted(set(level_dist_percentile.index) | set(level_dist_knn.index))
        for level in all_levels:
            count_percentile = level_dist_percentile.get(level, 0)
            count_knn = level_dist_knn.get(level, 0)
            diff = count_knn - count_percentile
            print(f"{level:<8} {count_percentile:<12} {count_knn:<12} {diff:+<12}")
        
        # Merge dataframes for user-level comparison
        comparison_df = df_percentile[['user_id', 'crew_level', 'composite_score']].merge(
            df_knn[['user_id', 'crew_level', 'composite_score']], 
            on='user_id', 
            suffixes=('_percentile', '_knn')
        )
        
        # Calculate level changes
        comparison_df['level_change'] = comparison_df['crew_level_knn'] - comparison_df['crew_level_percentile']
        
        print("\\n2. USER-LEVEL CHANGES")
        print("-" * 40)
        
        level_changes = comparison_df['level_change'].value_counts().sort_index()
        total_users = len(comparison_df)
        
        print(f"Total users compared: {total_users}")
        print(f"\\n{'Change':<12} {'Users':<8} {'Percentage':<12}")
        print("-" * 35)
        
        for change, count in level_changes.items():
            percentage = (count / total_users) * 100
            if change == 0:
                change_str = "No change"
            elif change > 0:
                change_str = f"+{change} levels"
            else:
                change_str = f"{change} levels"
            print(f"{change_str:<12} {count:<8} {percentage:>6.1f}%")
        
        # Statistical comparison
        print("\\n3. STATISTICAL COMPARISON")
        print("-" * 40)
        
        print(f"{'Metric':<25} {'Percentile':<12} {'KNN':<12}")
        print("-" * 50)
        
        metrics = [
            ('Mean Level', df_percentile['crew_level'].mean(), df_knn['crew_level'].mean()),
            ('Std Level', df_percentile['crew_level'].std(), df_knn['crew_level'].std()),
            ('Mean Composite Score', df_percentile['composite_score'].mean(), df_knn['composite_score'].mean()),
            ('Std Composite Score', df_percentile['composite_score'].std(), df_knn['composite_score'].std()),
        ]
        
        for metric_name, val_percentile, val_knn in metrics:
            print(f"{metric_name:<25} {val_percentile:>8.3f}    {val_knn:>8.3f}")
        
        return comparison_df
    
    def visualize_comparison(self) -> None:
        """Create visualizations comparing the two methods."""
        if not self.results or not all(self.results.values()):
            print("Cannot create visualizations - one or both methods failed")
            return
        
        df_percentile = self.results['percentile']['dataframe']
        df_knn = self.results['knn']['dataframe']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Threshold Method Comparison: Percentile vs KNN', fontsize=16)
        
        # Plot 1: Level distribution comparison
        ax1 = axes[0, 0]
        level_dist_p = df_percentile['crew_level'].value_counts().sort_index()
        level_dist_k = df_knn['crew_level'].value_counts().sort_index()
        
        x = np.arange(1, 6)
        width = 0.35
        
        ax1.bar(x - width/2, [level_dist_p.get(i, 0) for i in x], width, 
                label='Percentile', alpha=0.8)
        ax1.bar(x + width/2, [level_dist_k.get(i, 0) for i in x], width, 
                label='KNN', alpha=0.8)
        
        ax1.set_title('Level Distribution Comparison')
        ax1.set_xlabel('Crew Level')
        ax1.set_ylabel('Number of Users')
        ax1.set_xticks(x)
        ax1.legend()
        
        # Plot 2: Score distribution by method
        ax2 = axes[0, 1]
        ax2.hist(df_percentile['composite_score'], bins=30, alpha=0.7, label='Percentile', density=True)
        ax2.hist(df_knn['composite_score'], bins=30, alpha=0.7, label='KNN', density=True)
        ax2.set_title('Composite Score Distribution')
        ax2.set_xlabel('Composite Score')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # Plot 3: Level changes scatter plot
        ax3 = axes[1, 0]
        merged_df = df_percentile[['user_id', 'crew_level', 'composite_score']].merge(
            df_knn[['user_id', 'crew_level']], on='user_id', suffixes=('_p', '_k'))
        
        ax3.scatter(merged_df['crew_level_p'], merged_df['crew_level_k'], alpha=0.6)
        ax3.plot([1, 5], [1, 5], 'r--', label='Perfect Agreement')
        ax3.set_title('Level Assignment Comparison')
        ax3.set_xlabel('Percentile Method Level')
        ax3.set_ylabel('KNN Method Level')
        ax3.legend()
        
        # Plot 4: Level change histogram
        ax4 = axes[1, 1]
        level_changes = merged_df['crew_level_k'] - merged_df['crew_level_p']
        ax4.hist(level_changes, bins=range(-3, 4), alpha=0.7, edgecolor='black')
        ax4.set_title('Level Changes (KNN - Percentile)')
        ax4.set_xlabel('Level Change')
        ax4.set_ylabel('Number of Users')
        ax4.axvline(x=0, color='red', linestyle='--', label='No Change')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(current_dir, 'threshold_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\\nComparison visualization saved to: {output_path}")
        
        plt.show()
    
    def save_comparison_results(self, comparison_df: pd.DataFrame = None) -> None:
        """Save comparison results to CSV."""
        if not self.results or not all(self.results.values()):
            print("Cannot save results - one or both methods failed")
            return
        
        # Save individual results
        output_dir = current_dir
        
        percentile_path = os.path.join(output_dir, 'crew_levels_percentile.csv')
        knn_path = os.path.join(output_dir, 'crew_levels_knn.csv')
        
        self.results['percentile']['dataframe'].to_csv(percentile_path, index=False)
        self.results['knn']['dataframe'].to_csv(knn_path, index=False)
        
        print(f"\\nResults saved:")
        print(f"  Percentile method: {percentile_path}")
        print(f"  KNN method: {knn_path}")
        
        # Save comparison if provided
        if comparison_df is not None:
            comparison_path = os.path.join(output_dir, 'threshold_comparison_results.csv')
            comparison_df.to_csv(comparison_path, index=False)
            print(f"  Comparison: {comparison_path}")

def main():
    """Main function to run the threshold comparison."""
    analyzer = ThresholdComparisonAnalyzer()
    
    # Run the comparison
    results = analyzer.run_comparison()
    
    if not results or not all(results.values()):
        print("\\n❌ Comparison failed - check the error messages above")
        return
    
    # Analyze differences
    comparison_df = analyzer.analyze_differences()
    
    # Create visualizations
    try:
        analyzer.visualize_comparison()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Save results
    analyzer.save_comparison_results(comparison_df)
    
    print("\\n" + "="*60)
    print("THRESHOLD COMPARISON COMPLETE")
    print("="*60)
    print("\\nSummary:")
    print("- Both percentile and KNN threshold methods have been tested")
    print("- Results show the differences in level assignments")
    print("- KNN method provides more data-driven clustering")
    print("- Percentile method ensures even distribution")
    print("\\nConsider the trade-offs when choosing a method for production use.")

if __name__ == "__main__":
    main()
