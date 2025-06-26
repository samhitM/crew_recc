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
        print("\n1. Running PERCENTILE-based threshold calculation...")
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
        print("\n2. Running KNN-based threshold calculation...")
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
    
    def analyze_differences(self) -> None:
        """Analyze the differences between threshold methods."""
        if not self.results or not all(self.results.values()):
            print("Cannot analyze differences - one or both methods failed")
            return
        
        df_percentile = self.results['percentile']['dataframe']
        df_knn = self.results['knn']['dataframe']
        
        print("\n" + "=" * 60)
        print("THRESHOLD METHOD COMPARISON ANALYSIS")
        print("=" * 60)
        
        # Compare level distributions
        print("\n1. LEVEL DISTRIBUTION COMPARISON")
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
        
        print("\n2. USER-LEVEL CHANGES")\n        print(\"-\" * 40)\n        \n        level_changes = comparison_df['level_change'].value_counts().sort_index()\n        total_users = len(comparison_df)\n        \n        print(f\"Total users compared: {total_users}\")\n        print(f\"\\n{'Change':<12} {'Users':<8} {'Percentage':<12}\")\n        print(\"-\" * 35)\n        \n        for change, count in level_changes.items():\n            percentage = (count / total_users) * 100\n            if change == 0:\n                change_str = \"No change\"\n            elif change > 0:\n                change_str = f\"+{change} levels\"\n            else:\n                change_str = f\"{change} levels\"\n            print(f\"{change_str:<12} {count:<8} {percentage:>6.1f}%\")\n        \n        # Statistical comparison\n        print(f\"\\n3. STATISTICAL COMPARISON\")\n        print(\"-\" * 40)\n        \n        print(f\"{'Metric':<25} {'Percentile':<12} {'KNN':<12}\")\n        print(\"-\" * 50)\n        \n        metrics = [\n            ('Mean Level', df_percentile['crew_level'].mean(), df_knn['crew_level'].mean()),\n            ('Std Level', df_percentile['crew_level'].std(), df_knn['crew_level'].std()),\n            ('Mean Composite Score', df_percentile['composite_score'].mean(), df_knn['composite_score'].mean()),\n            ('Std Composite Score', df_percentile['composite_score'].std(), df_knn['composite_score'].std()),\n        ]\n        \n        for metric_name, val_percentile, val_knn in metrics:\n            print(f\"{metric_name:<25} {val_percentile:>8.3f}    {val_knn:>8.3f}\")\n        \n        # Threshold comparison\n        print(f\"\\n4. THRESHOLD COMPARISON\")\n        print(\"-\" * 40)\n        \n        calc_percentile = self.results['percentile']['calculator']\n        calc_knn = self.results['knn']['calculator']\n        \n        # Get composite scores to calculate thresholds again for display\n        composite_scores_p = dict(zip(df_percentile['user_id'], df_percentile['composite_score']))\n        composite_scores_k = dict(zip(df_knn['user_id'], df_knn['composite_score']))\n        \n        thresholds_percentile = calc_percentile._calculate_percentile_fallback(\n            list(composite_scores_p.values()), 5)\n        \n        try:\n            thresholds_knn = calc_knn._calculate_knn_thresholds(\n                composite_scores_k, 5)\n        except:\n            thresholds_knn = calc_knn._calculate_percentile_fallback(\n                list(composite_scores_k.values()), 5)\n        \n        print(f\"{'Threshold':<12} {'Percentile':<12} {'KNN':<12} {'Difference':<12}\")\n        print(\"-\" * 50)\n        \n        max_thresholds = max(len(thresholds_percentile), len(thresholds_knn))\n        for i in range(max_thresholds):\n            thresh_p = thresholds_percentile[i] if i < len(thresholds_percentile) else \"N/A\"\n            thresh_k = thresholds_knn[i] if i < len(thresholds_knn) else \"N/A\"\n            \n            if thresh_p != \"N/A\" and thresh_k != \"N/A\":\n                diff = thresh_k - thresh_p\n                print(f\"Level {i+2:<7} {thresh_p:>8.3f}    {thresh_k:>8.3f}    {diff:>+8.3f}\")\n            else:\n                print(f\"Level {i+2:<7} {str(thresh_p):>8}    {str(thresh_k):>8}    {'N/A':>8}\")\n        \n        return comparison_df\n    \n    def visualize_comparison(self) -> None:\n        \"\"\"Create visualizations comparing the two methods.\"\"\"\n        if not self.results or not all(self.results.values()):\n            print(\"Cannot create visualizations - one or both methods failed\")\n            return\n        \n        df_percentile = self.results['percentile']['dataframe']\n        df_knn = self.results['knn']['dataframe']\n        \n        # Create comparison plots\n        fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n        fig.suptitle('Threshold Method Comparison: Percentile vs KNN', fontsize=16)\n        \n        # Plot 1: Level distribution comparison\n        ax1 = axes[0, 0]\n        level_dist_p = df_percentile['crew_level'].value_counts().sort_index()\n        level_dist_k = df_knn['crew_level'].value_counts().sort_index()\n        \n        x = np.arange(1, 6)\n        width = 0.35\n        \n        ax1.bar(x - width/2, [level_dist_p.get(i, 0) for i in x], width, \n                label='Percentile', alpha=0.8)\n        ax1.bar(x + width/2, [level_dist_k.get(i, 0) for i in x], width, \n                label='KNN', alpha=0.8)\n        \n        ax1.set_title('Level Distribution Comparison')\n        ax1.set_xlabel('Crew Level')\n        ax1.set_ylabel('Number of Users')\n        ax1.set_xticks(x)\n        ax1.legend()\n        \n        # Plot 2: Score distribution by method\n        ax2 = axes[0, 1]\n        ax2.hist(df_percentile['composite_score'], bins=30, alpha=0.7, label='Percentile', density=True)\n        ax2.hist(df_knn['composite_score'], bins=30, alpha=0.7, label='KNN', density=True)\n        ax2.set_title('Composite Score Distribution')\n        ax2.set_xlabel('Composite Score')\n        ax2.set_ylabel('Density')\n        ax2.legend()\n        \n        # Plot 3: Level changes scatter plot\n        ax3 = axes[0, 2]\n        merged_df = df_percentile[['user_id', 'crew_level', 'composite_score']].merge(\n            df_knn[['user_id', 'crew_level']], on='user_id', suffixes=('_p', '_k'))\n        \n        ax3.scatter(merged_df['crew_level_p'], merged_df['crew_level_k'], alpha=0.6)\n        ax3.plot([1, 5], [1, 5], 'r--', label='Perfect Agreement')\n        ax3.set_title('Level Assignment Comparison')\n        ax3.set_xlabel('Percentile Method Level')\n        ax3.set_ylabel('KNN Method Level')\n        ax3.legend()\n        \n        # Plot 4: Box plot of scores by level for each method\n        ax4 = axes[1, 0]\n        percentile_data = [df_percentile[df_percentile['crew_level'] == i]['composite_score'].values \n                          for i in range(1, 6)]\n        knn_data = [df_knn[df_knn['crew_level'] == i]['composite_score'].values \n                   for i in range(1, 6)]\n        \n        positions = np.arange(1, 6)\n        bp1 = ax4.boxplot(percentile_data, positions=positions-0.2, widths=0.3, \n                         patch_artist=True, boxprops=dict(facecolor='lightblue'))\n        bp2 = ax4.boxplot(knn_data, positions=positions+0.2, widths=0.3, \n                         patch_artist=True, boxprops=dict(facecolor='lightcoral'))\n        \n        ax4.set_title('Score Distribution by Level')\n        ax4.set_xlabel('Crew Level')\n        ax4.set_ylabel('Composite Score')\n        ax4.legend([bp1[\"boxes\"][0], bp2[\"boxes\"][0]], ['Percentile', 'KNN'])\n        \n        # Plot 5: Level change histogram\n        ax5 = axes[1, 1]\n        level_changes = merged_df['crew_level_k'] - merged_df['crew_level_p']\n        ax5.hist(level_changes, bins=range(-3, 4), alpha=0.7, edgecolor='black')\n        ax5.set_title('Level Changes (KNN - Percentile)')\n        ax5.set_xlabel('Level Change')\n        ax5.set_ylabel('Number of Users')\n        ax5.axvline(x=0, color='red', linestyle='--', label='No Change')\n        ax5.legend()\n        \n        # Plot 6: Threshold comparison\n        ax6 = axes[1, 2]\n        \n        # Get thresholds for visualization\n        calc_percentile = self.results['percentile']['calculator']\n        calc_knn = self.results['knn']['calculator']\n        \n        composite_scores_p = dict(zip(df_percentile['user_id'], df_percentile['composite_score']))\n        composite_scores_k = dict(zip(df_knn['user_id'], df_knn['composite_score']))\n        \n        thresholds_p = calc_percentile._calculate_percentile_fallback(\n            list(composite_scores_p.values()), 5)\n        try:\n            thresholds_k = calc_knn._calculate_knn_thresholds(\n                composite_scores_k, 5)\n        except:\n            thresholds_k = calc_knn._calculate_percentile_fallback(\n                list(composite_scores_k.values()), 5)\n        \n        x_thresh = range(2, len(thresholds_p) + 2)\n        width = 0.35\n        \n        ax6.bar([x - width/2 for x in x_thresh], thresholds_p, width, \n                label='Percentile', alpha=0.8)\n        ax6.bar([x + width/2 for x in x_thresh], thresholds_k[:len(thresholds_p)], width, \n                label='KNN', alpha=0.8)\n        \n        ax6.set_title('Threshold Values Comparison')\n        ax6.set_xlabel('Level Threshold')\n        ax6.set_ylabel('Threshold Value')\n        ax6.set_xticks(x_thresh)\n        ax6.set_xticklabels([f'L{i}' for i in x_thresh])\n        ax6.legend()\n        \n        plt.tight_layout()\n        \n        # Save the plot\n        output_path = os.path.join(current_dir, 'threshold_comparison.png')\n        plt.savefig(output_path, dpi=300, bbox_inches='tight')\n        print(f\"\\nComparison visualization saved to: {output_path}\")\n        \n        plt.show()\n    \n    def save_comparison_results(self, comparison_df: pd.DataFrame = None) -> None:\n        \"\"\"Save comparison results to CSV.\"\"\"\n        if not self.results or not all(self.results.values()):\n            print(\"Cannot save results - one or both methods failed\")\n            return\n        \n        # Save individual results\n        output_dir = current_dir\n        \n        percentile_path = os.path.join(output_dir, 'crew_levels_percentile.csv')\n        knn_path = os.path.join(output_dir, 'crew_levels_knn.csv')\n        \n        self.results['percentile']['dataframe'].to_csv(percentile_path, index=False)\n        self.results['knn']['dataframe'].to_csv(knn_path, index=False)\n        \n        print(f\"\\nResults saved:\")\n        print(f\"  Percentile method: {percentile_path}\")\n        print(f\"  KNN method: {knn_path}\")\n        \n        # Save comparison if provided\n        if comparison_df is not None:\n            comparison_path = os.path.join(output_dir, 'threshold_comparison_results.csv')\n            comparison_df.to_csv(comparison_path, index=False)\n            print(f\"  Comparison: {comparison_path}\")\n\ndef main():\n    \"\"\"Main function to run the threshold comparison.\"\"\"\n    analyzer = ThresholdComparisonAnalyzer()\n    \n    # Run the comparison\n    results = analyzer.run_comparison()\n    \n    if not results or not all(results.values()):\n        print(\"\\n❌ Comparison failed - check the error messages above\")\n        return\n    \n    # Analyze differences\n    comparison_df = analyzer.analyze_differences()\n    \n    # Create visualizations\n    try:\n        analyzer.visualize_comparison()\n    except Exception as e:\n        print(f\"Visualization failed: {e}\")\n    \n    # Save results\n    analyzer.save_comparison_results(comparison_df)\n    \n    print(\"\\n\" + \"=\"*60)\n    print(\"THRESHOLD COMPARISON COMPLETE\")\n    print(\"=\"*60)\n    print(\"\\nSummary:\")\n    print(\"- Both percentile and KNN threshold methods have been tested\")\n    print(\"- Results show the differences in level assignments\")\n    print(\"- KNN method provides more data-driven clustering\")\n    print(\"- Percentile method ensures even distribution\")\n    print(\"\\nConsider the trade-offs when choosing a method for production use.\")\n\nif __name__ == \"__main__\":\n    main()\n
