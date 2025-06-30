"""
Utility functions for impression scoring.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from datetime import datetime, timedelta


class NormalizationUtils:
    """Utility functions for score normalization."""
    
    @staticmethod
    def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to have mean 0 and std 1."""
        values = list(scores.values())
        if not values or np.std(values) == 0:
            return {k: 0.0 for k in scores.keys()}
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        return {k: (v - mean_val) / std_val for k, v in scores.items()}
    
    @staticmethod
    def rescale_scores(normalized_scores: Dict[str, float], target_mean: float = 50, target_std: float = 15) -> Dict[str, float]:
        """Rescale normalized scores to a target range."""
        return {k: round(v * target_std + target_mean) for k, v in normalized_scores.items()}


class DataAggregationUtils:
    """Utility functions for data aggregation."""
    
    @staticmethod
    def aggregate_daily_impressions(impression_data: pd.DataFrame) -> Dict[str, float]:
        """Aggregate impressions by day."""
        print("Aggregating daily impressions...")
        
        # Create sample daily data (since we don't have timestamps in current data)
        daily_impressions = {}
        current_date = datetime.now()
        
        # Simulate daily aggregation for demonstration
        for i in range(7):  # Last 7 days
            date_str = (current_date - timedelta(days=i)).strftime('%Y-%m-%d')
            total_impressions = impression_data['total_impression_score'].sum() / 7  # Distribute evenly
            daily_impressions[date_str] = total_impressions
        
        print(f"Calculated daily impressions for {len(daily_impressions)} days")
        return daily_impressions
    
    @staticmethod
    def aggregate_weekly_impressions(daily_impressions: Dict[str, float]) -> Dict[str, float]:
        """Aggregate impressions by week."""
        print("Aggregating weekly impressions...")
        
        # Group daily impressions into weeks
        weekly_impressions = {}
        total_weekly = sum(daily_impressions.values())
        current_date = datetime.now()
        week_start = current_date - timedelta(days=current_date.weekday())
        week_str = week_start.strftime('Week of %Y-%m-%d')
        weekly_impressions[week_str] = total_weekly
        
        print(f"Calculated weekly impressions for {len(weekly_impressions)} weeks")
        return weekly_impressions
    
    @staticmethod
    def aggregate_monthly_impressions(daily_impressions: Dict[str, float]) -> Dict[str, float]:
        """Aggregate impressions by month."""
        print("Aggregating monthly impressions...")
        
        # Group daily impressions into months
        monthly_impressions = {}
        total_monthly = sum(daily_impressions.values())
        current_date = datetime.now()
        month_str = current_date.strftime('%Y-%m')
        monthly_impressions[month_str] = total_monthly
        
        print(f"Calculated monthly impressions for {len(monthly_impressions)} months")
        return monthly_impressions


class VisualizationUtils:
    """Utility functions for score visualization."""
    
    @staticmethod
    def plot_score_distributions(topological_scores: Dict[str, float], 
                                user_feature_scores: Dict[str, float], 
                                total_scores: Dict[str, float],
                                filename: str = "score_distributions.png"):
        """Visualize distribution of scores using histograms."""
        print("Creating score distribution visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Score Distributions for Impression Scoring', fontsize=16, fontweight='bold')
        
        # Topological Scores
        topological_values = list(topological_scores.values())
        axes[0, 0].hist(topological_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Topological Scores Distribution')
        axes[0, 0].set_xlabel('Scores')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # User Feature Scores
        feature_values = list(user_feature_scores.values())
        axes[0, 1].hist(feature_values, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('User Feature Scores Distribution')
        axes[0, 1].set_xlabel('Scores')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Impression Scores
        total_values = list(total_scores.values())
        axes[1, 0].hist(total_values, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Total Impression Scores Distribution')
        axes[1, 0].set_xlabel('Scores')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Combined comparison
        axes[1, 1].hist(topological_values, bins=15, alpha=0.5, color='blue', label='Topological', edgecolor='black')
        axes[1, 1].hist(feature_values, bins=15, alpha=0.5, color='green', label='Feature', edgecolor='black')
        axes[1, 1].hist(total_values, bins=15, alpha=0.5, color='red', label='Total', edgecolor='black')
        axes[1, 1].set_title('Score Comparison')
        axes[1, 1].set_xlabel('Scores')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Score distribution visualization saved as {filename}")
        plt.close()
    
    @staticmethod
    def plot_aggregated_impressions(daily_impressions: Dict[str, float],
                                   weekly_impressions: Dict[str, float],
                                   monthly_impressions: Dict[str, float],
                                   filename: str = "aggregated_impressions.png"):
        """Visualize aggregated impressions over time."""
        print("Creating aggregated impressions visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Aggregated Impressions Analysis', fontsize=16, fontweight='bold')
        
        # Daily Impressions
        dates = list(daily_impressions.keys())
        daily_values = list(daily_impressions.values())
        axes[0].bar(dates, daily_values, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title('Daily Impressions')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Total Impressions')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Weekly Impressions
        weeks = list(weekly_impressions.keys())
        weekly_values = list(weekly_impressions.values())
        axes[1].bar(weeks, weekly_values, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1].set_title('Weekly Impressions')
        axes[1].set_xlabel('Week')
        axes[1].set_ylabel('Total Impressions')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # Monthly Impressions
        months = list(monthly_impressions.keys())
        monthly_values = list(monthly_impressions.values())
        axes[2].bar(months, monthly_values, alpha=0.7, color='coral', edgecolor='black')
        axes[2].set_title('Monthly Impressions')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('Total Impressions')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Aggregated impressions visualization saved as {filename}")
        plt.close()


class FileUtils:
    """Utility functions for file operations."""
    
    @staticmethod
    def save_results_with_fallback(df: pd.DataFrame, output_file: str):
        """Save results with error handling and fallback filename."""
        try:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        except PermissionError:
            print(f"Permission denied saving to {output_file}. File may be open in another program.")
            # Try alternative filename
            alt_file = f"crew_impressions_revised_{int(pd.Timestamp.now().timestamp())}.csv"
            df.to_csv(alt_file, index=False)
            print(f"Results saved to alternative file: {alt_file}")
