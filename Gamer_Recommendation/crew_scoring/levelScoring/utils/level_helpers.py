"""
Utility functions for level scoring.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class LevelFileUtils:
    """Utility functions for file operations in level scoring."""
    
    @staticmethod
    def save_results_with_fallback(df: pd.DataFrame, output_file: str):
        """Save results with error handling and fallback filename."""
        try:
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        except PermissionError:
            print(f"Permission denied saving to {output_file}. File may be open in another program.")
            # Try alternative filename
            alt_file = f"crew_levels_revised_{int(pd.Timestamp.now().timestamp())}.csv"
            df.to_csv(alt_file, index=False)
            print(f"Results saved to alternative file: {alt_file}")


class LevelValidationUtils:
    """Utility functions for level validation."""
    
    @staticmethod
    def validate_level_distribution(level_assignments: Dict[str, int], target_levels: int = 3):
        """Validate and print level distribution."""
        level_counts = {}
        for level in level_assignments.values():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print("Level distribution:")
        for level in range(1, target_levels + 1):
            count = level_counts.get(level, 0)
            percentage = (count / len(level_assignments)) * 100 if level_assignments else 0
            print(f"  Level {level}: {count} users ({percentage:.1f}%)")
        
        return level_counts


class LevelAggregationUtils:
    """Utility functions for level data aggregation."""
    
    @staticmethod
    def aggregate_level_data(df: pd.DataFrame, 
                           user_id_col: str = 'user_id',
                           level_col: str = 'level',
                           time_col: str = 'timestamp',
                           period: str = 'daily') -> pd.DataFrame:
        """
        Aggregate level data by time period.
        
        Args:
            df: DataFrame with level data
            user_id_col: Column name for user ID
            level_col: Column name for level
            time_col: Column name for timestamp
            period: Aggregation period ('daily', 'weekly', 'monthly')
        
        Returns:
            Aggregated DataFrame
        """
        if time_col not in df.columns:
            print(f"Warning: No '{time_col}' column found for aggregation")
            return df
        
        # Convert timestamp to datetime if it's not already
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Set the grouping frequency
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M'
        }
        
        if period not in freq_map:
            raise ValueError(f"Period must be one of {list(freq_map.keys())}")
        
        # Group by time period and calculate aggregations
        grouped = df.groupby([
            pd.Grouper(key=time_col, freq=freq_map[period]),
            user_id_col
        ]).agg({
            level_col: ['mean', 'max', 'min', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
        
        return grouped
    
    @staticmethod
    def get_level_trends(df: pd.DataFrame, 
                        user_id_col: str = 'user_id',
                        level_col: str = 'level',
                        time_col: str = 'timestamp',
                        top_n: int = 10) -> Dict:
        """
        Get level trends for top users.
        
        Args:
            df: DataFrame with level data
            user_id_col: Column name for user ID
            level_col: Column name for level
            time_col: Column name for timestamp
            top_n: Number of top users to analyze
        
        Returns:
            Dictionary with trend data
        """
        if time_col not in df.columns:
            print(f"Warning: No '{time_col}' column found for trends")
            return {}
        
        # Get top users by average level
        top_users = df.groupby(user_id_col)[level_col].mean().nlargest(top_n).index.tolist()
        
        # Get trends for top users
        trends = {}
        for user in top_users:
            user_data = df[df[user_id_col] == user].sort_values(time_col)
            trends[user] = {
                'timestamps': user_data[time_col].tolist(),
                'levels': user_data[level_col].tolist(),
                'avg_level': user_data[level_col].mean(),
                'max_level': user_data[level_col].max()
            }
        
        return trends


class LevelVisualizationUtils:
    """Utility functions for level visualization."""
    
    @staticmethod
    def plot_level_distribution(df: pd.DataFrame, 
                              level_col: str = 'level',
                              title: str = "Level Distribution",
                              save_path: Optional[str] = None) -> None:
        """
        Plot distribution of levels.
        
        Args:
            df: DataFrame with level data
            level_col: Column name for level
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(df[level_col], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Level')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_level = df[level_col].mean()
        median_level = df[level_col].median()
        plt.axvline(mean_level, color='red', linestyle='--', label=f'Mean: {mean_level:.2f}')
        plt.axvline(median_level, color='green', linestyle='--', label=f'Median: {median_level:.2f}')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Level distribution plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_level_trends(trends: Dict, 
                         title: str = "Level Trends for Top Users",
                         save_path: Optional[str] = None) -> None:
        """
        Plot level trends over time for top users.
        
        Args:
            trends: Dictionary with user trend data
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if not trends:
            print("No trend data available to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(trends)))
        
        for i, (user, data) in enumerate(trends.items()):
            plt.plot(data['timestamps'], data['levels'], 
                    marker='o', label=f"User {user} (avg: {data['avg_level']:.2f})",
                    color=colors[i], linewidth=2)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Level')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Level trends plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_level_aggregations(aggregated_df: pd.DataFrame,
                              period: str = 'daily',
                              level_col: str = 'level',
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot aggregated level data.
        
        Args:
            aggregated_df: Aggregated DataFrame
            period: Aggregation period
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if aggregated_df.empty:
            print("No aggregated data available to plot")
            return
        
        if title is None:
            title = f"Level Aggregations ({period.title()})"
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Mean levels over time
        time_col = 'timestamp'
        level_mean_col = f'{level_col}_mean'
        
        if time_col in aggregated_df.columns and level_mean_col in aggregated_df.columns:
            time_grouped = aggregated_df.groupby(time_col)[level_mean_col].mean().reset_index()
            axes[0, 0].plot(time_grouped[time_col], time_grouped[level_mean_col], marker='o')
            axes[0, 0].set_title('Average Level Over Time')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Average Level')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Distribution of mean levels
        if level_mean_col in aggregated_df.columns:
            axes[0, 1].hist(aggregated_df[level_mean_col], bins=20, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Distribution of Mean Levels')
            axes[0, 1].set_xlabel('Mean Level')
            axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Distribution of max levels
        level_max_col = f'{level_col}_max'
        if level_max_col in aggregated_df.columns:
            axes[1, 0].hist(aggregated_df[level_max_col], bins=20, alpha=0.7, color='lightcoral')
            axes[1, 0].set_title('Distribution of Max Levels')
            axes[1, 0].set_xlabel('Max Level')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Level count distribution
        level_count_col = f'{level_col}_count'
        if level_count_col in aggregated_df.columns:
            axes[1, 1].hist(aggregated_df[level_count_col], bins=20, alpha=0.7, color='gold')
            axes[1, 1].set_title('Distribution of Level Counts')
            axes[1, 1].set_xlabel('Level Count')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Level aggregations plot saved to {save_path}")
        
        plt.show()
