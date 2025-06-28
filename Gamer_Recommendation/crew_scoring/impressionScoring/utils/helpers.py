"""
Utility functions for impression scoring.
"""
import numpy as np
import pandas as pd
from typing import Dict


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
