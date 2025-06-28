"""
Utility functions for level scoring.
"""
import pandas as pd
from typing import Dict


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
