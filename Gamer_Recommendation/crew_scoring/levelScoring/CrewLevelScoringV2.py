"""
Refactored Crew Level Scoring - Now uses modular components.
This file has been refactored from 1149+ lines to use modular components.

The original monolithic code has been split into:
- database/level_db_manager.py - Database operations
- graph/level_graph_manager.py - Graph construction and metrics
- scoring/level_scoring_manager.py - Scoring calculations
- clustering/level_clustering_manager.py - KNN and hybrid clustering
- utils/level_helpers.py - Helper functions
- main_level_calculator.py - Main orchestrator
"""
import warnings
from main_level_calculator import LevelCalculator

warnings.filterwarnings("ignore")

class StandaloneCrewLevelCalculator:
    """
    Refactored wrapper class that uses modular components.
    Original 1149+ lines reduced to simple delegation.
    """
    
    def __init__(self):
        self.calculator = LevelCalculator()
    
    def calculate_final_levels(self):
        """Calculate final levels using modular components."""
        return self.calculator.calculate_crew_levels()

if __name__ == "__main__":
    # Use the modular calculator
    calculator = StandaloneCrewLevelCalculator()
    results_df = calculator.calculate_final_levels()
    
    if not results_df.empty:
        print(f"Successfully calculated levels for {len(results_df)} users")
        print("Results saved to crew_levels_revised.csv")
    else:
        print("No results to save")
