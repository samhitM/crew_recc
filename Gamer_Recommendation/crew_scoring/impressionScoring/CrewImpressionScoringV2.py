"""
Refactored Crew Impression Scoring - Now uses modular components.
This file has been refactored from 800+ lines to use modular components.

The original monolithic code has been split into:
- database/db_manager.py - Database operations
- graph/graph_manager.py - Graph construction and metrics
- scoring/scoring_manager.py - Scoring calculations
- utils/helpers.py - Helper functions
- main_impression_calculator.py - Main orchestrator
"""
import warnings
from main_impression_calculator import ImpressionCalculator

warnings.filterwarnings("ignore")

class StandaloneCrewImpressionCalculator:
    """
    Refactored wrapper class that uses modular components.
    Original 800+ lines reduced to simple delegation.
    """
    
    def __init__(self):
        self.calculator = ImpressionCalculator()
    
    def calculate_final_impressions(self):
        """Calculate final impressions using modular components."""
        return self.calculator.calculate_final_impressions()

if __name__ == "__main__":
    # Use the modular calculator
    calculator = StandaloneCrewImpressionCalculator()
    results_df = calculator.calculate_final_impressions()
    
    if not results_df.empty:
        print(f"Successfully calculated impressions for {len(results_df)} users")
        print("Results saved to crew_impressions_revised.csv")
    else:
        print("No results to save")
