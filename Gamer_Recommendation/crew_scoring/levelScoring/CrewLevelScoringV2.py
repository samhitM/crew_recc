"""
Crew Level Scoring Calculator
Calculates user level assignments based on gaming activity, impressions, and community metrics.
"""
import warnings
from main_level_calculator import LevelCalculator

warnings.filterwarnings("ignore")

class CrewLevelCalculator:
    """Main calculator for crew member level assignments."""
    
    def __init__(self):
        # Initialize the modular level calculator
        self.calculator = LevelCalculator()
    
    def calculate_final_levels(self):
        """Calculate level assignments for all users."""
        return self.calculator.calculate_crew_levels()

if __name__ == "__main__":
    # Run level scoring
    calculator = CrewLevelCalculator()
    results_df = calculator.calculate_final_levels()
    
    if not results_df.empty:
        print(f"Successfully calculated levels for {len(results_df)} users")
        print("Results saved to crew_levels_revised.csv")
    else:
        print("No results to save")
