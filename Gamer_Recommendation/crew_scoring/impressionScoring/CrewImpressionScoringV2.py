"""
Crew Impression Scoring Calculator
Calculates user impression scores based on graph metrics and user features.
"""
import warnings
from main_impression_calculator import ImpressionCalculator

warnings.filterwarnings("ignore")

class CrewImpressionCalculator:
    """Main calculator for crew member impression scores."""
    
    def __init__(self):
        # Initialize the modular impression calculator
        self.calculator = ImpressionCalculator()
    
    def calculate_final_impressions(self):
        """Calculate impression scores for all users."""
        return self.calculator.calculate_final_impressions()

if __name__ == "__main__":
    # Run impression scoring
    calculator = CrewImpressionCalculator()
    results_df = calculator.calculate_final_impressions()
    
    if not results_df.empty:
        print(f"Successfully calculated impressions for {len(results_df)} users")
        print("Results saved to crew_impressions_revised.csv")
    else:
        print("No results to save")
