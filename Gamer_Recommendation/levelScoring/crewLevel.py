import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.crew_score_manager import CrewScoreManager
from utils.threshold_calculator import ThresholdCalculator
from core.database import fetch_all_user_ids
from utils.crew_level_assigner import CrewLevelAssigner
from core.database import update_crew_level

def main():
    # Initialize CrewScoreManager
    manager = CrewScoreManager()

    # Fetch and normalize data for all users
    print("Fetching and normalizing data for all users...")
    manager.fetch_and_normalize_all_data()

    # Example: Get score for a specific user
    user_id = "7irNPR6kOia"
    score = manager.get_user_score(user_id)
    print(f"Composite score for User ID {user_id}: {score:.2f}")

    # Fetch all user scores to determine the best threshold method
    print("Fetching scores for all users...")
    all_user_ids = fetch_all_user_ids()[:3]  # Fetch a subset of user IDs for demonstration
    all_scores = [manager.get_user_score(user) for user in all_user_ids]
    print(f"All scores: {all_scores}")

    # Select the best threshold calculation method
    print("Selecting the best threshold calculation method...")
    best_method = ThresholdCalculator.select_best_method(all_scores)
    print(f"Best method selected: {best_method}")

    # Calculate thresholds using the selected method
    print("Calculating thresholds...")
    thresholds = ThresholdCalculator.calculate_thresholds(all_scores, method=best_method, clusters=2)
    print(f"Calculated thresholds: {thresholds}")

    # Assign crew level for the specific user
    print("Assigning crew level...")
    crew_level = CrewLevelAssigner.assign_crew_level(all_scores, thresholds)
    print(f"Crew Level for User ID {user_id}: {crew_level}")

    # Update the database with the new level and score
    print("Updating database with the new level and score...")
    
    update_crew_level(user_id, crew_level[0], score)
    print(f"Updated User ID {user_id}: Composite Score = {score:.2f}, Crew Level = {crew_level[0]}")

if __name__ == "__main__":
    main()
