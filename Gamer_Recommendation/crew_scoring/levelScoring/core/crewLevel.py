import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from scoring import CrewScoreManager
from thresholds import ThresholdCalculator
from database import fetch_all_user_ids
from assigner import CrewLevelAssigner
from updater import CrewLevelUpdater

def main():
    """
    Main function to fetch user data, calculate crew levels, and update the database with the new levels.
    """
    # Initialize CrewScoreManager
    manager = CrewScoreManager()

    # Fetch and normalize data for all users
    print("Fetching and normalizing data for all users...")
    manager.fetch_and_normalize_all_data()

    # Fetch all user scores
    print("Fetching scores for all users...")
    all_user_ids = fetch_all_user_ids()
    all_scores = [manager.get_user_score(user) for user in all_user_ids]
    print(f"All scores: {all_scores}")

    # Select the best threshold method
    print("Selecting the best threshold calculation method...")
    best_method = ThresholdCalculator.select_best_method(all_scores)
    print(f"Best method selected: {best_method}")

    # Calculate thresholds
    print("Calculating thresholds...")
    thresholds = ThresholdCalculator.calculate_thresholds(all_scores, method=best_method, clusters=2)
    print(f"Calculated thresholds: {thresholds}")

    # Assign crew levels
    print("Assigning crew levels...")
    crew_assigner = CrewLevelAssigner()  # Create an instance
    crew_levels = crew_assigner.assign_crew_level(all_user_ids, all_scores, thresholds)

    # Prepare data for update
    print("Preparing data for update...")
    update_records = [
        {"user_id": user_id, "crew_level": crew_levels[idx]}
        for idx, user_id in enumerate(all_user_ids)
    ]

    # Perform the update
    print("Updating database with the new levels...")
    crew_updater = CrewLevelUpdater()

    # Call the method on the instance
    crew_updater.update_crew_levels(update_records)
    print(f"Updated {len(update_records)} users' crew levels.")

if __name__ == "__main__":
    main()