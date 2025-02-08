import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scoring import CrewScoreManager
from scoring import CrewScoreCalculator
from impressionScoring.utils.aggregator import Aggregator
from impressionScoring.utils.plotter import Plotter
from database import fetch_all_user_ids
from updater import CrewImpressionUpdater
from core.config import API_BASE_URL

if __name__ == "__main__":
    api_url = API_BASE_URL
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI4UHpJOG5FUXU1TCIsImVtYWlsIjoieWFzaHlhZGF2MDBAZmxhc2guY28iLCJpYXQiOjE3MzIxOTE3MjYsImV4cCI6MTczNzM3NTcyNn0.DeRiGzNUflr6_8CSqrw3K7UkybEb8pJe9ocD9Gs5Axs"
    user_ids = fetch_all_user_ids()[:2]

    manager = CrewScoreManager(api_url=api_url,jwt_token=jwt_token,user_ids=user_ids)
    all_user_data = manager.fetch_complete_user_data()
    print("All_users_data")
    print(all_user_data)
    
    crew_score_calculator = CrewScoreCalculator()

    users_scores=crew_score_calculator.get_user_scores(all_user_data=all_user_data)
    print("User score:",users_scores)
    
    # Update the database with the new level and score
    print("Updating database with the new impression...")
    updates = []
    for user_id, score_data in users_scores.items():
        updates.append({
            "user_id": user_id,
            "crew_impression": score_data["Total_Score"]
        })
    
    aggregates = []
    for user_id, score_data in users_scores.items():
        aggregates.append({
            "user_id": user_id,
            "crew_impression": score_data["Total_Score"],
            "Timestamp": "2024-07-25 11:15:00"
        })
    
    print(aggregates)
    
    crew_updater = CrewImpressionUpdater()
    crew_updater.update_crew_impressions(updates=updates)
    print("Update completed...")
    
    # Aggregate impressions
    aggregator=Aggregator()
    aggregated_data = aggregator.aggregate_impressions(aggregates)
    print("Aggregated Impressions:")
    print(aggregated_data)

   # Create a Plotter instance
    plotter = Plotter()

    # Call the method to plot the impression analysis and score distributions
    plotter.plot_impression_and_score_analysis(aggregated_data, users_scores, user_ids)
        