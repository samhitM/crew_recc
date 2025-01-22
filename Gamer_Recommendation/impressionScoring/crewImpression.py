import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.crew_score_manager import CrewScoreManager
from utils.score_calculator import CrewScoreCalculator
from utils.aggregator import Aggregator
from utils.plotter import Plotter
from core.database import fetch_all_user_ids, update_crew_impressions

if __name__ == "__main__":
    api_url = "https://localhost:3000/api"
    jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiI4UHpJOG5FUXU1TCIsImVtYWlsIjoieWFzaHlhZGF2MDBAZmxhc2guY28iLCJpYXQiOjE3MzIxOTE3MjYsImV4cCI6MTczNzM3NTcyNn0.DeRiGzNUflr6_8CSqrw3K7UkybEb8pJe9ocD9Gs5Axs"
    user_ids = fetch_all_user_ids()[:3]
    print(user_ids)

    manager = CrewScoreManager(api_url=api_url,jwt_token=jwt_token)
    user_data = manager.fetch_complete_user_data(user_ids[0])
    print(user_data)
    
    crew_score_calculator = CrewScoreCalculator()

    user_score_list=[crew_score_calculator.update_user_scores(user_data=user_data)]
    print("User score:",user_score_list)
    
    # Aggregate impressions
    aggregator=Aggregator()
    aggregated_data = aggregator.aggregate_impressions(user_score_list)
    print("Aggregated Impressions:")
    print(aggregated_data)

    # Plot distributions
    plotter=Plotter()
    plotter.plot_distributions(user_score_list, user_ids)
    
    # Update the database with the new level and score
    print("Updating database with the new impression...")
    update_crew_impressions(user_ids[0], user_score_list[0]['Total_Score'])
    print(f"Updated User ID {user_ids[0]}: Impression Score = {user_score_list[0]['Total_Score']:.2f}")
    
    