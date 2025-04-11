import threading
import time
from datetime import datetime, timedelta
import pytz
import sys
import os


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from crew_scoring.impressionScoring.core.scoring.score_calculator import CrewScoreCalculator
from crew_scoring.impressionScoring.core.scoring.crew_score_manager import CrewScoreManager
from crew_scoring.impressionScoring.core.updater.crew_impression_updater import CrewImpressionUpdater
from crew_scoring.impressionScoring.utils.aggregator import Aggregator
from crew_scoring.impressionScoring.utils.plotter import Plotter
from core.config import API_BASE_URL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from database import fetch_all_user_ids

def periodic_crew_impression_update():
    ist = pytz.timezone('Asia/Kolkata')
    interval_hours = 8

    while True:
        now = datetime.now(ist)
        first_run = now.replace(hour=23, minute=00, second=0, microsecond=0)
        if now > first_run:
            first_run += timedelta(days=1)

        next_run_time = first_run
        while next_run_time <= now:
            next_run_time += timedelta(hours=interval_hours)

        wait_seconds = (next_run_time - now).total_seconds()
        print(f"[Crew] Next update at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
        time.sleep(wait_seconds)

        try:
            print(f"[Crew] Update started at: {datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')}")
            user_ids = fetch_all_user_ids()
            manager = CrewScoreManager(api_url=API_BASE_URL, user_ids=user_ids)
            all_user_data = manager.fetch_complete_user_data()
            
            calculator = CrewScoreCalculator()
            user_scores = calculator.get_user_scores(all_user_data=all_user_data)

            # Prepare DB updates
            updates = [{"user_id": uid, "crew_impression": score["Total_Score"]}
                       for uid, score in user_scores.items()]
            updater = CrewImpressionUpdater()
            updater.update_crew_impressions(updates=updates)
            print("[Crew] DB update done.")

            # Prepare aggregates for plotting
            timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
            aggregates = [{"user_id": uid, "crew_impression": score["Total_Score"], "Timestamp": timestamp}
                          for uid, score in user_scores.items()]
            aggregator = Aggregator()
            aggregated_data = aggregator.aggregate_impressions(aggregates)

            # # Plot results
            # plotter = Plotter()
            # plotter.plot_impression_and_score_analysis(aggregated_data, user_scores, user_ids)

            print("[Crew] Aggregation and plotting completed.\n")

        except Exception as e:
            print(f"[Crew] Error in periodic update: {e}")