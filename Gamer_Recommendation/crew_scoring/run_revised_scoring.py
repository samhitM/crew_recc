#!/usr/bin/env python3
"""
Main execution script for the revised crew scoring system.
This script runs both impression scoring and level calculation.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "impressionScoring")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "levelScoring")))

def main():
    """Main execution function."""
    print("="*60)
    print("REVISED CREW SCORING SYSTEM")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Step 1: Calculate Impression Scores
    print("STEP 1: CALCULATING IMPRESSION SCORES")
    print("-" * 40)
    
    try:
        from revised_impression_calculator import RevisedCrewImpressionCalculator
        
        impression_calc = RevisedCrewImpressionCalculator()
        impression_results = impression_calc.calculate_final_impressions()
        
        if not impression_results.empty:
            # Save impression results
            impression_file = "crew_impressions_revised.csv"
            impression_results.to_csv(impression_file, index=False)
            print(f"✅ Impression scores calculated for {len(impression_results)} users")
            print(f"✅ Results saved to {impression_file}")
            
            # Plot impression results
            try:
                impression_calc.plot_impressions(impression_results, "impression_plots.png")
                print("✅ Impression plots generated")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate impression plots: {e}")
        else:
            print("❌ No impression results generated")
            return
            
    except Exception as e:
        print(f"❌ Error in impression calculation: {e}")
        return
    
    print()
    
    # Step 2: Calculate Crew Levels
    print("STEP 2: CALCULATING CREW LEVELS")
    print("-" * 40)
    
    try:
        from revised_level_calculator import RevisedCrewLevelCalculator
        
        level_calc = RevisedCrewLevelCalculator()
        level_results = level_calc.calculate_crew_levels()
        
        if not level_results.empty:
            # Save level results
            level_file = "crew_levels_revised.csv"
            level_results.to_csv(level_file, index=False)
            print(f"✅ Crew levels calculated for {len(level_results)} users")
            print(f"✅ Results saved to {level_file}")
            
            # Plot level results
            try:
                level_calc.plot_level_distribution(level_results, "level_plots.png")
                print("✅ Level distribution plots generated")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate level plots: {e}")
        else:
            print("❌ No level results generated")
            return
            
    except Exception as e:
        print(f"❌ Error in level calculation: {e}")
        return
    
    print()
    
    # Step 3: Generate Combined Report
    print("STEP 3: GENERATING COMBINED REPORT")
    print("-" * 40)
    
    try:
        # Merge impression and level results
        combined_results = pd.merge(
            impression_results[['user_id', 'pagerank', 'total_impression_score']], 
            level_results[['user_id', 'crew_level', 'composite_score', 'gaming_time']], 
            on='user_id', 
            how='outer'
        )
        
        # Save combined results
        combined_file = "crew_scoring_combined_report.csv"
        combined_results.to_csv(combined_file, index=False)
        print(f"✅ Combined report generated for {len(combined_results)} users")
        print(f"✅ Combined results saved to {combined_file}")
        
        # Print summary statistics
        print("\n📊 SUMMARY STATISTICS")
        print("-" * 25)
        
        if not impression_results.empty:
            print(f"Impression Scores:")
            print(f"  Average: {impression_results['total_impression_score'].mean():.2f}")
            print(f"  Range: {impression_results['total_impression_score'].min():.2f} - {impression_results['total_impression_score'].max():.2f}")
        
        if not level_results.empty:
            print(f"\nCrew Levels:")
            level_distribution = level_results['crew_level'].value_counts().sort_index()
            for level, count in level_distribution.items():
                print(f"  Level {level}: {count} users")
            
            print(f"\nComposite Scores:")
            print(f"  Average: {level_results['composite_score'].mean():.3f}")
            print(f"  Range: {level_results['composite_score'].min():.3f} - {level_results['composite_score'].max():.3f}")
        
        if not combined_results.empty:
            print(f"\nTotal Users Processed: {len(combined_results)}")
        
    except Exception as e:
        print(f"❌ Error generating combined report: {e}")
    
    print()
    print("="*60)
    print("REVISED CREW SCORING SYSTEM COMPLETED")
    print("="*60)
    print(f"Finished at: {datetime.now()}")

if __name__ == "__main__":
    main()
