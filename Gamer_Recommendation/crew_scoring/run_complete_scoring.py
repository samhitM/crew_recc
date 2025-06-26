#!/usr/bin/env python3
"""
Combined execution script for the revised crew scoring system.
This script runs both impression scoring and level calculation without dependencies.
"""

import os
import pandas as pd
from datetime import datetime
import sys

def run_impression_calculator():
    """Run the impression calculator."""
    print("STEP 1: CALCULATING IMPRESSION SCORES")
    print("-" * 40)
    
    try:
        # Import and run the standalone impression calculator
        from standalone_impression_calculator import StandaloneCrewImpressionCalculator
        
        impression_calc = StandaloneCrewImpressionCalculator()
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
            
            return True
        else:
            print("❌ No impression results generated")
            return False
            
    except Exception as e:
        print(f"❌ Error in impression calculation: {e}")
        return False

def run_level_calculator():
    """Run the level calculator."""
    print("STEP 2: CALCULATING CREW LEVELS")
    print("-" * 40)
    
    try:
        # Import and run the standalone level calculator
        from standalone_level_calculator import StandaloneCrewLevelCalculator
        
        level_calc = StandaloneCrewLevelCalculator()
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
            
            return True
        else:
            print("❌ No level results generated")
            return False
            
    except Exception as e:
        print(f"❌ Error in level calculation: {e}")
        return False

def generate_combined_report():
    """Generate a combined report from both calculations."""
    print("STEP 3: GENERATING COMBINED REPORT")
    print("-" * 40)
    
    try:
        # Read both CSV files
        impression_file = "crew_impressions_revised.csv"
        level_file = "crew_levels_revised.csv"
        
        if not os.path.exists(impression_file) or not os.path.exists(level_file):
            print("❌ Missing required CSV files for combined report")
            return False
        
        impression_df = pd.read_csv(impression_file)
        level_df = pd.read_csv(level_file)
        
        # Merge the results
        combined_results = pd.merge(
            impression_df[['user_id', 'pagerank', 'total_impression_score']], 
            level_df[['user_id', 'crew_level', 'composite_score', 'gaming_time']], 
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
        
        if not impression_df.empty:
            print(f"Impression Scores:")
            print(f"  Average: {impression_df['total_impression_score'].mean():.2f}")
            print(f"  Range: {impression_df['total_impression_score'].min():.2f} - {impression_df['total_impression_score'].max():.2f}")
        
        if not level_df.empty:
            print(f"\nCrew Levels:")
            level_distribution = level_df['crew_level'].value_counts().sort_index()
            for level, count in level_distribution.items():
                print(f"  Level {level}: {count} users")
            
            print(f"\nComposite Scores:")
            print(f"  Average: {level_df['composite_score'].mean():.3f}")
            print(f"  Range: {level_df['composite_score'].min():.3f} - {level_df['composite_score'].max():.3f}")
        
        if not combined_results.empty:
            print(f"\nTotal Users Processed: {len(combined_results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating combined report: {e}")
        return False

def update_database():
    """Update the database with the calculated scores."""
    print("STEP 4: UPDATING DATABASE")
    print("-" * 40)
    
    try:
        # Import and run the database updater
        from database_updater import CrewScoringUpdater
        
        updater = CrewScoringUpdater()
        
        # Update from CSV files
        impression_file = "crew_impressions_revised.csv"
        level_file = "crew_levels_revised.csv"
        
        updater.update_from_csv_files(impression_file, level_file)
        print("✅ Database update completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        return False

def main():
    """Main execution function."""
    print("="*60)
    print("REVISED CREW SCORING SYSTEM")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success_count = 0
    
    # Step 1: Calculate Impression Scores
    if run_impression_calculator():
        success_count += 1
    print()
    
    # Step 2: Calculate Crew Levels
    if run_level_calculator():
        success_count += 1
    print()
    
    # Step 3: Generate Combined Report
    if generate_combined_report():
        success_count += 1
    print()
    
    # Final summary
    print("="*60)
    print("REVISED CREW SCORING SYSTEM COMPLETED")
    print("="*60)
    print(f"Successfully completed {success_count}/3 steps")
    print(f"Finished at: {datetime.now()}")
    
    if success_count == 3:
        print("🎉 All steps completed successfully!")
        print("\n📁 Generated Files:")
        print("  - crew_impressions_revised.csv (impression scores)")
        print("  - crew_levels_revised.csv (crew levels)")
        print("  - crew_scoring_combined_report.csv (combined results)")
        print("  - impression_plots.png (impression visualizations)")
        print("  - level_plots.png (level distribution plots)")
    else:
        print("⚠️  Some steps failed. Check the output above for details.")

if __name__ == "__main__":
    main()
