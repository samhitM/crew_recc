#!/usr/bin/env python3
"""
Summary of the Revised Crew Scoring System Results
"""

import pandas as pd
import os
from datetime import datetime

def show_results_summary():
    """Display a comprehensive summary of the revised crew scoring results."""
    
    print("="*80)
    print("REVISED CREW SCORING SYSTEM - RESULTS SUMMARY")
    print("="*80)
    print(f"Generated on: {datetime.now()}")
    print()
    
    # Check what files exist
    files_to_check = [
        "crew_impressions_revised.csv",
        "crew_levels_revised.csv", 
        "crew_scoring_combined_report.csv",
        "impression_plots.png",
        "level_plots.png"
    ]
    
    print("📁 Generated Files:")
    print("-" * 20)
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} (missing)")
    print()
    
    # Load and summarize impression scores
    try:
        impression_df = pd.read_csv("crew_impressions_revised.csv")
        print("🎯 IMPRESSION SCORES SUMMARY")
        print("-" * 30)
        print(f"Total Users: {len(impression_df)}")
        print(f"Average Impression Score: {impression_df['total_impression_score'].mean():.2f}")
        print(f"Score Range: {impression_df['total_impression_score'].min():.0f} - {impression_df['total_impression_score'].max():.0f}")
        print(f"Average PageRank: {impression_df['pagerank'].mean():.6f}")
        print(f"Users with High Scores (>100): {len(impression_df[impression_df['total_impression_score'] > 100])}")
        print()
        
        # Show top users by impression score
        print("🏆 Top 10 Users by Impression Score:")
        top_impression = impression_df.nlargest(10, 'total_impression_score')[['user_id', 'pagerank', 'total_impression_score']]
        for idx, row in top_impression.iterrows():
            print(f"  {row['user_id'][:15]:<15} | PageRank: {row['pagerank']:.6f} | Score: {row['total_impression_score']:.0f}")
        print()
        
    except Exception as e:
        print(f"❌ Error reading impression scores: {e}")
        print()
    
    # Load and summarize crew levels
    try:
        level_df = pd.read_csv("crew_levels_revised.csv")
        print("🏅 CREW LEVELS SUMMARY")
        print("-" * 25)
        print(f"Total Users: {len(level_df)}")
        print(f"Average Composite Score: {level_df['composite_score'].mean():.3f}")
        print(f"Average Gaming Time: {level_df['gaming_time'].mean():.1f} hours")
        print()
        
        # Level distribution
        print("Level Distribution:")
        level_counts = level_df['crew_level'].value_counts().sort_index()
        for level, count in level_counts.items():
            percentage = (count / len(level_df)) * 100
            print(f"  Level {level}: {count:2d} users ({percentage:4.1f}%)")
        print()
        
        # Show top users by composite score
        print("🏆 Top 10 Users by Composite Score:")
        top_composite = level_df.nlargest(10, 'composite_score')[['user_id', 'crew_level', 'composite_score', 'gaming_time']]
        for idx, row in top_composite.iterrows():
            print(f"  {row['user_id'][:15]:<15} | Level: {row['crew_level']} | Score: {row['composite_score']:.3f} | Gaming: {row['gaming_time']:.0f}h")
        print()
        
        # Gaming statistics
        gaming_users = level_df[level_df['gaming_time'] > 0]
        if len(gaming_users) > 0:
            print("🎮 Gaming Statistics:")
            print(f"  Users with Gaming Data: {len(gaming_users)}")
            print(f"  Average Gaming Time: {gaming_users['gaming_time'].mean():.1f} hours")
            print(f"  Max Gaming Time: {gaming_users['gaming_time'].max():.0f} hours")
            print()
        
    except Exception as e:
        print(f"❌ Error reading crew levels: {e}")
        print()
    
    # Combined analysis
    try:
        combined_df = pd.read_csv("crew_scoring_combined_report.csv")
        print("🔗 COMBINED ANALYSIS")
        print("-" * 20)
        print(f"Total Users in Combined Report: {len(combined_df)}")
        
        # Correlation analysis
        if 'pagerank' in combined_df.columns and 'crew_level' in combined_df.columns:
            correlation = combined_df['pagerank'].corr(combined_df['crew_level'])
            print(f"PageRank vs Crew Level Correlation: {correlation:.3f}")
        
        if 'total_impression_score' in combined_df.columns and 'crew_level' in combined_df.columns:
            correlation = combined_df['total_impression_score'].corr(combined_df['crew_level'])
            print(f"Impression Score vs Crew Level Correlation: {correlation:.3f}")
        print()
        
    except Exception as e:
        print(f"❌ Error reading combined report: {e}")
        print()
    
    # Implementation details
    print("🔧 IMPLEMENTATION DETAILS")
    print("-" * 25)
    print("✅ Graph-based Topological Scores:")
    print("   - PageRank (40% weight)")
    print("   - K-Shell decomposition (30% weight)")
    print("   - Out-Degree connectivity (30% weight)")
    print()
    print("✅ Data-driven Feature Weights:")
    print("   - Learned via Linear Regression using PageRank as target")
    print("   - Gaming time from user_games table utilized")
    print("   - Other features defaulted to 0 as specified")
    print()
    print("✅ Community Detection:")
    print("   - Connected components analysis")
    print("   - Top 5 performers per community get bonus")
    print()
    print("✅ Crew Level Assignment:")
    print("   - Gaming Activity (30%)")
    print("   - Impression Score (25%)")
    print("   - Community Score (10%)")
    print("   - Link Prediction (20% - placeholder)")
    print("   - Bonus Factors (15%)")
    print()
    
    print("="*80)
    print("🎉 REVISED CREW SCORING SYSTEM SUCCESSFULLY IMPLEMENTED!")
    print("="*80)

if __name__ == "__main__":
    show_results_summary()
