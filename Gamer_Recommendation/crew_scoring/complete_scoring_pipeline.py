#!/usr/bin/env python3
"""
Complete Crew Scoring Pipeline
Runs impression scoring first, then level scoring with link prediction.
"""

import sys
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def main():
    """Run the complete scoring pipeline."""
    print("=" * 70)
    print("COMPLETE CREW SCORING PIPELINE")
    print("=" * 70)
    
    # Step 1: Run impression scoring
    print("\n🔄 STEP 1: Running Impression Scoring...")
    print("-" * 50)
    
    try:
        # Import and run the revised impression calculator
        sys.path.append(os.path.join(os.path.dirname(__file__), 'impressionScoring'))
        from revised_impression_calculator import RevisedCrewImpressionCalculator
        
        # Check if impression CSV already exists
        impression_file = "crew_impressions_revised.csv"
        if os.path.exists(impression_file):
            print(f"✓ Impression scores already exist: {impression_file}")
            print("  Loading existing file...")
            impression_df = pd.read_csv(impression_file)
            print(f"  Loaded {len(impression_df)} user impression scores")
        else:
            print("  Calculating new impression scores...")
            calculator = RevisedCrewImpressionCalculator()
            impression_df = calculator.calculate_final_impressions()
            
            if not impression_df.empty:
                impression_df.to_csv(impression_file, index=False)
                print(f"✓ Impression scores saved to {impression_file}")
                print(f"  Calculated for {len(impression_df)} users")
            else:
                print("✗ No impression scores calculated")
                return
    
    except Exception as e:
        print(f"✗ Impression scoring failed: {e}")
        return
    
    # Step 2: Run level scoring with link prediction
    print("\\n🔄 STEP 2: Running Level Scoring with Link Prediction...")
    print("-" * 50)
    
    try:
        from standalone_level_calculator import StandaloneCrewLevelCalculator
        
        # Check if level CSV already exists
        level_file = "crew_levels_revised.csv"
        if os.path.exists(level_file):
            print(f"✓ Level scores already exist: {level_file}")
            print("  Loading existing file...")
            level_df = pd.read_csv(level_file)
            print(f"  Loaded {len(level_df)} user level assignments")
        else:
            print("  Calculating new level scores...")
            calculator = StandaloneCrewLevelCalculator()
            
            # Use KNN thresholding as default for better results
            calculator.threshold_method = 'knn'
            
            level_df = calculator.calculate_crew_levels()
            
            if not level_df.empty:
                level_df.to_csv(level_file, index=False)
                print(f"✓ Level scores saved to {level_file}")
                print(f"  Calculated for {len(level_df)} users")
            else:
                print("✗ No level scores calculated")
                return
    
    except Exception as e:
        print(f"✗ Level scoring failed: {e}")
        return
    
    # Step 3: Create combined report
    print("\\n🔄 STEP 3: Creating Combined Report...")
    print("-" * 50)
    
    try:
        # Merge impression and level data
        combined_df = impression_df.merge(
            level_df[['user_id', 'crew_level', 'gaming_score', 'community_score', 
                     'link_prediction_score', 'bonus_score', 'composite_score']], 
            on='user_id', 
            how='outer'
        )
        
        # Save combined report
        combined_file = "crew_scoring_complete_report.csv"
        combined_df.to_csv(combined_file, index=False)
        
        print(f"✓ Combined report saved to {combined_file}")
        print(f"  Combined data for {len(combined_df)} users")
        
        # Print summary statistics
        print("\\n📊 SCORING SUMMARY:")
        print("-" * 30)
        print(f"Total users processed: {len(combined_df)}")
        print(f"Impression scores range: {impression_df['total_impression_score'].min():.3f} - {impression_df['total_impression_score'].max():.3f}")
        print(f"Composite scores range: {level_df['composite_score'].min():.3f} - {level_df['composite_score'].max():.3f}")
        
        # Level distribution
        print(f"\\nLevel distribution:")
        level_counts = level_df['crew_level'].value_counts().sort_index()
        for level, count in level_counts.items():
            percentage = (count / len(level_df)) * 100
            print(f"  Level {level}: {count} users ({percentage:.1f}%)")
        
        # Component score averages
        print(f"\\nAverage component scores:")
        components = ['gaming_score', 'impression_score', 'community_score', 'link_prediction_score', 'bonus_score']
        for component in components:
            if component in level_df.columns:
                avg_score = level_df[component].mean()
                print(f"  {component.replace('_', ' ').title()}: {avg_score:.3f}")
        
    except Exception as e:
        print(f"✗ Combined report failed: {e}")
        return
    
    # Step 4: Generate visualizations
    print("\\n🔄 STEP 4: Generating Visualizations...")
    print("-" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Complete Crew Scoring Analysis', fontsize=16)
        
        # Plot 1: Impression Score Distribution
        axes[0, 0].hist(impression_df['total_impression_score'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('Impression Score Distribution')
        axes[0, 0].set_xlabel('Total Impression Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Level Distribution
        level_counts = level_df['crew_level'].value_counts().sort_index()
        axes[0, 1].bar(level_counts.index, level_counts.values, alpha=0.7, color='green')
        axes[0, 1].set_title('Crew Level Distribution')
        axes[0, 1].set_xlabel('Crew Level')
        axes[0, 1].set_ylabel('Number of Users')
        
        # Plot 3: Component Scores by Level
        components = ['gaming_score', 'impression_score', 'community_score', 'link_prediction_score']
        level_means = level_df.groupby('crew_level')[components].mean()
        
        x = range(len(level_means.index))
        width = 0.2
        
        for i, component in enumerate(components):
            if component in level_means.columns:
                axes[0, 2].bar([pos + i*width for pos in x], level_means[component], 
                              width, label=component.replace('_', ' ').title(), alpha=0.8)
        
        axes[0, 2].set_title('Average Component Scores by Level')
        axes[0, 2].set_xlabel('Crew Level')
        axes[0, 2].set_ylabel('Average Score')
        axes[0, 2].set_xticks([pos + width*1.5 for pos in x])
        axes[0, 2].set_xticklabels(level_means.index)
        axes[0, 2].legend()
        
        # Plot 4: Impression vs Composite Score Correlation
        merged_scores = combined_df[['total_impression_score', 'composite_score']].dropna()
        axes[1, 0].scatter(merged_scores['total_impression_score'], merged_scores['composite_score'], alpha=0.6)
        axes[1, 0].set_title('Impression vs Composite Score')
        axes[1, 0].set_xlabel('Total Impression Score')
        axes[1, 0].set_ylabel('Composite Score')
        
        # Plot 5: Link Prediction Score Distribution
        if 'link_prediction_score' in level_df.columns:
            axes[1, 1].hist(level_df['link_prediction_score'], bins=20, alpha=0.7, color='orange')
            axes[1, 1].set_title('Link Prediction Score Distribution')
            axes[1, 1].set_xlabel('Link Prediction Score')
            axes[1, 1].set_ylabel('Frequency')
        
        # Plot 6: PageRank vs Crew Level
        if 'pagerank' in combined_df.columns and 'crew_level' in combined_df.columns:
            for level in sorted(combined_df['crew_level'].unique()):
                level_data = combined_df[combined_df['crew_level'] == level]
                axes[1, 2].scatter([level] * len(level_data), level_data['pagerank'], 
                                 alpha=0.6, label=f'Level {level}')
            axes[1, 2].set_title('PageRank by Crew Level')
            axes[1, 2].set_xlabel('Crew Level')
            axes[1, 2].set_ylabel('PageRank Score')
            axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = "crew_scoring_complete_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualizations saved to {viz_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
    
    print("\\n" + "=" * 70)
    print("✅ COMPLETE CREW SCORING PIPELINE FINISHED")
    print("=" * 70)
    print("\\nFiles generated:")
    print(f"  📄 {impression_file} - Impression scores with PageRank, topological scores")
    print(f"  📄 {level_file} - Level assignments with link prediction")
    print(f"  📄 {combined_file} - Combined report")
    print(f"  📊 {viz_file} - Comprehensive visualizations")
    print("\\n🔬 Key Features Implemented:")
    print("  ✓ Graph-based impression scoring (PageRank, K-Shell, Out-Degree)")
    print("  ✓ Data-driven feature weights (Linear Regression)")
    print("  ✓ Link prediction using Jaccard coefficient and Katz centrality")
    print("  ✓ Enhanced community detection with link prediction")
    print("  ✓ KNN-based threshold selection for level assignment")
    print("  ✓ Integrated impression scores in level calculation")

if __name__ == "__main__":
    main()
