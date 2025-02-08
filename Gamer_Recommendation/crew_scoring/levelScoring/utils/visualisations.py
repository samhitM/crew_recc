import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from levelScoring.core.thresholds import ThresholdCalculator
from levelScoring.core.assigner import CrewLevelAssigner

# Example Scores and Levels (Using Threshold Calculation)
scores = np.random.normal(50, 15, 10000)  # Larger sample size for visualization

# Calculate thresholds and levels
thresholds = ThresholdCalculator.calculate_thresholds(scores, clusters=4)[:3]
levels = CrewLevelAssigner.assign_crew_level(scores, thresholds)

# Visualization Functions

def plot_score_distribution(scores, thresholds):
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=20, kde=True, color='skyblue')
    for t in thresholds:
        plt.axvline(t, color='red', linestyle='--', label=f'Threshold: {t:.2f}')
    plt.title('Score Distribution with Crew Level Thresholds')
    plt.xlabel('Scores')
    plt.ylabel('Number of Players')
    plt.legend()
    plt.show()

def plot_boxplot_by_level(scores, levels):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=levels, y=scores, hue=levels, palette='Set2', legend=False)
    plt.title('Box Plot of Scores by Crew Level')
    plt.xlabel('Crew Level')
    plt.ylabel('Scores')
    plt.show()

def plot_progress(player_scores, thresholds):
    plt.figure(figsize=(10, 6))
    plt.plot(player_scores, marker='o', color='blue')
    for t in thresholds:
        plt.axhline(t, color='red', linestyle='--')
    plt.title('Player Score Progression Over Time')
    plt.xlabel('Time (e.g., Game Sessions)')
    plt.ylabel('Score')
    plt.show()

def plot_radar_chart(metrics, player_name='Player 1'):
    from math import pi
    categories = list(metrics.keys())
    values = list(metrics.values())

    # Number of variables
    N = len(categories)

    # Create the radar chart
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories)

    # Plot the metrics
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)

    plt.title(f'{player_name} Performance Metrics')
    plt.show()

# Metrics for radar chart
player_metrics = {"Playtime": 80, "Achievements": 60, "Crew Participation": 70, "Consistency": 65}

# Call the functions
plot_score_distribution(scores, thresholds)
plot_boxplot_by_level(scores, levels)
plot_progress([45, 50, 55, 60, 70], thresholds)
plot_radar_chart(player_metrics)
