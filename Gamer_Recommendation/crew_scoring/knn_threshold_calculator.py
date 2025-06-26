import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class KNNBasedThresholdCalculator:
    """
    KNN-based threshold calculation for crew levels using clustering and neighborhood analysis.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.optimal_clusters = 5
        self.knn_neighbors = 5
    
    def find_optimal_clusters(self, scores: List[float], max_clusters: int = 8) -> int:
        """Find optimal number of clusters using silhouette score and elbow method."""
        if len(scores) < 4:
            return min(len(scores), 3)
        
        scores_array = np.array(scores).reshape(-1, 1)
        
        # Normalize scores
        scores_normalized = self.scaler.fit_transform(scores_array)
        
        silhouette_scores = []
        inertias = []
        cluster_range = range(2, min(max_clusters + 1, len(scores)))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scores_normalized)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(scores_normalized, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
        
        # Find optimal clusters using silhouette score
        if silhouette_scores:
            optimal_idx = np.argmax(silhouette_scores)
            optimal_clusters = list(cluster_range)[optimal_idx]
            
            print(f"Silhouette scores: {dict(zip(cluster_range, silhouette_scores))}")
            print(f"Optimal clusters (silhouette): {optimal_clusters}")
            
            return optimal_clusters
        
        return 5  # Default fallback
    
    def calculate_knn_thresholds(self, composite_scores: Dict[str, float], 
                                num_levels: int = None) -> List[float]:
        """Calculate thresholds using KNN-based clustering approach."""
        print("Calculating KNN-based thresholds...")
        
        if not composite_scores:
            return []
        
        scores = list(composite_scores.values())
        user_ids = list(composite_scores.keys())
        
        if len(scores) < 4:
            print("Too few scores for KNN clustering, using percentile method")
            return self._fallback_percentile_thresholds(scores, num_levels or 5)
        
        # Step 1: Find optimal number of clusters
        if num_levels is None:
            num_levels = self.find_optimal_clusters(scores)
        else:
            num_levels = min(num_levels, len(scores))
        
        print(f"Using {num_levels} levels for threshold calculation")
        
        # Step 2: Perform K-means clustering on scores
        scores_array = np.array(scores).reshape(-1, 1)
        scores_normalized = self.scaler.fit_transform(scores_array)
        
        kmeans = KMeans(n_clusters=num_levels, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scores_normalized)
        
        # Step 3: Create feature matrix for KNN
        # Use multiple features for better clustering
        features = []
        for i, score in enumerate(scores):
            features.append([
                score,  # Original composite score
                i / len(scores),  # Rank position (normalized)
                scores_normalized[i][0],  # Normalized score
            ])
        
        features_array = np.array(features)
        
        # Step 4: Train KNN classifier on cluster assignments
        knn = KNeighborsClassifier(n_neighbors=min(self.knn_neighbors, len(scores) // 2))
        knn.fit(features_array, cluster_labels)
        
        # Step 5: Calculate cluster centroids and boundaries
        cluster_info = {}
        for level in range(num_levels):
            cluster_mask = cluster_labels == level
            cluster_scores = np.array(scores)[cluster_mask]
            
            if len(cluster_scores) > 0:
                cluster_info[level] = {
                    'min_score': cluster_scores.min(),
                    'max_score': cluster_scores.max(),
                    'mean_score': cluster_scores.mean(),
                    'size': len(cluster_scores)
                }
        
        # Step 6: Calculate thresholds as boundaries between clusters
        # Sort clusters by mean score
        sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['mean_score'])
        
        thresholds = []
        for i in range(len(sorted_clusters) - 1):
            current_cluster = sorted_clusters[i][1]
            next_cluster = sorted_clusters[i + 1][1]
            
            # Threshold is midpoint between cluster boundaries
            threshold = (current_cluster['max_score'] + next_cluster['min_score']) / 2
            thresholds.append(threshold)
        
        print(f"KNN-based thresholds: {thresholds}")
        print("Cluster information:")
        for i, (cluster_id, info) in enumerate(sorted_clusters):
            print(f"  Level {i+1}: {info['size']} users, "
                  f"score range: {info['min_score']:.3f}-{info['max_score']:.3f}, "
                  f"mean: {info['mean_score']:.3f}")
        
        return thresholds
    
    def _fallback_percentile_thresholds(self, scores: List[float], num_levels: int) -> List[float]:
        """Fallback to percentile-based thresholds if KNN fails."""
        print("Using fallback percentile-based thresholds")
        scores = sorted(scores)
        thresholds = []
        
        for i in range(1, num_levels):
            percentile = i * (100 / num_levels)
            threshold = np.percentile(scores, percentile)
            thresholds.append(threshold)
        
        return thresholds
    
    def calculate_adaptive_knn_thresholds(self, composite_scores: Dict[str, float],
                                        gaming_scores: Dict[str, float] = None,
                                        impression_scores: Dict[str, float] = None) -> List[float]:
        """
        Calculate thresholds using multi-dimensional KNN with multiple score components.
        """
        print("Calculating adaptive KNN-based thresholds with multiple features...")
        
        if not composite_scores:
            return []
        
        user_ids = list(composite_scores.keys())
        
        # Create multi-dimensional feature matrix
        features = []
        for user_id in user_ids:
            feature_vector = [
                composite_scores[user_id],  # Primary composite score
                gaming_scores.get(user_id, 0) if gaming_scores else 0,  # Gaming component
                impression_scores.get(user_id, 0) if impression_scores else 0,  # Impression component
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        if len(features_array) < 4:
            return self._fallback_percentile_thresholds([s for s in composite_scores.values()], 5)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features_array)
        
        # Find optimal clusters using multi-dimensional data
        optimal_clusters = self.find_optimal_clusters_multidim(features_normalized)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_normalized)
        
        # Use KNN to refine cluster boundaries
        knn = KNeighborsClassifier(n_neighbors=min(7, len(features_array) // 3))
        knn.fit(features_normalized, cluster_labels)
        
        # Calculate thresholds based on composite scores within clusters
        cluster_scores = {}
        for i, user_id in enumerate(user_ids):
            cluster_id = cluster_labels[i]
            if cluster_id not in cluster_scores:
                cluster_scores[cluster_id] = []
            cluster_scores[cluster_id].append(composite_scores[user_id])
        
        # Sort clusters by mean composite score
        cluster_means = {}
        for cluster_id, scores in cluster_scores.items():
            cluster_means[cluster_id] = np.mean(scores)
        
        sorted_cluster_ids = sorted(cluster_means.keys(), key=lambda x: cluster_means[x])
        
        # Calculate thresholds
        thresholds = []
        for i in range(len(sorted_cluster_ids) - 1):
            current_cluster_scores = cluster_scores[sorted_cluster_ids[i]]
            next_cluster_scores = cluster_scores[sorted_cluster_ids[i + 1]]
            
            threshold = (max(current_cluster_scores) + min(next_cluster_scores)) / 2
            thresholds.append(threshold)
        
        print(f"Adaptive KNN thresholds: {thresholds}")
        return thresholds
    
    def find_optimal_clusters_multidim(self, features_normalized: np.ndarray, max_clusters: int = 8) -> int:
        """Find optimal clusters for multi-dimensional features."""
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(features_normalized)))
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_normalized)
            silhouette_avg = silhouette_score(features_normalized, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        if silhouette_scores:
            optimal_idx = np.argmax(silhouette_scores)
            return list(cluster_range)[optimal_idx]
        
        return 5
    
    def visualize_thresholds(self, composite_scores: Dict[str, float], 
                           thresholds: List[float], output_path: str = "knn_thresholds.png"):
        """Visualize the threshold distribution."""
        scores = list(composite_scores.values())
        
        plt.figure(figsize=(12, 6))
        
        # Plot score distribution
        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        for i, threshold in enumerate(thresholds):
            plt.axvline(threshold, color='red', linestyle='--', 
                       label=f'Threshold {i+1}: {threshold:.3f}')
        plt.title('Score Distribution with KNN Thresholds')
        plt.xlabel('Composite Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot level assignments
        plt.subplot(1, 2, 2)
        levels = []
        for score in scores:
            level = 1
            for threshold in thresholds:
                if score >= threshold:
                    level += 1
                else:
                    break
            levels.append(level)
        
        level_counts = pd.Series(levels).value_counts().sort_index()
        plt.bar(level_counts.index, level_counts.values)
        plt.title('Level Distribution with KNN Thresholds')
        plt.xlabel('Crew Level')
        plt.ylabel('Number of Users')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Threshold visualization saved to {output_path}")

# Example usage and comparison
if __name__ == "__main__":
    # Load existing data for comparison
    import pandas as pd
    
    try:
        level_df = pd.read_csv("crew_levels_revised.csv")
        composite_scores = dict(zip(level_df['user_id'], level_df['composite_score']))
        gaming_scores = dict(zip(level_df['user_id'], level_df['gaming_score']))
        impression_scores = dict(zip(level_df['user_id'], level_df['impression_score']))
        
        calculator = KNNBasedThresholdCalculator()
        
        print("=" * 60)
        print("COMPARING THRESHOLD CALCULATION METHODS")
        print("=" * 60)
        
        # Method 1: Original percentile-based
        print("\n1. PERCENTILE-BASED THRESHOLDS:")
        scores = list(composite_scores.values())
        percentile_thresholds = []
        for i in range(1, 5):
            percentile = i * 20
            threshold = np.percentile(scores, percentile)
            percentile_thresholds.append(threshold)
        print(f"Thresholds: {percentile_thresholds}")
        
        # Method 2: KNN-based clustering
        print("\n2. KNN-BASED CLUSTERING THRESHOLDS:")
        knn_thresholds = calculator.calculate_knn_thresholds(composite_scores)
        
        # Method 3: Adaptive multi-dimensional KNN
        print("\n3. ADAPTIVE MULTI-DIMENSIONAL KNN THRESHOLDS:")
        adaptive_thresholds = calculator.calculate_adaptive_knn_thresholds(
            composite_scores, gaming_scores, impression_scores
        )
        
        # Visualize results
        calculator.visualize_thresholds(composite_scores, knn_thresholds)
        
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY:")
        print("=" * 60)
        print(f"Percentile method: {len(percentile_thresholds)} thresholds")
        print(f"KNN clustering:    {len(knn_thresholds)} thresholds")
        print(f"Adaptive KNN:      {len(adaptive_thresholds)} thresholds")
        
    except FileNotFoundError:
        print("Please run the crew scoring system first to generate data for comparison")
