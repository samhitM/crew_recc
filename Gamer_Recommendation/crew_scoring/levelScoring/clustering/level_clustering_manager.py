"""
Clustering operations for level assignment.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict


class LevelClusteringManager:
    """Manages KNN clustering for level assignment."""
    
    def __init__(self, target_levels: int = 3):
        self.target_levels = target_levels
    
    def find_optimal_clusters_elbow(self, features: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        print("Finding optimal number of clusters using elbow method...")
        
        if len(features) < 2:
            return 1
        
        # Limit max clusters to reasonable range
        max_clusters = min(max_clusters, len(features) // 2, 10)
        
        inertias = []
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            if k > len(features):
                break
                
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(features, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0)
                    
            except Exception as e:
                print(f"Error with k={k}: {e}")
                break
        
        if not inertias:
            return 2  # Default fallback
        
        # Find elbow using rate of change
        optimal_k = 2
        if len(inertias) >= 3:
            # Calculate second derivative to find elbow
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            
            if len(second_diffs) > 0:
                # Find the point where the rate of decrease slows down most
                elbow_idx = np.argmax(second_diffs) + 2  # +2 because we start from k=2
                optimal_k = elbow_idx
        
        # Validate with silhouette score
        if silhouette_scores and len(silhouette_scores) > 0:
            best_silhouette_idx = np.argmax(silhouette_scores)
            best_silhouette_k = best_silhouette_idx + 2  # +2 because we start from k=2
            
            # Use silhouette if it's reasonably close to elbow method
            if abs(best_silhouette_k - optimal_k) <= 1:
                optimal_k = best_silhouette_k
        
        optimal_k = max(2, min(optimal_k, max_clusters))  # Ensure reasonable range
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Inertias: {inertias}")
        if silhouette_scores:
            print(f"Silhouette scores: {[round(s, 3) for s in silhouette_scores]}")
        
        return optimal_k
    
    def assign_levels_with_knn_clustering(self, composite_scores: Dict[str, float]) -> Dict[str, int]:
        """Assign crew levels using KNN clustering with improved balance."""
        print("Assigning crew levels using improved KNN clustering...")
        
        if not composite_scores:
            return {}
        
        # Prepare data for clustering
        user_ids = list(composite_scores.keys())
        scores = np.array(list(composite_scores.values())).reshape(-1, 1)
        
        if len(scores) <= self.target_levels:
            # If we have fewer users than target levels, assign incrementally
            level_assignments = {}
            for i, user_id in enumerate(user_ids):
                level_assignments[user_id] = (i % self.target_levels) + 1
            return level_assignments
        
        # Find optimal number of clusters
        optimal_clusters = self.find_optimal_clusters_elbow(scores, max_clusters=min(8, len(scores)//2))
        
        try:
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scores)
            
            # Check cluster balance
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            min_cluster_size = min(counts)
            max_cluster_size = max(counts)
            
            # If clustering is too unbalanced, use hybrid approach
            if max_cluster_size > min_cluster_size * 3:
                print("Clustering too unbalanced, applying hybrid approach...")
                return self._apply_hybrid_clustering(user_ids, scores, self.target_levels)
            
            # Map clusters to levels based on average scores
            cluster_means = {}
            for cluster_id in unique_labels:
                cluster_mask = cluster_labels == cluster_id
                cluster_scores = scores[cluster_mask]
                cluster_means[cluster_id] = np.mean(cluster_scores)
            
            # Sort clusters by mean score and assign levels
            sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
            cluster_to_level = {}
            
            for i, (cluster_id, _) in enumerate(sorted_clusters):
                # Map clusters to levels (1 to target_levels)
                level = min(i + 1, self.target_levels)
                cluster_to_level[cluster_id] = level
            
            # Assign levels to users
            level_assignments = {}
            for i, user_id in enumerate(user_ids):
                cluster_id = cluster_labels[i]
                level_assignments[user_id] = cluster_to_level[cluster_id]
            
        except Exception as e:
            print(f"Error in KMeans clustering: {e}")
            return self._apply_hybrid_clustering(user_ids, scores, self.target_levels)
        
        # Print level distribution
        level_counts = {}
        for level in level_assignments.values():
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print("Level distribution:")
        for level, count in sorted(level_counts.items()):
            print(f"  Level {level}: {count} users")
        
        return level_assignments
    
    def _apply_hybrid_clustering(self, user_ids: list, scores: np.ndarray, num_levels: int) -> Dict[str, int]:
        """Apply hybrid clustering approach when standard clustering fails."""
        print("Applying hybrid clustering approach...")
        
        # Combine percentile-based and KMeans approaches
        level_assignments = {}
        
        # Sort users by score
        user_score_pairs = list(zip(user_ids, scores.flatten()))
        user_score_pairs.sort(key=lambda x: x[1])
        
        # Divide into roughly equal groups
        users_per_level = len(user_ids) // num_levels
        remainder = len(user_ids) % num_levels
        
        current_index = 0
        for level in range(1, num_levels + 1):
            # Add one extra user to first 'remainder' levels
            level_size = users_per_level + (1 if level <= remainder else 0)
            
            for i in range(level_size):
                if current_index < len(user_score_pairs):
                    user_id = user_score_pairs[current_index][0]
                    level_assignments[user_id] = level
                    current_index += 1
        
        print("Hybrid clustering completed")
        return level_assignments
