from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

class ThresholdCalculator:
    @staticmethod
    def calculate_thresholds(scores: List[float], method: str = "logarithmic", clusters: int = 2, widening_factor: float = 1.2) -> List[float]:
        """
        Calculate thresholds using the specified method with widening thresholds to make higher levels harder to reach.

        Parameters:
        - scores (List[float]): List of composite scores.
        - method (str): Method for threshold calculation ("logarithmic", "exponential", "clustering").
        - clusters (int): Number of clusters for clustering-based thresholds.
        - widening_factor (float): Factor to control the widening of thresholds at higher levels.

        Returns:
        - List[float]: Calculated widening thresholds.
        """
        if len(scores) < 2:
            raise ValueError("At least two scores are required to calculate thresholds.")

        min_score, max_score = min(scores), max(scores)
        if max_score == min_score:
            return [min_score + i for i in range(len(scores))]  # Handle edge case with equal scores

        if method == "clustering":
            kmeans = KMeans(n_clusters=clusters, random_state=42,n_init=10)
            reshaped_scores = np.array(scores).reshape(-1, 1)
            kmeans.fit(reshaped_scores)
            thresholds = sorted(kmeans.cluster_centers_.flatten())
        else:
            if method == "logarithmic":
                base_thresholds = [min_score + (max_score - min_score) * np.log1p(i) / np.log1p(clusters - 1) for i in range(1, clusters)]
                thresholds = [min_score + (t - min_score) * widening_factor ** i for i, t in enumerate(base_thresholds)]
            elif method == "exponential":
                base_thresholds = [min_score + (max_score - min_score) * (1 - np.exp(-i)) for i in range(1, clusters)]
                thresholds = [min_score + (t - min_score) * widening_factor ** i for i, t in enumerate(base_thresholds)]

        # Ensure consistent length with scores
        if len(thresholds) < len(scores):
            thresholds += [thresholds[-1]] * (len(scores) - len(thresholds))  # Pad with last threshold
        elif len(thresholds) > len(scores):
            thresholds = thresholds[:len(scores)]  # Truncate to match scores

        return thresholds

    @staticmethod
    def select_best_method(scores: List[float], clusters: int = 2, widening_factor: float = 1.2) -> str:
        """
        Select the best thresholding method based on score distribution or other metrics.

        Parameters:
        - scores (List[float]): List of composite scores.
        - clusters (int): Number of clusters for clustering-based thresholds.
        - widening_factor (float): Factor to control the widening of thresholds.

        Returns:
        - str: The name of the best threshold calculation method.
        """
        methods = ["logarithmic", "exponential", "clustering"]
        errors = {}

        for method in methods:
            thresholds = ThresholdCalculator.calculate_thresholds(scores, method, clusters, widening_factor)

            # Adjust thresholds length to match scores
            if len(thresholds) < len(scores):
                thresholds += [thresholds[-1]] * (len(scores) - len(thresholds))
            elif len(thresholds) > len(scores):
                thresholds = thresholds[:len(scores)]

            # Compute error
            if method == "clustering":
                kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
                reshaped_scores = np.array(scores).reshape(-1, 1)
                kmeans.fit(reshaped_scores)
                inertia = kmeans.inertia_
                errors[method] = inertia
            else:
                rmse = mean_squared_error(scores, thresholds, squared=False)
                errors[method] = rmse
        
        print(errors)
        best_method = min(errors, key=errors.get)
        print(f"Best thresholding method selected: {best_method}")
        return best_method
