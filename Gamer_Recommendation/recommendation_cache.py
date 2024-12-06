class RecommendationCache:
    """Cache class to store preprocessed data, mapping layer, and recommendation system for quick access."""
    def __init__(self):
        self.preprocessed_data = None
        self.mapping_layer = None
        self.recommendation_system = None

    def reset_cache(self):
        """Clears all cache components."""
        self.preprocessed_data = None
        self.mapping_layer = None
        self.recommendation_system = None

    def update_recommendation_system(self, new_recommendation_system):
        """Updates the recommendation system in the cache."""
        self.recommendation_system = new_recommendation_system

    def update_preprocessed_data(self, new_preprocessed_data):
        """Updates the preprocessed data in the cache."""
        self.preprocessed_data = new_preprocessed_data

    def update_mapping_layer(self, new_mapping_layer):
        """Updates the mapping layer in the cache."""
        self.mapping_layer = new_mapping_layer

# Instantiate the cache object
recommendation_cache = RecommendationCache()
