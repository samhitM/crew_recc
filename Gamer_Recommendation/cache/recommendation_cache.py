class RecommendationCache:
    """Singleton class to store preprocessed data, mapping layer, and recommendation system."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RecommendationCache, cls).__new__(cls)
            cls._instance.preprocessed_data = None
            cls._instance.mapping_layer = None
            cls._instance.recommendation_system = None
        return cls._instance

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

# Use this shared instance everywhere
recommendation_cache = RecommendationCache()