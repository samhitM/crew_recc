import tensorflow as tf
import numpy as np

class DatasetPreparer:
    def __init__(self, batch_size=32):
        # Initialize DatasetPreparer with a specified batch size
        self.batch_size = batch_size

    def create_input_tensors(self, df, mapping_layer):
        # Map player_id and game_id to integer indices using the mapping_layer
        user_id = df['player_id'].map(mapping_layer.user_id_mapping).values.astype(np.int32)
        game_id = df['game_id'].map(mapping_layer.game_id_mapping).values.astype(np.int32)

        # Extract game-specific features and convert to float32
        game_features = df[['playtime_forever', 'achievements_unlocked', 'num_sessions', 'last_played']].values.astype(np.float32)

        # Identify columns corresponding to categorical and playtime features
        genre_columns = [col for col in df.columns if col.startswith('genres_')]
        platform_columns = [col for col in df.columns if col.startswith('platforms_')]
        medium_columns = [col for col in df.columns if col.startswith('medium_')]
        genre_playtime_columns = [col for col in df.columns if col.startswith('total_playtime_per_genre_')]

        # Compile global features by combining categorical and aggregated playtime features
        global_features = df[genre_columns + platform_columns + medium_columns + genre_playtime_columns + 
                            ['total_playtime_across_games', 'game_diversity']].values.astype(np.float32)

        # Set playtime_forever as the target labels
        labels = df['playtime_forever'].values.astype(np.float32)
        return (user_id, game_id, game_features, global_features), labels
    
    def prepare_tf_dataset(self, df, mapping_layer):
        # Generate input tensors and labels
        inputs, labels = self.create_input_tensors(df, mapping_layer)
        
        # Create TensorFlow dataset from input tensors and labels, then batch it
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        return dataset.batch(self.batch_size)
