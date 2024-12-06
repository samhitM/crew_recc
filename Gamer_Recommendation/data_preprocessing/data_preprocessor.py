import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime

class DataPreprocessor:
    def __init__(self):
        # Initialize scaler and one-hot encoder for preprocessing
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def preprocess(self, data):
        df_list = []
        genre_set = set()
        platform_set = set()
        medium_set = set()

        # Process each player's data
        for player_data in data:
            player_id, games, total_playtime_across_games, genre_playtime_dict, games_played, age, recommendation_expertise, country, user_interests = self.process_player(player_data)

            # Process each game in the player's data
            for game in games:
                game_data, updated_genres, updated_platforms, updated_mediums = self.process_game(player_id, game)
                game_data['age'] = age
                game_data['recommendation_expertise'] = recommendation_expertise
                game_data['country'] = country
                game_data['user_interests'] = user_interests
                df_list.append(game_data)

                # Track unique genres, platforms, and mediums
                genre_set.update(updated_genres)
                platform_set.update(updated_platforms)
                medium_set.update(updated_mediums)

            # Add global features for the player
            game_diversity = len(games_played)
            self.add_global_features(df_list, player_id, total_playtime_across_games, genre_playtime_dict, genre_set, game_diversity)

        # Convert list to DataFrame and handle missing values
        df = pd.DataFrame(df_list)
        df.fillna(0, inplace=True)
        
        # Expand categorical features and normalize numerical ones
        df = self.expand_categorical_features(df, genre_set, platform_set, medium_set)
        df = self.normalize_features(df, genre_set)
        return df

    def calculate_age(self, dob):
        # Calculate age based on date of birth
        if dob is None:
            return None
        today = datetime.today()
        birth_date = datetime.strptime(dob, '%Y-%m-%dT%H:%M:%S.%fZ')
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age

    def process_player(self, player_data):
        # Extract and process player-level data
        player_id = player_data['userId']
        endpoint_data = player_data['endpoint_data']['endpointData']
        games = endpoint_data['games']
        player_summary = endpoint_data.get('playerSummary', {})
        dob = player_summary.get('dob', None)
        age = self.calculate_age(dob) if dob else None
        recommendation_expertise = player_summary.get('recommendation_expertise', 'beginner')
        country = player_summary.get('country', '')
        user_interests = player_summary.get('user_interests', [])

        # Track total playtime and genre-specific playtime
        total_playtime_across_games = 0
        genre_playtime_dict = {}
        games_played = set()

        for game in games:
            playtime_forever = game['playtime_forever']
            genres = [g['description'] for g in game['details']['genres']]

            total_playtime_across_games += playtime_forever

            for genre in genres:
                genre_playtime_dict[genre] = genre_playtime_dict.get(genre, 0) + playtime_forever

            games_played.add(game['appid'])
        
        return player_id, games, total_playtime_across_games, genre_playtime_dict, games_played, age, recommendation_expertise, country, user_interests

    def process_game(self, player_id, game):
        # Extract and process game-level data
        game_id = game['appid']
        game_name = game['name']
        playtime_forever = game['playtime_forever']
        achievements_unlocked = sum([ach['achieved'] for ach in game['achievements']])
        num_sessions = len(game['achievements'])

        # Extract genres, platforms, and mediums for the game
        genres = [g['description'] for g in game['details']['genres']]
        platforms = [platform for platform, supported in game['details']['platforms'].items() if supported]
        mediums = [cat['description'] for cat in game['details']['categories']]
        last_played = game.get('rtime_last_played', 0)
        
        return {
            'player_id': player_id,
            'game_id': game_id,
            'game_name': game_name,
            'playtime_forever': playtime_forever,
            'achievements_unlocked': achievements_unlocked,
            'num_sessions': num_sessions,
            'genres': genres,
            'platforms': platforms,
            'medium': mediums,
            'last_played': last_played  
        }, genres, platforms, mediums

    def add_global_features(self, df_list, player_id, total_playtime_across_games, genre_playtime_dict, genre_set, game_diversity):
        # Add player-level global features to each game entry
        for game_data in df_list:
            if game_data['player_id'] == player_id:
                game_data['total_playtime_across_games'] = total_playtime_across_games
                for genre in genre_set:
                    game_data[f'total_playtime_per_genre_{genre}'] = genre_playtime_dict.get(genre, 0)
                game_data['game_diversity'] = game_diversity

    def normalize_features(self, df, genre_set):
        # Normalize numerical features
        numerical_features = ['playtime_forever', 'achievements_unlocked', 'num_sessions', 'total_playtime_across_games', 'game_diversity', 'last_played']
        genre_playtime_columns = [f'total_playtime_per_genre_{genre}' for genre in genre_set]
        
        df[numerical_features + genre_playtime_columns] = self.scaler.fit_transform(df[numerical_features + genre_playtime_columns])
        return df

    def expand_categorical_features(self, df, genre_set, platform_set, medium_set):
        # One-hot encode categorical features
        for genre in genre_set:
            df[f'genres_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)

        for platform in platform_set:
            df[f'platforms_{platform}'] = df['platforms'].apply(lambda x: 1 if platform in x else 0)

        for medium in medium_set:
            df[f'medium_{medium}'] = df['medium'].apply(lambda x: 1 if medium in x else 0)

        # Drop original columns for one-hot encoded features
        df = df.drop(columns=['genres', 'platforms', 'medium'])
        
        # Check for any NaN values
        assert not df.isnull().any().any(), "Data contains NaN values."
        return df
