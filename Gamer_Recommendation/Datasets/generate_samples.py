import json
import random
import time
import copy
from datetime import datetime, timedelta

# Load the existing player data
with open('Datasets/data_new.json', 'r') as f:
    data = json.load(f)

# Extract the base player template to clone and modify
base_player = data["playersStats"][0]

# Predefined friend types and countries
friend_types = ["Beginner", "Intermediate", "Pro"]
countries = ["Germany", "USA", "India", "UK", "France", "Brazil", "Canada", "Australia"]
interest_categories = ["Action", "MOBA", "Strategy", "Indie", "RPG"]

# Function to generate a random date of birth
def generate_random_dob(min_age=18, max_age=50):
    today = datetime.today()
    random_age = random.randint(min_age, max_age)
    dob = today - timedelta(days=random_age * 365)
    return dob.strftime('%Y-%m-%d')

# Function to generate random player data
def generate_random_player(user_id):
    new_player = copy.deepcopy(base_player)
    new_player["userId"] = str(user_id)
    
    # Access the playerSummary within endpoint_data["endpointData"]
    player_summary = new_player["endpoint_data"]["endpointData"]["playerSummary"]
    player_summary["steamid"] = str(random.randint(76561198000000000, 76561199999999999))
    player_summary["personaname"] = "Player-" + str(user_id)
    player_summary["profileurl"] = f"https://steamcommunity.com/id/player_{user_id}/"
    player_summary["avatar"] = f"https://avatars.steamstatic.com/avatar_{user_id}.jpg"
    player_summary["avatarmedium"] = f"https://avatars.steamstatic.com/avatar_{user_id}_medium.jpg"
    player_summary["avatarfull"] = f"https://avatars.steamstatic.com/avatar_{user_id}_full.jpg"
    player_summary["realname"] = f"Player {user_id}"
    player_summary["lastlogoff"] = int(time.time()) - random.randint(0, 100000)
    
    # Add new fields
    player_summary["friend_type"] = random.choice(friend_types)
    player_summary["country"] = random.choice(countries)
    player_summary["user_interests"] = random.sample(interest_categories, random.randint(1, 3))
    player_summary["dob"] = generate_random_dob()

    # Generate games data
    new_player["endpoint_data"]["endpointData"]["games"] = generate_random_games()

    return new_player

# Function to generate random game data for each player
def generate_random_games():
    games_list = []
    for _ in range(random.randint(1, 10)):  # Players can own 1 to 10 games
        appid = random.randint(100, 1000)
        game = {
            "appid": appid,
            "name": "Game-" + str(appid),
            "playtime_forever": random.randint(0, 5000),
            "img_icon_url": f"icon_{appid}",
            "has_community_visible_stats": random.choice([True, False]),
            "playtime_windows_forever": random.randint(0, 5000),
            "playtime_mac_forever": random.randint(0, 5000),
            "playtime_linux_forever": random.randint(0, 5000),
            "playtime_deck_forever": random.randint(0, 5000),
            "rtime_last_played": random.randint(0, int(time.time())),
            "playtime_disconnected": random.randint(0, 5000),
            "achievements": generate_random_achievements(),
            "details": {
                "name": "Game-" + str(appid),
                "steam_appid": appid,
                "platforms": {
                    "windows": random.choice([True, False]),
                    "mac": random.choice([True, False]),
                    "linux": random.choice([True, False])
                }, 
                "categories": [
                    {"id": 2, "description": "Single-player"},
                    {"id": 22, "description": "Steam Achievements"}
                ],
                "genres": [
                    {"id": "1", "description": random.choice(["Action", "Adventure", "Indie", "Strategy"])}
                ]
            }
        }
        games_list.append(game)
    return games_list

# Function to generate random achievements for a game
def generate_random_achievements():
    achievements = []
    for ach_id in range(random.randint(1, 5)):  # 1 to 5 achievements per game
        achievement = {
            "apiname": "ACH_" + str(ach_id),
            "achieved": random.choice([0, 1]),
            "unlocktime": random.randint(0, int(time.time())) if random.choice([True, False]) else 0
        }
        achievements.append(achievement)
    return achievements

# Add new player samples to the dataset
for i in range(5, 505):  # Add 500 new players (userId 5 to 504)
    new_player = generate_random_player(i)
    data["playersStats"].append(new_player)

# Create a new dictionary with num_recommendations first
updated_data = {
    "playersStats": data["playersStats"]  # Keep the playerStats after num_recommendations
}

# Save the updated data back to the JSON file
with open('request.json', 'w') as f:
    json.dump(updated_data, f, indent=4)

print(f"Added 500 new player samples successfully.")
