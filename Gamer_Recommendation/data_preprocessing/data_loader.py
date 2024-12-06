import json

class DataLoader:
    def __init__(self, file_path):
        # Initialize DataLoader with the file path to the JSON data
        self.file_path = file_path

    def load_data(self):
        # Open and read the JSON file
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        
        # Return player statistics from the loaded JSON data, defaulting to an empty list if not found
        return {
            "playersStats": data.get("playersStats", [])
        }
