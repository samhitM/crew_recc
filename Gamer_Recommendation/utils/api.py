import requests
import warnings
from urllib3.exceptions import InsecureRequestWarning
from core.config import VERIFY_SSL

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

class APIClient:
    def __init__(self, base_url, jwt_token):
        self.base_url = base_url.rstrip('/')
        self.jwt_token = jwt_token
        self.headers = {
            'Authorization': f'Bearer {self.jwt_token}'
        }

    def fetch_user_relations(self, limit=50, offset=0, relation=None):
        """
        Fetch user relations with optional pagination and filtering.

        Parameters:
            limit (int): The number of records to fetch per request (default: 50).
            offset (int): The offset for pagination (default: 0).
            relation (str, optional): The type of relation to filter (e.g., "friends", "blocked_list", "report_list").

        Returns:
            list[dict]: A list of user relation records with keys:
                - "player_id" (str): The user ID.
                - "relation" (str): The type of relation.
                - "user_interests" (list): Interests of the user.
                - "played_games" (list): Games the user has played.
                - "last_active_ts" (str): Last active timestamp.
        
        Raises:
            ValueError: If an error occurs during the API request.
        """
        params = {
            'limit': limit,
            'offset': offset,
        }
        data = {"relation": relation} if relation else {}
        
        try:
            response = requests.get(
                f'{self.base_url}/api/users/relations',
                headers=self.headers,
                params=params,
                json=data,
                verify=VERIFY_SSL
            )

            if response.status_code == 200:
                return [
                    {
                        "player_id": r["id"],
                        "relation": relation,
                        "user_interests": r.get("user_interests", []),
                        "played_games": r.get("played_games", []),
                        "last_active_ts": r.get("last_active_ts", ""),
                    }
                    for r in response.json()
                ]
            elif response.status_code == 404:
                return []
            else:
                raise ValueError(f"Failed to fetch relations. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"An error occurred while fetching relations: {str(e)}")
