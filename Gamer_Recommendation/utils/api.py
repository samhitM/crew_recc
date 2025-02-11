import requests
import warnings
from services.token_utils import generate_jwt_token
from urllib3.exceptions import InsecureRequestWarning
from core.config import VERIFY_SSL, JWT_SECRET_KEY

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

class APIClient:
    def __init__(self, base_url,user_id):
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id,
        self.jwt_token = generate_jwt_token(self.user_id)  # Automatically generate token
        self.headers = {
            'Authorization': f'Bearer {self.jwt_token}'
        }

    def fetch_user_relations(self,limit=50, offset=0, relation=None):
        params = {
            'limit': limit,
            'offset': offset,
        }
        data = {
            "user_id": self.user_id,
            "relation": relation if relation else {}
        }

        try:
            response = requests.get(
                f'{self.base_url}/api/users/relations',
                headers=self.headers,
                params=params,
                json=data,
                verify=VERIFY_SSL
            )
            # print(f"Response Status Code: {response.status_code}")  # Debugging
            # print(f"Response Content: {response.text}")  # Debugging

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return []
            else:
                raise ValueError(f"Failed to fetch relations. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"An error occurred while fetching relations: {str(e)}")
