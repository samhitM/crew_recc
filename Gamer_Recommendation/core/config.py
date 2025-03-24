from google.cloud import secretmanager
import os

# Database connection settings
DB_CONFIG = {
    "crewdb": {"host": "34.44.52.84", "port": 5432, "user": "admin_crew", "password": "xV/nI2+=uOI&KL1P"},
    # Add other databases here
}

# Secret key for token verification
JWT_SECRET_KEY = "942bb82561c615e0d67a27538c8e203ea60423c0508956a134e4f747d88189bd"
INTERNAL_SECRET_KEY = "1c75f472b1e52c582c8ed3f4d88af9c0137f9a2eeeb1d63e97ecedd1be8f1a3c"
CREW_USER_ID = "7irNPR6kOia"  # Add valid user ID as a variable for easy management
# VALID_SECRET_KEY = '7irNPR6kOia'  # Add valid secret key as a variable for easy management

API_BASE_URL = "http://cygbg8gyg5d2fxbm.fz25.alb.azure.com"
VERIFY_SSL = False  # Set to True if you want to verify SSL certificates

def get_secret(secret_name):
    """Fetches a secret's value from Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GCP_PROJECT_ID")  # Set this environment variable in GCP deployment
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})
    return response.payload.data.decode("UTF-8")

# # Secret keys for token verification, retrieved securely
# SECRET_KEY = get_secret("SECRET_KEY")
# VALID_SECRET_KEY = get_secret("VALID_SECRET_KEY")