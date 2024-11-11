from google.cloud import secretmanager
import os

# Database connection settings
DB_HOST = '34.44.52.84'
DB_PORT = 5432
DATABASE = 'crewdb'
DB_USER = 'admin_crew'
DB_PASSWORD = 'xV/nI2+=uOI&KL1P'

# # Secret key for token verification
# SECRET_KEY = "1c75f472b1e52c582c8ed3f4d88af9c0137f9a2eeeb1d63e97ecedd1be8f1a3c"
# VALID_SECRET_KEY = '7irNPR6kOia'  # Add valid secret key as a variable for easy management

def get_secret(secret_name):
    """Fetches a secret's value from Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GCP_PROJECT_ID")  # Set this environment variable in GCP deployment
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})
    return response.payload.data.decode("UTF-8")

# Secret keys for token verification, retrieved securely
SECRET_KEY = get_secret("SECRET_KEY")
VALID_SECRET_KEY = get_secret("VALID_SECRET_KEY")