import json
import os

KAGGLE_CONFIG_DIR = os.environ.get('KAGGLE_CONFIG_DIR', '.kaggle')
SECRETS_PATH = os.path.join(KAGGLE_CONFIG_DIR, 'secrets.json')

class UserSecretsClient:
    def __init__(self, secrets_path=SECRETS_PATH):
        self.secrets_path = secrets_path

    def get_secret(self, secret_name):
        with open(self.secrets_path) as f:
            secrets = json.load(f)
            return secrets[secret_name]
