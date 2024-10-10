# config.py

import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the path to the config file
CONFIG_FILE = Path(__file__).parent / "config.yaml"

def load_config(config_path=CONFIG_FILE):
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def validate_config(config):
    # Example validation
    if config['api_costs']['whisper']['cost_per_minute'] < 0:
        raise ValueError("Whisper cost per minute must be non-negative.")
    if config['api_costs']['pyannote']['cost_per_hour'] < 0:
        raise ValueError("Pyannote cost per hour must be non-negative.")
    # Add more validations as needed

# Load and validate the configuration
CONFIG = load_config()
validate_config(CONFIG)

# Extract API Costs
MAX_TARGET_POSITIONS = 448  # This might remain constant or can also be moved to config.yaml
BUFFER_TOKENS = 3
MAX_CHUNK_DURATION = 30  # seconds
GENERATE_TIMEOUT = CONFIG['timeouts']['diarization_timeout']  # Example usage
COST_PER_MINUTE = CONFIG['api_costs']['whisper']['cost_per_minute']
PYANNOTE_COST_PER_HOUR = CONFIG['api_costs']['pyannote']['cost_per_hour']

# Logging Configuration
LOG_DIRECTORY = CONFIG['logging']['log_directory']
MAX_LOG_SIZE = CONFIG['logging']['max_log_size']
BACKUP_COUNT = CONFIG['logging']['backup_count']

# Model Configuration
DEFAULT_MODEL_ID = CONFIG['model']['default_model_id']

# Supported Services
SUPPORTED_UPLOAD_SERVICES = set(CONFIG['supported_upload_services'])

# Timeout Settings
DOWNLOAD_TIMEOUT = CONFIG['timeouts']['download_timeout']
UPLOAD_TIMEOUT = CONFIG['timeouts']['upload_timeout']
DIARIZATION_TIMEOUT = CONFIG['timeouts']['diarization_timeout']
JOB_STATUS_TIMEOUT = CONFIG['timeouts']['job_status_timeout']

# Environment Variables for Sensitive Information
PYANNOTE_TOKEN = os.getenv("PYANNOTE_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_KEY")