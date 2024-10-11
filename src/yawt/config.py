# config.py

import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the path to the default config file
DEFAULT_CONFIG_FILE = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"

def load_config(config_path="config/default_config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

DOWNLOAD_TIMEOUT = CONFIG.get('timeouts', {}).get('download_timeout', 60)  # {{ edit: Load download_timeout with a default of 60 seconds }}

def validate_config(config):
    """
    Validates the loaded configuration.
    
    Args:
        config (dict): The configuration dictionary to validate.
    
    Raises:
        ValueError: If any validation checks fail.
    """
    # Existing validations
    if config['api_costs']['whisper']['cost_per_minute'] < 0:
        raise ValueError("Whisper cost per minute must be non-negative.")
    if config['api_costs']['pyannote']['cost_per_hour'] < 0:
        raise ValueError("Pyannote cost per hour must be non-negative.")
    
    # Validate Transcription Settings
    transcription = config.get('transcription')
    if transcription is None:
        raise ValueError("Missing 'transcription' section in configuration.")
    
    if transcription.get('max_target_positions', 0) <= 0:
        raise ValueError("max_target_positions must be a positive integer.")
    if transcription.get('generate_timeout', 0) <= 0:
        raise ValueError("generate_timeout must be a positive integer.")
    if transcription.get('buffer_tokens', 0) < 0:
        raise ValueError("buffer_tokens must be non-negative.")
    
    # Add more validations as needed

def load_and_prepare_config(config_path=None):
    """
    Loads and validates the configuration from the specified file.

    Args:
        config_path (Path or str, optional): The path to the configuration file. Defaults to None.

    Returns:
        dict: The validated configuration dictionary.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE
    config = load_config(config_path)
    validate_config(config)
    return config

# Removed Global CONFIG Variable and Related Extractions
# CONFIG = load_config()
# validate_config(CONFIG)

# Removed Global Variables Extraction
# All configurations are now handled within main.py