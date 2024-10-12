# config.py

import yaml
import importlib.resources
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def load_config(config_path=None):
    """
    Loads the configuration from the specified file or the default config.

    Args:
        config_path (str, optional): The path to the configuration file. Defaults to None.

    Returns:
        dict: The configuration dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    if config_path:
        # Expand user and environment variables in the path
        config_path = os.path.expandvars(os.path.expanduser(config_path))
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Load the default configuration from the package
        with importlib.resources.open_text('yawt', 'default_config.yaml') as f:
            config = yaml.safe_load(f)
    return config

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
        config_path (str, optional): The path to the configuration file. Defaults to None.

    Returns:
        dict: The validated configuration dictionary.
    """
    config = load_config(config_path)
    validate_config(config)
    return config

