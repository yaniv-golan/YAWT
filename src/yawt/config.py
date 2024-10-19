# config.py

import yaml
from dataclasses import dataclass, field
from typing import List, Optional
import os
import logging
import requests

SAMPLING_RATE = 16000

# New constants
APP_NAME = "YAWT"
REPO_OWNER = "yaniv-golan"
REPO_NAME = "YAWT"

# Fetch the latest release tag
try:
    response = requests.get(f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest")
    if response.status_code == 200:
        APP_VERSION = response.json()['tag_name']
    else:
        APP_VERSION = "unknown"
except Exception:
    APP_VERSION = "unknown"

CONTACT_INFO = f"https://github.com/{REPO_OWNER}/{REPO_NAME}"

@dataclass
class APICosts:
    """
    Configuration for API costs associated with transcription and diarization services.
    """
    whisper_cost_per_minute: float = 0.006  # USD per minute for Whisper transcription service
    pyannote_cost_per_hour: float = 0.18    # USD per hour for Pyannote diarization service

@dataclass
class LoggingConfig:
    """
    Configuration for logging behavior.
    """
    log_directory: str = "logs"          # Directory where log files will be stored
    max_log_size: int = 10485760         # Maximum size of a log file before rotation (10 MB)
    backup_count: int = 5                 # Number of backup log files to keep
    debug: bool = False                   # Flag to enable debug-level logging
    verbose: bool = False                 # Flag to enable verbose logging output

@dataclass
class ModelConfig:
    """
    Configuration for machine learning models used in transcription.
    """
    default_model_id: str = "openai/whisper-large-v3"  # Identifier for the default Whisper model

@dataclass
class TimeoutSettings:
    """
    Configuration for various timeout settings in seconds.
    """
    download_timeout: int = 60           # Timeout for downloading audio files
    upload_timeout: int = 120            # Timeout for uploading files
    diarization_timeout: int = 3600      # Timeout for the diarization process
    job_status_timeout: int = 60         # Timeout for checking job status

@dataclass
class TranscriptionSettings:
    """
    Configuration for transcription process settings.
    """
    generate_timeout: int = 300
    max_target_positions: int = 448
    buffer_tokens: int = 10
    confidence_threshold: float = 0.6
    max_retries: int = 1

    def __post_init__(self):
        if not (0 <= self.confidence_threshold <= 1):
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if self.max_target_positions <= 0:
            raise ValueError("max_target_positions must be a positive integer.")
        if self.buffer_tokens < 0:
            raise ValueError("buffer_tokens must be zero or a positive integer.")
        if self.max_retries < 0:
            raise ValueError("max_retries must be zero or a positive integer.")

@dataclass
class Config:
    """
    Comprehensive configuration class that aggregates all configuration sections.
    """
    api_costs: APICosts = field(default_factory=APICosts)                     # API cost configurations
    logging: LoggingConfig = field(default_factory=LoggingConfig)             # Logging configurations
    model: ModelConfig = field(default_factory=ModelConfig)                   # Model configurations
    supported_upload_services: List[str] = field(default_factory=lambda: ["0x0.st", "file.io"])  # Supported file upload services
    timeouts: TimeoutSettings = field(default_factory=TimeoutSettings)        # Timeout settings configurations
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)  # Transcription process settings
    pyannote_token: Optional[str] = None                                      # Optional Pyannote API token
    openai_key: Optional[str] = None                                           # Optional OpenAI API key

    def load_and_log_tokens(self, args):  # Removed 'logger' parameter
        import logging  # Use the logging module directly
        # Check Pyannote token
        if args.pyannote_token:
            self.pyannote_token = args.pyannote_token
            logging.debug("Pyannote token loaded from command-line arguments.")  # Updated to use logging
        elif self.pyannote_token:
            if os.getenv("PYANNOTE_TOKEN") == self.pyannote_token:
                logging.debug("Pyannote token loaded from environment variable.")
            else:
                logging.debug("Pyannote token loaded from config file.")
        elif os.getenv("PYANNOTE_TOKEN"):
            self.pyannote_token = os.getenv("PYANNOTE_TOKEN")
            logging.debug("Pyannote token loaded from environment variable.")
        else:
            logging.error("Pyannote token not found in args, config, or environment variables.")

        # Check OpenAI key
        if args.openai_key:
            self.openai_key = args.openai_key
            logging.debug("OpenAI key loaded from command-line arguments.")
        elif self.openai_key:
            if os.getenv("OPENAI_KEY") == self.openai_key:
                logging.debug("OpenAI key loaded from environment variable.")
            else:
                logging.debug("OpenAI key loaded from config file.")
        elif os.getenv("OPENAI_KEY"):
            self.openai_key = os.getenv("OPENAI_KEY")
            logging.debug("OpenAI key loaded from environment variable.")
        else:
            logging.error("OpenAI key not found in args, config, or environment variables.")

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Loads the configuration from the default settings and overrides with a config file if provided.

    Args:
        config_path (str, optional): Path to the YAML configuration file. Defaults to None.

    Returns:
        Config: The resulting configuration object with all settings loaded.
    """
    config = Config()  # Initialize configuration with default values

    if config_path:
        # Open and load user-provided YAML configuration file
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        if user_config is None:
            user_config = {}

        def update_dataclass(dc, updates):
            """
            Recursively updates dataclass fields with values from the user configuration.

            Args:
                dc (dataclass instance): The dataclass instance to update.
                updates (dict): Dictionary containing updates to apply.
            """
            for key, value in updates.items():
                if hasattr(dc, key):
                    attr = getattr(dc, key)
                    if isinstance(attr, (APICosts, LoggingConfig, ModelConfig, TimeoutSettings, TranscriptionSettings)):
                        update_dataclass(attr, value)  # Recursively update nested dataclasses
                    else:
                        setattr(dc, key, value)  # Update the field with the new value
                else:
                    setattr(dc, key, value)      # Add new attributes if they don't exist

        update_dataclass(config, user_config)  # Apply user configuration overrides

    # Override configuration with environment variables if they are set
    config.pyannote_token = config.pyannote_token or os.getenv("PYANNOTE_TOKEN")
    config.openai_key = config.openai_key or os.getenv("OPENAI_KEY")
    
    # Override logging debug and verbose flags with environment variables if set
    config.logging.debug = config.logging.debug or os.getenv("DEBUG") == "true"
    config.logging.verbose = config.logging.verbose or os.getenv("VERBOSE") == "true"

    return config  # Return the fully loaded and configured Config object

# Example usage:
# config = load_config(args.config)

def validate_config(config: Config):
    """
    Validates the given configuration.

    Args:
        config (Config): The configuration object to validate.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if not config.api_costs:
        raise ValueError("Configuration must include 'api_costs'.")
    if config.api_costs.whisper_cost_per_minute < 0:
        raise ValueError("Whisper cost per minute must be non-negative.")
    if config.api_costs.pyannote_cost_per_hour < 0:
        raise ValueError("Pyannote cost per hour must be non-negative.")
    if not config.logging:
        raise ValueError("Configuration must include 'logging'.")
    if not config.model:
        raise ValueError("Configuration must include 'model'.")
    if not config.supported_upload_services:
        raise ValueError("Configuration must include 'supported_upload_services'.")
    if not config.timeouts:
        raise ValueError("Configuration must include 'timeouts'.")
    if not config.transcription:
        raise ValueError("Configuration must include 'transcription'.")
    # Add additional validation rules as needed

__all__ = ["load_config", "validate_config"]
