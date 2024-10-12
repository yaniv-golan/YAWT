# config.py

import yaml
from dataclasses import dataclass, field
from typing import List, Optional
import os

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
    generate_timeout: int = 300          # Timeout for transcription generation
    max_target_positions: int = 448      # Maximum number of target positions for the model
    buffer_tokens: int = 10              # Number of buffer tokens to use during transcription

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
