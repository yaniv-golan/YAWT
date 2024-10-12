# config.py

import yaml
from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class APICosts:
    whisper_cost_per_minute: float = 0.006  # USD per minute for Whisper
    pyannote_cost_per_hour: float = 0.18    # USD per hour for diarization

@dataclass
class LoggingConfig:
    log_directory: str = "logs"
    max_log_size: int = 10485760  # 10 MB in bytes
    backup_count: int = 5
    debug: bool = False
    verbose: bool = False

@dataclass
class ModelConfig:
    default_model_id: str = "openai/whisper-large-v3"

@dataclass
class TimeoutSettings:
    download_timeout: int = 60       # seconds
    upload_timeout: int = 120        # seconds
    diarization_timeout: int = 3600  # seconds
    job_status_timeout: int = 60     # seconds

@dataclass
class TranscriptionSettings:
    generate_timeout: int = 300        # seconds
    max_target_positions: int = 448
    buffer_tokens: int = 10            # Reduced from 445 to 10

@dataclass
class Config:
    api_costs: APICosts = field(default_factory=APICosts)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    supported_upload_services: List[str] = field(default_factory=lambda: ["0x0.st", "file.io"])
    timeouts: TimeoutSettings = field(default_factory=TimeoutSettings)
    transcription: TranscriptionSettings = field(default_factory=TranscriptionSettings)
    pyannote_token: Optional[str] = None
    openai_key: Optional[str] = None

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Loads the configuration from the default settings and overrides with a config file if provided.

    Args:
        config_path (str, optional): Path to the YAML configuration file. Defaults to None.

    Returns:
        Config: The resulting configuration object.
    """
    config = Config()

    if config_path:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)

        if user_config is None:
            user_config = {}

        # Helper function to recursively update dataclass fields
        def update_dataclass(dc, updates):
            for key, value in updates.items():
                if hasattr(dc, key):
                    attr = getattr(dc, key)
                    if isinstance(attr, (APICosts, LoggingConfig, ModelConfig, TimeoutSettings, TranscriptionSettings)):
                        update_dataclass(attr, value)
                    else:
                        setattr(dc, key, value)
                else:
                    setattr(dc, key, value)

        update_dataclass(config, user_config)

    # Override with environment variables if present
    config.pyannote_token = config.pyannote_token or os.getenv("PYANNOTE_TOKEN")
    config.openai_key = config.openai_key or os.getenv("OPENAI_KEY")
    
    # Ensure global debug and verbose flags override environment variables if needed
    config.logging.debug = config.logging.debug or os.getenv("DEBUG") == "true"
    config.logging.verbose = config.logging.verbose or os.getenv("VERBOSE") == "true"

    return config

# Example usage:
# config = load_config(args.config)