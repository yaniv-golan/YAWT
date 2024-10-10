import pytest
from yawt.config import load_config, validate_config
import os
import yaml

def test_load_config():
    config = load_config("config/default_config.yaml")
    assert config is not None, "Configuration should not be None"
    assert "api_costs" in config, "Configuration should contain 'api_costs'"
    assert "logging" in config, "Configuration should contain 'logging'"
    assert "model" in config, "Configuration should contain 'model'"
    assert "supported_upload_services" in config, "Configuration should contain 'supported_upload_services'"
    assert "timeouts" in config, "Configuration should contain 'timeouts'"

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("config/nonexistent_config.yaml")

def test_validate_config_valid():
    config = load_config("config/default_config.yaml")
    # If load_config already validates, this should pass without exceptions
    validate_config(config)  # Should not raise

def test_validate_config_invalid_whisper_cost(tmp_path):
    # Create an invalid config with negative whisper cost
    invalid_config = {
        "api_costs": {
            "whisper": {
                "cost_per_minute": -0.01  # Invalid negative value
            },
            "pyannote": {
                "cost_per_hour": 0.18
            }
        },
        "logging": {
            "log_directory": "logs",
            "max_log_size": 10485760,
            "backup_count": 5
        },
        "model": {
            "default_model_id": "openai/whisper-large-v3"
        },
        "supported_upload_services": ["0x0.st", "file.io"],
        "timeouts": {
            "download_timeout": 120,
            "upload_timeout": 120,
            "diarization_timeout": 3600,
            "job_status_timeout": 60
        }
    }
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(invalid_config, f)
    
    with pytest.raises(ValueError) as exc_info:
        config = load_config(str(config_path))
        validate_config(config)
    assert "Whisper cost per minute must be non-negative." in str(exc_info.value)