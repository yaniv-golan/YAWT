import pytest
from yawt.config import Config, SAMPLING_RATE, APICosts, LoggingConfig, ModelConfig, TimeoutSettings, TranscriptionSettings

def test_config_initialization():
    config = Config()
    assert isinstance(config, Config)

def test_sampling_rate():
    assert SAMPLING_RATE == 16000

def test_config_attributes():
    config = Config()
    expected_attributes = [
        'api_costs',
        'logging',
        'model',
        'supported_upload_services',
        'timeouts',
        'transcription',
        'pyannote_token',
        'openai_key'
    ]
    for attr in expected_attributes:
        assert hasattr(config, attr), f"Config should have attribute '{attr}'"

def test_api_costs():
    config = Config()
    assert isinstance(config.api_costs, APICosts)
    assert hasattr(config.api_costs, 'whisper_cost_per_minute')
    assert hasattr(config.api_costs, 'pyannote_cost_per_hour')

def test_logging_config():
    config = Config()
    assert isinstance(config.logging, LoggingConfig)
    assert hasattr(config.logging, 'log_directory')
    assert hasattr(config.logging, 'max_log_size')
    assert hasattr(config.logging, 'backup_count')
    assert hasattr(config.logging, 'debug')
    assert hasattr(config.logging, 'verbose')

def test_model_config():
    config = Config()
    assert isinstance(config.model, ModelConfig)
    assert hasattr(config.model, 'default_model_id')

def test_timeout_settings():
    config = Config()
    assert isinstance(config.timeouts, TimeoutSettings)
    assert hasattr(config.timeouts, 'download_timeout')
    assert hasattr(config.timeouts, 'upload_timeout')
    assert hasattr(config.timeouts, 'diarization_timeout')
    assert hasattr(config.timeouts, 'job_status_timeout')

def test_transcription_settings():
    config = Config()
    assert isinstance(config.transcription, TranscriptionSettings)
    assert hasattr(config.transcription, 'generate_timeout')
    assert hasattr(config.transcription, 'max_target_positions')
    assert hasattr(config.transcription, 'buffer_tokens')
    assert hasattr(config.transcription, 'confidence_threshold')
    assert hasattr(config.transcription, 'max_retries')

def test_supported_upload_services():
    config = Config()
    assert isinstance(config.supported_upload_services, list)
    assert len(config.supported_upload_services) > 0

def test_token_and_key():
    config = Config()
    assert hasattr(config, 'pyannote_token')
    assert hasattr(config, 'openai_key')

@pytest.mark.parametrize("attr,invalid_value", [
    ('confidence_threshold', 1.5),
    ('max_target_positions', 0),
    ('buffer_tokens', -1),
    ('max_retries', -1)
])
def test_invalid_transcription_settings(attr, invalid_value):
    with pytest.raises(ValueError) as exc_info:
        TranscriptionSettings(**{attr: invalid_value})
    
    assert str(exc_info.value) == (
        "confidence_threshold must be between 0 and 1." if attr == 'confidence_threshold' else
        "max_target_positions must be a positive integer." if attr == 'max_target_positions' else
        "buffer_tokens must be zero or a positive integer." if attr == 'buffer_tokens' else
        "max_retries must be zero or a positive integer."
    )

def test_custom_config():
    custom_values = {
        'api_costs': APICosts(whisper_cost_per_minute=0.007, pyannote_cost_per_hour=0.2),
        'logging': LoggingConfig(log_directory="/custom/log/dir", debug=True),
        'model': ModelConfig(default_model_id="custom_model"),
        'timeouts': TimeoutSettings(download_timeout=120),
        'transcription': TranscriptionSettings(max_target_positions=1024, buffer_tokens=15),
        'pyannote_token': "custom_token",
        'openai_key': "custom_key"
    }
    custom_config = Config(**custom_values)
    
    assert custom_config.api_costs.whisper_cost_per_minute == 0.007
    assert custom_config.api_costs.pyannote_cost_per_hour == 0.2
    assert custom_config.logging.log_directory == "/custom/log/dir"
    assert custom_config.logging.debug == True
    assert custom_config.model.default_model_id == "custom_model"
    assert custom_config.timeouts.download_timeout == 120
    assert custom_config.transcription.max_target_positions == 1024
    assert custom_config.transcription.buffer_tokens == 15
    assert custom_config.pyannote_token == "custom_token"
    assert custom_config.openai_key == "custom_key"
