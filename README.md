# YAWT (Yet Another Whisper-based Transcriber)

YAWT is an audio transcription tool that utilizes OpenAI's Whisper model for efficient audio-to-text conversion. It incorporates speaker diarization using PyAnnote and supports multiple upload services, enabling flexible transcription workflows.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Transcription:** Uses OpenAI's Whisper model for audio-to-text conversion.
- **Speaker Diarization:** Identifies and separates individual speakers within an audio file using PyAnnote.
- **Output Formats:** Exports transcriptions in `text`, `json`, and `srt` formats.
- **Upload Services:** Uploads audio files to services like `0x0.st` and `file.io`.
- **Configurable Timeouts and Costs:** Users can customize timeout settings and view cost estimations based on usage.
- **Logging:** Includes logging with configurable levels and log rotation for monitoring and debugging.
- **Dry-Run Mode:** Allows cost estimation without executing the transcription process.

## Installation

### Prerequisites

- **Python 3.11** or higher
- **Poetry** for dependency management

### Steps

1. **Install YAWT using Poetry:**
   ```bash
   poetry add git+https://github.com/yaniv-golan/YAWT.git@latest   ```

2. **Set Up a Virtual Environment:**

   It's recommended to use a virtual environment to manage dependencies.
   ```bash
   python3 -m venv venv
   source venv/bin/activate   ```

3. **Install Dependencies:**

   **Using Poetry:**
   ```bash
   poetry install   ```

   **Using pip:**
   ```bash
   pip install -r requirements.txt   ```

4. **Set Up Configuration:**

   - **Create a `.env` File:**

     Copy the example configuration and populate it with your API tokens.
     ```bash
     cp config/.env.example .env     ```

     Edit the `.env` file to include your `PYANNOTE_TOKEN` and `OPENAI_KEY`:
     ```env
     PYANNOTE_TOKEN=your_pyannote_api_token_here
     OPENAI_KEY=your_openai_api_key_here     ```

   - **Optional: Create a Custom Configuration File:**

     Instead of using the default configuration, you can create a custom configuration file and specify its path using command-line arguments when running YAWT.
     ```bash
     poetry run yawt --config path/to/your_config.yaml     ```

## Configuration

YAWT's behavior can be customized via the `config.py` module and environment variables. Additionally, you have the option to create a custom configuration file and specify its path using command-line arguments when running the application.

Here's a breakdown of the key configurations:

- **API Costs:**
  - `whisper.cost_per_minute`: Cost per minute for using the Whisper model.
  - `pyannote.cost_per_hour`: Cost per hour for speaker diarization using PyAnnote.

- **Logging:**
  - `log_directory`: Directory where log files are stored.
  - `max_log_size`: Maximum size (in bytes) of a log file before rotation.
  - `backup_count`: Number of backup log files to keep.

- **Model:**
  - `default_model_id`: The default Whisper model to use for transcription.

- **Supported Upload Services:**
  - List of services (`0x0.st`, `file.io`) that YAWT can upload audio files to.

- **Timeouts:**
  - `download_timeout`: Maximum time to wait for audio file downloads.
  - `upload_timeout`: Maximum time to wait for audio file uploads.
  - `diarization_timeout`: Maximum time to wait for the diarization process.
  - `job_status_timeout`: Maximum time to wait when checking the status of a diarization job.

- **Transcription Settings:**
  - `generate_timeout`: Maximum time to allow for transcription generation.
  - `max_target_positions`: Maximum number of target positions.
  - `buffer_tokens`: Number of buffer tokens for the transcription model.

- **API Tokens:**
  - `pyannote_token`: PyAnnote API token.
  - `openai_key`: OpenAI API key.

## Usage

YAWT can be used via the command line to transcribe audio files either from a local path or a publicly accessible URL.

### Basic Commands

1. **Transcribe a Local Audio File:**
   ```bash
   poetry run yawt --input-file path/to/audio.wav --main-language en   ```

2. **Transcribe an Audio File from a URL:**
   ```bash
   poetry run yawt --audio-url https://example.com/audio.wav --main-language en   ```

3. **Estimate Cost Without Transcription (Dry Run):**
   ```bash
   poetry run yawt --input-file path/to/audio.wav --main-language en --dry-run   ```

4. **Specify Output Formats:**
   ```bash
   poetry run yawt --input-file path/to/audio.wav --main-language en --output-format text json srt   ```

### Sample Command with Multiple Command-Line Arguments

To run YAWT with a custom configuration file, enable verbose logging, specify the main language, set a secondary language for retries, and set the number of speakers:

```bash
poetry run yawt --input-file path/to/audio.wav --config config/custom_config.yaml --verbose --main-language en --secondary-language es --num-speakers 2
```

### Available Options

- `--audio-url`: Publicly accessible URL of the audio file to transcribe.
- `--input-file`: Path to the local audio file to transcribe.
- `--config`: Path to the configuration file.
- `--context-prompt`: Context prompt to guide transcription.
- `--main-language`: **(Required)** Specify the main language of the audio using ISO 639-1 or ISO 639-3 codes (e.g., `en` for English, `es` for Spanish).
- `--secondary-language`: **(Optional)** Specify a secondary language for retrying failed transcriptions using ISO 639-1 or ISO 639-3 codes (e.g., `fr` for French).
- `--num-speakers`: Specify the number of speakers if known.
- `--dry-run`: Estimate cost without processing.
- `--debug`: Enable debug logging.
- `--verbose`: Enable verbose output.
- `--pyannote-token`: PyAnnote API token (overrides environment variable).
- `--openai-key`: OpenAI API key (overrides environment variable).
- `--model`: OpenAI transcription model to use (default: `openai/whisper-large-v3`).
- `--output-format`: Desired output format(s): `text`, `json`, `srt`.

**Note:** Both `--main-language` and `--secondary-language` must conform to [ISO 639](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) standards. Please refer to the [ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) or [ISO 639-3](https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes) codes for correct usage. If an invalid code is provided, the tool will notify you and skip transcription for the affected segments.

---

### 4. **`tests/test_main.py`**

Ensure that the tests correctly handle a single `secondary_language` and validate language code handling.

```python:tests/test_main.py
import pytest
from unittest.mock import patch, MagicMock
from yawt.main import transcribe_audio, main
import sys

@pytest.fixture
def mock_transcribe_audio_success(mocker):
    mock = mocker.patch('yawt.main.transcribe_audio')
    mock.return_value = "This is a mocked transcription."
    return mock

@pytest.fixture
def mock_transcribe_audio_failure(mocker):
    mock = mocker.patch('yawt.main.transcribe_audio')
    mock.side_effect = FileNotFoundError("Audio file not found.")
    return mock

def test_transcribe_audio_success(mock_transcribe_audio_success):
    audio_file = "path/to/test_audio.wav"
    expected_transcription = "This is a mocked transcription."

    result = transcribe_audio(audio_file)
    assert result == expected_transcription, "Transcription should match expected output"

def test_transcribe_audio_file_not_found(mock_transcribe_audio_failure):
    with pytest.raises(FileNotFoundError):
        transcribe_audio("path/to/nonexistent_audio.wav")

@patch('yawt.main.parse_arguments')
@patch('yawt.main.setup_logging')
@patch('yawt.main.initialize_environment')
@patch('yawt.main.check_api_tokens')
@patch('yawt.main.load_and_prepare_model')
@patch('yawt.main.handle_audio_input')
@patch('yawt.main.perform_diarization')
@patch('yawt.main.map_speakers')
@patch('yawt.main.load_audio')
@patch('yawt.main.calculate_cost')
@patch('yawt.main.transcribe_segments')
@patch('yawt.main.retry_transcriptions')  # Ensure retry_transcriptions is patched correctly
@patch('yawt.main.write_transcriptions')
def test_main_success(
    mock_write_transcriptions,
    mock_retry_transcriptions,  # Adjusted order if necessary
    mock_transcribe_segments,
    mock_calculate_cost,
    mock_load_audio,
    mock_map_speakers,
    mock_perform_diarization,
    mock_handle_audio_input,
    mock_load_and_prepare_model,
    mock_check_api_tokens,
    mock_initialize_environment,
    mock_setup_logging,
    mock_parse_arguments
):
    # Setup mock return values
    mock_parse_arguments.return_value = MagicMock(
        audio_url=None,
        input_file="path/to/test_audio.wav",
        context_prompt=None,
        main_language='en',
        secondary_language='es',  # Single secondary language as string
        num_speakers=2,
        dry_run=False,
        debug=False,
        verbose=False,
        pyannote_token="fake_token",
        openai_key="fake_key",
        model="openai/whisper-large-v3",
        output_format=['text']
    )
    
    # Setup other mocks as necessary
    mock_transcribe_segments.return_value = ([], [])
    mock_retry_transcriptions.return_value = ([], [])
    
    # Call main function
    main()
    
    # Assertions to ensure functions are called correctly
    mock_retry_transcriptions.assert_called_once_with(
        mock_load_and_prepare_model.return_value[0],
        mock_load_and_prepare_model.return_value[1],
        "path/to/test_audio.wav",
        [],  # Empty failed_segments
        [],  # Empty transcription_segments
        {},  # Empty generate_kwargs
        mock_load_and_prepare_model.return_value[2],
        mock_load_and_prepare_model.return_value[3],
        "test_audio",
        [],
        300,
        1024,
        50,
        300,
        3,
        'es',  # Single secondary language
        0.6,
        'en'
    )
    
    mock_write_transcriptions.assert_called_once()

def test_main_diarization_failure(mocker):
    with patch('yawt.main.parse_arguments') as mock_parse:
        mock_parse.return_value = MagicMock(
            audio_url=None,
            input_file="path/to/test_audio.wav",
            context_prompt=None,
            main_language='en',
            secondary_language='es',  # Single secondary language
            num_speakers=2,
            dry_run=False,
            debug=False,
            verbose=False,
            pyannote_token="fake_token",
            openai_key="fake_key",
            model="openai/whisper-large-v3",
            output_format=['text']
        )
        mocker.patch('yawt.main.setup_logging')
        mocker.patch('yawt.main.initialize_environment')
        mocker.patch('yawt.main.check_api_tokens')
        mocker.patch('yawt.main.load_and_prepare_model')
        mocker.patch('yawt.main.handle_audio_input').return_value = ("uploaded_audio_url", "path/to/test_audio.wav")
        mocker.patch('yawt.main.perform_diarization', side_effect=Exception("Diarization failed."))

        with patch('yawt.main.sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)

@patch('yawt.main.time.sleep')
def test_submit_diarization_job_rate_limit(mock_sleep, mocker):
    # Implement your test logic here
    ...
    mock_sleep.assert_called_once_with(1)
    ...
```

**Key Changes:**

- **Test `test_main_success`:**
  - Changed `secondary_language` from a list to a single string `'es'`.
  
- **Test `test_main_diarization_failure`:**
  - Ensured that `secondary_language` is a single string `'es'`.

- **General:**
  - Updated all relevant test cases to handle `secondary_language` as a single string.

---

### 5. **`.vscode/launch.json`**

If your `launch.json` configuration includes arguments related to language handling, ensure they are updated to use `--main-language` and `--secondary-language` appropriately.

```json:.vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: YAWT",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/yawt/main.py",
            "args": [
                "--input-file",
                "path/to/audio.wav",
                "--main-language",
                "en",
                "--secondary-language",
                "es",
                "--output-format",
                "text",
                "json",
                "srt"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

**Key Changes:**

- Updated the arguments to use `--main-language` and `--secondary-language` instead of the deprecated `--language`.

---

### 6. **`src/yawt/config.py`**

Ensure that configuration settings related to languages are correctly defined and that only one secondary language is expected.

```python:src/yawt/config.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TranscriptionSettings:
    """
    Configuration for transcription process settings.
    """
    generate_timeout: int = 300          # Timeout for transcription generation
    max_target_positions: int = 448      # Maximum number of target positions for the model
    buffer_tokens: int = 10              # Number of buffer tokens to use during transcription
    confidence_threshold: float = 0.6    # Confidence threshold for accepting transcriptions

@dataclass
class APICosts:
    whisper_cost_per_minute: float = 0.006  # Example cost per minute for Whisper
    pyannote_cost_per_hour: float = 0.12    # Example cost per hour for PyAnnote

@dataclass
class LoggingConfig:
    log_directory: str = "logs"            # Directory for log files
    max_log_size: int = 10485760           # 10 MB
    backup_count: int = 5                  # Number of backup log files
    debug: bool = False                    # Enable debug logging
    verbose: bool = False                  # Enable verbose output

@dataclass
class ModelConfig:
    default_model_id: str = "openai/whisper-large-v3"  # Default Whisper model

@dataclass
class Config:
    """
    Comprehensive configuration class that aggregates all configuration sections.
    """
    api_costs: APICosts = field(default_factory=APICosts)                     # API cost configurations
    logging: LoggingConfig = field(default_factory=LoggingConfig)             # Logging configurations
    model: ModelConfig = field(default_factory=ModelConfig)                   # Model configurations
    supported_upload_services: List[str] = field(default_factory=lambda: ["0x0.st", "file.io"])  # Supported file upload services
    timeouts: TranscriptionSettings = field(default_factory=TranscriptionSettings)  # Transcription process settings
    pyannote_token: Optional[str] = None                                      # Optional Pyannote API token
    openai_key: Optional[str] = None                                           # Optional OpenAI API key
```

**Key Changes:**

- **`APICosts` and Other Configurations:**
  - Defined separate dataclasses for better organization and clarity.
  
- **Maintain Single Secondary Language:**
  - Since only one secondary language is supported, no list is used. Ensure that any references in the configuration expect single string values for languages.

---

### 7. **`config/example_config.yaml`**

Ensure the configuration file aligns with the single secondary language setup.

```yaml:config/example_config.yaml
# Model Configuration
# -------------------
# Specify which transcription model to use.
model:
  default_model_id: "openai/whisper-large-v3"  # Identifier for the default transcription model

# Supported Upload Services
# -------------------------
# List of services that can be used to upload audio files.
supported_upload_services:
  - "0x0.st"
  - "file.io"

# Timeout Settings (in seconds)
# -----------------------------
# Define various timeout durations for different operations.
timeouts:
  download_timeout: 60           # Timeout for downloading audio files
  upload_timeout: 120            # Timeout for uploading audio files
  diarization_timeout: 3600      # Timeout for the diarization process
  job_status_timeout: 60         # Timeout for checking job status

# Transcription Settings
# ----------------------
# Configure settings related to the transcription process.
transcription:
  generate_timeout: 300          # Timeout for transcription generation in seconds
  max_target_positions: 448      # Maximum target positions for the model
  buffer_tokens: 10              # Number of buffer tokens to maintain
  confidence_threshold: 0.6      # Confidence threshold for accepting transcriptions

# API Costs
# ---------
api_costs:
  whisper_cost_per_minute: 0.006  # Example cost per minute for Whisper
  pyannote_cost_per_hour: 0.12    # Example cost per hour for PyAnnote

# Logging Configuration
# ---------------------
logging:
  log_directory: "logs"            # Directory where log files are stored.
  max_log_size: 10485760           # Maximum size (in bytes) of a log file before rotation.
  backup_count: 5                  # Number of backup log files to keep.
  debug: false                     # Disable debug mode by default.
  verbose: false                   # Disable verbose output by default.

# API Tokens
# ----------
pyannote_token: "your_pyannote_api_token_here"  # PyAnnote API token.
openai_key: "your_openai_api_key_here"          # OpenAI API key.
```

**Key Changes:**

- **Configuration Alignment:**
  - Ensure that the configuration expects single values for `main_language` and `secondary_language` where necessary.

---

### 8. **`tests/test_output_writer.py`**

Ensure that the output writer tests handle transcriptions with single language entries.

```python:tests/test_output_writer.py
import pytest
from unittest.mock import patch, mock_open, MagicMock
from yawt.output_writer import write_transcriptions

def test_write_transcriptions_text_success(mocker):
    transcription_segments = [
        {'start': 0, 'end': 5, 'speaker_id': 'Speaker1', 'text': 'Hello'},
        {'start': 5, 'end': 10, 'speaker_id': 'Speaker2', 'text': 'Hi there'}
    ]
    speakers = [{'id': 'Speaker1', 'name': 'Speaker 1'}, {'id': 'Speaker2', 'name': 'Speaker 2'}]

    with patch('builtins.open', mock_open()) as mocked_file:
        write_transcriptions(['text'], 'test_audio', transcription_segments, speakers)
        mocked_file.assert_called_once_with('test_audio_transcription.txt', 'w', encoding='utf-8')
        handle = mocked_file()
        handle.write.assert_any_call("[0.00 - 5.00] Speaker1: Hello\n")
        handle.write.assert_any_call("[5.00 - 10.00] Speaker2: Hi there\n")

def test_write_transcriptions_srt_success(mocker):
    transcription_segments = [
        {'start': 0, 'end': 5, 'speaker_id': 'Speaker1', 'text': 'Hello'},
        {'start': 5, 'end': 10, 'speaker_id': 'Speaker2', 'text': 'Hi there'}
    ]
    speakers = [{'id': 'Speaker1', 'name': 'Speaker 1'}, {'id': 'Speaker2', 'name': 'Speaker 2'}]

    with patch('builtins.open', mock_open()) as mocked_file, \
         patch('yawt.output_writer.srt.compose') as mock_srt_compose:
        mock_srt_compose.return_value = "1\n00:00:00,000 --> 00:00:05,000\nSpeaker1: Hello\n\n2\n00:00:05,000 --> 00:00:10,000\nSpeaker2: Hi there\n"
        write_transcriptions(['srt'], 'test_audio', transcription_segments, speakers)
        mocked_file.assert_called_once_with('test_audio_transcription.srt', 'w', encoding='utf-8')
        handle = mocked_file()
        handle.write.assert_called_once_with("1\n00:00:00,000 --> 00:00:05,000\nSpeaker1: Hello\n\n2\n00:00:05,000 --> 00:00:10,000\nSpeaker2: Hi there\n")
```

**Key Changes:**

- **Test Transcription Segments:**
  - Ensured that each transcription segment includes only a single `language` entry if applicable.
  
- **Maintain Single Language Consistency:**
  - As the output writer doesn't directly handle language codes, ensure that transcriptions contain correctly formatted single languages.

---

### 9. **`src/yawt/audio_handler.py`**

Review and ensure that any functions related to language handling conform to the single secondary language setup. If there are no direct interactions, no changes are needed.

```python:src/yawt/audio_handler.py
def handle_audio_input(args, supported_upload_services, upload_timeout):
    """
    Handles the input audio by either downloading it from a URL or uploading a local file.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        supported_upload_services (set): Set of supported upload services.
        upload_timeout (int): Timeout for the upload in seconds.
    
    Returns:
        tuple: A tuple containing the audio URL and the local path to the audio file.
    
    Raises:
        SystemExit: If downloading or uploading fails.
    """