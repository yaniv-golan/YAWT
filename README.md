# YAWT (Yet Another Whisper-based Transcriber)

YAWT is a powerful and flexible audio transcription tool that leverages OpenAI's Whisper model to provide accurate and efficient audio-to-text conversion. With built-in speaker diarization using PyAnnote and support for multiple upload services, YAWT is designed to cater to diverse transcription needs with ease.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Accurate Transcription:** Utilizes OpenAI's Whisper model for high-quality audio-to-text conversion.
- **Speaker Diarization:** Differentiates between multiple speakers in an audio file using PyAnnote.
- **Multiple Output Formats:** Supports exporting transcriptions in `text`, `json`, and `srt` formats.
- **Upload Services Integration:** Seamlessly uploads audio files to services like `0x0.st` and `file.io`.
- **Configurable Timeouts and Costs:** Allows customization of timeout settings and provides cost estimations based on usage.
- **Logging:** Comprehensive logging with configurable log levels and log rotation to monitor and debug processes.
- **Dry-Run Mode:** Estimate processing costs without executing the actual transcription.

## Installation

### Prerequisites

- **Python 3.11.4** or higher
- **Poetry** for dependency management (optional but recommended)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/yawt.git
   cd yawt
   ```

2. **Set Up a Virtual Environment:**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   **Using Poetry:**

   ```bash
   poetry install
   ```

   **Using pip:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Configuration:**

   - **Create a `.env` File:**

     Copy the example configuration and populate it with your API tokens.

     ```bash
     cp config/.env.example .env
     ```

     Edit the `.env` file to include your `PYANNOTE_TOKEN` and `OPENAI_KEY`:

     ```env
     PYANNOTE_TOKEN=your_pyannote_api_token_here
     OPENAI_KEY=your_openai_api_key_here
     ```

   - **Configure `config/default_config.yaml`:**

     Adjust the configurations as needed, such as API costs, logging settings, and supported upload services.

     ```yaml
     # API Costs
     api_costs:
       whisper:
         cost_per_minute: 0.006  # USD per minute for Whisper
       pyannote:
         cost_per_hour: 0.18     # USD per hour for diarization

     # Logging Configuration
     logging:
       log_directory: "logs"
       max_log_size: 10485760      # 10 MB in bytes
       backup_count: 5

     # Model Configuration
     model:
       default_model_id: "openai/whisper-large-v3"

     # Supported Services
     supported_upload_services:
       - "0x0.st"
       - "file.io"

     # Timeout Settings (in seconds)
     timeouts:
       download_timeout: 60  # Default download timeout
       upload_timeout: 120    # Default upload timeout
       diarization_timeout: 3600
       job_status_timeout: 60

     # Transcription Settings
     transcription:
       generate_timeout: 300  # Timeout for transcription in seconds
       max_target_positions: 448
       buffer_tokens: 10  # Reduced from 445 to 10

     # API Tokens
     # These can also be set via environment variables in the .env file
     # pyannote_token: "your_pyannote_api_token_here"
     # openai_key: "your_openai_api_key_here"
     ```

## Configuration

YAWT's behavior can be customized via the `config/default_config.yaml` file and environment variables. Here's a breakdown of the key configurations:

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
   python src/yawt/main.py --input-file path/to/audio.wav
   ```

2. **Transcribe an Audio File from a URL:**

   ```bash
   python src/yawt/main.py --audio-url https://example.com/audio.wav
   ```

3. **Estimate Cost Without Transcription (Dry Run):**

   ```bash
   python src/yawt/main.py --input-file path/to/audio.wav --dry-run
   ```

4. **Specify Output Formats:**

   ```bash
   python src/yawt/main.py --input-file path/to/audio.wav --output-format text json srt
   ```

### Available Options

- `--audio-url`: Publicly accessible URL of the audio file to transcribe.
- `--input-file`: Path to the local audio file to transcribe.
- `--config`: Path to the configuration file.
- `--context-prompt`: Context prompt to guide transcription.
- `--language`: Specify the language(s) of the audio.
- `--num-speakers`: Specify the number of speakers if known.
- `--dry-run`: Estimate cost without processing.
- `--debug`: Enable debug logging.
- `--verbose`: Enable verbose output.
- `--pyannote-token`: PyAnnote API token (overrides environment variable).
- `--openai-key`: OpenAI API key (overrides environment variable).
- `--model`: OpenAI transcription model to use (default: `openai/whisper-large-v3`).
- `--output-format`: Desired output format(s): `text`, `json`, `srt`.

## Testing

YAWT includes a comprehensive test suite to ensure reliability and correctness. Tests are written using `pytest` and utilize `unittest.mock` for mocking external dependencies.

### Running Tests

1. **Ensure All Dependencies Are Installed:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run All Tests:**

   ```bash
   pytest
   ```

3. **Run a Specific Test:**

   ```bash
   pytest tests/test_audio_handler.py::test_load_audio_ffmpeg_error
   ```

### Test Coverage

The test suite covers various components, including:

- **Audio Handling (`tests/test_audio_handler.py`):**  
  Tests for loading audio, handling FFmpeg errors, uploading files to supported services, and downloading audio files.

- **Diarization (`tests/test_diarization.py`):**  
  Tests for submitting diarization jobs, handling rate limits, and checking job statuses.

- **Main Application (`tests/test_main.py`):**  
  Tests for the main transcription flow, including successful transcriptions and handling failures.

- **Logging Setup (`tests/test_logging_setup.py`):**  
  Tests for configuring logging based on different settings.

- **Transcription (`tests/test_transcription.py`):**  
  Tests for transcription generation, handling timeouts, and retry mechanisms.

## Contributing

Contributions are welcome! If you'd like to enhance YAWT, please follow these steps:

1. **Fork the Repository.**

2. **Create a New Branch:**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Make Your Changes and Commit:**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to Your Fork:**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request.**

Please ensure your code adheres to the existing style and passes all tests.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Note:** Ensure that you replace placeholder URLs, paths, and other details with actual values relevant to your project.