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
   poetry add git+https://github.com/yaniv-golan/YAWT.git@latest
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

   - **Optional: Create a Custom Configuration File:**

     Instead of using the default configuration, you can create a custom configuration file and specify its path using command-line arguments when running YAWT.

     ```bash
     poetry run yawt --config path/to/your_config.yaml
     ```

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
   poetry run yawt --input-file path/to/audio.wav
   ```

2. **Transcribe an Audio File from a URL:**

   ```bash
   poetry run yawt --audio-url https://example.com/audio.wav
   ```

3. **Estimate Cost Without Transcription (Dry Run):**

   ```bash
   poetry run yawt --input-file path/to/audio.wav --dry-run
   ```

4. **Specify Output Formats:**

   ```bash
   poetry run yawt --input-file path/to/audio.wav --output-format text json srt
   ```

### Sample Command with Multiple Command-Line Arguments

To run YAWT with a custom configuration file, enable verbose logging, specify the language, and set the number of speakers:

```bash
poetry run yawt --input-file path/to/audio.wav --config config/custom_config.yaml --verbose --language English --num-speakers 2
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

YAWT includes a set of tests to ensure basic functionality. Tests are written using `pytest` and utilize `unittest.mock` for mocking external dependencies. The test suite is not yet comprehensive and has not been integrated into the build process.

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
