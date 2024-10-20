# YAWT (Yet Another Whisper-based Transcriber)

YAWT is a Python-based audio transcription tool that uses OpenAI's Whisper model for converting audio files to text. It integrates speaker diarization via PyAnnote and supports multiple languages for transcription, making it adaptable to multilingual audio content. It also supports various output formats and upload services.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Command-line Arguments](#command-line-arguments)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Multilingual Transcription:** Supports both primary and secondary languages for transcription, ensuring flexibility in handling multilingual audio files.
- **Speaker Diarization:** Identifies and separates speakers using PyAnnote for clear, speaker-specific transcription.
- **Output Formats:** Supports `text`, `stj`, and `srt` formats.
- **STJ Output Format:** Exports transcriptions in Standard Transcription JSON format for structured data interchange.
- **Upload Services:** Allows uploading audio files to services such as `0x0.st` and `file.io`.
- **Customizable Timeouts & Cost Estimation:** Adjustable timeout settings and estimated costs based on usage.
- **Logging:** Provides configurable logging levels and log rotation for monitoring and debugging.
- **Dry-Run Mode:** Offers cost estimation without executing the transcription.
- **Retry Mechanism:** Automatically retries failed transcription segments.
- **Flexible API Key Management:** OpenAI and PyAnnote API keys can be specified via command-line arguments or environment variables.

## Installation

### Prerequisites
- Python 3.11 or higher
- Poetry for managing dependencies

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yaniv-golan/YAWT.git
   cd YAWT
   ```
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
3. Set up environment variables (optional if not using command-line arguments for API keys):
   Create a `.env` file in the project root with the following keys:
   ```env
   PYANNOTE_TOKEN=your_pyannote_api_token
   OPENAI_KEY=your_openai_api_key
   ```

## Configuration

YAWT uses a YAML-based configuration file to control various options. A sample configuration is available in the repository at `config/example_config.yaml`.

To use this sample configuration:
```bash
cp config/example_config.yaml config/config.yaml
```
Edit `config.yaml` to suit your requirements.

## Usage

YAWT can be executed directly from the command line using the `yawt` command without needing `python` or `poetry` prefixes, thanks to the entry point defined in `pyproject.toml`.

### Transcribing a Local Audio File to Text:
```bash
yawt --input-file path/to/audio.wav --output-format text --main-language en
```

### Transcribing from a URL and Exporting to SRT:
```bash
yawt --audio-url https://example.com/audio.mp3 --output-format srt --main-language en --num-speakers 3
```

### Transcribing and Exporting to Standard Transcription JSON (STJ):
```bash
yawt --input-file path/to/audio.wav --output-format stj --main-language en
```

The transcription will be saved in a file with the `.stj.json` extension.

### Estimating Costs Without Running the Transcription:
```bash
yawt --dry-run --input-file path/to/audio.wav
```

## Command-line Arguments

- `--input-file`: Path to the local audio file to transcribe (required if no URL is provided).
- `--audio-url`: URL of the publicly accessible audio file to transcribe (required if no input file is provided).
- `--config`: Path to the configuration file (optional).
- `--context-prompt`: Context prompt to guide the transcription process (optional).
- `--main-language`: Main language of the audio (required).
- `--secondary-language`: Secondary language of the audio for retry attempts (optional).
- `--num-speakers`: Number of speakers to detect (auto-detect if not specified).
- `--dry-run`: Estimate costs without processing.
- `--debug`: Enable debug logging.
- `--verbose`: Enable verbose output.
- `--pyannote-token`: PyAnnote API token (overrides the environment variable).
- `--openai-key`: OpenAI API key (overrides the environment variable).
- `--model`: Specify the OpenAI Whisper model to use (default is `openai/whisper-large-v3`).
- `--output-format`: Desired output format(s), can specify one or more formats (default: `text`, options: `text`, `stj`, `srt`).
- `-o`, `--output`: Base path for output files (without extension).

For more information about the STJ format, see the [STJ Specification](https://github.com/yaniv-golan/STJ).

## Testing

YAWT uses `pytest` for testing. Run the test suite using the following command:
```bash
pytest
```
Ensure that all tests pass before submitting changes.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please ensure that your code follows PEP8 guidelines, and add or update tests as necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.