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
- [Acknowledgments](#acknowledgments)
- [Building Executables](#building-executables)

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

- Python 3.10
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

### Transcribing a Local Audio File to Text

```bash
yawt --input-file path/to/audio.wav --output-format text --main-language en
```

### Transcribing from a URL and Exporting to SRT

```bash
yawt --audio-url https://example.com/audio.mp3 --output-format srt --main-language en --num-speakers 3
```

### Transcribing and Exporting to Standard Transcription JSON (STJ)

```bash
yawt --input-file path/to/audio.wav --output-format stj --main-language en
```

The transcription will be saved in a file with the `.stj.json` extension.

### Estimating Costs Without Running the Transcription

```bash
yawt --dry-run --input-file path/to/audio.wav
```

## Command-line Arguments

- `--audio-url`: Publicly accessible URL of the audio file to transcribe (required if no input file provided).
- `--input-file`: Path to the local audio file to transcribe (required if no URL provided).
- `--config`: Path to the configuration file.
- `--context-prompt`: Context prompt to guide transcription.
- `--main-language`: Main language of the audio (required).
- `--secondary-language`: Secondary language of the audio for retry attempts.
- `--num-speakers`: Number of speakers to detect (auto-detect if not specified).
- `--dry-run`: Estimate costs without processing.
- `--debug`: Enable debug logging.
- `--verbose`: Enable verbose output.
- `--pyannote-token`: Pyannote API token (overrides environment variable).
- `--openai-key`: OpenAI API key (overrides environment variable).
- `--model`: Specify the OpenAI Whisper model to use (default: "openai/whisper-large-v3", choices: "openai/whisper-large-v3" or "openai/whisper-large-v3-turbo").
- `--output-format`: Desired output format(s): text, stj, srt (default: "text").
- `-o`, `--output`: Base path for output files (without extension).

Note: Either `--audio-url` or `--input-file` must be provided, but not both.

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

## Acknowledgments

Special thanks to [Lior Atias](https://www.linkedin.com/in/lioratias/) for allowing the inclusion of our podcast conversation as a test file in this project. The episode is in Hebrew, with intermixed English terms, so `--main-language he --secondary-language he` :

- Local test file: `samples/multi-lang/podcast - Lior Atias - Executive Summary/Lior Atias - Executive Summary - Oct 22, 2024 - with Yaniv Golan.mp3`
- Original episode: [Listen on Spotify](https://open.spotify.com/episode/3EpOe03k99TJPdy2HXILRN?si=9wt19jz9T8GQTwd1bG-fUg)

## Building Executables

Build for the current platform:

```bash
python freeze.py build
```

The executable will be created in `build/` directory with a platform-specific name:

- macOS ARM: `yawt-macos-arm64`
- macOS Intel: `yawt-macos-x86_64`
- Linux: `yawt-linux-x86_64`
- Windows: `yawt-win-x86_64.exe`

Note: For best compatibility, build on each target platform separately.
