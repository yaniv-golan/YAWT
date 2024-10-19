#!/usr/bin/env python3

# 1. Import warnings and configure them before any other imports to suppress specific warnings from transformers
import warnings
warnings.filterwarnings("ignore", message=".*transformers.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*transformers.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*transformers.*", category=FutureWarning)

# 2. Now import the setup_logging function from the logging_setup module
from yawt.logging_setup import setup_logging

# 3. Initialize logging with specified parameters
# setup_logging(
#     log_directory="logs",
#     max_log_size=10 * 1024 * 1024,  # 10 MB maximum log file size
#     backup_count=5,                  # Keep up to 5 backup log files
#     debug=False,                     # Disable debug mode by default
#     verbose=False                    # Disable verbose output by default
# )

# 4. Import transformers and other necessary modules after logging is configured
import transformers

import sys
import os

# Add the parent directory to PYTHONPATH to ensure modules can be imported correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import requests
from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from dotenv import load_dotenv
import tempfile
from datetime import datetime, timedelta
import srt
import logging  # Import logging module to use logging throughout the script

# Constants and Configuration
from yawt.config import (
    load_config,
    Config,
    SAMPLING_RATE  # Import the SAMPLING_RATE constant
)

# Setup environment variable to disable file validation in pydev debugger
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

from yawt.audio_handler import load_audio, upload_file, download_audio, handle_audio_input
from yawt.diarization import submit_diarization_job, wait_for_diarization, perform_diarization
from yawt.transcription import (
    transcribe_segments,
    retry_transcriptions,
    load_and_optimize_model,
    ModelResources,          # Import ModelResources
    TranscriptionConfig      # Import TranscriptionConfig
)
from yawt.output_writer import write_transcriptions

from yawt.exceptions import ModelLoadError, DiarizationError, TranscriptionError  # Import custom exceptions

def check_api_tokens(pyannote_token, openai_key):
    """
    Checks if the required API tokens are set.

    Args:
        pyannote_token (str): Pyannote API token.
        openai_key (str): OpenAI API key.

    Raises:
        SystemExit: If any of the tokens are not set.
    """
    if not pyannote_token:
        logging.error("PYANNOTE_TOKEN is not set. Please provide it via the config file or environment variable.")
        sys.exit(1)
    
    if not openai_key:
        logging.error("OPENAI_KEY is not set. Please provide it via the config file or environment variable.")
        sys.exit(1)

def integrate_context_prompt(args, processor, device, torch_dtype):
    """
    Integrates context prompt into transcription by tokenizing and preparing decoder input ids.
    
    Args:
        args: Parsed command-line arguments.
        processor: The processor for the transcription model.
        device: The device to run the model on.
        torch_dtype: The data type for torch tensors.
    
    Returns:
        torch.Tensor or None: The decoder input ids if context prompt is provided, else None.
    """
    if args.context_prompt:
        logging.info("Integrating context prompt into transcription.")
        # Tokenize the context prompt without adding special tokens
        prompt_encoded = processor.tokenizer(args.context_prompt, return_tensors="pt", add_special_tokens=False)
        # Move the input ids to the specified device and dtype
        decoder_input_ids = prompt_encoded['input_ids'].to(device).to(torch_dtype)
        return decoder_input_ids
    return None

def map_speakers(diarization_segments):
    """
    Maps speaker labels to unique speaker IDs.

    Args:
        diarization_segments (list): List of diarization segments with 'speaker' key.

    Returns:
        list: List of speaker dictionaries with 'id' and 'name'.
    """
    speaker_mapping = {}
    speakers = []
    speaker_counter = 1
    for segment in diarization_segments:
        speaker = segment['speaker']
        if speaker not in speaker_mapping:
            # Assign a unique ID to each new speaker
            speaker_id = f"Speaker{speaker_counter}"
            speaker_mapping[speaker] = {'id': speaker_id, 'name': f'Speaker {speaker_counter}'}
            speakers.append({'id': speaker_id, 'name': f'Speaker {speaker_counter}'})
            speaker_counter += 1
        # Add speaker ID to the segment
        segment['speaker_id'] = speaker_mapping[speaker]['id']
    return speakers

def validate_output_formats(formats):
    """
    Validates the output formats specified by the user.

    Args:
        formats (str or list): Desired output formats.

    Returns:
        list: Validated list of output formats.

    Raises:
        argparse.ArgumentTypeError: If invalid formats are provided.
    """
    valid = {'text', 'json', 'srt'}
    if isinstance(formats, list):
        # Join all elements and split to handle comma-separated or space-separated inputs
        formats = ' '.join(formats).replace(',', ' ').split()
    elif isinstance(formats, str):
        formats = formats.replace(',', ' ').split()
    # Clean and lowercase the format strings
    formats = [fmt.strip().lower() for fmt in formats if fmt.strip()]
    invalid = set(formats) - valid
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid formats: {', '.join(invalid)}. Choose from text, json, srt.")
    return formats

def calculate_cost(duration_seconds, cost_per_minute, pyannote_cost_per_hour):
    """
    Calculates the estimated cost based on audio duration.

    Args:
        duration_seconds (float): Duration of the audio in seconds.
        cost_per_minute (float): Cost per minute for transcription.
        pyannote_cost_per_hour (float): Cost per hour for diarization.

    Returns:
        tuple: Whisper cost, Diarization cost, Total cost.
    """
    minutes = duration_seconds / 60
    hours = duration_seconds / 3600
    whisper = minutes * cost_per_minute
    diarization = hours * pyannote_cost_per_hour
    total = whisper + diarization
    return whisper, diarization, total

def parse_arguments():
    """
    Parses command-line arguments provided by the user.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    import argparse  # Ensure argparse is imported if not already

    # Custom ArgumentParser to override the default error message
    class CustomArgumentParser(argparse.ArgumentParser):
        def error(self, message):
            if 'one of the arguments --audio-url --input-file is required' in message:
                self.print_usage(sys.stderr)
                self.exit(2, 'Error: You must provide either --audio-url or --input-file.\n')
            else:
                super().error(message)
    
    # Use the CustomArgumentParser instead of the default ArgumentParser
    parser = CustomArgumentParser(description="Transcribe audio with speaker diarization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio-url', type=str, help='Publicly accessible URL of the audio file to transcribe.')
    group.add_argument('--input-file', type=str, help='Path to the local audio file to transcribe.')

    # Added the --config argument for configuration file path
    parser.add_argument('--config', type=str, help='Path to the configuration file.')

    parser.add_argument('--context-prompt', type=str, help='Context prompt to guide transcription.')
    parser.add_argument('--main-language', type=str, required=True, help='Main language of the audio.')
    parser.add_argument('--secondary-language', type=str, help='Secondary language of the audio.')  # Remains optional
    parser.add_argument('--num-speakers', type=int, help='Specify the number of speakers if known.')
    parser.add_argument('--dry-run', action='store_true', help='Estimate cost without processing.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument("--pyannote-token", help="Pyannote API token (overrides environment variable)")
    parser.add_argument("--openai-key", help="OpenAI API key (overrides environment variable)")
    parser.add_argument("--model", default="openai/whisper-large-v3",   # Corrected default model
                        help="OpenAI transcription model to use")
    parser.add_argument('--output-format', type=str, nargs='+',
                        default=['text'], help='Desired output format(s): text, json, srt.')
    parser.add_argument("-o", "--output", help="Base path for output files (without extension)")
    return parser.parse_args()

def main():
    """
    Main function to orchestrate the transcription and diarization process.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Load and validate configurations from the config file
        config = load_config(args.config)
                
        # Setup logging using command-line arguments and configuration
        setup_logging(
            log_directory=config.logging.log_directory,
            max_log_size=config.logging.max_log_size,
            backup_count=config.logging.backup_count,
            debug=args.debug or config.logging.debug,      # Override with --debug if provided
            verbose=args.verbose or config.logging.verbose # Override with --verbose if provided
        )
        
        logging.info("Script started.")
    
        # Load and log API tokens
        config.load_and_log_tokens(args)  # Removed the 'logging' argument
    
        # Check if API tokens are set, exit if not
        check_api_tokens(config.pyannote_token, config.openai_key)
    
        # Validate output formats specified by the user
        try:
            args.output_format = validate_output_formats(args.output_format)
        except argparse.ArgumentTypeError as e:
            parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
            parser.error(str(e))
        
        print(f"Output formats: {args.output_format}")  # Debugging line
    
        # Load and optimize the transcription model
        model_id = args.model or config.model.default_model_id  # Use config default if args.model is None
        model, processor, device, torch_dtype = load_and_optimize_model(model_id)

        # Integrate context prompt into the transcription process if provided
        decoder_input_ids = integrate_context_prompt(args, processor, device, torch_dtype)

        # Prepare generate_kwargs for initial transcription
        generate_kwargs = {}
        if decoder_input_ids is not None:
            generate_kwargs["decoder_input_ids"] = decoder_input_ids

        # Do not set generate_kwargs["language"] here
        # The language will be set within the transcription functions based on the main_language parameter

        # Create ModelResources instance
        model_resources = ModelResources(
            model=model,
            processor=processor,
            device=device,
            torch_dtype=torch_dtype,
            generate_kwargs=generate_kwargs
        )

        # Create TranscriptionConfig instance
        transcription_config = TranscriptionConfig(
            transcription_timeout=config.transcription.generate_timeout,
            max_target_positions=config.transcription.max_target_positions,
            buffer_tokens=config.transcription.buffer_tokens,
            confidence_threshold=config.transcription.confidence_threshold
        )
    
        # Handle audio input, either from URL or local file
        audio_url, local_audio_path = handle_audio_input(
            args=args,
            supported_upload_services=config.supported_upload_services,
            upload_timeout=config.timeouts.upload_timeout
        )
    
        # Determine base name for output files
        if args.output:
            base_name = args.output
        else:
            base_name = os.path.splitext(os.path.basename(audio_url if args.audio_url else args.input_file))[0]
        logging.info(f"Base name for outputs: {base_name}")

        # Submit diarization job and wait for its completion
        try:
            diarization_segments = perform_diarization(
                config.pyannote_token, 
                audio_url, 
                args.num_speakers, 
                config.timeouts.diarization_timeout,
                config.timeouts.job_status_timeout  # Pass job_status_timeout
            )
        except Exception as e:
            logging.exception(f"Diarization error: {e}")  # Capture stack trace
            if args.audio_url and os.path.exists(local_audio_path):
                try:
                    os.remove(local_audio_path)
                    logging.info(f"Deleted temporary file: {local_audio_path}")
                except Exception as cleanup_error:
                    logging.warning(f"Cleanup failed: {cleanup_error}")
            sys.exit(1)
    
        logging.debug(f"Diarization Segments Before Mapping: {diarization_segments}")  # Added back the debug line
    
        # Map speakers to unique identifiers for clarity in outputs
        speakers = map_speakers(diarization_segments)

        # Load audio data into an array for processing
        audio_array = load_audio(local_audio_path)
        total_duration = len(audio_array) / SAMPLING_RATE  
        whisper_cost, diarization_cost, total_cost = calculate_cost(
            total_duration, config.api_costs.whisper_cost_per_minute, config.api_costs.pyannote_cost_per_hour
        )

        # Handle dry-run option to estimate costs without actual processing
        if args.dry_run:
            print(f"Estimated cost: ${total_cost:.4f} USD")
            sys.exit(0)

        logging.info(f"Processing cost: Whisper=${whisper_cost:.4f}, Diarization=${diarization_cost:.4f}, Total=${total_cost:.4f}")

        # Initial transcription without secondary languages
        transcription_segments, failed_segments = transcribe_segments(
            diarization_segments=diarization_segments,
            audio_array=audio_array,
            model_resources=model_resources,
            config=transcription_config,
            main_language=args.main_language  # Now mandatory
        )

        # Retry transcription for any failed segments using secondary language if provided
        if failed_segments and args.secondary_language:
            logging.info("Retrying failed segments with secondary language...")
            transcription_segments, failed_segments = retry_transcriptions(
                audio_array=audio_array,
                diarization_segments=diarization_segments,
                failed_segments=failed_segments,
                transcription_segments=transcription_segments,
                model_resources=model_resources,
                config=transcription_config,
                secondary_language=args.secondary_language  # Now optional
            )

        # Write the transcriptions to the specified output formats after handling retries
        write_transcriptions(args.output_format, base_name, transcription_segments, speakers)

        if failed_segments:
            logging.warning(f"{len(failed_segments)} segments failed to transcribe after all retry attempts.")
            for failed_segment in failed_segments:
                logging.warning(f"Failed segment: {failed_segment}")

        # Recalculate costs if retries were attempted
        whisper_cost, diarization_cost, total_cost = calculate_cost(
            total_duration, config.api_costs.whisper_cost_per_minute, config.api_costs.pyannote_cost_per_hour
        )
        print(f"\nTotal Duration: {total_duration:.2f}s")
        print(f"Transcription Cost: ${whisper_cost:.4f} USD")
        print(f"Diarization Cost: ${diarization_cost:.4f} USD")
        print(f"Total Estimated Cost: ${total_cost:.4f} USD\n")

        # Cleanup temporary audio file if processing from a URL
        if args.audio_url:
            try:
                os.remove(local_audio_path)
                logging.info(f"Deleted temporary file: {local_audio_path}")
            except Exception as e:
                logging.warning(f"Failed to delete temporary file: {e}")

        logging.info("Process completed successfully.")
    except ModelLoadError as e:
        logging.error(f"Model loading failed: {e}")
        sys.exit(2)  # Exit code 2 for model loading issues
    except DiarizationError as e:
        logging.error(f"Diarization failed: {e}")
        sys.exit(3)  # Exit code 3 for diarization issues
    except TranscriptionError as e:
        logging.error(f"Transcription failed: {e}")
        sys.exit(4)  # Exit code 4 for transcription issues
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")
        sys.exit(1)  # Exit code 1 for general errors

if __name__ == '__main__':
    main()

