#!/usr/bin/env python3

# 1. Import warnings and configure them before any other imports to suppress specific warnings from transformers
import warnings
warnings.filterwarnings("ignore", message=".*transformers.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*transformers.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*transformers.*", category=FutureWarning)

# 2. Now import the setup_logging function from the logging_setup module
from yawt.logging_setup import setup_logging

# 3. Initialize logging with specified parameters
setup_logging(
    log_directory="logs",
    max_log_size=10 * 1024 * 1024,  # 10 MB maximum log file size
    backup_count=5,                # Keep up to 5 backup log files
    debug=False,                   # Disable debug mode by default
    verbose=False                  # Disable verbose output by default
)

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
import ffmpeg
import tempfile
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from datetime import datetime, timedelta
import srt
from logging.handlers import RotatingFileHandler
import concurrent.futures

# Constants and Configuration
from yawt.config import (
    load_config,
    Config
)

# Setup environment variable to disable file validation in pydev debugger
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

from yawt.audio_handler import load_audio, upload_file, download_audio, handle_audio_input
from yawt.diarization import submit_diarization_job, wait_for_diarization, perform_diarization
from yawt.transcription import (
    transcribe_single_segment,
    retry_transcriptions,
    load_and_optimize_model,
    transcribe_segments  # Added transcribe_segments to the import
)
from yawt.output_writer import write_transcriptions

import logging

# Configure basic logging settings
logging.basicConfig(level=logging.INFO)

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
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio-url', type=str, help='Publicly accessible URL of the audio file to transcribe.')
    group.add_argument('--input-file', type=str, help='Path to the local audio file to transcribe.')

    # Added the --config argument for configuration file path
    parser.add_argument('--config', type=str, help='Path to the configuration file.')

    parser.add_argument('--context-prompt', type=str, help='Context prompt to guide transcription.')
    parser.add_argument('--language', type=str, nargs='+', help='Specify the language(s) of the audio.')
    parser.add_argument('--num-speakers', type=int, help='Specify the number of speakers if known.')
    parser.add_argument('--dry-run', action='store_true', help='Estimate cost without processing.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument("--pyannote-token", help="Pyannote API token (overrides environment variable)")
    parser.add_argument("--openai-key", help="OpenAI API key (overrides environment variable)")
    parser.add_argument("--model", default="openai/whisper-large-v3",  # Set a default model
                        help="OpenAI transcription model to use")
    parser.add_argument('--output-format', type=str, nargs='+',
                        default=['text'], help='Desired output format(s): text, json, srt.')
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
                
        # Setup logging based on configuration and command-line overrides
        setup_logging(
            log_directory=config.logging.log_directory,
            max_log_size=config.logging.max_log_size,
            backup_count=config.logging.backup_count,
            debug=args.debug or config.logging.debug,      # Override with --debug if provided
            verbose=args.verbose or config.logging.verbose # Override with --verbose if provided
        )
        
        logging.info("Script started.")
    
        # **Retrieve API tokens with correct precedence**
        pyannote_token = args.pyannote_token or config.pyannote_token or os.getenv("PYANNOTE_TOKEN")
        openai_key = args.openai_key or config.openai_key or os.getenv("OPENAI_KEY")
    
        # **Add logging to verify where tokens are loaded from**
        if args.pyannote_token:
            logging.debug("Pyannote token loaded from command-line arguments.")
        elif config.pyannote_token:
            logging.debug("Pyannote token loaded from config file.")
        elif os.getenv("PYANNOTE_TOKEN"):
            logging.debug("Pyannote token loaded from environment variable.")
        else:
            logging.error("Pyannote token not found in args, config, or environment variables.")
    
        if args.openai_key:
            logging.debug("OpenAI key loaded from command-line arguments.")
        elif config.openai_key:
            logging.debug("OpenAI key loaded from config file.")
        elif os.getenv("OPENAI_KEY"):
            logging.debug("OpenAI key loaded from environment variable.")
        else:
            logging.error("OpenAI key not found in args, config, or environment variables.")
    
        # Check if API tokens are set, exit if not
        check_api_tokens(pyannote_token, openai_key)
    
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
    
        # Handle audio input, either from URL or local file
        audio_url, local_audio_path = handle_audio_input(
            args=args,
            supported_upload_services=config.supported_upload_services,
            upload_timeout=config.timeouts.upload_timeout
        )
    
        # Determine base name for output files based on the input source
        base_name = os.path.splitext(os.path.basename(audio_url if args.audio_url else args.input_file))[0]
        logging.info(f"Base name for outputs: {base_name}")
    
        # Submit diarization job and wait for its completion
        try:
            diarization_segments = perform_diarization(
                pyannote_token, 
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
    
        logging.debug(f"Diarization Segments Before Mapping: {diarization_segments}")  
    
        # Map speakers to unique identifiers for clarity in outputs
        speakers = map_speakers(diarization_segments)
    
        # Load audio data into an array for processing
        audio_array = load_audio(local_audio_path)
        total_duration = len(audio_array) / 16000  # Assuming 16kHz sampling rate
        whisper_cost, diarization_cost, total_cost = calculate_cost(
            total_duration, config.api_costs.whisper_cost_per_minute, config.api_costs.pyannote_cost_per_hour
        )
    
        # Handle dry-run option to estimate costs without actual processing
        if args.dry_run:
            print(f"Estimated cost: ${total_cost:.4f} USD")
            sys.exit(0)
    
        logging.info(f"Processing cost: Whisper=${whisper_cost:.4f}, Diarization=${diarization_cost:.4f}, Total=${total_cost:.4f}")
    
        # Transcribe all segments of the audio
        transcription_segments, failed_segments = transcribe_segments(
            args,
            diarization_segments,
            audio_array,
            model,
            processor,
            device,
            torch_dtype,
            generate_timeout=config.transcription.generate_timeout,
            max_target_positions=config.transcription.max_target_positions,
            buffer_tokens=config.transcription.buffer_tokens,
            transcription_timeout=config.transcription.generate_timeout,  # Assuming transcription_timeout is same as generate_timeout
            generate_kwargs={"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}
        )
    
        # Retry transcription for any failed segments
        if failed_segments:
            logging.info("Retrying failed segments...")
            failed_segments = retry_transcriptions(
                model,
                processor,
                audio_array,
                diarization_segments,
                failed_segments,
                {"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {},
                device,
                torch_dtype,
                base_name,
                transcription_segments,
                generate_timeout=config.transcription.generate_timeout,
                max_target_positions=config.transcription.max_target_positions,
                buffer_tokens=config.transcription.buffer_tokens,
                transcription_timeout=config.transcription.generate_timeout,
            )
    
        # Write the transcriptions to the specified output formats after handling retries
        write_transcriptions(args.output_format, base_name, transcription_segments, speakers)
    
        # Report any remaining failed segments after all retry attempts
        if failed_segments:
            logging.warning("Some segments failed to transcribe after all retry attempts:")
            for failure in failed_segments:
                logging.warning(f"Segment {failure['segment']}: {failure['reason']}")
            print("Some segments failed to transcribe after retries. Check logs for details.")
    
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
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()