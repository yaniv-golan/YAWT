#!/usr/bin/env python3

import sys
import os

# Add the parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import os
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
from datetime import datetime

# Constants and Configuration
from config import (
    load_and_prepare_config
)

# Setup environment variable
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Logging setup
from yawt.logging_setup import setup_logging

from audio_handler import load_audio, upload_file, download_audio, handle_audio_input
from diarization import submit_diarization_job, wait_for_diarization, perform_diarization
from yawt.transcription import (
    transcribe_single_segment,
    retry_transcriptions,
    load_and_optimize_model,
    transcribe_segments  # Added transcribe_segments to the import
)
from output_writer import write_transcriptions

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
        # Tokenize the context prompt
        prompt_encoded = processor.tokenizer(args.context_prompt, return_tensors="pt", add_special_tokens=False)
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
            speaker_id = f"Speaker{speaker_counter}"
            speaker_mapping[speaker] = {'id': speaker_id, 'name': f'Speaker {speaker_counter}'}
            speakers.append({'id': speaker_id, 'name': f'Speaker {speaker_counter}'})
            speaker_counter += 1
        segment['speaker_id'] = speaker_mapping[speaker]['id']
    return speakers

def validate_output_formats(formats):
    valid = {'text', 'json', 'srt'}
    if isinstance(formats, list):
        # Join all elements and split again to handle cases where formats are passed as separate arguments
        formats = ' '.join(formats).replace(',', ' ').split()
    elif isinstance(formats, str):
        formats = formats.replace(',', ' ').split()
    formats = [fmt.strip().lower() for fmt in formats if fmt.strip()]
    invalid = set(formats) - valid
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid formats: {', '.join(invalid)}. Choose from text, json, srt.")
    return formats

def calculate_cost(duration_seconds, cost_per_minute, pyannote_cost_per_hour):
    minutes = duration_seconds / 60
    hours = duration_seconds / 3600
    whisper = minutes * cost_per_minute
    diarization = hours * pyannote_cost_per_hour
    total = whisper + diarization
    return whisper, diarization, total

def parse_arguments():
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio-url', type=str, help='Publicly accessible URL of the audio file to transcribe.')
    group.add_argument('--input-file', type=str, help='Path to the local audio file to transcribe.')

    # Added the --config argument
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
    try:
        # Load and validate configurations
        config = load_and_prepare_config()
        
        # Parse command-line arguments
        args = parse_arguments()
        
        # Setup logging based on configuration
        setup_logging(
            log_directory=config['logging']['log_directory'],
            max_log_size=config['logging']['max_log_size'],
            backup_count=config['logging']['backup_count'],
            debug=config['transcription'].get('debug', False),
            verbose=config['transcription'].get('verbose', False)
        )
        
        logging.info("Script started.")
    
        # **Retrieve API tokens with correct precedence**
        pyannote_token = args.pyannote_token or config.get('pyannote_token') or os.getenv("PYANNOTE_TOKEN")
        openai_key = args.openai_key or config.get('openai_key') or os.getenv("OPENAI_KEY")
    
        # **Add logging to verify where tokens are loaded from**
        if args.pyannote_token:
            logging.debug("Pyannote token loaded from command-line arguments.")
        elif config.get('pyannote_token'):
            logging.debug("Pyannote token loaded from config file.")
        elif os.getenv("PYANNOTE_TOKEN"):
            logging.debug("Pyannote token loaded from environment variable.")
        else:
            logging.error("Pyannote token not found in args, config, or environment variables.")
    
        if args.openai_key:
            logging.debug("OpenAI key loaded from command-line arguments.")
        elif config.get('openai_key'):
            logging.debug("OpenAI key loaded from config file.")
        elif os.getenv("OPENAI_KEY"):
            logging.debug("OpenAI key loaded from environment variable.")
        else:
            logging.error("OpenAI key not found in args, config, or environment variables.")
    
        # Check if API tokens are set
        check_api_tokens(pyannote_token, openai_key)
    
        # Validate output formats
        try:
            args.output_format = validate_output_formats(args.output_format)
        except argparse.ArgumentTypeError as e:
            parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
            parser.error(str(e))
    
        print(f"Output formats: {args.output_format}")  # Debugging line
    
        # Load and optimize the model
        model_id = args.model or config['model']['default_model_id']  # Use config default if args.model is None
        model, processor, device, torch_dtype = load_and_optimize_model(model_id)
    
        # Integrate context prompt if provided
        decoder_input_ids = integrate_context_prompt(args, processor, device, torch_dtype)
    
        # Handle audio input
        audio_url, local_audio_path = handle_audio_input(
            args=args,
            supported_upload_services=config['supported_upload_services'],
            upload_timeout=config['timeouts']['upload_timeout']
        )
    
        # Determine base name for output files
        base_name = os.path.splitext(os.path.basename(audio_url if args.audio_url else args.input_file))[0]
        logging.info(f"Base name for outputs: {base_name}")
    
        # Submit and wait for diarization job
        try:
            diarization_segments = perform_diarization(
                pyannote_token, 
                audio_url, 
                args.num_speakers, 
                config['timeouts']['diarization_timeout'],
                config['timeouts']['job_status_timeout']  # Pass job_status_timeout
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
    
        # Map speakers to identifiers
        speakers = map_speakers(diarization_segments)
    
        # Load audio data
        audio_array = load_audio(local_audio_path)
        total_duration = len(audio_array) / 16000  # Assuming 16kHz sampling rate
        whisper_cost, diarization_cost, total_cost = calculate_cost(
            total_duration, config['api_costs']['whisper']['cost_per_minute'], config['api_costs']['pyannote']['cost_per_hour']
        )
    
        # Handle dry-run option
        if args.dry_run:
            print(f"Estimated cost: ${total_cost:.4f} USD")
            sys.exit(0)
    
        logging.info(f"Processing cost: Whisper=${whisper_cost:.4f}, Diarization=${diarization_cost:.4f}, Total=${total_cost:.4f}")
    
        # Transcribe all segments
        transcription_segments, failed_segments = transcribe_segments(
            args,
            diarization_segments,
            audio_array,
            model,
            processor,
            device,
            torch_dtype,
            generate_timeout=config['transcription']['generate_timeout'],
            max_target_positions=config['transcription']['max_target_positions'],
            buffer_tokens=config['transcription']['buffer_tokens'],
            transcription_timeout=config['transcription']['generate_timeout'],  # Assuming transcription_timeout is same as generate_timeout
            generate_kwargs={"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}  # {{ edit: pass generate_kwargs }}
        )
    
        # Retry failed segments
        if failed_segments:
            logging.info("Retrying failed segments...")
            failed_segments = retry_transcriptions(
                model,
                processor,
                audio_array,
                diarization_segments,
                failed_segments,
                {"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {},  # {{ edit: pass generate_kwargs }}
                device,
                torch_dtype,
                base_name,
                transcription_segments,
                generate_timeout=config['transcription']['generate_timeout'],
                max_target_positions=config['transcription']['max_target_positions'],
                buffer_tokens=config['transcription']['buffer_tokens'],
                transcription_timeout=config['transcription']['generate_timeout'],  # Assuming same as generate_timeout
            )
    
        # Write transcriptions to specified formats AFTER handling retries
        write_transcriptions(args.output_format, base_name, transcription_segments, speakers)
    
        # Report any remaining failed segments after retries
        if failed_segments:
            logging.warning("Some segments failed to transcribe after all retry attempts:")
            for failure in failed_segments:
                logging.warning(f"Segment {failure['segment']}: {failure['reason']}")
            print("Some segments failed to transcribe after retries. Check logs for details.")
    
        # Recalculate costs if retries were attempted
        whisper_cost, diarization_cost, total_cost = calculate_cost(
            total_duration, config['api_costs']['whisper']['cost_per_minute'], config['api_costs']['pyannote']['cost_per_hour']
        )
        print(f"\nTotal Duration: {total_duration:.2f}s")
        print(f"Transcription Cost: ${whisper_cost:.4f} USD")
        print(f"Diarization Cost: ${diarization_cost:.4f} USD")
        print(f"Total Estimated Cost: ${total_cost:.4f} USD\n")
    
        # Cleanup temporary audio file if using audio URL
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
