#!/usr/bin/env python3

import argparse
import logging
import os
import sys
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
from datetime import datetime
import signal
import srt  # Import the srt module for subtitle generation
from datetime import timedelta
from logging.handlers import RotatingFileHandler

# At the beginning of your script, after importing the necessary modules:
MAX_TARGET_POSITIONS = 448  # This is the max_target_positions for the Whisper model
BUFFER_TOKENS = 3  # This accounts for the special tokens and prompt tokens

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Set up logging
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_directory, f"transcription_log_{current_time}.log")

def load_audio(input_file, sampling_rate=16000):
    try:
        # Use ffmpeg to read the audio file and resample it to the target sampling rate
        out, _ = (
            ffmpeg.input(input_file, threads=0)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ar=sampling_rate, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.float32)
        return audio
    except ffmpeg.Error as e:
        logging.error(f"Failed to load audio file with ffmpeg: {e.stderr.decode()}")
        sys.exit(1)

def upload_file(file_path, service='0x0.st', secret=None, expires=None):
    """
    Uploads the file to the specified service and returns the direct download URL.
    Supported services: '0x0.st', 'file.io'
    Optional parameters:
      - secret: If provided, generates a longer, hard-to-guess URL.
      - expires: Sets the file's expiration time in hours or milliseconds since the UNIX epoch.
    """
    try:
        if service == '0x0.st':
            url = 'https://0x0.st'
            headers = {
                'User-Agent': 'TranscriberWithContext/1.0 (your_email@example.com)'  # Replace with your info
            }

            logging.info(f"Preparing to upload '{file_path}' to '{service}'...")

            # Initialize MultipartEncoder with the file and optional fields
            with open(file_path, 'rb') as f:
                encoder = MultipartEncoder(
                    fields={
                        'file': (os.path.basename(file_path), f),
                    }
                )

                # Add optional fields if provided
                fields = {'Content-Type': encoder.content_type}
                if secret:
                    encoder.fields['secret'] = secret
                if expires:
                    encoder.fields['expires'] = expires

                # Define a callback to update the progress bar
                def progress_callback(monitor):
                    pbar.update(monitor.bytes_read - pbar.n)

                # Wrap the encoder with a monitor
                monitor = MultipartEncoderMonitor(encoder, progress_callback)

                # Initialize progress bar
                with tqdm(total=encoder.len, unit='B', unit_scale=True, desc="Uploading") as pbar:
                    logging.info(f"Starting upload to '{service}'...")
                    response = requests.post(
                        url,
                        data=monitor,
                        headers={'User-Agent': headers['User-Agent'], 'Content-Type': monitor.content_type},
                        timeout=120
                    )

            if response.status_code == 200:
                file_url = response.text.strip()
                logging.info(f"File uploaded successfully to '{service}': {file_url}")
                return file_url
            else:
                logging.error(f"Failed to upload file to '{service}': {response.status_code} {response.text}")
                raise Exception(f"Upload failed with status code {response.status_code}: {response.text}")

        elif service == 'file.io':
            url = 'https://file.io/'
            headers = {
                'User-Agent': 'TranscriberWithContext/1.0 (your_email@example.com)'  # Replace with your info
            }

            logging.info(f"Preparing to upload '{file_path}' to '{service}'...")

            with open(file_path, 'rb') as f:
                encoder = MultipartEncoder(
                    fields={
                        'file': (os.path.basename(file_path), f),
                    }
                )

                # Add optional fields if provided
                if secret:
                    encoder.fields['secret'] = secret
                if expires:
                    encoder.fields['expires'] = expires

                logging.info(f"Starting upload to '{service}'...")
                response = requests.post(
                    url,
                    data=encoder,
                    headers={'Content-Type': encoder.content_type},
                    timeout=120
                )

            if response.status_code == 200 and response.json().get('success'):
                file_url = response.json().get('link')
                logging.info(f"File uploaded successfully to '{service}': {file_url}")
                return file_url
            else:
                logging.error(f"Failed to upload file to '{service}': {response.status_code} {response.text}")
                raise Exception(f"Upload failed with status code {response.status_code}: {response.text}")
        else:
            logging.error(f"Unsupported upload service: '{service}'")
            raise ValueError(f"Unsupported upload service: '{service}'")
    except requests.exceptions.Timeout:
        logging.error(f"Request timed out while uploading to '{service}'.")
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception occurred during upload to '{service}': {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during upload: {e}")
        raise

def download_audio(audio_url):
    try:
        logging.info(f"Starting download of audio from URL: {audio_url}")
        with requests.get(audio_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
        logging.info(f"Audio downloaded successfully to temporary file: {tmp_file_path}")

        # Verify integrity by checking file size and attempting to load
        file_size = os.path.getsize(tmp_file_path)
        if file_size == 0:
            logging.error(f"Downloaded audio file '{tmp_file_path}' is empty.")
            os.remove(tmp_file_path)
            raise Exception("Downloaded audio file is empty.")

        try:
            load_audio(tmp_file_path)
            logging.info(f"Integrity check passed for audio file '{tmp_file_path}'.")
        except Exception as e:
            logging.error(f"Integrity check failed for audio file '{tmp_file_path}': {e}")
            os.remove(tmp_file_path)
            raise Exception("Downloaded audio file failed integrity check.")

        return tmp_file_path
    except requests.exceptions.HTTPError as errh:
        logging.error(f"HTTP Error while downloading audio: {errh}")
        raise
    except requests.exceptions.ConnectionError as errc:
        logging.error(f"Connection Error while downloading audio: {errc}")
        raise
    except requests.exceptions.Timeout as errt:
        logging.error(f"Timeout Error while downloading audio: {errt}")
        raise
    except requests.exceptions.RequestException as err:
        logging.error(f"Request Exception while downloading audio: {err}")
        raise
    except Exception as e:
        logging.error(f"Failed to download and verify audio file: {e}")
        raise

def submit_diarization_job(api_token, audio_url, num_speakers=None):
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    data = {'url': audio_url}
    if num_speakers:
        data['numSpeakers'] = num_speakers
    try:
        response = requests.post(
            'https://api.pyannote.ai/v1/diarize',
            headers=headers,
            json=data,
            timeout=60  # Set a timeout for the request
        )
        if response.status_code == 200:
            job_info = response.json()
            logging.info(f"Diarization job submitted: {job_info['jobId']}")
            return job_info['jobId']
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return submit_diarization_job(api_token, audio_url, num_speakers)
        else:
            logging.error(f"Failed to submit diarization job: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        logging.error("Request timed out while submitting diarization job.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to submit diarization job: {e}")
        sys.exit(1)

def get_job_status(api_token, job_id):
    headers = {
        'Authorization': f'Bearer {api_token}',
    }
    try:
        response = requests.get(
            f'https://api.pyannote.ai/v1/jobs/{job_id}',
            headers=headers,
            timeout=60  # Set a timeout for the request
        )
        if response.status_code == 200:
            job_info = response.json()
            return job_info
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return get_job_status(api_token, job_id)
        else:
            logging.error(f"Failed to get job status: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        logging.error("Request timed out while getting job status.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to get job status: {e}")
        sys.exit(1)

def wait_for_diarization_job(api_token, job_id, check_interval=5, max_retries=3, timeout=3600):
    start_time = time.time()
    retries = 0
    
    print("Waiting for speaker recognition to complete...")
    while True:
        job_info = get_job_status(api_token, job_id)
        job_status = job_info['status']
        elapsed_time = int(time.time() - start_time)
        
        if job_status == 'succeeded':
            logging.info(f"\nDiarization job completed successfully after {elapsed_time} seconds.")
            print(f"\nSpeaker recogition complete.")
            return job_info
        elif job_status == 'failed':
            logging.error(f"\nDiarization job failed after {elapsed_time} seconds.")
            print(f"\nSpeaker recogition failed.")
            raise Exception("Diarization job failed")
        elif job_status == 'cancelled':
            print(f"\nSpeaker recogition failed, retrying...")
            logging.warning(f"\nDiarization job was cancelled after {elapsed_time} seconds. Retrying...")
            retries += 1
            if retries > max_retries:
                logging.error(f"\nDiarization job cancelled {max_retries} times. Aborting.")
                print(f"\nSpeaker recogition failed after multiple retries, aborting.")
                raise Exception(f"Diarization job cancelled {max_retries} times")
            job_id = submit_diarization_job(api_token, audio_url, num_speakers)
            logging.info(f"Submitted new diarization job with ID: {job_id}")
            print(f"Trying speaker recognition again...")
            start_time = time.time()  # Reset the timer for the new job
            continue
        else:
            logging.info(f"\rDiarization job status: {job_status}. Time elapsed: {elapsed_time}s")
            print(f"\rPerforming speaker recognition... Time elapsed: {elapsed_time}s", end='', flush=True)
            time.sleep(check_interval)
        
        if elapsed_time > timeout:
            logging.error(f"\rDiarization job status: {job_status}. Time elapsed: {elapsed_time}s")
            print(f"\nSpeaker recognition timed out after {elapsed_time} seconds.")
            raise Exception("Diarization job timed out")

def extract_text_for_timeframe(transcription, start_time, end_time):
    words = transcription.split()
    relevant_words = []
    current_time = 0
    for word in words:
        if '[' in word and ']' in word:
            timestamp = float(word.strip('[]'))
            if start_time <= timestamp < end_time:
                relevant_words.append(word)
            current_time = timestamp
        elif start_time <= current_time < end_time:
            relevant_words.append(word)
    return ' '.join(relevant_words)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Generation timed out")

# Set the timeout (e.g., 5 minutes)
GENERATE_TIMEOUT = 300  # seconds

def load_and_optimize_model(model_id):
    try:
        logging.info(f"Loading Whisper model '{model_id}'...")
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"  # Use scaled dot product attention
        )
        model = model.to(device)

        # Ensure model is in float16 for CUDA and MPS
        if device in ["cuda", "mps"] and model.dtype != torch.float16:
            model = model.half()
            logging.info(f"Converted model to half precision (float16) for {device}")

        processor = AutoProcessor.from_pretrained(model_id)

        logging.info(f"Whisper model '{model_id}' loaded successfully on {device} with dtype {model.dtype}.")
        
        # Suppress specific warnings
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.whisper.modeling_whisper")

        logging.info(f"Model device before optimization: {model.device}")
        logging.info("Starting model optimization...")

        # torch.compile optimization
        logging.info("Applying torch.compile optimization...")
        try:
            original_device = model.device
            model = torch.compile(model, mode="reduce-overhead")
            model = model.to(original_device)
            logging.info(f"torch.compile optimization completed successfully. Model device: {model.device}")
        except Exception as e:
            logging.error(f"torch.compile optimization failed: {str(e)}")
            logging.info("Continuing with the model without torch.compile optimization.")

        logging.info(f"Model device after optimization: {model.device}")
        logging.info("Model optimization process finished.")

        # Additional logging for model configuration
        logging.info(f"Model config after optimization: {model.config}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

        # Existing code for removing forced_decoder_ids
        logging.info("Removing forced_decoder_ids from model config")
        if hasattr(model.config, 'forced_decoder_ids'):
            model.config.forced_decoder_ids = None
        logging.info(f"Model config forced_decoder_ids after removal: {getattr(model.config, 'forced_decoder_ids', None)}")

        # Log the final model configuration
        logging.info(f"Final model configuration: {model.config}")

        return model, processor, device, torch_dtype
    except Exception as e:
        logging.error(f"Failed to load and optimize Whisper model '{model_id}': {e}")
        sys.exit(1)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Get the appropriate device
device = get_device()
logging.info(f"Using device: {device}")

def validate_output_formats(formats):
    valid_formats = {'text', 'json', 'srt'}
    formats = [fmt.strip().lower() for fmt in formats if fmt.strip()]
    invalid_formats = set(formats) - valid_formats
    if invalid_formats:
        raise argparse.ArgumentTypeError(f"Invalid output format(s): {', '.join(invalid_formats)}. Supported formats are: text, json, srt.")
    return formats

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with speaker diarization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audio-url', type=str, help='Publicly accessible URL of the audio file to transcribe.')
    group.add_argument('--input-file', type=str, help='Path to the local audio file to transcribe.')
    parser.add_argument('--context-prompt', type=str, help='Context prompt to guide transcription.')
    parser.add_argument('--language', type=str, nargs='+', help='Specify the language(s) of the audio.')
    parser.add_argument('--num-speakers', type=int, help='Specify the number of speakers if known.')
    parser.add_argument('--dry-run', action='store_true', help='Estimate cost without processing.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument("--pyannote-token", help="Pyannote API token (overrides environment variable)")
    parser.add_argument("--openai-key", help="OpenAI API key (overrides environment variable)")
    parser.add_argument("--model", default="openai/whisper-large-v3", 
                        help="OpenAI transcription model to use (default: openai/whisper-large-v3)")
    parser.add_argument('--output-format', type=str, nargs='+',
                        default=['text'], help='Specify desired output format(s). Supported formats: text, json, srt (default: text). Can specify multiple formats separated by spaces or commas.')
    
    args = parser.parse_args()

    # {{ edit_4: Define log_level based on debug and verbose flags }}
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    # {{ edit_2: Create log file only if --verbose or --debug is specified }}
    if args.debug or args.verbose:
        log_file = os.path.join(log_directory, f"transcription_log_{current_time}.log")
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    # Capture warnings in the log
    logging.captureWarnings(True)

    logging.info("Script started")

    # Load environment variables
    load_dotenv()

    # Use command line args if provided, otherwise fall back to environment variables
    pyannote_token = args.pyannote_token or os.getenv("PYANNOTE_TOKEN")
    openai_key = args.openai_key or os.getenv("OPENAI_KEY")

    if not pyannote_token:
        logging.error("PYANNOTE_TOKEN is not set. Please set it in .env file or provide it via --pyannote-token")
        sys.exit(1)

    if not openai_key:
        logging.error("OPENAI_KEY is not set. Please set it in .env file or provide it via --openai-key")
        sys.exit(1)

    # Use the specified model or the default one
    model_id = args.model
    logging.info(f"Using transcription model: {model_id}")

    # Load the specified model (only once)
    model, processor, device, torch_dtype = load_and_optimize_model(model_id)

    # {{ edit_1: Conditional handling based on user input }}
    if args.audio_url:
        audio_url = args.audio_url
        logging.info(f"Using provided audio URL: {audio_url}")

        # {{ edit_2: Download the audio file locally for Whisper model }}
        local_audio_path = download_audio(audio_url)
        logging.info(f"Downloaded audio for transcription: {local_audio_path}")
    else:
        input_file = args.input_file
        if not os.path.isfile(input_file):
            logging.error(f"Input file {input_file} does not exist.")
            sys.exit(1)
        logging.info(f"Using local audio file: {input_file}")

        # {{ edit_3: Upload local file for diarization }}
        upload_service = '0x0.st'  # Set to '0x0.st' as per user request
        logging.info(f"Uploading file to {upload_service} for diarization...")
        try:
            audio_url = upload_file(input_file, service=upload_service)
            logging.info(f"File uploaded successfully. Public URL: {audio_url}")
        except Exception as upload_error:
            logging.error(f"Failed to upload audio file: {upload_error}")
            sys.exit(1)

        # {{ edit_4: Use local file directly for Whisper model }}
        local_audio_path = input_file
        logging.info(f"Using local audio file for transcription: {local_audio_path}")

    # Extract the base name for output
    base_name = ''
    if args.audio_url:
        base_name = os.path.splitext(os.path.basename(audio_url))[0]
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    logging.info(f"Base name for output files: {base_name}")

    try:
        print(f"Starting speaker recognition")
        logging.info(f"Submitting diarization job for audio: {audio_url}")
        job_id = submit_diarization_job(pyannote_token, audio_url, args.num_speakers)
        logging.info(f"Diarization job submitted. Job ID: {job_id}")
        
        job_info = wait_for_diarization_job(pyannote_token, job_id, max_retries=3)
        
        # Process the diarization results
        diarization_output = job_info.get('output', {})
        diarization_segments = diarization_output.get('diarization', [])
        if not diarization_segments:
            raise Exception("No diarization results found.")
        
        logging.info(f"Diarization completed. Found {len(diarization_segments)} segments.")
        print(f"Speaker recognition completed.")
    except Exception as e:
        logging.error(f"Diarization failed: {e}")
        # Cleanup downloaded audio if it was downloaded
        if args.audio_url and 'local_audio_path' in locals():
            logging.info(f"Cleaning up downloaded audio file: {local_audio_path}")
            try:
                os.remove(local_audio_path)
                logging.info(f"Deleted temporary audio file: {local_audio_path}")
            except Exception as cleanup_error:
                logging.warning(f"Failed to delete temporary audio file '{local_audio_path}': {cleanup_error}")
        sys.exit(1)

    # Build speaker mapping
    speaker_mapping = {}
    speaker_counter = 1  # Start numbering at 1
    for segment in diarization_segments:
        speaker_label = segment['speaker']
        if speaker_label not in speaker_mapping:
            speaker_id = f"Speaker{speaker_counter}"  # ID as string
            speaker_mapping[speaker_label] = {
                'id': speaker_id,
                'name': f'Speaker {speaker_counter}'  # Name follows "Speaker 1" pattern
            }
            speaker_counter += 1
        segment['speaker_id'] = speaker_mapping[speaker_label]['id']

    # Load and optimize the Whisper model
    model, processor, device, torch_dtype = load_and_optimize_model(model_id)
    
    # At the beginning of your script, after model initialization:
    if hasattr(model.config, 'forced_decoder_ids'):
        logging.info("Removing forced_decoder_ids from model config")
        model.config.forced_decoder_ids = None

    # Prepare context prompt
    prompt = args.context_prompt
    if prompt:
        # Tokenize prompt to get decoder_input_ids
        decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False).input_ids
        max_length = model.config.max_decoder_input_length or model.config.decoder.max_position_embeddings
        # Adjusting for potential special tokens added during generation
        max_prompt_length = max_length - 1  # Reserve space for at least one new token

        if len(decoder_input_ids) > max_prompt_length:
            logging.warning(f"Context prompt is too long ({len(decoder_input_ids)} tokens). Truncating to {max_prompt_length} tokens.")
            # Truncate intelligently (from the beginning)
            decoder_input_ids = decoder_input_ids[-max_prompt_length:]
    else:
        decoder_input_ids = None

    # Prepare language settings
    languages = args.language
    if languages:
        if len(languages) == 1:
            language = languages[0]
            logging.info(f"Using language: {language}")
        else:
            language = None
            logging.warning("Multiple languages specified. Language detection will be used for each segment.")
    else:
        language = None
        logging.info("No language specified. Language detection will be used.")

    # Load the downloaded audio for segmentation
    logging.info(f"Loading audio file {local_audio_path} for segmentation...")
    audio_array = load_audio(local_audio_path, sampling_rate=16000)
    sampling_rate = 16000

    # Estimate cost
    cost_per_minute = 0.006  # USD per minute, as per OpenAI Whisper API pricing
    pyannote_cost_per_hour = 0.18  # USD per hour for speaker diarization (converted from €0.15)

    total_duration = len(audio_array) / sampling_rate  # in seconds
    total_minutes = total_duration / 60
    diarization_hours = total_duration / 3600
    whisper_cost = total_minutes * cost_per_minute
    diarization_cost = diarization_hours * pyannote_cost_per_hour
    total_cost = diarization_cost + whisper_cost

    if args.dry_run:
        print(f"Estimated cost for processing {local_audio_path}: ${diarization_cost + whisper_cost:.4f} USD")
        sys.exit(0)

    logging.info(f"Estimated cost for processing: ${total_cost:.4f} USD")

    MAX_CHUNK_DURATION = 30  # seconds

    logging.info("Starting transcription process...")
    transcription_segments = []
    total_segments = len(diarization_segments)
    processed_duration = 0.0  # Initialize at the beginning of the main function

    with tqdm(total=total_segments, desc="Transcribing", unit="segment") as pbar:
        for idx, segment_info in enumerate(diarization_segments):
            start_time = segment_info['start']
            end_time = segment_info['end']
            segment_duration = end_time - start_time
            
            segment_start_time = time.time()
            
            logging.info(f"Processing segment {idx+1}/{len(diarization_segments)}: duration {segment_duration:.2f}s")
            
            chunk_transcriptions = []
            # Process the segment in smaller chunks if it's too long
            for chunk_start in range(int(start_time), int(end_time), MAX_CHUNK_DURATION):
                chunk_end = min(chunk_start + MAX_CHUNK_DURATION, end_time)
                
                # Extract audio chunk for this segment
                start_sample = int(chunk_start * sampling_rate)
                end_sample = int(chunk_end * sampling_rate)
                chunk = audio_array[start_sample:end_sample]
                
                inputs = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
                inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}

                # Explicitly set the attention mask
                inputs['attention_mask'] = torch.ones_like(inputs['input_features'])

                generate_kwargs = {
                    "task": "transcribe",
                    "return_timestamps": True,
                    "max_new_tokens": MAX_TARGET_POSITIONS - BUFFER_TOKENS
                }

                if language:
                    generate_kwargs["language"] = language
                else:
                    generate_kwargs["language"] = None

                # Ensure forced_decoder_ids is not set
                generate_kwargs.pop('forced_decoder_ids', None)

                logging.info(f"Generate kwargs: {generate_kwargs}")
                logging.info(f"Model config forced_decoder_ids: {getattr(model.config, 'forced_decoder_ids', None)}")

                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(GENERATE_TIMEOUT)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            **generate_kwargs
                        )
                    
                    signal.alarm(0)  # Cancel the alarm
                    
                    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                except TimeoutException:
                    logging.error(f"Generation timed out for chunk in segment {idx+1}")
                    continue
                except RuntimeError as e:
                    if "MPS" in str(e):
                        logging.warning(f"MPS error encountered. Falling back to CPU for this chunk.")
                        # Move model and inputs to CPU for this chunk
                        model_cpu = model.to('cpu')
                        inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
                        with torch.no_grad():
                            generated_ids = model_cpu.generate(**inputs_cpu, **generate_kwargs)
                        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        # Move model back to original device
                        model.to(device)
                    else:
                        logging.error(f"Error processing chunk in segment {idx+1}: {str(e)}")
                        continue
                finally:
                    signal.alarm(0)  # Ensure the alarm is cancelled
                
                chunk_transcriptions.append(transcription.strip())
            
            # Combine all chunk transcriptions for this segment
            full_transcription = ' '.join(chunk_transcriptions)
            
            transcription_segments.append({
                'start': start_time,
                'end': end_time,
                'speaker_id': segment_info['speaker_id'],
                'text': full_transcription
            })
            
            segment_processing_time = time.time() - segment_start_time
            logging.info(f"Segment {idx+1} processed in {segment_processing_time:.2f}s")
            pbar.update(1)

            # Update processed_duration if transcription was successful
            processed_duration += segment_duration

    logging.info("Transcription process completed.")
    logging.info(f"Preparing to write transcription in specified format(s): {args.output_format}")

    # Build speaker list
    speakers = [
        {'id': info['id'], 'name': info['name']}
        for info in speaker_mapping.values()
    ]

    # Build and write text transcription if requested
    if 'text' in args.output_format:
        text_output_file = f"{base_name}_transcription.txt"
        try:
            with open(text_output_file, 'w', encoding='utf-8') as f:
                for segment in transcription_segments:
                    f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['speaker_id']}: {segment['text']}\n")
            logging.info(f"Text transcription written to {text_output_file}")
        except Exception as e:
            logging.error(f"Failed to write text transcription to file: {e}")
    
    # Build and write SRT transcription if requested
    if 'srt' in args.output_format:
        srt_output_file = f"{base_name}_transcription.srt"
        try:
            subtitles = []
            for idx, segment in enumerate(transcription_segments, 1):
                start = timedelta(seconds=segment['start'])
                end = timedelta(seconds=segment['end'])
                content = f"{segment['speaker_id']}: {segment['text']}"
                subtitle = srt.Subtitle(index=idx, start=start, end=end, content=content)
                subtitles.append(subtitle)
            srt_content = srt.compose(subtitles)
            with open(srt_output_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            logging.info(f"SRT transcription written to {srt_output_file}")
        except Exception as e:
            logging.error(f"Failed to write SRT transcription to file: {e}")

    # Build output JSON if requested
    if 'json' in args.output_format:
        output_data = {
            'speakers': speakers,
            'transcript': transcription_segments
        }

        # Write transcription to JSON file
        json_output_file = f"{base_name}_transcript.json"
        try:
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)
            logging.info(f"JSON transcription written to {json_output_file}")
        except Exception as e:
            logging.error(f"Failed to write JSON transcription to file: {e}")

    # Output the actual cost based on processed duration
    print(f"\n---\nTotal audio duration: {total_duration:.2f} seconds")
    print(f"Processed duration for transcription: {processed_duration:.2f} seconds")
    print(f"Diarization cost (@ ${pyannote_cost_per_hour}/hour): ${diarization_cost:.4f} USD")
    print(f"Whisper transcription cost (@ ${cost_per_minute}/minute): ${whisper_cost:.4f} USD")
    print(f"Total estimated cost: ${total_cost:.4f} USD\n")

    if not args.audio_url:
        logging.info("No need to delete downloaded audio since local file was used.")
    else:
        # Existing cleanup code for downloaded audio
        logging.info("Cleaning up temporary downloaded audio file...")
        try:
            os.remove(local_audio_path)
            logging.info(f"Deleted temporary audio file: {local_audio_path}")
        except Exception as e:
            logging.warning(f"Failed to delete temporary audio file: {e}")

    logging.info("Transcription process completed successfully.")
    logging.info(f"Transcription completed. Total segments: {len(transcription_segments)}")

    # {{ edit_3: Clean up the format handling code to be more robust }}
    if args.output_format:
        # Split by comma and/or space, remove empty strings, and convert to lowercase
        formats = [fmt.strip().lower() for fmt in ','.join(args.output_format).replace(',', ' ').split() if fmt.strip()]
        try:
            args.output_format = validate_output_formats(formats)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
    else:
        args.output_format = ['text']  # Default format
    
    # {{ edit_1: List the output files that were created }}
    output_files = []
    if 'text' in args.output_format:
        output_files.append(text_output_file)
    if 'srt' in args.output_format:
        output_files.append(srt_output_file)
    if 'json' in args.output_format:
        output_files.append(json_output_file)
    
    if output_files:
        print("\nGenerated Output Files:")
        for file in output_files:
            print(f"- {file}")
    
    # Ensure cleanup after writing transcription
    if args.audio_url and 'local_audio_path' in locals():
        logging.info(f"Final cleanup of temporary audio file: {local_audio_path}")
        try:
            os.remove(local_audio_path)
            logging.info(f"Deleted temporary audio file: {local_audio_path}")
        except Exception as cleanup_error:
            logging.warning(f"Failed to delete temporary audio file '{local_audio_path}': {cleanup_error}")

if __name__ == '__main__':
    main()