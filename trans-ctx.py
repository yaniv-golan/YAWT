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
from datetime import datetime, timedelta
import srt
from logging.handlers import RotatingFileHandler
import concurrent.futures

# Constants
MAX_TARGET_POSITIONS = 448  # Max target positions for the Whisper model
BUFFER_TOKENS = 3  # Accounts for special and prompt tokens
MAX_CHUNK_DURATION = 30  # seconds
GENERATE_TIMEOUT = 300  # seconds
COST_PER_MINUTE = 0.006  # USD per minute for Whisper
PYANNOTE_COST_PER_HOUR = 0.18  # USD per hour for diarization

# Setup environment variable
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Logging setup
def setup_logging(debug=False, verbose=False):
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_directory, f"transcription_log_{current_time}.log")

    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    handlers = [
        RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ] if debug or verbose else [
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.captureWarnings(True)
    logging.info("Logging setup complete.")

setup_logging()

def load_audio(input_file, sampling_rate=16000):
    try:
        out, _ = (
            ffmpeg.input(input_file, threads=0)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ar=sampling_rate, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.float32)
        logging.info(f"Audio loaded from {input_file}.")
        return audio
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        sys.exit(1)

def upload_file(file_path, service='0x0.st', secret=None, expires=None):
    try:
        if service == '0x0.st':
            url = 'https://0x0.st'
            headers = {'User-Agent': 'TranscriberWithContext/1.0 (your_email@example.com)'}
            logging.info(f"Uploading '{file_path}' to '{service}'...")
            with open(file_path, 'rb') as f:
                encoder = MultipartEncoder(fields={'file': (os.path.basename(file_path), f)})
                if secret:
                    encoder.fields['secret'] = secret
                if expires:
                    encoder.fields['expires'] = expires

                def progress_callback(monitor):
                    pbar.update(monitor.bytes_read - pbar.n)

                monitor = MultipartEncoderMonitor(encoder, progress_callback)
                with tqdm(total=encoder.len, unit='B', unit_scale=True, desc="Uploading") as pbar:
                    response = requests.post(
                        url,
                        data=monitor,
                        headers={'Content-Type': monitor.content_type, 'User-Agent': headers['User-Agent']},
                        timeout=120
                    )

            if response.status_code == 200:
                file_url = response.text.strip()
                logging.info(f"Uploaded to '{service}': {file_url}")
                return file_url
            else:
                logging.error(f"Upload failed: {response.status_code} {response.text}")
                raise Exception(f"Upload failed: {response.status_code} {response.text}")

        elif service == 'file.io':
            url = 'https://file.io/'
            headers = {'User-Agent': 'TranscriberWithContext/1.0 (your_email@example.com)'}
            logging.info(f"Uploading '{file_path}' to '{service}'...")
            with open(file_path, 'rb') as f:
                encoder = MultipartEncoder(fields={'file': (os.path.basename(file_path), f)})
                if secret:
                    encoder.fields['secret'] = secret
                if expires:
                    encoder.fields['expires'] = expires

                response = requests.post(
                    url,
                    data=encoder,
                    headers={'Content-Type': encoder.content_type},
                    timeout=120
                )

            if response.status_code == 200 and response.json().get('success'):
                file_url = response.json().get('link')
                logging.info(f"Uploaded to '{service}': {file_url}")
                return file_url
            else:
                logging.error(f"Upload failed: {response.status_code} {response.text}")
                raise Exception(f"Upload failed: {response.status_code} {response.text}")
        else:
            logging.error(f"Unsupported service: '{service}'")
            raise ValueError(f"Unsupported service: '{service}'")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error during upload: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during upload: {e}")
        raise

def download_audio(audio_url):
    try:
        logging.info(f"Downloading audio from {audio_url}...")
        with requests.get(audio_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
        logging.info(f"Downloaded to {tmp_file_path}")

        if os.path.getsize(tmp_file_path) == 0:
            logging.error("Downloaded audio file is empty.")
            os.remove(tmp_file_path)
            raise Exception("Empty audio file.")

        load_audio(tmp_file_path)
        logging.info("Audio integrity verified.")
        return tmp_file_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Download error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during audio download: {e}")
        raise

def submit_diarization_job(api_token, audio_url, num_speakers=None):
    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    data = {'url': audio_url}
    if num_speakers:
        data['numSpeakers'] = num_speakers
    try:
        response = requests.post('https://api.pyannote.ai/v1/diarize', headers=headers, json=data, timeout=60)
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
            logging.error(f"Diarization submission failed: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        logging.error("Diarization submission timed out.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Diarization submission error: {e}")
        sys.exit(1)

def get_job_status(api_token, job_id):
    headers = {'Authorization': f'Bearer {api_token}'}
    try:
        response = requests.get(f'https://api.pyannote.ai/v1/jobs/{job_id}', headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return get_job_status(api_token, job_id)
        else:
            logging.error(f"Failed to get job status: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        logging.error("Getting job status timed out.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error getting job status: {e}")
        sys.exit(1)

def wait_for_diarization(api_token, job_id, audio_url, check_interval=5, max_retries=3, timeout=3600):
    start_time = time.time()
    retries = 0
    print("Waiting for speaker recognition to complete...")

    while True:
        job_info = get_job_status(api_token, job_id)
        status = job_info['status']
        elapsed = int(time.time() - start_time)

        if status == 'succeeded':
            logging.info(f"Diarization succeeded in {elapsed} seconds.")
            print("\nSpeaker recognition complete.")
            return job_info
        elif status in ['failed', 'cancelled']:
            if status == 'cancelled':
                retries += 1
                logging.warning(f"Diarization job cancelled. Retry {retries}/{max_retries}.")
                if retries > max_retries:
                    logging.error("Max retries reached. Aborting.")
                    raise Exception("Diarization job cancelled multiple times.")
                job_id = submit_diarization_job(api_token, audio_url)
                continue
            logging.error(f"Diarization failed after {elapsed} seconds.")
            raise Exception("Diarization job failed.")
        else:
            logging.info(f"Diarization status: {status}. Elapsed time: {elapsed}s")
            print(f"\rPerforming speaker recognition... {elapsed}s elapsed.", end='', flush=True)
            time.sleep(check_interval)

        if elapsed > timeout:
            logging.error("Diarization job timed out.")
            raise Exception("Diarization job timed out.")

def extract_text(transcription, start_time, end_time):
    words = transcription.split()
    relevant = []
    current = 0
    for word in words:
        if '[' in word and ']' in word:
            timestamp = float(word.strip('[]'))
            if start_time <= timestamp < end_time:
                relevant.append(word)
            current = timestamp
        elif start_time <= current < end_time:
            relevant.append(word)
    return ' '.join(relevant)

class TimeoutException(Exception):
    pass

def model_generate_with_timeout(model, inputs, generate_kwargs, timeout):
    """
    Generates transcription with a timeout using ThreadPoolExecutor.
    
    Args:
        model: The speech-to-text model.
        inputs: Input features for the model.
        generate_kwargs: Keyword arguments for the model's generate method.
        timeout: Timeout in seconds.
    
    Returns:
        generated_ids: The generated token IDs from the model.
    
    Raises:
        TimeoutException: If the generation exceeds the specified timeout.
    """
    def generate():
        with torch.no_grad():  # Ensure gradients are not tracked
            return model.generate(**inputs, **generate_kwargs)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate)  # Submit the wrapped generate function
        try:
            generated_ids = future.result(timeout=timeout)
            return generated_ids
        except concurrent.futures.TimeoutError:
            raise TimeoutException("Generation timed out")

def load_and_optimize_model(model_id):
    try:
        logging.info(f"Loading model '{model_id}'...")
        device = get_device()
        torch_dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        ).to(device)

        if device.type in ["cuda", "mps"] and model.dtype != torch.float16:
            model = model.half()
            logging.info("Converted model to float16.")

        processor = AutoProcessor.from_pretrained(model_id)
        logging.info(f"Model loaded on {device} with dtype {model.dtype}.")

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.whisper.modeling_whisper")

        # Optimize model
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logging.info("Model optimized with torch.compile.")
        except Exception as e:
            logging.error(f"Model optimization failed: {e}")
        
        # Remove forced_decoder_ids if present
        if hasattr(model.config, 'forced_decoder_ids'):
            model.config.forced_decoder_ids = None
            logging.info("Removed forced_decoder_ids from model config.")

        return model, processor, device, torch_dtype
    except Exception as e:
        logging.error(f"Failed to load and optimize model '{model_id}': {e}")
        sys.exit(1)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

def calculate_cost(duration_seconds):
    minutes = duration_seconds / 60
    hours = duration_seconds / 3600
    whisper = minutes * COST_PER_MINUTE
    diarization = hours * PYANNOTE_COST_PER_HOUR
    total = whisper + diarization
    return whisper, diarization, total

def write_transcriptions(output_format, base_name, transcription_segments, speakers):
    output_files = []
    if 'text' in output_format:
        text_file = f"{base_name}_transcription.txt"
        try:
            with open(text_file, 'w', encoding='utf-8') as f:
                for seg in transcription_segments:
                    f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['speaker_id']}: {seg['text']}\n")
            logging.info(f"Text transcription saved to {text_file}")
            output_files.append(text_file)
        except Exception as e:
            logging.error(f"Failed to write text file: {e}")

    if 'srt' in output_format:
        srt_file = f"{base_name}_transcription.srt"
        try:
            subtitles = [
                srt.Subtitle(index=i, start=timedelta(seconds=seg['start']),
                            end=timedelta(seconds=seg['end']),
                            content=f"{seg['speaker_id']}: {seg['text']}")
                for i, seg in enumerate(transcription_segments, 1)
            ]
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(srt.compose(subtitles))
            logging.info(f"SRT transcription saved to {srt_file}")
            output_files.append(srt_file)
        except Exception as e:
            logging.error(f"Failed to write SRT file: {e}")

    if 'json' in output_format:
        json_file = f"{base_name}_transcript.json"
        data = {'speakers': speakers, 'transcript': transcription_segments}
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logging.info(f"JSON transcription saved to {json_file}")
            output_files.append(json_file)
        except Exception as e:
            logging.error(f"Failed to write JSON file: {e}")

    if output_files:
        print("\nGenerated Output Files:")
        for file in output_files:
            print(f"- {file}")

def transcribe_single_segment(model, processor, inputs, generate_kwargs, idx, chunk_start, chunk_end):
    """
    Transcribes a single audio segment using the provided model and inputs.
    
    Args:
        model: The speech-to-text model.
        processor: The processor for decoding.
        inputs: Input features for the model.
        generate_kwargs: Keyword arguments for the model's generate method.
        idx: Segment index.
        chunk_start: Start time of the chunk.
        chunk_end: End time of the chunk.
    
    Returns:
        transcription (str) if successful, None otherwise.
    """
    try:
        generated_ids = model_generate_with_timeout(model, inputs, generate_kwargs, GENERATE_TIMEOUT)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except TimeoutException:
        logging.error(f"Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")
        return None
    except RuntimeError as e:
        if "MPS" in str(e):
            logging.warning("MPS error encountered. Falling back to CPU.")
            model_cpu = model.to('cpu')
            inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
            try:
                generated_ids = model_generate_with_timeout(model_cpu, inputs_cpu, generate_kwargs, GENERATE_TIMEOUT)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return transcription.strip()
            except TimeoutException:
                logging.error(f"CPU Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")
                return None
            except RuntimeError as cpu_e:
                logging.error(f"Runtime error on CPU during transcription: {cpu_e}")
                return None
            finally:
                model.to(device)  # Move the model back to the original device
        else:
            logging.error(f"Runtime error: {e}")
            return None

def retry_transcriptions(model, processor, audio_array, diarization_segments, failed_segments, generate_kwargs, device, torch_dtype, base_name, MAX_RETRIES=3):
    """
    Retries transcription for failed segments up to MAX_RETRIES times.
    
    Args:
        model: The speech-to-text model.
        processor: The processor for decoding.
        audio_array: Numpy array of the audio data.
        diarization_segments: List of all diarization segments.
        failed_segments: List of dictionaries containing failed segment indices and reasons.
        generate_kwargs: Keyword arguments for the model's generate method.
        device: The device used for computation.
        torch_dtype: The torch data type.
        base_name: Base name for output files.
        MAX_RETRIES: Maximum number of retry attempts.
    
    Returns:
        Updated failed_segments after retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        if not failed_segments:
            break  # No more segments to retry
        logging.info(f"Starting retry attempt {attempt} for failed segments.")
        logging.info(f"Number of segments to retry: {len(failed_segments)}")
        retry_failed_segments = []
        for failure in failed_segments:
            idx = failure['segment']
            seg = diarization_segments[idx - 1]
            start, end = seg['start'], seg['end']
            logging.info(f"Retrying transcription for segment {idx}: {start}-{end}s")
            chunk_start = int(start)
            chunk_end = min(chunk_start + MAX_CHUNK_DURATION, end)
            chunk = audio_array[int(chunk_start*16000):int(chunk_end*16000)]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}
            inputs['attention_mask'] = torch.ones_like(inputs['input_features'])
    
            # Add decoder_input_ids if available
            if 'decoder_input_ids' in generate_kwargs and generate_kwargs['decoder_input_ids'] is not None:
                inputs['decoder_input_ids'] = generate_kwargs['decoder_input_ids']
    
            transcription = transcribe_single_segment(model, processor, inputs, generate_kwargs, idx, chunk_start, chunk_end)
            if transcription:
                # Update the transcription segments with the new transcription
                for t_seg in transcription_segments:
                    if t_seg['speaker_id'] == dia.segment['speaker_id'] and t_seg['start'] == start and t_seg['end'] == end:
                        t_seg['text'] = transcription
                        break
            else:
                # If transcription fails again, log it
                logging.warning(f"Retry {attempt}: Failed to transcribe segment {idx}-{chunk_start}-{chunk_end}s")
                retry_failed_segments.append(failure)
        failed_segments = retry_failed_segments  # Update the list of failed segments
    return failed_segments

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
                        default=['text'], help='Desired output format(s): text, json, srt.')

    args = parser.parse_args()

    # Update logging based on arguments
    setup_logging(debug=args.debug, verbose=args.verbose)

    logging.info("Script started.")

    load_dotenv()

    pyannote_token = args.pyannote_token or os.getenv("PYANNOTE_TOKEN")
    openai_key = args.openai_key or os.getenv("OPENAI_KEY")

    if not pyannote_token:
        logging.error("PYANNOTE_TOKEN not set.")
        sys.exit(1)
    if not openai_key:
        logging.error("OPENAI_KEY not set.")
        sys.exit(1)

    try:
        args.output_format = validate_output_formats(args.output_format)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    print(f"Output formats: {args.output_format}")  # Add this line for debugging

    model, processor, device, torch_dtype = load_and_optimize_model(args.model)

    # Handle --context-prompt
    if args.context_prompt:
        logging.info("Integrating context prompt into transcription.")
        # Tokenize the context prompt
        prompt_encoded = processor.tokenizer(args.context_prompt, return_tensors="pt", add_special_tokens=False)
        decoder_input_ids = prompt_encoded['input_ids'].to(device).to(torch_dtype)
    else:
        decoder_input_ids = None

    if args.audio_url:
        audio_url = args.audio_url
        logging.info(f"Using audio URL: {audio_url}")
        local_audio_path = download_audio(audio_url)
    else:
        input_file = args.input_file
        if not os.path.isfile(input_file):
            logging.error(f"Input file {input_file} does not exist.")
            sys.exit(1)
        logging.info(f"Using local file: {input_file}")
        try:
            audio_url = upload_file(input_file, service='0x0.st')
            logging.info(f"Uploaded to '0x0.st': {audio_url}")
            local_audio_path = input_file
        except Exception as e:
            logging.error(f"File upload failed: {e}")
            sys.exit(1)

    base_name = os.path.splitext(os.path.basename(audio_url if args.audio_url else args.input_file))[0]
    logging.info(f"Base name for outputs: {base_name}")

    try:
        logging.info("Submitting diarization job.")
        job_id = submit_diarization_job(pyannote_token, audio_url, args.num_speakers)
        job_info = wait_for_diarization(pyannote_token, job_id, audio_url)
        diarization_segments = job_info.get('output', {}).get('diarization', [])
        if not diarization_segments:
            raise Exception("No diarization results found.")

        logging.info(f"Diarization completed: {len(diarization_segments)} segments found.")
    except Exception as e:
        logging.error(f"Diarization error: {e}")
        if args.audio_url and 'local_audio_path' in locals():
            try:
                os.remove(local_audio_path)
                logging.info(f"Deleted temporary file: {local_audio_path}")
            except Exception as cleanup_error:
                logging.warning(f"Cleanup failed: {cleanup_error}")
        sys.exit(1)

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

    audio_array = load_audio(local_audio_path)
    total_duration = len(audio_array) / 16000
    whisper_cost, diarization_cost, total_cost = calculate_cost(total_duration)

    if args.dry_run:
        print(f"Estimated cost: ${total_cost:.4f} USD")
        sys.exit(0)

    logging.info(f"Processing cost: Whisper=${whisper_cost:.4f}, Diarization=${diarization_cost:.4f}, Total=${total_cost:.4f}")

    transcription_segments = []
    # Initialize a list to track failed segments
    failed_segments = []

    with tqdm(total=len(diarization_segments), desc="Transcribing") as pbar:
        for idx, seg in enumerate(diarization_segments, 1):
            start, end = seg['start'], seg['end']
            logging.info(f"Transcribing segment {idx}: {start}-{end}s")
            chunk_transcriptions = []
            for chunk_start in range(int(start), int(end), MAX_CHUNK_DURATION):
                chunk_end = min(chunk_start + MAX_CHUNK_DURATION, end)
                chunk = audio_array[int(chunk_start*16000):int(chunk_end*16000)]
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}
                inputs['attention_mask'] = torch.ones_like(inputs['input_features'])

                generate_kwargs = {
                    "task": "transcribe",
                    "return_timestamps": True,
                    "max_new_tokens": MAX_TARGET_POSITIONS - BUFFER_TOKENS
                }
                if args.language and len(args.language) == 1:
                    generate_kwargs["language"] = args.language[0]
                else:
                    generate_kwargs["language"] = None

                # Add decoder_input_ids if available
                if 'decoder_input_ids' in generate_kwargs and generate_kwargs['decoder_input_ids'] is not None:
                    generate_kwargs["decoder_input_ids"] = generate_kwargs["decoder_input_ids"]

                # Transcribe the segment
                transcription = transcribe_single_segment(model, processor, inputs, generate_kwargs, idx, chunk_start, chunk_end)
                if transcription == 'fallback':
                    # Fallback to CPU using the same helper function
                    try:
                        model_cpu = model.to('cpu')
                        inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
                        transcription = transcribe_single_segment(model_cpu, processor, inputs_cpu, generate_kwargs, idx, chunk_start, chunk_end)
                        if transcription:
                            chunk_transcriptions.append(transcription.strip())
                        else:
                            # If transcription fails again, log it
                            failed_segments.append({'segment': idx, 'reason': 'CPU fallback failed'})
                    except Exception as e:
                        logging.error(f"CPU fallback failed for segment {idx}-{chunk_start}-{chunk_end}s: {e}")
                        failed_segments.append({'segment': idx, 'reason': f'CPU fallback exception: {e}'})
                        continue
                    finally:
                        model.to(device)  # Move the model back to the original device
                elif transcription:
                    chunk_transcriptions.append(transcription)
                else:
                    # Track failed segment
                    failed_segments.append({'segment': idx, 'reason': 'Transcription failed due to Timeout or Runtime Error'})
            full_transcription = ' '.join(chunk_transcriptions)
            transcription_segments.append({
                'start': start,
                'end': end,
                'speaker_id': seg['speaker_id'],
                'text': full_transcription
            })
            pbar.update(1)

    logging.info("Transcription completed.")
    write_transcriptions(args.output_format, base_name, transcription_segments, speakers)

    # Retry failed segments
    if failed_segments:
        logging.info("Retrying failed segments...")
        failed_segments = retry_transcriptions(
            model, 
            processor, 
            audio_array, 
            diarization_segments, 
            failed_segments, 
            generate_kwargs, 
            device, 
            torch_dtype, 
            base_name
        )

    # Report any remaining failed segments after retries
    if failed_segments:
        logging.warning("Some segments failed to transcribe after all retry attempts:")
        for failure in failed_segments:
            logging.warning(f"Segment {failure['segment']}: {failure['reason']}")
        print("Some segments failed to transcribe after retries. Check logs for details.")

    whisper_cost, diarization_cost, total_cost = calculate_cost(total_duration)
    print(f"\nTotal Duration: {total_duration:.2f}s")
    print(f"Transcription Cost: ${whisper_cost:.4f} USD")
    print(f"Diarization Cost: ${diarization_cost:.4f} USD")
    print(f"Total Estimated Cost: ${total_cost:.4f} USD\n")

    if args.audio_url:
        try:
            os.remove(local_audio_path)
            logging.info(f"Deleted temporary file: {local_audio_path}")
        except Exception as e:
            logging.warning(f"Failed to delete temporary file: {e}")

    logging.info("Process completed successfully.")

if __name__ == '__main__':
    main()