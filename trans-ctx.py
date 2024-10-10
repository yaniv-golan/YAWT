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
from config import MAX_TARGET_POSITIONS, BUFFER_TOKENS, MAX_CHUNK_DURATION, GENERATE_TIMEOUT, COST_PER_MINUTE, PYANNOTE_COST_PER_HOUR, DEFAULT_MODEL_ID, SUPPORTED_UPLOAD_SERVICES, DOWNLOAD_TIMEOUT, UPLOAD_TIMEOUT, DIARIZATION_TIMEOUT, JOB_STATUS_TIMEOUT

# Setup environment variable
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Logging setup
from logging_setup import setup_logging

from audio_handler import load_audio, upload_file, download_audio
from diarization import submit_diarization_job, wait_for_diarization
from transcription import transcribe_single_segment, retry_transcriptions

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
                transcription = transcribe_single_segment(model, processor, inputs, generate_kwargs, idx, chunk_start, chunk_end, device, torch_dtype)
                if transcription == 'fallback':
                    # Fallback to CPU using the same helper function
                    try:
                        model_cpu = model.to('cpu')
                        inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
                        transcription = transcribe_single_segment(model_cpu, processor, inputs_cpu, generate_kwargs, idx, chunk_start, chunk_end, device, torch_dtype)
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
            base_name, 
            transcription_segments
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