#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
from dotenv import load_dotenv
from pyannote.audio import Pipeline

def main():
    parser = argparse.ArgumentParser(description='Transcribe m4a file with speaker diarization using Whisper large-v3 model.')
    parser.add_argument('input_file', help='Input m4a file to transcribe.')
    parser.add_argument('--context-prompt', type=str, help='Context prompt to guide transcription.')
    parser.add_argument('--language', type=str, nargs='+', help='Specify the language(s) of the audio.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--debug', action='store_true', help='Enable debug output.')

    args = parser.parse_args()

    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Load environment variables from .env file
    load_dotenv()

    input_file = args.input_file

    if not os.path.isfile(input_file):
        logging.error(f"Input file {input_file} does not exist.")
        sys.exit(1)

    output_file = os.path.splitext(input_file)[0] + '.json'

    logging.info(f"Transcribing {input_file} to {output_file}")

    # Device configuration
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    logging.info(f"Using device: {device}")

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    model_id = "openai/whisper-large-v3"

    logging.info(f"Loading model {model_id}...")
    # Load model and processor
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Load audio file
    logging.info(f"Loading audio file {input_file}...")
    try:
        dataset = load_dataset("audiofolder", data_files={'train': input_file}, split='train')
        audio = dataset[0]['audio']
    except Exception as e:
        logging.error(f"Failed to load audio file: {e}")
        sys.exit(1)

    sampling_rate = audio["sampling_rate"]
    audio_array = audio["array"]

    # Perform speaker diarization
    logging.info("Performing speaker diarization...")
    try:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        diarization = diarization_pipeline(input_file)
    except Exception as e:
        logging.error(f"Failed to perform speaker diarization: {e}")
        sys.exit(1)

    # Build speaker mapping
    speaker_mapping = {}
    speaker_counter = 0

    # Collect all segments with speaker labels
    diarization_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_mapping:
            speaker_mapping[speaker] = {
                'id': speaker_counter,
                'name': f'Speaker {speaker_counter}'
            }
            speaker_counter += 1
        diarization_segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker_mapping[speaker]['id']
        })

    # Sort segments by start time
    diarization_segments.sort(key=lambda x: x['start'])

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

    # Transcribe each segment
    transcription_segments = []
    total_segments = len(diarization_segments)
    with tqdm(total=total_segments, desc="Transcribing", unit="segment") as pbar:
        for idx, segment_info in enumerate(diarization_segments):
            start_time = segment_info['start']
            end_time = segment_info['end']
            speaker_id = segment_info['speaker']

            # Extract audio segment
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            chunk = audio_array[start_sample:end_sample]

            inputs = processor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Prepare generation arguments
            generate_kwargs = {
                "max_new_tokens": 448,
                "task": "transcribe",
                "return_timestamps": False,
            }

            if decoder_input_ids is not None:
                # Convert to tensor and move to device
                generate_kwargs["decoder_input_ids"] = torch.tensor([decoder_input_ids], device=device)

            if language:
                generate_kwargs["language"] = language

            with torch.no_grad():
                try:
                    generated_ids = model.generate(
                        **inputs,
                        **generate_kwargs
                    )
                except Exception as e:
                    logging.error(f"Failed during generation: {e}")
                    sys.exit(1)

            # Remove prompt tokens from generated output
            if decoder_input_ids is not None:
                generated_ids = generated_ids[:, len(decoder_input_ids):]

            chunk_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Append transcription segment
            transcription_segments.append({
                'start': start_time,
                'end': end_time,
                'speaker_id': speaker_id,
                'text': chunk_transcription.strip()
            })

            pbar.update(1)

    # Build output JSON
    output_data = {
        'speakers': [
            {'id': info['id'], 'name': info['name']}
            for info in speaker_mapping.values()
        ],
        'transcript': transcription_segments
    }

    # Write transcription to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to write transcription to file: {e}")
        sys.exit(1)

    logging.info(f"Transcription written to {output_file}")

if __name__ == '__main__':
    main()
