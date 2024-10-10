import json
import os
import logging
import srt
from datetime import timedelta

def write_transcriptions(output_format, base_name, transcription_segments, speakers):
    """
    Writes transcriptions to the specified formats.
    
    Args:
        output_format (list): List of desired output formats ('text', 'json', 'srt').
        base_name (str): Base name for the output files.
        transcription_segments (list): List of transcription segments.
        speakers (list): List of speakers with their identifiers and names.
    """
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
                srt.Subtitle(
                    index=i, 
                    start=timedelta(seconds=seg['start']),
                    end=timedelta(seconds=seg['end']),
                    content=f"{seg['speaker_id']}: {seg['text']}"
                )
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
        data = {
            'speakers': speakers,
            'transcript': transcription_segments
        }
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