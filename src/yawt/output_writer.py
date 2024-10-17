import json
import os
import logging
import srt
from datetime import timedelta

def ensure_directory_exists(file_path):
    """
    Ensure that the directory for the given file path exists.
    If it doesn't exist, create it.
    
    Args:
        file_path (str): The full path of the file.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def write_transcriptions(output_format, base_name, transcription_segments, speakers):
    """
    Writes transcriptions to the specified formats.
    
    Args:
        output_format (list): List of desired output formats ('text', 'json', 'srt').
        base_name (str): Base name for the output files.
        transcription_segments (list): List of transcription segments.
        speakers (list): List of speakers with their identifiers and names.
    """
    output_files = []  # Initialize a list to keep track of generated output files
    
    # Ensure base_name is an absolute path
    base_name = os.path.abspath(base_name)
    
    # Ensure the directory exists once for all output files
    ensure_directory_exists(base_name)
    
    if 'text' in output_format:
        text_file = f"{base_name}.txt"
        try:
            # Open the text file in write mode with UTF-8 encoding
            with open(text_file, 'w', encoding='utf-8') as f:
                # Iterate through each transcription segment and write to the file
                for seg in transcription_segments:
                    f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['speaker_id']}: {seg['text']}\n")
            logging.info(f"Text transcription saved to {text_file}")  # Log successful save with full path
            output_files.append(text_file)  # Add the full path to output_files list
        except Exception as e:
            logging.error(f"Failed to write text file: {e}")  # Log any errors during writing

    if 'srt' in output_format:
        srt_file = f"{base_name}.srt"
        try:
            # Create a list of srt.Subtitle objects from transcription segments
            subtitles = [
                srt.Subtitle(
                    index=i, 
                    start=timedelta(seconds=seg['start']),
                    end=timedelta(seconds=seg['end']),
                    content=f"{seg['speaker_id']}: {seg['text']}"
                )
                for i, seg in enumerate(transcription_segments, 1)
            ]
            # Open the SRT file in write mode and write the composed subtitles
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(srt.compose(subtitles))
            logging.info(f"SRT transcription saved to {srt_file}")  # Log successful save with full path
            output_files.append(srt_file)  # Add the full path to output_files list
        except Exception as e:
            logging.error(f"Failed to write SRT file: {e}")  # Log any errors during writing

    if 'json' in output_format:
        json_file = f"{base_name}.json"
        data = {
            'speakers': speakers,  # Include speaker information
            'transcript': transcription_segments  # Include transcription segments
        }
        try:
            # Open the JSON file in write mode and dump the data with indentation
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logging.info(f"JSON transcription saved to {json_file}")  # Log successful save with full path
            output_files.append(json_file)  # Add the full path to output_files list
        except Exception as e:
            logging.error(f"Failed to write JSON file: {e}")  # Log any errors during writing

    if output_files:
        # If any output files were generated, print a list of them with full paths
        print("\nGenerated Output Files:")
        for file in output_files:
            print(f"- {file}")
