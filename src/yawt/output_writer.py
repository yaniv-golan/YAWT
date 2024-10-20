import json
import os
import logging
import srt
from datetime import timedelta
from yawt.stj import StandardTranscriptionJSON, STJError, InvalidConfidenceError, InvalidLanguageCodeError

def ensure_directory_exists(file_path):
    """
    Ensure that the directory for the given file path exists.
    If it doesn't exist, create it.
    
    Args:
        file_path (str): The full path of the file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")

def write_transcriptions(output_format, base_name, stj: StandardTranscriptionJSON):
    """
    Writes transcriptions to the specified formats.
    
    Args:
        output_format (list): List of desired output formats ('text', 'json', 'srt').
        base_name (str): Base name for the output files.
        stj (StandardTranscriptionJSON): The STJ instance to extract data from.
    """
    output_files = []  # Initialize a list to keep track of generated output files
    
    # Ensure base_name is an absolute path
    base_name = os.path.abspath(base_name)
    
    # Ensure the directory exists once for all output files
    ensure_directory_exists(base_name)
    
    # Extract data from the STJ instance
    speakers = {speaker.id: speaker.name or speaker.id for speaker in stj.transcript.speakers}
    segments = stj.transcript.segments

    # Generate Text Output
    if 'text' in output_format:
        text_file = f"{base_name}.txt"
        try:
            with open(text_file, 'w', encoding='utf-8') as f:
                for seg in segments:
                    speaker_name = speakers.get(seg.speaker_id, seg.speaker_id or '')
                    f.write(f"[{seg.start:.2f} - {seg.end:.2f}] {speaker_name}: {seg.text}\n")
            logging.info(f"Text transcription saved to {text_file}")
            output_files.append(text_file)
        except Exception as e:
            logging.error(f"Failed to write text file: {e}")

    # Generate SRT Output
    if 'srt' in output_format:
        srt_file = f"{base_name}.srt"
        try:
            subtitles = []
            for i, seg in enumerate(segments, 1):
                speaker_name = speakers.get(seg.speaker_id, seg.speaker_id or '')
                content = f"{speaker_name}: {seg.text}" if speaker_name else seg.text
                subtitle = srt.Subtitle(
                    index=i,
                    start=timedelta(seconds=seg.start),
                    end=timedelta(seconds=seg.end),
                    content=content
                )
                subtitles.append(subtitle)
            with open(srt_file, 'w', encoding='utf-8') as f:
                f.write(srt.compose(subtitles))
            logging.info(f"SRT transcription saved to {srt_file}")
            output_files.append(srt_file)
        except Exception as e:
            logging.error(f"Failed to write SRT file: {e}")

    # Generate STJ Output
    if 'stj' in output_format:
        stj_file = f"{base_name}.stj.json"
        try:
            stj.save(stj_file)
            logging.info(f"STJ transcription saved to {stj_file}")
            output_files.append(stj_file)
        except (InvalidConfidenceError, InvalidLanguageCodeError, STJError) as e:
            logging.error(f"Failed to write STJ JSON file: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while writing STJ file: {e}")

    if output_files:
        # If any output files were generated, print a list of them with full paths
        print("\nGenerated Output Files:")
        for file in output_files:
            print(f"- {file}")
