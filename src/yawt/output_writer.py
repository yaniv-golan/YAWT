import json
import os
import logging
import srt
from datetime import timedelta
from stjlib import StandardTranscriptionJSON

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

def write_transcriptions(output_format, base_name, transcription_doc: StandardTranscriptionJSON):
    """
    Writes transcriptions to the specified formats.
    
    Args:
        output_format (list): List of desired output formats ('text', 'json', 'srt').
        base_name (str): Base name for the output files.
        transcription_doc (StandardTranscriptionJSON): The STJ instance to extract data from.
    """
    output_files = []  # Initialize a list to keep track of generated output files
    
    # Ensure base_name is an absolute path
    base_name = os.path.abspath(base_name)
    
    # Ensure the directory exists once for all output files
    ensure_directory_exists(base_name)
    
    # Extract data from the STJ instance
    speakers = {speaker.id: speaker.name or speaker.id for speaker in transcription_doc.transcript.speakers}
    segments = transcription_doc.transcript.segments

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
        stj_file = f"{base_name}.stjson"
        try:
            # Convert to dictionary and write as JSON
            stj_data = transcription_doc.to_dict()
            with open(stj_file, 'w', encoding='utf-8') as f:
                json.dump({"stj": stj_data}, f, indent=2, ensure_ascii=False)
            logging.info(f"STJ transcription saved to {stj_file}")
            output_files.append(stj_file)
        except Exception as e:
            logging.error(f"Failed to write STJ file: {e}")

    if output_files:
        # If any output files were generated, print a list of them with full paths
        print("\nGenerated Output Files:")
        for file in output_files:
            print(f"- {file}")

class STJWriter:
    def __init__(self, config, context=None):
        self.config = config
        self.context = context

    def create_metadata(self):
        metadata = {
            "title": self.config.get("title", ""),
            "language": self.config.get("language", ""),
            "extensions": {
                "YAWT": {
                    "model": {
                        "name": self.config.get("model", ""),
                        "parameters": self.config.get("model_parameters", {})
                    },
                    "speaker_recognition": {
                        "api": self.config.get("speaker_recognition_api", ""),
                        "parameters": self.config.get("speaker_recognition_parameters", {})
                    },
                    "context": self.context if self.context else None,
                    "version": "1.0"
                }
            }
        }
        return metadata

    def write(self, transcription_data, output_file):
        stj_data = {
            "metadata": self.create_metadata(),
            "segments": [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment.get("speaker", ""),
                    "text": segment["text"]
                }
                for segment in transcription_data
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stj_data, f, indent=2, ensure_ascii=False)
