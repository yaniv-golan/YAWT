import os
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock
from yawt.output_writer import ensure_directory_exists, write_transcriptions
from stjlib import StandardTranscriptionJSON, Metadata, Transcriber, Transcript, Speaker, Segment, Word
from datetime import datetime, timezone

@pytest.fixture
def mock_stj():
    metadata = Metadata(
        transcriber=Transcriber(name="TestTranscriber", version="0.1"),
        created_at=datetime.now(timezone.utc),
    )
    transcript = Transcript(
        speakers=[
            Speaker(id="Speaker1", name="John Doe"),
            Speaker(id="Speaker2", name="Jane Smith")
        ],
        segments=[
            Segment(
                start=0.0,
                end=2.5,
                text="Hello, world!",
                speaker_id="Speaker1",
                confidence=0.95,
                language=None
            ),
            Segment(
                start=2.5,
                end=5.0,
                text="How are you?",
                speaker_id="Speaker2",
                confidence=0.90,
                language=None
            )
        ]
    )
    return StandardTranscriptionJSON(metadata=metadata, transcript=transcript)

def test_ensure_directory_exists(tmp_path):
    test_dir = tmp_path / "test_dir"
    file_path = test_dir / "test_file.txt"
    ensure_directory_exists(str(file_path))
    assert test_dir.exists()

@patch('os.path.exists')
@patch('os.makedirs')
def test_ensure_directory_exists_already_exists(mock_makedirs, mock_exists):
    mock_exists.return_value = True
    ensure_directory_exists("/path/to/file.txt")
    mock_makedirs.assert_not_called()

@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
def test_write_transcriptions_text(mock_file, mock_makedirs, mock_stj, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['text'], base_name, mock_stj)

    mock_file.assert_called_once_with(f"{base_name}.txt", 'w', encoding='utf-8')
    handle = mock_file()

    handle.write.assert_any_call("[0.00 - 2.50] John Doe: Hello, world!\n")
    handle.write.assert_any_call("[2.50 - 5.00] Jane Smith: How are you?\n")

@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
def test_write_transcriptions_stj(mock_file, mock_makedirs, mock_stj, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['stj'], base_name, mock_stj)

    mock_file.assert_called_once_with(f"{base_name}.stjson", 'w', encoding='utf-8')
    handle = mock_file()

    # Get the actual written data
    written_data = ''.join(args[0] for args, kwargs in handle.write.call_args_list)
    written_json = json.loads(written_data)

    # Verify the STJ structure
    assert 'stj' in written_json
    stj_data = written_json['stj']['stj']  # Navigate to the actual STJ content
    
    # Verify metadata
    assert 'metadata' in stj_data
    assert 'transcriber' in stj_data['metadata']
    assert stj_data['metadata']['transcriber']['name'] == 'TestTranscriber'
    
    # Verify transcript
    assert 'transcript' in stj_data
    assert 'speakers' in stj_data['transcript']
    assert 'segments' in stj_data['transcript']
    
    # Verify segments content
    segments = stj_data['transcript']['segments']
    assert len(segments) == 2
    
    # Verify first segment
    assert segments[0]['start'] == 0.0
    assert segments[0]['end'] == 2.5
    assert segments[0]['text'] == "Hello, world!"
    assert segments[0]['speaker_id'] == "Speaker1"
    
    # Verify second segment
    assert segments[1]['start'] == 2.5
    assert segments[1]['end'] == 5.0
    assert segments[1]['text'] == "How are you?"
    assert segments[1]['speaker_id'] == "Speaker2"

@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
def test_write_transcriptions_srt(mock_file, mock_makedirs, mock_stj, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['srt'], base_name, mock_stj)

    mock_file.assert_called_once_with(f"{base_name}.srt", 'w', encoding='utf-8')
    handle = mock_file()

    written_content = ''.join(args[0] for args, kwargs in handle.write.call_args_list)
    assert "1\n00:00:00,000 --> 00:00:02,500\nJohn Doe: Hello, world!\n\n" in written_content
    assert "2\n00:00:02,500 --> 00:00:05,000\nJane Smith: How are you?\n\n" in written_content

@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
@patch('builtins.print')
def test_write_transcriptions_multiple_formats(mock_print, mock_file, mock_makedirs, mock_stj, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['text', 'stj', 'srt'], base_name, mock_stj)

    assert mock_file.call_count == 3
    mock_print.assert_any_call("\nGenerated Output Files:")
    mock_print.assert_any_call(f"- {base_name}.txt")
    mock_print.assert_any_call(f"- {base_name}.stjson")
    mock_print.assert_any_call(f"- {base_name}.srt")

@patch('os.makedirs')
@patch('builtins.open', side_effect=IOError("Test error"))
@patch('logging.error')
def test_write_transcriptions_error_handling(mock_logging_error, mock_open, mock_makedirs, mock_stj, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['text'], base_name, mock_stj)

    mock_logging_error.assert_called_once_with("Failed to write text file: Test error")