import os
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock, call
from yawt.output_writer import ensure_directory_exists, write_transcriptions

@pytest.fixture
def sample_transcription_segments():
    return [
        {'start': 0.0, 'end': 2.5, 'speaker_id': 'Speaker 1', 'text': 'Hello, world!'},
        {'start': 2.5, 'end': 5.0, 'speaker_id': 'Speaker 2', 'text': 'How are you?'}
    ]

@pytest.fixture
def sample_speakers():
    return [
        {'id': 'Speaker 1', 'name': 'John'},
        {'id': 'Speaker 2', 'name': 'Jane'}
    ]

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
def test_write_transcriptions_text(mock_file, mock_makedirs, sample_transcription_segments, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['text'], base_name, sample_transcription_segments, [])
    
    mock_file.assert_called_once_with(f"{base_name}.txt", 'w', encoding='utf-8')
    handle = mock_file()
    handle.write.assert_any_call("[0.00 - 2.50] Speaker 1: Hello, world!\n")
    handle.write.assert_any_call("[2.50 - 5.00] Speaker 2: How are you?\n")

@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
def test_write_transcriptions_json(mock_file, mock_makedirs, sample_transcription_segments, sample_speakers, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['json'], base_name, sample_transcription_segments, sample_speakers)
    
    mock_file.assert_called_once_with(f"{base_name}.json", 'w', encoding='utf-8')
    handle = mock_file()
    expected_data = {
        'speakers': sample_speakers,
        'transcript': sample_transcription_segments
    }
    # Instead of checking for a single write call, we'll check if the JSON content is correct
    written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
    assert json.loads(written_data) == expected_data

@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
def test_write_transcriptions_srt(mock_file, mock_makedirs, sample_transcription_segments, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['srt'], base_name, sample_transcription_segments, [])
    
    mock_file.assert_called_once_with(f"{base_name}.srt", 'w', encoding='utf-8')
    handle = mock_file()
    
    # Check if the entire SRT content is written in a single call
    written_content = handle.write.call_args[0][0]
    assert "1\n00:00:00,000 --> 00:00:02,500\nSpeaker 1: Hello, world!\n\n" in written_content
    assert "2\n00:00:02,500 --> 00:00:05,000\nSpeaker 2: How are you?\n\n" in written_content

@patch('os.makedirs')
@patch('builtins.open', new_callable=mock_open)
@patch('builtins.print')
def test_write_transcriptions_multiple_formats(mock_print, mock_file, mock_makedirs, sample_transcription_segments, sample_speakers, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['text', 'json', 'srt'], base_name, sample_transcription_segments, sample_speakers)
    
    assert mock_file.call_count == 3
    mock_print.assert_any_call("\nGenerated Output Files:")
    mock_print.assert_any_call(f"- {base_name}.txt")
    mock_print.assert_any_call(f"- {base_name}.json")
    mock_print.assert_any_call(f"- {base_name}.srt")

@patch('os.makedirs')
@patch('builtins.open', side_effect=IOError("Test error"))
@patch('logging.error')
def test_write_transcriptions_error_handling(mock_logging_error, mock_open, mock_makedirs, sample_transcription_segments, tmp_path):
    base_name = str(tmp_path / "output")
    write_transcriptions(['text'], base_name, sample_transcription_segments, [])
    
    mock_logging_error.assert_called_once_with("Failed to write text file: Test error")
