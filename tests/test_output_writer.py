import pytest
from unittest.mock import patch, mock_open, MagicMock
from yawt.output_writer import write_transcriptions

def test_write_transcriptions_text_success(mocker):
    transcription_segments = [
        {'start': 0, 'end': 5, 'speaker_id': 'Speaker1', 'text': 'Hello'},
        {'start': 5, 'end': 10, 'speaker_id': 'Speaker2', 'text': 'Hi there'}
    ]
    speakers = [{'id': 'Speaker1', 'name': 'Speaker 1'}, {'id': 'Speaker2', 'name': 'Speaker 2'}]

    with patch('builtins.open', mock_open()) as mocked_file:
        write_transcriptions(['text'], 'test_audio', transcription_segments, speakers)
        mocked_file.assert_called_once_with('test_audio_transcription.txt', 'w', encoding='utf-8')
        handle = mocked_file()
        handle.write.assert_any_call("[0.00 - 5.00] Speaker1: Hello\n")
        handle.write.assert_any_call("[5.00 - 10.00] Speaker2: Hi there\n")

def test_write_transcriptions_srt_success(mocker):
    transcription_segments = [
        {'start': 0, 'end': 5, 'speaker_id': 'Speaker1', 'text': 'Hello'},
        {'start': 5, 'end': 10, 'speaker_id': 'Speaker2', 'text': 'Hi there'}
    ]
    speakers = [{'id': 'Speaker1', 'name': 'Speaker 1'}, {'id': 'Speaker2', 'name': 'Speaker 2'}]

    with patch('builtins.open', mock_open()) as mocked_file, \
         patch('yawt.output_writer.srt.compose') as mock_srt_compose:
        mock_srt_compose.return_value = "1\n00:00:00,000 --> 00:00:05,000\nSpeaker1: Hello\n\n2\n00:00:05,000 --> 00:00:10,000\nSpeaker2: Hi there\n"
        write_transcriptions(['srt'], 'test_audio', transcription_segments, speakers)
        mocked_file.assert_called_once_with('test_audio_transcription.srt', 'w', encoding='utf-8')
        handle = mocked_file()
        handle.write.assert_called_once_with("1\n00:00:00,000 --> 00:00:05,000\nSpeaker1: Hello\n\n2\n00:00:05,000 --> 00:00:10,000\nSpeaker2: Hi there\n")

def test_write_transcriptions_json_success(mocker):
    transcription_segments = [
        {'start': 0, 'end': 5, 'speaker_id': 'Speaker1', 'text': 'Hello'},
        {'start': 5, 'end': 10, 'speaker_id': 'Speaker2', 'text': 'Hi there'}
    ]
    speakers = [{'id': 'Speaker1', 'name': 'Speaker 1'}, {'id': 'Speaker2', 'name': 'Speaker 2'}]

    with patch('builtins.open', mock_open()) as mocked_file:
        write_transcriptions(['json'], 'test_audio', transcription_segments, speakers)
        mocked_file.assert_called_once_with('test_audio_transcript.json', 'w', encoding='utf-8')
        handle = mocked_file()
        expected_data = {
            'speakers': speakers,
            'transcript': transcription_segments
        }
        import json
        handle.write.assert_called_once_with(json.dumps(expected_data, indent=2))

def test_write_transcriptions_multiple_formats_success(mocker):
    transcription_segments = [
        {'start': 0, 'end': 5, 'speaker_id': 'Speaker1', 'text': 'Hello'}
    ]
    speakers = [{'id': 'Speaker1', 'name': 'Speaker 1'}]

    with patch('builtins.open', mock_open()) as mocked_file, \
         patch('yawt.output_writer.srt.compose') as mock_srt_compose:
        mock_srt_compose.return_value = "1\n00:00:00,000 --> 00:00:05,000\nSpeaker1: Hello\n"
        write_transcriptions(['text', 'srt', 'json'], 'test_audio', transcription_segments, speakers)
        assert mocked_file.call_count == 3