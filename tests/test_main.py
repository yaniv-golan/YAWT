import pytest
from unittest.mock import patch, MagicMock
from yawt.main import transcribe_audio, main
import sys

@pytest.fixture
def mock_transcribe_audio_success(mocker):
    mock = mocker.patch('yawt.main.transcribe_audio')
    mock.return_value = "This is a mocked transcription."
    return mock

@pytest.fixture
def mock_transcribe_audio_failure(mocker):
    mock = mocker.patch('yawt.main.transcribe_audio')
    mock.side_effect = FileNotFoundError("Audio file not found.")
    return mock

def test_transcribe_audio_success(mock_transcribe_audio_success):
    audio_file = "path/to/test_audio.wav"
    expected_transcription = "This is a mocked transcription."

    result = transcribe_audio(audio_file)
    assert result == expected_transcription, "Transcription should match expected output"

def test_transcribe_audio_file_not_found(mock_transcribe_audio_failure):
    with pytest.raises(FileNotFoundError):
        transcribe_audio("path/to/nonexistent_audio.wav")

@patch('yawt.main.parse_arguments')
@patch('yawt.main.setup_logging')
@patch('yawt.main.initialize_environment')
@patch('yawt.main.check_api_tokens')
@patch('yawt.main.load_and_prepare_model')
@patch('yawt.main.handle_audio_input')
@patch('yawt.main.perform_diarization')
@patch('yawt.main.map_speakers')
@patch('yawt.main.load_audio')
@patch('yawt.main.calculate_cost')
@patch('yawt.main.transcribe_segments')
@patch('yawt.main.retry_transcriptions')  # Ensure retry_transcriptions is patched correctly
@patch('yawt.main.write_transcriptions')
def test_main_success(
    mock_write_transcriptions,
    mock_retry_transcriptions,  # Adjusted order if necessary
    mock_transcribe_segments,
    mock_calculate_cost,
    mock_load_audio,
    mock_map_speakers,
    mock_perform_diarization,
    mock_handle_audio_input,
    mock_load_and_prepare_model,
    mock_check_api_tokens,
    mock_initialize_environment,
    mock_setup_logging,
    mock_parse_arguments
):
    # Setup mock return values
    mock_parse_arguments.return_value = MagicMock(
        audio_url=None,
        input_file="path/to/test_audio.wav",
        context_prompt=None,
        main_language='en',
        secondary_language='es',  # Changed from list to single string
        num_speakers=2,
        dry_run=False,
        debug=False,
        verbose=False,
        pyannote_token="fake_token",
        openai_key="fake_key",
        model="openai/whisper-large-v3",
        output_format=['text']
    )
    
    # Setup other mocks as necessary
    mock_transcribe_segments.return_value = ([], [])
    mock_retry_transcriptions.return_value = ([], [])
    
    # Call main function
    main()
    
    # Assertions to ensure functions are called correctly
    mock_retry_transcriptions.assert_called_once_with(
        mock_load_and_prepare_model.return_value[0],
        mock_load_and_prepare_model.return_value[1],
        "path/to/test_audio.wav",
        [],
        [],
        {},
        mock_load_and_prepare_model.return_value[2],
        mock_load_and_prepare_model.return_value[3],
        "test_audio",
        [],
        300,
        1024,
        50,
        300,
        3,
        'es',  # Single secondary language
        0.6,
        'en'
    )
    
    mock_write_transcriptions.assert_called_once()

def test_main_diarization_failure(mocker):
    with patch('yawt.main.parse_arguments') as mock_parse:
        mock_parse.return_value = MagicMock(
            audio_url=None,
            input_file="path/to/test_audio.wav",
            context_prompt=None,
            language=['en'],
            num_speakers=2,
            dry_run=False,
            debug=False,
            verbose=False,
            pyannote_token="fake_token",
            openai_key="fake_key",
            model="openai/whisper-large-v3",
            output_format=['text']
        )
        mocker.patch('yawt.main.setup_logging')
        mocker.patch('yawt.main.initialize_environment')
        mocker.patch('yawt.main.check_api_tokens')
        mocker.patch('yawt.main.load_and_prepare_model')
        mocker.patch('yawt.main.handle_audio_input').return_value = ("uploaded_audio_url", "path/to/test_audio.wav")
        mocker.patch('yawt.main.perform_diarization', side_effect=Exception("Diarization failed."))

        with patch('yawt.main.sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)

@patch('yawt.main.time.sleep')
def test_submit_diarization_job_rate_limit(mock_sleep, mocker):
    ...
    mock_sleep.assert_called_once_with(1)
    ...
