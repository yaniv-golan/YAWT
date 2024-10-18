import pytest
from unittest.mock import patch, MagicMock
from yawt.main import main

def test_full_transcription_flow(mocker):
    # Mock external dependencies
    mocker.patch('yawt.main.load_audio', return_value=MagicMock())
    mocker.patch('yawt.main.upload_file', return_value='https://0x0.st/abc123')
    mocker.patch('yawt.main.submit_diarization_job', return_value='job123')
    mocker.patch('yawt.main.get_job_status', return_value={'status': 'succeeded', 'output': {'diarization': []}})
    mock_transcribe_audio = mocker.patch('yawt.main.transcribe_audio', return_value='Mocked transcription.')
    mock_write_transcriptions = mocker.patch('yawt.main.write_transcriptions')
    
    # Call the main function
    main()
    
    # Assertions to ensure the flow was executed
    mock_write_transcriptions.assert_called_once()
    mock_transcribe_audio.assert_called_once()
