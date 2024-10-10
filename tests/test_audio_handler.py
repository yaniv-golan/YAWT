import pytest
from unittest.mock import patch, mock_open, MagicMock
from yawt.audio_handler import load_audio, upload_file, download_audio
import os

def test_load_audio_success(mocker):
    mock_ffmpeg = mocker.patch('yawt.audio_handler.ffmpeg')
    mock_ffmpeg.input.return_value.output.return_value.run.return_value = (b'\x00\x01', None)
    audio = load_audio("path/to/audio.wav")
    assert isinstance(audio, list)
    assert len(audio) == 2  # Based on mock data

def test_load_audio_ffmpeg_error(mocker):
    mock_ffmpeg = mocker.patch('yawt.audio_handler.ffmpeg')
    mock_ffmpeg.input.return_value.output.return_value.run.side_effect = ffmpeg.Error('ffmpeg', 'Error', 'Error message')
    
    with pytest.raises(SystemExit):
        load_audio("path/to/invalid_audio.wav")

@patch('yawt.audio_handler.requests.post')
def test_upload_file_0x0_st_success(mock_post, mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "https://0x0.st/abc123"
    mock_post.return_value = mock_response

    with patch('builtins.open', mock_open(read_data=b"data")):
        file_url = upload_file("path/to/file.wav", service='0x0.st')
        assert file_url == "https://0x0.st/abc123"

@patch('yawt.audio_handler.requests.post')
def test_upload_file_0x0_st_failure(mock_post, mocker):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response

    with patch('builtins.open', mock_open(read_data=b"data")), pytest.raises(Exception):
        upload_file("path/to/file.wav", service='0x0.st')

@patch('yawt.audio_handler.requests.post')
def test_upload_file_file_io_success(mock_post, mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "link": "https://file.io/xyz789"}
    mock_post.return_value = mock_response

    with patch('builtins.open', mock_open(read_data=b"data")):
        file_url = upload_file("path/to/file.wav", service='file.io')
        assert file_url == "https://file.io/xyz789"

@patch('yawt.audio_handler.requests.get')
def test_download_audio_success(mock_get, mocker, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b'data']
    mock_response.__enter__.return_value = mock_response
    mock_get.return_value = mock_response

    with patch('builtins.open', mock_open()) as mocked_file:
        file_path = download_audio("https://example.com/audio.wav")
        mocked_file.assert_called_once_with(os.path.join(tmp_path, 'temp_audio.wav'), 'wb')
        assert file_path.endswith('.wav')

def test_download_audio_empty_file(mocker, tmp_path):
    mock_get = mocker.patch('yawt.audio_handler.requests.get')
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = []
    mock_response.__enter__.return_value = mock_response
    mock_get.return_value = mock_response

    with patch('builtins.open', mock_open()), \
         patch('yawt.audio_handler.os.path.getsize', return_value=0), \
         pytest.raises(Exception):
        download_audio("https://example.com/empty_audio.wav")