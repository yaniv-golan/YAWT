import pytest
from unittest.mock import patch, MagicMock
from yawt.audio_handler import load_audio, upload_file, download_audio
import os
from io import BytesIO
import ffmpeg
import numpy as np

def test_load_audio_success(mocker):
    mock_ffmpeg = mocker.patch('yawt.audio_handler.ffmpeg')
    mock_ffmpeg.input.return_value.output.return_value.run.return_value = (b'\x00\x01', None)
    audio = load_audio("path/to/audio.wav")
    assert isinstance(audio, np.ndarray)
    assert len(audio) == 2  # Based on mock data

def test_load_audio_ffmpeg_error(mocker):
    mock_ffmpeg = mocker.patch('yawt.audio_handler.ffmpeg')
    mock_error = ffmpeg._run.Error('ffmpeg', 'Error', 'Error message')
    mock_ffmpeg.input.return_value.output.return_value.run.side_effect = mock_error
    
    with pytest.raises(ffmpeg._run.Error) as exc_info:
        load_audio("path/to/invalid_audio.wav")
    
    assert str(exc_info.value) == str(mock_error)

@patch('yawt.audio_handler.requests.post')
def test_upload_file_0x0_st_success(mock_post, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "https://0x0.st/abc123"
    mock_post.return_value = mock_response

    # Create a temporary file to upload
    temp_file = tmp_path / "file.wav"
    temp_file.write_bytes(b"data")

    # Call the upload_file function with the temporary file
    file_url = upload_file(str(temp_file), service='0x0.st', supported_upload_services={'0x0.st', 'file.io'})
    assert file_url == "https://0x0.st/abc123"

@patch('yawt.audio_handler.requests.post')
def test_upload_file_0x0_st_failure(mock_post, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response

    # Create a temporary file to upload
    temp_file = tmp_path / "file.wav"
    temp_file.write_bytes(b"data")

    # Expect an exception due to failed upload
    with pytest.raises(Exception) as exc_info:
        upload_file(str(temp_file), service='0x0.st', supported_upload_services={'0x0.st', 'file.io'})
    assert "Upload failed: 400 Bad Request" in str(exc_info.value)

@patch('yawt.audio_handler.requests.post')
def test_upload_file_file_io_success(mock_post, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True, "link": "https://file.io/xyz789"}
    mock_post.return_value = mock_response

    # Create a temporary file to upload
    temp_file = tmp_path / "file.wav"
    temp_file.write_bytes(b"data")

    # Call the upload_file function with the temporary file
    file_url = upload_file(str(temp_file), service='file.io', supported_upload_services={'0x0.st', 'file.io'})
    assert file_url == "https://file.io/xyz789"

@patch('yawt.audio_handler.requests.get')
def test_download_audio_success(mock_get, mocker, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b'data']
    mock_response.__enter__.return_value = mock_response
    mock_get.return_value = mock_response

    file_path = download_audio("https://example.com/audio.wav", destination_dir=tmp_path)
    assert file_path.endswith('.wav')  # Ensure the file has the correct extension

    # Verify that the file was written correctly
    assert (tmp_path / 'audio.wav').exists()
    assert (tmp_path / 'audio.wav').read_bytes() == b'data'

@patch('yawt.audio_handler.requests.get')
def test_download_audio_empty_file(mock_get, mocker, tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = []  # Simulate empty content
    mock_response.__enter__.return_value = mock_response
    mock_get.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        download_audio("https://example.com/empty_audio.wav", destination_dir=tmp_path)
    
    # Optionally, verify the exception message
    assert "Downloaded audio file is empty." in str(exc_info.value)
