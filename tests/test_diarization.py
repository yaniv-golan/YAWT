import pytest
from unittest.mock import patch, MagicMock
from yawt.diarization import submit_diarization_job, wait_for_diarization, get_job_status

@patch('yawt.diarization.requests.post')
def test_submit_diarization_job_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"jobId": "job123"}
    mock_post.return_value = mock_response

    job_id = submit_diarization_job("fake_token", "https://example.com/audio.wav", num_speakers=2)
    assert job_id == "job123"

@patch('yawt.diarization.requests.post')
def test_submit_diarization_job_rate_limit(mock_post, mocker):
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {'Retry-After': '1'}
    mock_post.return_value = mock_response

    with patch('time.sleep') as mock_sleep:
        mock_post.side_effect = [mock_response, MagicMock(status_code=200, json=lambda: {"jobId": "job123"})]
        job_id = submit_diarization_job("fake_token", "https://example.com/audio.wav")
        assert job_id == "job123"
        mock_sleep.assert_called_once_with(1)

@patch('yawt.diarization.requests.post')
def test_submit_diarization_job_failure(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_post.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        submit_diarization_job("fake_token", "https://example.com/audio.wav")
    assert "Diarization submission failed" in str(exc_info.value)

@patch('yawt.diarization.requests.get')
def test_get_job_status_success(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "succeeded", "output": {"diarization": []}}
    mock_get.return_value = mock_response

    job_info = get_job_status("fake_token", "job123")
    assert job_info['status'] == "succeeded"

@patch('yawt.diarization.requests.get')
def test_get_job_status_rate_limit(mock_get, mocker):
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {'Retry-After': '1'}
    mock_get.return_value = mock_response

    with patch('time.sleep') as mock_sleep:
        mock_get.side_effect = [mock_response, MagicMock(status_code=200, json=lambda: {"status": "succeeded"})]
        job_info = get_job_status("fake_token", "job123")
        assert job_info['status'] == "succeeded"
        mock_sleep.assert_called_once_with(1)

@patch('yawt.diarization.requests.get')
def test_get_job_status_failure(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_get.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        get_job_status("fake_token", "job123")
    assert "Failed to get job status" in str(exc_info.value)

@patch('yawt.diarization.get_job_status')
def test_wait_for_diarization_success(mock_get_job_status, mocker):
    mock_get_job_status.return_value = {"status": "succeeded", "output": {"diarization": []}}
    job_info = wait_for_diarization("fake_token", "job123", "https://example.com/audio.wav")
    assert job_info['status'] == "succeeded"

@patch('yawt.diarization.get_job_status')
def test_wait_for_diarization_timeout(mock_get_job_status, mocker):
    mock_get_job_status.return_value = {"status": "in_progress"}
    
    with patch('time.time', side_effect=[0, 4000]):
        with patch('time.sleep'):
            with pytest.raises(Exception) as exc_info:
                wait_for_diarization("fake_token", "job123", "https://example.com/audio.wav", timeout=3600)
            assert "Diarization job timed out." in str(exc_info.value)