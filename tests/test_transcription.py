import pytest
from unittest.mock import patch, MagicMock
from yawt.transcription import (
    load_and_optimize_model,
    model_generate_with_timeout,
    transcribe_single_segment,
    retry_transcriptions,
    TimeoutException  # Ensure TimeoutException is imported
)
import concurrent.futures  # {{ edit: Import concurrent.futures }}

@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.generate.return_value = [1, 2, 3]
    return mock

@pytest.fixture
def mock_processor():
    mock = MagicMock()
    mock.batch_decode.return_value = ["Transcribed text"]
    return mock

def test_load_and_optimize_model_success(mocker):
    mocker.patch('yawt.transcription.AutoModelForSpeechSeq2Seq.from_pretrained', return_value=MagicMock())
    mocker.patch('yawt.transcription.AutoProcessor.from_pretrained', return_value=MagicMock())
    mocker.patch('yawt.transcription.get_device', return_value=MagicMock())
    mocker.patch('torch.compile')
    
    model, processor, device, torch_dtype = load_and_optimize_model("model_id")
    assert model is not None
    assert processor is not None
    assert device is not None
    assert torch_dtype is not None

def test_model_generate_with_timeout_success(mock_model):
    inputs = {"input_ids": MagicMock()}
    generate_kwargs = {}
    timeout = 5

    result = model_generate_with_timeout(mock_model, inputs, generate_kwargs, timeout)
    assert result == [1, 2, 3]
    mock_model.generate.assert_called_once_with(**inputs, **generate_kwargs)

def test_model_generate_with_timeout_failure(mock_model):
    inputs = {"input_ids": MagicMock()}
    generate_kwargs = {}
    timeout = 1

    with patch('concurrent.futures.ThreadPoolExecutor') as MockExecutor:
        instance = MockExecutor.return_value.__enter__.return_value
        instance.submit.side_effect = concurrent.futures.TimeoutError

        with pytest.raises(TimeoutException):
            model_generate_with_timeout(mock_model, inputs, generate_kwargs, timeout)

def test_transcribe_single_segment_success(mock_model, mock_processor):
    generate_kwargs = {}
    transcription = transcribe_single_segment(
        mock_model,
        mock_processor,
        {"input_ids": MagicMock()},
        generate_kwargs,
        idx=1,
        chunk_start=0,
        chunk_end=5,
        device=MagicMock(),
        torch_dtype=MagicMock()
    )
    assert transcription == "Transcribed text"
    mock_model.generate.assert_called_once()

def test_transcribe_single_segment_timeout(mock_model, mock_processor):
    mock_model.generate.side_effect = TimeoutException("Timeout")
    generate_kwargs = {}

    transcription = transcribe_single_segment(
        mock_model,
        mock_processor,
        {"input_ids": MagicMock()},
        generate_kwargs,
        idx=1,
        chunk_start=0,
        chunk_end=5,
        device=MagicMock(),
        torch_dtype=MagicMock()
    )
    assert transcription is None

def test_retry_transcriptions_success(mock_model, mock_processor):
    transcription_segments = []
    failed_segments = [{'segment': 1, 'reason': 'Initial failure'}]
    diarization_segments = [{'start': 0, 'end': 5}]
    audio_array = [0.0, 1.0, 2.0]

    with patch('yawt.transcription.transcribe_single_segment', return_value="Retry transcription"):
        failed = retry_transcriptions(
            model=mock_model,
            processor=mock_processor,
            audio_array=audio_array,
            diarization_segments=diarization_segments,
            failed_segments=failed_segments,
            generate_kwargs={},
            device=MagicMock(),
            torch_dtype=MagicMock(),
            base_name="test",
            transcription_segments=transcription_segments,
            MAX_RETRIES=2
        )
        assert failed == []
        assert len(transcription_segments) == 1
        assert transcription_segments[0]['text'] == "Retry transcription"

def test_retry_transcriptions_failure(mock_model, mock_processor):
    transcription_segments = []
    failed_segments = [{'segment': 1, 'reason': 'Initial failure'}]
    diarization_segments = [{'start': 0, 'end': 5}]
    audio_array = [0.0, 1.0, 2.0]

    with patch('yawt.transcription.transcribe_single_segment', return_value=None):
        failed = retry_transcriptions(
            model=mock_model,
            processor=mock_processor,
            audio_array=audio_array,
            diarization_segments=diarization_segments,
            failed_segments=failed_segments,
            generate_kwargs={},
            device=MagicMock(),
            torch_dtype=MagicMock(),
            base_name="test",
            transcription_segments=transcription_segments,
            MAX_RETRIES=2
        )
        assert len(failed) == 1  # Still failed after retries
        assert len(transcription_segments) == 0

def test_transcribe_audio_empty_input(mock_transcribe_audio_success):
    with pytest.raises(ValueError) as exc_info:
        transcribe_audio("")
    assert "Audio file path cannot be empty." in str(exc_info.value)

def test_transcribe_audio_invalid_format(mock_transcribe_audio_success):
    with pytest.raises(ValueError) as exc_info:
        transcribe_audio("path/to/invalid_audio.mp3")  # Assuming only .wav is supported
    assert "Unsupported audio format." in str(exc_info.value)
