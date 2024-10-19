import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from unittest.mock import patch, MagicMock, Mock
from yawt.transcription import (
    get_device,
    load_and_optimize_model,
    compute_per_token_confidence,
    aggregate_confidence,
    is_valid_language_code,
    evaluate_confidence,
    TimeoutException,
    ModelLoadError
)

import torch
import torch.nn.functional as F
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

def test_get_device_cuda_available():
    with patch('torch.cuda.is_available', return_value=True):
        device = get_device()
        assert device.type == 'cuda'

def test_get_device_mps_available():
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=True):
        device = get_device()
        assert device.type == 'mps'

def test_get_device_cpu():
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False):
        device = get_device()
        assert device.type == 'cpu'

@patch('yawt.transcription.AutoProcessor.from_pretrained')
@patch('yawt.transcription.AutoModelForSpeechSeq2Seq.from_pretrained')
@patch('torch.compile')
def test_load_and_optimize_model(mock_torch_compile, mock_model_pretrained, mock_proc_pretrained):
    mock_device = torch.device('cpu')
    
    # Create a MagicMock for the model without spec, but add necessary attributes
    mock_model_instance = MagicMock()
    mock_model_instance.dtype = torch.float32
    mock_model_instance.to.return_value = mock_model_instance  # Mock the 'to' method
    mock_model_pretrained.return_value = mock_model_instance
    
    # Mock torch.compile to return the model instance unchanged
    mock_torch_compile.return_value = mock_model_instance
    
    # Create a MagicMock for the processor
    mock_processor_instance = MagicMock()
    mock_proc_pretrained.return_value = mock_processor_instance
    
    with patch('yawt.transcription.get_device', return_value=mock_device):
        model, processor, device, dtype = load_and_optimize_model('test-model-id')
        
        # Assertions to ensure correct behavior
        mock_model_pretrained.assert_called_with(
            'test-model-id',
            torch_dtype=torch.float16 if mock_device.type in ["cuda", "mps"] else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        )
        mock_proc_pretrained.assert_called_with('test-model-id')
        mock_torch_compile.assert_called_once_with(mock_model_instance, mode="reduce-overhead")
        assert model == mock_model_instance
        assert processor == mock_processor_instance
        assert device == mock_device
        assert dtype == torch.float32

def test_is_valid_language_code():
    # Assuming 'en' is a valid code from iso639
    assert is_valid_language_code('en') is True
    assert is_valid_language_code('EN') is True  # case-insensitive
    assert is_valid_language_code('nonexistent') is False

def test_compute_per_token_confidence_with_scores():
    mock_output = Mock()
    # Define scores as a list of tensors, each tensor is [batch_size=1, vocab_size=2]
    mock_output.scores = [
        torch.tensor([[0.1, 0.9]]),  # Token 1
        torch.tensor([[0.8, 0.2]])   # Token 2
    ]
    confidences = compute_per_token_confidence(mock_output)
    
    # Calculate expected confidences using the same method as in the function
    expected_confidences = [
        F.softmax(score.float(), dim=-1).max().item()
        for score in mock_output.scores
    ]
    
    assert confidences == pytest.approx(expected_confidences, rel=1e-4)

def test_compute_per_token_confidence_no_scores():
    # Create a Mock without the 'scores' attribute using spec_set
    mock_output = Mock(spec_set=['sequences'])
    mock_output.sequences = torch.tensor([[1, 2, 3]])
    
    confidences = compute_per_token_confidence(mock_output)
    assert confidences == [1.0, 1.0, 1.0]

def test_aggregate_confidence():
    confidences = [0.8, 0.9, 0.85]
    overall = aggregate_confidence(confidences)
    assert overall == pytest.approx(0.85)

def test_evaluate_confidence_high():
    assert evaluate_confidence(0.9, 'en', threshold=0.6, main_language='en') is True

def test_evaluate_confidence_low_confidence():
    assert evaluate_confidence(0.5, 'en', threshold=0.6, main_language='en') is False

def test_evaluate_confidence_wrong_language():
    assert evaluate_confidence(0.9, 'fr', threshold=0.6, main_language='en') is False

def test_evaluate_confidence_no_language():
    assert evaluate_confidence(0.9, None, threshold=0.6, main_language='en') is False

def test_load_and_optimize_model_file_not_found():
    with patch('yawt.transcription.AutoModelForSpeechSeq2Seq.from_pretrained', side_effect=FileNotFoundError):
        with pytest.raises(ModelLoadError):
            load_and_optimize_model('invalid-model-id')

def test_load_and_optimize_model_invalid_value():
    with patch('yawt.transcription.AutoModelForSpeechSeq2Seq.from_pretrained', side_effect=ValueError):
        with pytest.raises(ModelLoadError):
            load_and_optimize_model('invalid-model-id')
