import pytest
from yawt.transcription import transcribe_audio
import time
import concurrent.futures

def test_transcribe_audio_performance():
    start_time = time.time()
    # Assuming 'large_audio.wav' is a large file for testing
    transcription = transcribe_audio("tests/fixtures/large_audio.wav")
    end_time = time.time()
    duration = end_time - start_time
    assert transcription is not None
    assert duration < 300  # Transcription should complete within 5 minutes
