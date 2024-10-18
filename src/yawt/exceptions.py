class ModelLoadError(Exception):
    """Exception raised when the model fails to load or optimize."""
    pass

class DiarizationError(Exception):
    """Exception raised during the diarization process."""
    pass

class TranscriptionError(Exception):
    """Exception raised during the transcription process."""
    pass

# ... Add more custom exceptions as needed
