# Model generation parameters
MODEL_RETURN_DICT_IN_GENERATE = True  # Required for confidence calculation
MODEL_OUTPUT_SCORES = True            # Required for confidence calculation
MODEL_USE_CACHE = True               # Performance optimization

# Model Identifiers
WHISPER_LARGE_V3 = "openai/whisper-large-v3"
WHISPER_LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"

# Model Parameters
MODEL_SETTINGS = {
    WHISPER_LARGE_V3: {
        "batch_size": 16,
        "chunk_length_s": 30,
    },
    WHISPER_LARGE_V3_TURBO: {
        "batch_size": 24,
        "chunk_length_s": 30,
    },
}

# Speaker recognition API name
SPEAKER_RECOGNITION_API = "pyannote"