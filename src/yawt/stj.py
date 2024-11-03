"""
stj.py

A wrapper module that re-exports functionality from stjlib 0.4.0.
This maintains compatibility with existing code while using the official STJ implementation.
"""

from stjlib import (
    StandardTranscriptionJSON,
    Metadata,
    Transcript,
    Speaker,
    Segment,
    Word,
    Style,
    Transcriber,
    Source,
    WordTimingMode,
    ValidationError,
    STJError
)

# Re-export all the classes and utilities we need
__all__ = [
    'StandardTranscriptionJSON',
    'Metadata',
    'Transcript',
    'Speaker',
    'Segment',
    'Word',
    'Style',
    'Transcriber',
    'Source',
    'WordTimingMode',
    'ValidationError',
    'STJError'
]
