
import pytest
from yawt.stj import (
    Transcriber,
    Metadata,
    Speaker,
    Segment,
    Word,
    Transcript,
    StandardTranscriptionJSON,
    InvalidConfidenceError,
    InvalidLanguageCodeError
)
from datetime import datetime, timezone
from iso639 import Lang, exceptions as iso639_exceptions

def test_stj_serialization():
    metadata = Metadata(
        transcriber=Transcriber(name="YAWT", version="0.2.0"),
        created_at=datetime.now(timezone.utc)
    )
    transcript = Transcript(
        speakers=[Speaker(id="Speaker1", name="John Doe")],
        segments=[
            Segment(
                start=0.0,
                end=5.0,
                text="Hello, world!",
                speaker_id="Speaker1",
                confidence=0.95,
                language=Lang('en'),
                words=[Word(start=0.0, end=1.0, text="Hello", confidence=0.9)]
            )
        ]
    )
    stj = StandardTranscriptionJSON(metadata=metadata, transcript=transcript)
    stj_dict = stj.to_dict()
    assert stj_dict['metadata']['transcriber']['name'] == "YAWT"
    assert stj_dict['transcript']['speakers'][0]['name'] == "John Doe"
    assert stj_dict['transcript']['segments'][0]['text'] == "Hello, world!"

def test_invalid_confidence_error():
    with pytest.raises(InvalidConfidenceError):
        Word(start=0.0, end=1.0, text="Test", confidence=1.5)

def test_invalid_language_code_error():
    with pytest.raises(InvalidLanguageCodeError):
        Segment(start=0.0, end=1.0, text="Test", language='invalid_code')

def test_stj_from_dict():
    data = {
        "metadata": {
            "transcriber": {"name": "YAWT", "version": "0.2.0"},
            "created_at": "2023-10-10T12:00:00Z"
        },
        "transcript": {
            "speakers": [{"id": "Speaker1", "name": "John Doe"}],
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "text": "Hello, world!",
                "speaker_id": "Speaker1",
                "confidence": 0.95,
                "language": "en",
                "words": [{"start": 0.0, "end": 1.0, "text": "Hello", "confidence": 0.9}]
            }]
        }
    }
    stj = StandardTranscriptionJSON.from_dict(data)
    assert stj.metadata.transcriber.name == "YAWT"
    assert stj.transcript.speakers[0].name == "John Doe"