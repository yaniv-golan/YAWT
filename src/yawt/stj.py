"""
stj.py

A Python wrapper for the Standard Transcription JSON (STJ) format.

This module provides data classes and utilities for working with STJ files,
which are used to represent transcribed audio and video data in a structured,
machine-readable JSON format.

For more information about the STJ format, please refer to the STJ Specification:
https://github.com/yaniv-golan/STJ

"""
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import json
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
from decimal import Decimal, InvalidOperation


# Custom Exceptions
class STJError(Exception):
    """Base class for exceptions in the STJ module."""
    pass

class InvalidConfidenceError(STJError, ValueError):
    """Raised when a confidence value is out of the valid range."""
    pass

class InvalidLanguageCodeError(STJError, ValueError):
    """Raised when an invalid language code is provided."""
    pass

class InvalidTimestampError(STJError, ValueError):
    """Raised when a timestamp is invalid."""
    pass

class InvalidMetadataError(STJError, ValueError):
    """Raised when metadata fields are invalid."""
    pass

class InvalidSegmentError(STJError, ValueError):
    """Raised when a segment has invalid data."""
    pass

class InvalidWordError(STJError, ValueError):
    """Raised when a word has invalid data."""
    pass


# Data Classes
@dataclass
class Transcriber:
    name: str
    version: str

@dataclass
class Source:
    uri: Optional[str] = None
    duration: Optional[float] = None
    languages: Optional[List[Lang]] = None

@dataclass
class Metadata:
    transcriber: Transcriber
    created_at: datetime  # Use datetime type
    source: Optional[Source] = None
    languages: Optional[List[Lang]] = None
    confidence_threshold: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate confidence_threshold
        if self.confidence_threshold is not None:
            try:
                self.confidence_threshold = float(self.confidence_threshold)
                if not (0.0 <= self.confidence_threshold <= 1.0):
                    raise InvalidConfidenceError(f"confidence_threshold must be between 0.0 and 1.0 (got {self.confidence_threshold})")
            except (ValueError, InvalidOperation):
                raise InvalidConfidenceError("confidence_threshold must be a float between 0.0 and 1.0")
        
        # Validate created_at
        if not isinstance(self.created_at, datetime):
            raise InvalidTimestampError("created_at must be a datetime instance")
        if self.created_at.tzinfo is None:
            # Assume UTC if no timezone is provided
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)
        
        # Validate transcriber
        if not isinstance(self.transcriber, Transcriber):
            raise InvalidMetadataError("transcriber must be an instance of Transcriber")

        # Validate languages
        if self.languages:
            for lang in self.languages:
                if not isinstance(lang, Lang):
                    raise InvalidLanguageCodeError(f"Invalid language code in languages: {lang}")
        

@dataclass
class Speaker:
    id: str
    name: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Word:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None

    def __post_init__(self):
        # Validate confidence
        if self.confidence is not None:
            try:
                self.confidence = float(self.confidence)
                if not (0.0 <= self.confidence <= 1.0):
                    raise InvalidConfidenceError(f"Word confidence must be between 0.0 and 1.0 (got {self.confidence})")
            except (ValueError, InvalidOperation):
                raise InvalidConfidenceError("Word confidence must be a float between 0.0 and 1.0")

        # Validate start and end times
        if self.start > self.end:
            raise InvalidWordError("Word start time must be less than or equal to end time")
        if self.start < 0 or self.end < 0:
            raise InvalidWordError("Word start and end times must be non-negative")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[Lang] = None
    style_id: Optional[str] = None
    words: Optional[List[Word]] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate start and end times
        if self.start > self.end:
            raise InvalidSegmentError("Segment start time must be less than or equal to end time")
        if self.start < 0 or self.end < 0:
            raise InvalidSegmentError("Segment start and end times must be non-negative")
        
        # Validate confidence
        if self.confidence is not None:
            try:
                self.confidence = float(self.confidence)
                if not (0.0 <= self.confidence <= 1.0):
                    raise InvalidConfidenceError(f"Segment confidence must be between 0.0 and 1.0 (got {self.confidence})")
            except (ValueError, InvalidOperation):
                raise InvalidConfidenceError("Segment confidence must be a float between 0.0 and 1.0")
        
        # Validate language
        if self.language is not None and not isinstance(self.language, Lang):
            raise InvalidLanguageCodeError(f"language must be an instance of Lang from iso639 (got {self.language})")
        
        # Validate words
        if self.words:
            for word in self.words:
                if not isinstance(word, Word):
                    raise InvalidWordError("words must be a list of Word instances")


@dataclass
class Transcript:
    speakers: List[Speaker] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    styles: Optional[List[Dict[str, Any]]] = None  # Define Style class if needed

@dataclass
class StandardTranscriptionJSON:
    metadata: Metadata
    transcript: Transcript

    @classmethod
    def from_file(cls, filename: str) -> 'StandardTranscriptionJSON':
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            raise STJError(f"File not found: {filename}")
        except json.JSONDecodeError as e:
            raise STJError(f"JSON decode error: {e}")

    def save(self, filename: str):
        data = self.to_dict()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            raise STJError(f"Error writing to file {filename}: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardTranscriptionJSON':
        try:
            # Deserialize metadata
            transcriber = Transcriber(**data['metadata']['transcriber'])
            created_at_str = data['metadata']['created_at']
            created_at = datetime.fromisoformat(created_at_str.rstrip('Z'))
            if created_at.tzinfo is None:
                # Assume UTC if timezone is missing
                created_at = created_at.replace(tzinfo=timezone.utc)
            source = None
            if 'source' in data['metadata']:
                source_data = data['metadata']['source']
                source_languages = [Lang(code) for code in source_data.get('languages', [])]
                source = Source(
                    uri=source_data.get('uri'),
                    duration=source_data.get('duration'),
                    languages=source_languages if source_languages else None
                )
            metadata_languages = [Lang(code) for code in data['metadata'].get('languages', [])] if data['metadata'].get('languages') else None
            metadata = Metadata(
                transcriber=transcriber,
                created_at=created_at,
                source=source,
                languages=metadata_languages,
                confidence_threshold=data['metadata'].get('confidence_threshold'),
                additional_info=data['metadata'].get('additional_info', {})
            )
            # Deserialize transcript
            speakers = [Speaker(**s) for s in data['transcript'].get('speakers', [])]
            segments = []
            for s in data['transcript']['segments']:
                words = [Word(**w) for w in s.get('words', [])] if s.get('words') else None
                language = Lang(s['language']) if s.get('language') else None
                segment = Segment(
                    start=s['start'],
                    end=s['end'],
                    text=s['text'],
                    speaker_id=s.get('speaker_id'),
                    confidence=s.get('confidence'),
                    language=language,
                    style_id=s.get('style_id'),
                    words=words,
                    additional_info=s.get('additional_info', {})
                )
                segments.append(segment)
            transcript = Transcript(
                speakers=speakers,
                segments=segments,
                styles=data['transcript'].get('styles')
            )
            return cls(metadata=metadata, transcript=transcript)
        except KeyError as e:
            raise InvalidMetadataError(f"Missing required field: {e}")
        except (TypeError, ValueError) as e:
            raise STJError(f"Error parsing STJ data: {e}")

    def to_dict(self) -> Dict[str, Any]:
        data = self._asdict()
        # Clean up optional fields that are None or empty
        self._cleanup_optional_fields(data['metadata'])
        if not data['transcript'].get('styles'):
            data['transcript'].pop('styles', None)
        # Clean up segments
        for segment in data['transcript']['segments']:
            if not segment.get('language'):
                segment.pop('language', None)
            if segment.get('words'):
                for word in segment['words']:
                    if word.get('confidence') is None:
                        word.pop('confidence', None)
            else:
                segment.pop('words', None)
            self._cleanup_optional_fields(segment)
        # Clean up speakers
        for speaker in data['transcript']['speakers']:
            self._cleanup_optional_fields(speaker)
        return data

    def _asdict(self) -> Dict[str, Any]:
        def serialize(obj: Any) -> Any:
            if isinstance(obj, Lang):
                return obj.pt1 or obj.pt3
            elif isinstance(obj, datetime):
                # Serialize datetime to ISO 8601 string in UTC
                return obj.replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')
            elif is_dataclass(obj):
                result = {}
                for field in fields(obj):
                    value = getattr(obj, field.name)
                    value = serialize(value)
                    if value is not None:
                        result[field.name] = value
                return result
            elif isinstance(obj, (list, tuple)):
                return type(obj)(serialize(v) for v in obj)
            elif isinstance(obj, dict):
                return {serialize(k): serialize(v) for k, v in obj.items()}
            else:
                return obj

        return serialize(self)

    def _cleanup_optional_fields(self, data_dict: Dict[str, Any]):
        keys_to_delete = [key for key, value in data_dict.items() if value in [None, {}, []]]
        for key in keys_to_delete:
            data_dict.pop(key)

    def _get_lang_by_code(self, code: str) -> Lang:
        try:
            return Lang(code)
        except InvalidLanguageValue:
            raise InvalidLanguageCodeError(f"Invalid language code: {code}")