# transcription.py

import logging
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Any
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperProcessor
from tqdm import tqdm
from yawt.config import SAMPLING_RATE
import torch.nn.functional as F
import numpy as np
from iso639 import iter_langs, Lang
from yawt.exceptions import ModelLoadError  # Import the custom exception
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from dataclasses import dataclass
import tenacity
from yawt.constants import (
    MODEL_RETURN_DICT_IN_GENERATE,
    MODEL_OUTPUT_SCORES,
    MODEL_USE_CACHE,
    WHISPER_LARGE_V3,
    WHISPER_LARGE_V3_TURBO,
    MODEL_SETTINGS,
)

# Define constants
DEFAULT_MAX_NEW_TOKENS = 256  # Maximum number of new tokens to generate during model inference

class TimeoutException(Exception):
    """
    Custom exception to indicate a timeout during transcription.
    """
    pass

def get_device() -> torch.device:
    """
    Determines the available device for computation: CUDA, MPS, or CPU.

    Returns:
        torch.device: The selected device based on availability.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Define ModelConfig dataclass
@dataclass
class ModelConfig:
    model: AutoModelForSpeechSeq2Seq
    processor: AutoProcessor
    device: torch.device
    torch_dtype: torch.dtype
    batch_size: int
    chunk_length_s: float

def load_and_optimize_model(model_id: str) -> ModelConfig:
    """
    Loads and optimizes the speech-to-text model.

    Args:
        model_id (str): The identifier for the model to load.

    Returns:
        ModelConfig: The loaded transcription model.
    """
    try:
        logging.info(f"Loading model '{model_id}'...")
        device = get_device()
        torch_dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32

        # Try loading with WhisperProcessor directly
        try:
            processor = WhisperProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
        except Exception as e:
            logging.error(f"Failed to load processor: {e}")
            raise

        # Basic parameters that work with all versions
        model_args = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True  # Add this to match processor loading
        }

        # Check transformers version for feature support
        import transformers
        version = tuple(map(int, transformers.__version__.split('.')))

        # use_safetensors supported from 4.26.0
        if version >= (4, 26, 0):
            model_args["use_safetensors"] = True

        # attn_implementation supported from 4.28.0 and requires PyTorch 2.0+
        if version >= (4, 28, 0) and torch.__version__ >= "2.0.0":
            if torch_dtype in [torch.float16, torch.bfloat16]:
                model_args["attn_implementation"] = "sdpa"
            else:
                logging.info("Using default attention implementation due to full precision mode")

        logging.debug(f"Loading model with args: {model_args}")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            **model_args
        ).to(device)

        # Convert model to half precision if on CUDA or MPS for performance
        if device.type in ["cuda", "mps"] and model.dtype != torch.float16:
            model = model.half()
            logging.info("Converted model to float16.")

        logging.info(f"Model loaded on {device} with dtype {model.dtype}.")

        import warnings
        # Suppress specific warnings from transformers library to reduce clutter
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.whisper.modeling_whisper")

        # Attempt to optimize the model using torch.compile for better performance
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logging.info("Model optimized with torch.compile.")
        except Exception as e:
            logging.warning(f"Failed to optimize model with torch.compile: {e}")

        # Retrieve model settings based on model_id
        model_settings = MODEL_SETTINGS.get(model_id)
        if not model_settings:
            raise ValueError(f"Unsupported model_id: {model_id}")

        batch_size = model_settings["batch_size"]
        chunk_length_s = model_settings["chunk_length_s"]

        logging.info(f"Model settings: batch_size={batch_size}, chunk_length_s={chunk_length_s}")

        return ModelConfig(
            model=model,
            processor=processor,
            device=device,
            torch_dtype=torch_dtype,
            batch_size=batch_size,
            chunk_length_s=chunk_length_s
        )

    except FileNotFoundError as fnf_error:
        logging.exception(f"Model file not found: {fnf_error}")
        raise ModelLoadError("Model file is missing.") from fnf_error
    except ValueError as val_error:
        logging.exception(f"Invalid value encountered: {val_error}")
        raise ModelLoadError("Invalid model configuration.") from val_error
    except Exception as e:
        logging.exception(f"An unexpected error occurred while loading the model: {e}")
        raise ModelLoadError("Failed to load and optimize the model.") from e

def model_generate_with_timeout(
    model: AutoModelForSpeechSeq2Seq,
    inputs: Dict[str, torch.Tensor],
    generate_kwargs: Dict[str, Any],
    transcription_timeout: int
) -> Any:
    """
    Generates output from the model with a timeout.

    Args:
        model: The transcription model.
        inputs: The input tensor dictionary.
        generate_kwargs: Generation keyword arguments.
        transcription_timeout: Timeout for transcription in seconds.

    Returns:
        GenerateOutput: Generated token sequences and scores.

    Raises:
        TimeoutException: If transcription exceeds the specified timeout.
    """
    def generate() -> Any:
        # Add necessary generation parameters
        adjusted_kwargs = generate_kwargs.copy()
        adjusted_kwargs['input_features'] = inputs['input_features']
        adjusted_kwargs['return_dict_in_generate'] = MODEL_RETURN_DICT_IN_GENERATE  # Ensure detailed output
        adjusted_kwargs['output_scores'] = MODEL_OUTPUT_SCORES                      # Include scores
        logging.debug(f"Final generate_kwargs before generation: {adjusted_kwargs}")
        return model.generate(**adjusted_kwargs, use_cache=MODEL_USE_CACHE)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate)
        try:
            return future.result(timeout=transcription_timeout)
        except concurrent.futures.TimeoutError:
            logging.error("Transcription timed out.")
            raise TimeoutException("Transcription timed out.")

def compute_per_token_confidence(outputs: Any) -> List[float]:
    """
    Computes per-token confidence scores from model outputs.

    Args:
        outputs: The GenerateOutput object from model.generate().

    Returns:
        List of per-token confidence scores.
    """
    if hasattr(outputs, 'scores'):
        token_confidences = []
        for score in outputs.scores:
            probabilities = F.softmax(score, dim=-1)
            top_prob, _ = torch.max(probabilities, dim=-1)
            token_confidences.extend(top_prob.tolist())
        return token_confidences
    else:
        logging.warning("Output does not contain scores. Returning full confidence.")
        # If scores are not available, return full confidence
        return [1.0] * outputs.sequences.shape[1]

def aggregate_confidence(token_confidences: List[float]) -> float:
    """
    Aggregates per-token confidence scores into an overall confidence score.

    Args:
        token_confidences: List of per-token confidence scores.

    Returns:
        The average confidence score.
    """
    if not token_confidences:
        return 0.0
    overall_confidence = sum(token_confidences) / len(token_confidences)
    return overall_confidence

# Create a set of valid language codes
valid_codes = set()
for lang in iter_langs():
    if lang.pt1:
        valid_codes.add(lang.pt1.lower())  # ISO 639-1 codes
    if lang.pt2b:
        valid_codes.add(lang.pt2b.lower())  # ISO 639-2/B codes
    if lang.pt2t:
        valid_codes.add(lang.pt2t.lower())  # ISO 639-2/T codes
    if lang.pt3:
        valid_codes.add(lang.pt3.lower())  # ISO 639-3 codes
    valid_codes.add(lang.name.lower())  # Language names (lowercase)

def is_valid_language_code(code: str) -> bool:
    return code.lower() in valid_codes

def extract_language_token(generated_ids: torch.Tensor, tokenizer: Any) -> Optional[str]:
    tokens = tokenizer.convert_ids_to_tokens(generated_ids.cpu().flatten().tolist())

    logging.debug(f"Generated tokens: {tokens}")

    for token in tokens[:5]:  # Check only the first few tokens
        if token.startswith('<|') and token.endswith('|>'):
            lang_code = token[2:-2]  # Remove '<|' and '|>'
            if is_valid_language_code(lang_code):
                return lang_code
        elif token != '<|startoftranscript|>':
            break

    return None

@dataclass
class ModelResources:
    model: AutoModelForSpeechSeq2Seq
    processor: AutoProcessor
    device: torch.device
    torch_dtype: torch.dtype
    generate_kwargs: Dict[str, Any]
    batch_size: int
    chunk_length_s: float

@dataclass
class TranscriptionConfig:
    transcription_timeout: int
    max_target_positions: int
    buffer_tokens: int
    confidence_threshold: float
    context_prompt: Optional[str] = None  # Store context prompt in config
    overlap_duration: float = 2.0        # Added overlap duration attribute

def prepare_input_ids(context, tokenizer, device):
    # Encode the context and ensure the tensor is of type Long
    input_ids = tokenizer.encode(context, return_tensors='pt').to(device).long()
    return input_ids

def transcribe_single_segment(
    idx: int,
    chunk_start: float,
    chunk_end: float,
    inputs: Dict[str, torch.Tensor],
    model_resources: ModelResources,
    config: TranscriptionConfig,
    main_language: str  # Now mandatory
) -> Tuple[Optional[str], float, Optional[str], Optional[torch.Tensor]]:
    try:
        model = model_resources.model
        processor = model_resources.processor
        device = model_resources.device
        torch_dtype = model_resources.torch_dtype
        generate_kwargs = model_resources.generate_kwargs.copy()
        transcription_timeout = config.transcription_timeout
        max_target_positions = config.max_target_positions
        buffer_tokens = config.buffer_tokens

        # Assert batch size is 1
        assert inputs['input_features'].shape[0] == 1, f"Batch size must be 1, got {inputs['input_features'].shape[0]}"
        # Ensures that each transcription is processed individually to maintain consistency and prevent unexpected behavior
        assert inputs['input_features'].ndim == 3, f"Expected 3D tensor for input_features, got {inputs['input_features'].ndim}D"

        # Verify other input tensors if present
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                assert tensor.shape[0] == 1, f"Batch size for {key} must be 1, got {tensor.shape[0]}"

        adjusted_generate_kwargs = generate_kwargs.copy()
        # adjusted_generate_kwargs["language"] = main_language

        forced_decoder_ids = processor.get_decoder_prompt_ids(language=main_language, task="transcribe")
        adjusted_generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

        input_length = inputs['input_features'].shape[1]

        # Add detailed debug logging for model config
        logging.debug(f"Segment {idx}: Model config type: {type(model.config)}")
        logging.debug(f"Segment {idx}: Model config attributes: {dir(model.config)}")
        
        # Check max_length in model config
        if hasattr(model.config, 'max_length'):
            logging.debug(f"Segment {idx}: Direct max_length from config: {model.config.max_length}")
        
        # Check decoder config
        if hasattr(model.config, 'decoder'):
            logging.debug(f"Segment {idx}: Decoder config type: {type(model.config.decoder)}")
            logging.debug(f"Segment {idx}: Decoder config attributes: {dir(model.config.decoder)}")
            if hasattr(model.config.decoder, 'max_length'):
                logging.debug(f"Segment {idx}: Decoder max_length: {model.config.decoder.max_length}")
        
        # Check max_target_positions
        logging.debug(f"Segment {idx}: max_target_positions from config: {max_target_positions}")

        # Updated max_length retrieval with logging
        max_length = max_target_positions
        logging.debug(f"Segment {idx}: Using max_target_positions: {max_length} (fixed value)")

        prompt_length = adjusted_generate_kwargs.get('decoder_input_ids', torch.tensor([], device=device, dtype=torch_dtype)).shape[-1]

        # Add debug prints
        logging.debug(f"Segment {idx}: input_length = {input_length}")
        logging.debug(f"Segment {idx}: prompt_length = {prompt_length}")
        logging.debug(f"Segment {idx}: buffer_tokens = {buffer_tokens}")
        logging.debug(f"Segment {idx}: Calculation: {max_length} - {input_length} - {prompt_length} - {buffer_tokens} - 1")
        
        remaining_length = max_length - input_length - prompt_length - buffer_tokens - 1
        max_new_tokens = min(DEFAULT_MAX_NEW_TOKENS, remaining_length)
        max_new_tokens = max(max_new_tokens, 1)
        
        logging.debug(f"Segment {idx}: remaining_length before min/max = {remaining_length}")
        logging.debug(f"Segment {idx}: DEFAULT_MAX_NEW_TOKENS = {DEFAULT_MAX_NEW_TOKENS}")
        logging.debug(f"Segment {idx}: final max_new_tokens = {max_new_tokens}")

        adjusted_generate_kwargs["max_new_tokens"] = min(
            adjusted_generate_kwargs.get("max_new_tokens", max_new_tokens),
            max_new_tokens
        )

        logging.debug(f"Segment {idx}: Input features shape: {inputs['input_features'].shape}")
        logging.debug(f"Segment {idx}: Generate kwargs: {adjusted_generate_kwargs}")

        # Prepare decoder_input_ids with correct dtype
        decoder_input_ids = prepare_input_ids(config.context_prompt, processor.tokenizer, device) if config.context_prompt else None

        if decoder_input_ids is not None:
            adjusted_generate_kwargs["decoder_input_ids"] = decoder_input_ids  # Ensure decoder_input_ids are integers and passed to model.generate

        with torch.no_grad():  # Disable gradient computation during inference
            outputs = model_generate_with_timeout(
                model=model,
                inputs=inputs,
                generate_kwargs=adjusted_generate_kwargs,
                transcription_timeout=transcription_timeout
            )

        # Debug logging for outputs
        logging.debug(f"Segment {idx}: Type of outputs: {type(outputs)}")
        logging.debug(f"Segment {idx}: Outputs attributes: {dir(outputs)}")

        # Ensure outputs have 'sequences' and 'scores'
        if hasattr(outputs, 'sequences') and hasattr(outputs, 'scores'):
            # Assert that we're only processing one sequence
            assert outputs.sequences.shape[0] == 1, f"Expected 1 sequence, got {outputs.sequences.shape[0]}"
            assert outputs.sequences.ndim == 2, f"Expected 2D tensor for output sequences, got {outputs.sequences.ndim}D"

            transcription = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            token_confidences = compute_per_token_confidence(outputs)
            overall_confidence = aggregate_confidence(token_confidences)
            language_token = extract_language_token(outputs.sequences, processor.tokenizer)
            generated_sequence = outputs.sequences[0]
        else:
            logging.error(f"Segment {idx}: Unexpected output format")
            return None, 0.0, None, None

        logging.debug(f"Segment {idx}: Transcription: '{transcription}', Confidence: {overall_confidence}, Language: {language_token}")

        return transcription, overall_confidence, language_token, generated_sequence
    except TimeoutException:
        logging.error(f"Transcription timed out for segment {idx} ({chunk_start}-{chunk_end}s)")
        return None, 0.0, None, None
    except AssertionError as ae:
        logging.error(f"Shape mismatch or assertion failed in segment {idx}: {ae}")
        return None, 0.0, None, None
    except Exception as e:
        logging.exception(f"Unexpected error during transcription of segment {idx}: {e}")
        raise

def evaluate_confidence(
    overall_confidence: float,
    language_token: Optional[str],
    threshold: float = 0.6,
    main_language: str = 'en'
) -> bool:
    if overall_confidence == 0.0:
        logging.warning(f"Zero confidence detected. Confidence: {overall_confidence}")
        return False

    is_high_confidence = overall_confidence >= threshold

    if language_token is None:
        is_main_language = False
    else:
        if not is_valid_language_code(language_token) or not is_valid_language_code(main_language):
            logging.warning(f"Unrecognized language code: {language_token} or {main_language}")
            is_main_language = False
        else:
            is_main_language = language_token.lower() == main_language.lower()

    logging.debug(f"Confidence evaluation: Overall confidence: {overall_confidence}, Detected Language: {language_token}, Main Language: {main_language}")
    logging.debug(f"Evaluation result: High confidence: {is_high_confidence}, Is main language: {is_main_language}")

    return is_high_confidence and is_main_language

def transcribe_with_retry(
    idx: int,
    chunk_start: float,
    chunk_end: float,
    inputs: Dict[str, torch.Tensor],
    model_resources: ModelResources,
    config: TranscriptionConfig,
    main_language: str
) -> Tuple[Optional[str], float, Optional[str], Optional[torch.Tensor]]:
    """Custom retry implementation without tenacity"""
    
    max_attempts = 3
    attempt = 1
    last_result = None

    while attempt <= max_attempts:
        try:
            logging.debug(f"TRANSCRIBE - Segment {idx}, Attempt {attempt}/{max_attempts}")
            result = transcribe_single_segment(
                idx=idx,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                inputs=inputs,
                model_resources=model_resources,
                config=config,
                main_language=main_language
            )

            if result is not None:
                transcription, confidence, language, tokens = result
                if transcription and confidence >= 0.3:
                    logging.debug(f"TRANSCRIBE - Segment {idx}: Successful transcription with confidence {confidence:.3f}")
                    return result
                else:
                    logging.debug(f"TRANSCRIBE - Segment {idx}: Low confidence ({confidence:.3f}) on attempt {attempt}")
                    last_result = result
                    attempt += 1  # Only increment attempt if we need to retry
                    continue  # Try next attempt
            else:
                logging.debug(f"TRANSCRIBE - Segment {idx}: Received None result on attempt {attempt}")
                last_result = None
                attempt += 1  # Only increment attempt if we need to retry
                continue

        except Exception as e:
            logging.error(f"TRANSCRIBE - Segment {idx}: Attempt {attempt} failed with error: {e}", exc_info=True)
            last_result = None
            attempt += 1  # Only increment attempt if we need to retry
            continue

    # If we get here, we've exhausted all attempts
    if last_result is not None and last_result[1] >= 0.3:
        logging.warning(f"TRANSCRIBE - Segment {idx}: Using last obtained result with confidence {last_result[1]:.3f}")
        return last_result
    else:
        logging.error(f"TRANSCRIBE - Segment {idx}: All {max_attempts} attempts failed or confidence below threshold.")
        return None, 0.0, None, None

def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right, sampling_rate):
    """
    Iterate over chunks of audio with dynamic stride adjustment.
    
    Args:
        inputs: Input audio array
        feature_extractor: Feature extractor for processing chunks
        chunk_len: Length of each chunk in samples
        stride_left: Left stride length in samples
        stride_right: Right stride length in samples
        sampling_rate: Audio sampling rate
    
    Yields:
        Dict containing processed chunk data
    """
    inputs_len = inputs.shape[0]
    chunk_start_idx = 0
    
    # Validate and adjust stride lengths if necessary
    total_stride = stride_left + stride_right
    if chunk_len <= total_stride:
        logging.debug(f"Adjusting stride lengths. Chunk length: {chunk_len}, Total stride: {total_stride}")
        # Reduce stride lengths proportionally
        adjusted_total_stride = chunk_len // 2
        stride_left = adjusted_total_stride // 2
        stride_right = adjusted_total_stride - stride_left
        logging.debug(f"Adjusted strides - Left: {stride_left}, Right: {stride_right}")

    while chunk_start_idx < inputs_len:
        chunk_end_idx = min(chunk_start_idx + chunk_len, inputs_len)
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        
        if len(chunk) == 0:
            break
            
        processed = feature_extractor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
        
        # Adjust strides for first and last chunks
        _stride_left = stride_left if chunk_start_idx != 0 else 0
        is_last = chunk_end_idx >= inputs_len
        _stride_right = stride_right if not is_last else 0

        chunk_len_actual = chunk.shape[0]
        stride = (chunk_len_actual, _stride_left, _stride_right)
        
        if chunk_len_actual > _stride_left:
            yield {"is_last": is_last, "stride": stride, **processed}

        # Calculate next chunk start position
        increment = chunk_len - _stride_left - _stride_right
        increment = max(increment, chunk_len // 4)  # Ensure minimum forward progress
        chunk_start_idx += increment
        
        logging.debug(f"Chunk progress - Start: {chunk_start_idx}, End: {chunk_end_idx}, Increment: {increment}")

def merge_sequences(sequences):
    if not sequences:
        return []

    merged_sequence = sequences[0].tolist()

    for next_seq in sequences[1:]:
        next_sequence = next_seq.tolist()
        # Simple overlap handling
        overlap_size = min(len(merged_sequence), len(next_sequence)) // 2
        overlap_found = False
        for i in range(overlap_size, 0, -1):
            if merged_sequence[-i:] == next_sequence[:i]:
                merged_sequence.extend(next_sequence[i:])
                overlap_found = True
                break
        if not overlap_found:
            merged_sequence.extend(next_sequence)
    return merged_sequence

def transcribe_segments(
    diarization_segments: List[Dict[str, Any]],
    audio_array: np.ndarray,
    model_resources: ModelResources,
    config: TranscriptionConfig,
    main_language: str,
    processed_segments: Optional[set] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Transcribe audio segments with dynamic stride adjustment for short segments.
    """
    if processed_segments is None:
        processed_segments = set()  # Initialize as empty set within the run

    transcription_segments = []
    failed_segments = []

    for idx, segment in enumerate(tqdm(diarization_segments, desc="Transcribing segments"), start=1):
        segment_id = (segment['speaker_id'], round(segment['start'], 3), round(segment['end'], 3))
        if segment_id in processed_segments:
            logging.info(f"Segment {idx}: Skipping already processed segment: {segment_id}")
            continue

        logging.debug(f"Segment {idx}: Starting transcription for segment: {segment_id}")
        try:
            original_start = segment['start']
            original_end = segment['end']

            # Adjust start and end times with overlap
            if not transcription_segments:
                adjusted_start = original_start
            else:
                adjusted_start = transcription_segments[-1]['end']

            adjusted_end = min(original_end + config.overlap_duration, audio_array.shape[0] / SAMPLING_RATE)

            start_sample = int(adjusted_start * SAMPLING_RATE)
            end_sample = int(adjusted_end * SAMPLING_RATE)
            chunk = audio_array[start_sample:end_sample]

            # Calculate chunk parameters with dynamic adjustment
            chunk_length_s = adjusted_end - adjusted_start
            stride_length_s = min(config.overlap_duration, chunk_length_s / 4)  # Limit stride to 1/4 of chunk length
            
            chunk_len_samples = int(chunk_length_s * SAMPLING_RATE)
            stride_left_samples = int(stride_length_s * SAMPLING_RATE)
            stride_right_samples = int(stride_length_s * SAMPLING_RATE)
            
            logging.debug(f"Segment {idx} parameters - Length: {chunk_length_s:.2f}s, "
                        f"Stride: {stride_length_s:.2f}s, "
                        f"Samples: {chunk_len_samples}")

            sequences = []
            overall_confidences = []
            language_tokens = []

            for chunk_data in chunk_iter(
                inputs=chunk,
                feature_extractor=model_resources.processor.feature_extractor,
                chunk_len=chunk_len_samples,
                stride_left=stride_left_samples,
                stride_right=stride_right_samples,
                sampling_rate=SAMPLING_RATE
            ):
                inputs = {k: v.to(model_resources.device).to(model_resources.torch_dtype) 
                          for k, v in chunk_data.items() if k not in ['is_last', 'stride']}
                chunk_stride = chunk_data['stride']
                transcription, overall_confidence, language_token, generated_sequence = transcribe_with_retry(
                    idx=idx,
                    chunk_start=adjusted_start,
                    chunk_end=adjusted_end,
                    inputs=inputs,
                    model_resources=model_resources,
                    config=config,
                    main_language=main_language
                )
                if transcription is not None and generated_sequence is not None:
                    sequences.append(generated_sequence)
                    overall_confidences.append(overall_confidence)
                    language_tokens.append(language_token)

            # Merge overlapping sequences
            if sequences:
                merged_sequence = merge_sequences(sequences)
                transcription = model_resources.processor.decode(merged_sequence, skip_special_tokens=True).strip()
                overall_confidence = np.mean(overall_confidences)
                language_token = language_tokens[0] if language_tokens else None
            else:
                transcription = ""
                overall_confidence = 0.0
                language_token = None

            # Evaluate confidence
            low_confidence = overall_confidence < config.confidence_threshold or not evaluate_confidence(
                overall_confidence, language_token, threshold=config.confidence_threshold, main_language=main_language
            )

            transcript = {
                'speaker_id': segment['speaker_id'],
                'start': original_start,
                'end': original_end,
                'text': transcription,
                'confidence': overall_confidence,
                'language': language_token,
                'low_confidence': low_confidence
            }
            transcription_segments.append(transcript)
            processed_segments.add(segment_id)  # Track processed segment

            logging.info(f"Segment {idx}: Successfully transcribed.")

        except RetryError:
            logging.error(f"Segment {idx}: Failed after multiple retry attempts due to timeout.")
            failed_segments.append(segment)
        except Exception as e:
            logging.exception(f"Segment {idx}: Unexpected error during transcription: {e}")
            failed_segments.append(segment)

    logging.info(f"Transcription completed: {len(transcription_segments)} successful, {len(failed_segments)} failed.")
    return transcription_segments, failed_segments

def retry_transcriptions(
    audio_array: np.ndarray,
    diarization_segments: List[Dict[str, Any]],
    failed_segments: List[Dict[str, Any]],
    transcription_segments: List[Dict[str, Any]],
    model_resources: ModelResources,
    config: TranscriptionConfig,
    secondary_language: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model = model_resources.model
    processor = model_resources.processor
    device = model_resources.device
    torch_dtype = model_resources.torch_dtype
    generate_kwargs = model_resources.generate_kwargs.copy()
    confidence_threshold = config.confidence_threshold
    sampling_rate = SAMPLING_RATE

    if not is_valid_language_code(secondary_language):
        logging.warning(f"Invalid secondary language code: {secondary_language}")
        return transcription_segments, failed_segments  # No valid secondary language to retry with

    try:
        lang_obj = Lang(secondary_language)
        lang = lang_obj.pt1.lower()
        if not lang:
            raise ValueError
    except (ValueError, AttributeError):
        logging.warning(f"Invalid secondary language code: {secondary_language}")
        return transcription_segments, failed_segments

    logging.info(f"Starting retry for failed segments with secondary language '{lang}'.")
    logging.info(f"Number of segments to retry: {len(failed_segments)}")

    retry_failed_segments = []
    for failure in tqdm(failed_segments, desc="Retrying Segments", unit="segment"):
        logging.debug(f"Starting retry for failed segment: {failure}")
        idx = failure['segment_index']
        segment = diarization_segments[idx]
        original_start = segment['start']
        original_end = segment['end']

        try:
            # Adjust start and end times with overlap
            if not transcription_segments:
                adjusted_start = segment['start']
            else:
                adjusted_start = transcription_segments[-1]['end']

            adjusted_end = min(original_end + config.overlap_duration, audio_array.shape[0] / sampling_rate)

            start_sample = int(adjusted_start * sampling_rate)
            end_sample = int(adjusted_end * sampling_rate)
            chunk = audio_array[start_sample:end_sample]

            # Determine chunk parameters
            chunk_length_s = adjusted_end - adjusted_start
            stride_length_s = config.overlap_duration

            # Prepare chunks with overlapping
            chunk_len_samples = int(chunk_length_s * sampling_rate)
            stride_left_samples = int(stride_length_s * sampling_rate)
            stride_right_samples = int(stride_length_s * sampling_rate)

            sequences = []
            overall_confidences = []
            language_tokens = []

            for chunk_data in chunk_iter(
                inputs=chunk,
                feature_extractor=processor.feature_extractor,
                chunk_len=chunk_len_samples,
                stride_left=stride_left_samples,
                stride_right=stride_right_samples,
                sampling_rate=sampling_rate
            ):
                inputs = {k: v.to(device).to(torch_dtype) for k, v in chunk_data.items() if k != 'is_last' and k != 'stride'}
                chunk_stride = chunk_data['stride']
                transcription, overall_confidence, language_token, generated_sequence = transcribe_with_retry(
                    idx=idx,
                    chunk_start=adjusted_start,
                    chunk_end=adjusted_end,
                    inputs=inputs,
                    model_resources=current_model_resources,
                    config=config,
                    main_language=lang  # Use secondary_language for retries
                )
                if transcription is not None and generated_sequence is not None:
                    sequences.append(generated_sequence)
                    overall_confidences.append(overall_confidence)
                    language_tokens.append(language_token)

            # Merge overlapping sequences
            if sequences:
                merged_sequence = merge_sequences(sequences)
                transcription = processor.decode(merged_sequence, skip_special_tokens=True).strip()
                overall_confidence = np.mean(overall_confidences)
                language_token = language_tokens[0] if language_tokens else None
            else:
                transcription = ""
                overall_confidence = 0.0
                language_token = None

            # Evaluate confidence
            low_confidence = overall_confidence < confidence_threshold or not evaluate_confidence(
                overall_confidence, language_token, threshold=confidence_threshold, main_language=lang
            )

            transcript = {
                'speaker_id': segment['speaker_id'],
                'start': original_start,
                'end': original_end,
                'text': transcription,
                'confidence': overall_confidence,
                'language': language_token,
                'low_confidence': low_confidence
            }
            transcription_segments.append(transcript)

            # Since transcription is successful, do not add to failed_segments
            continue  # Proceed to the next segment

        except RetryError:
            logging.error(f"Segment {idx} failed after multiple retry attempts due to timeout.")
            retry_failed_segments.append({'segment_index': idx, 'segment': segment, 'reason': "Timeout after multiple retries"})
        except Exception as e:
            logging.exception(f"Retry for segment {idx} failed: {e}")
            retry_failed_segments.append({'segment_index': idx, 'segment': segment, 'reason': str(e)})

    logging.info(f"After retry, {len(retry_failed_segments)} segments still failed.")
    return transcription_segments, retry_failed_segments

def integrate_context_prompt(context_prompt: Optional[str], processor, device, torch_dtype):
    """
    Integrates context prompt into transcription by tokenizing and preparing decoder input ids.

    Args:
        context_prompt: The context prompt string.
        processor: The processor for the transcription model.
        device: The device to run the model on.
        torch_dtype: The data type for torch tensors.

    Returns:
        torch.Tensor or None: The decoder input ids if context prompt is provided, else None.
    """
    if context_prompt:
        logging.info("Integrating context prompt into transcription.")
        # Tokenize the context prompt without adding special tokens
        prompt_encoded = processor.tokenizer(context_prompt, return_tensors="pt", add_special_tokens=False)
        # Move the input ids to the specified device and dtype
        decoder_input_ids = prompt_encoded['input_ids'].to(device).to(torch_dtype).long()
        return decoder_input_ids
    return None

# Make sure to export the necessary functions and classes
__all__ = [
    'load_and_optimize_model',
    'model_generate_with_timeout',
    'transcribe_single_segment',
    'retry_transcriptions',
    'TimeoutException',
    'extract_language_token',
    'is_valid_language_code',
    'ModelResources',
    'TranscriptionConfig',
    'transcribe_segments',
    'transcribe_with_retry',
    'evaluate_confidence',
    'prepare_input_ids',
    'integrate_context_prompt',
    'chunk_iter',
    'merge_sequences',
]
