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
    MODEL_USE_CACHE
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

def load_and_optimize_model(model_id: str) -> Tuple[AutoModelForSpeechSeq2Seq, AutoProcessor, torch.device, torch.dtype]:
    """
    Loads and optimizes the speech-to-text model.

    Args:
        model_id (str): The identifier for the model to load.

    Returns:
        tuple: Contains the model, processor, device, and torch data type.
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

        return model, processor, device, torch_dtype

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

@dataclass
class TranscriptionConfig:
    transcription_timeout: int
    max_target_positions: int
    buffer_tokens: int
    confidence_threshold: float
    context_prompt: Optional[str] = None  # Store context prompt in config
    overlap_duration: float = 1.0  # Overlap duration in seconds

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

        # Updated max_length retrieval
        if hasattr(model.config, 'max_length'):
            max_length = model.config.max_length
        elif hasattr(model.config, 'decoder') and hasattr(model.config.decoder, 'max_length'):
            max_length = model.config.decoder.max_length
        else:
            max_length = max_target_positions

        prompt_length = adjusted_generate_kwargs.get('decoder_input_ids', torch.tensor([], device=device, dtype=torch_dtype)).shape[-1]
        
        # Calculate the remaining capacity for new tokens
        remaining_length = max_length - input_length - prompt_length - buffer_tokens - 1
        # Set max_new_tokens to the minimum of DEFAULT_MAX_NEW_TOKENS and the remaining capacity
        max_new_tokens = min(DEFAULT_MAX_NEW_TOKENS, remaining_length)
        # Ensure max_new_tokens is at least 1
        max_new_tokens = max(max_new_tokens, 1)

        logging.debug(f"Segment {idx}: input_length: {input_length}, prompt_length: {prompt_length}, remaining_length: {remaining_length}, max_new_tokens: {max_new_tokens}")

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def transcribe_with_retry(
    idx: int,
    chunk_start: float,
    chunk_end: float,
    inputs: Dict[str, torch.Tensor],
    model_resources: ModelResources,
    config: TranscriptionConfig,
    main_language: str  # Now mandatory
) -> Tuple[Optional[str], float, Optional[str], Optional[torch.Tensor]]:
    try:
        return transcribe_single_segment(
            idx=idx,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            inputs=inputs,
            model_resources=model_resources,
            config=config,
            main_language=main_language
        )
    except TimeoutException:
        logging.warning(f"Timeout occurred for segment {idx}. Retrying...")
        raise

def chunk_iter(inputs, feature_extractor, chunk_len, stride_left, stride_right, sampling_rate):
    inputs_len = inputs.shape[0]
    step = chunk_len - stride_left - stride_right
    for chunk_start_idx in range(0, inputs_len, step):
        chunk_end_idx = chunk_start_idx + chunk_len
        chunk = inputs[chunk_start_idx:chunk_end_idx]
        if len(chunk) == 0:
            break
        processed = feature_extractor(chunk, sampling_rate=sampling_rate, return_tensors="pt")
        _stride_left = 0 if chunk_start_idx == 0 else stride_left
        is_last = chunk_end_idx >= inputs_len
        _stride_right = 0 if is_last else stride_right

        chunk_len_actual = chunk.shape[0]
        stride = (chunk_len_actual, _stride_left, _stride_right)
        if chunk_len_actual > _stride_left:
            yield {"is_last": is_last, "stride": stride, **processed}
        if is_last:
            break

def _find_longest_common_sequence(sequences, tokenizer):
    sequence = [tok_id for tok_id in sequences[0].tolist() if tok_id not in tokenizer.all_special_ids]
    for new_seq in sequences[1:]:
        new_sequence = [tok_id for tok_id in new_seq.tolist() if tok_id not in tokenizer.all_special_ids]

        index = 0
        max_match = 0.0
        for i in range(1, len(new_sequence) + 1):
            matches = np.sum(np.array(sequence[-i:]) == np.array(new_sequence[:i]))
            matching = matches / i
            if matches > 1 and matching > max_match:
                index = i
                max_match = matching
        sequence.extend(new_sequence[index:])
    return np.array(sequence)

def transcribe_segments(
    diarization_segments: List[Dict[str, Any]],
    audio_array: np.ndarray,
    model_resources: ModelResources,
    config: TranscriptionConfig,
    main_language: str  # Now mandatory
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model = model_resources.model
    processor = model_resources.processor
    device = model_resources.device
    torch_dtype = model_resources.torch_dtype
    confidence_threshold = config.confidence_threshold
    overlap_duration = config.overlap_duration
    sampling_rate = SAMPLING_RATE

    transcription_segments = []
    failed_segments = []

    for idx, segment in enumerate(tqdm(diarization_segments, desc="Transcribing Segments", unit="segment")):
        try:
            original_start = segment['start']
            original_end = segment['end']

            # Adjust start and end times with overlap
            adjusted_start = max(original_start - overlap_duration, 0.0)
            adjusted_end = min(original_end + overlap_duration, audio_array.shape[0] / sampling_rate)

            start_sample = int(adjusted_start * sampling_rate)
            end_sample = int(adjusted_end * sampling_rate)
            chunk = audio_array[start_sample:end_sample]

            # Determine chunk parameters
            chunk_length_s = adjusted_end - adjusted_start
            stride_length_s = overlap_duration

            # Prepare chunks with overlapping
            chunk_len_samples = int((chunk_length_s + stride_length_s * 2) * sampling_rate)
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
                merged_sequence = _find_longest_common_sequence(sequences, processor.tokenizer)
                transcription = processor.decode(merged_sequence, skip_special_tokens=True).strip()
                overall_confidence = np.mean(overall_confidences)
                language_token = language_tokens[0] if language_tokens else None
            else:
                transcription = ""
                overall_confidence = 0.0
                language_token = None

            transcript = {
                'speaker_id': segment['speaker_id'],
                'start': original_start,
                'end': original_end,
                'text': transcription,
                'confidence': overall_confidence,
                'language': language_token,
                'low_confidence': overall_confidence < confidence_threshold or not evaluate_confidence(
                    overall_confidence, language_token, threshold=confidence_threshold, main_language=main_language
                )
            }
            transcription_segments.append(transcript)

        except RetryError:
            logging.error(f"Segment {idx} failed after multiple retry attempts due to timeout.")
            failed_segments.append({'segment_index': idx, 'segment': segment, 'reason': "Timeout after multiple retries"})
            continue
        except Exception as e:
            logging.exception(f"Failed to transcribe segment {idx} ({segment}): {e}")
            failed_segments.append({'segment_index': idx, 'segment': segment, 'reason': str(e)})
            continue

    return transcription_segments, failed_segments

def retry_transcriptions(
    audio_array: np.ndarray,
    diarization_segments: List[Dict[str, Any]],
    failed_segments: List[Dict[str, Any]],
    transcription_segments: List[Dict[str, Any]],
    model_resources: ModelResources,
    config: TranscriptionConfig,
    secondary_language: str  # Now optional
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
        idx = failure['segment_index']
        segment = diarization_segments[idx]
        original_start = segment['start']
        original_end = segment['end']

        try:
            # Adjust generate_kwargs for secondary language
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=lang, task="transcribe")
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

            # Create a new ModelResources instance with updated generate_kwargs
            current_model_resources = ModelResources(
                model=model,
                processor=processor,
                device=device,
                torch_dtype=torch_dtype,
                generate_kwargs=generate_kwargs
            )

            # Adjust start and end times with overlap
            adjusted_start = max(original_start - config.overlap_duration, 0.0)
            adjusted_end = min(original_end + config.overlap_duration, audio_array.shape[0] / sampling_rate)

            start_sample = int(adjusted_start * sampling_rate)
            end_sample = int(adjusted_end * sampling_rate)
            chunk = audio_array[start_sample:end_sample]

            # Determine chunk parameters
            chunk_length_s = adjusted_end - adjusted_start
            stride_length_s = config.overlap_duration

            # Prepare chunks with overlapping
            chunk_len_samples = int((chunk_length_s + stride_length_s * 2) * sampling_rate)
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
                merged_sequence = _find_longest_common_sequence(sequences, processor.tokenizer)
                transcription = processor.decode(merged_sequence, skip_special_tokens=True).strip()
                overall_confidence = np.mean(overall_confidences)
                language_token = language_tokens[0] if language_tokens else None
            else:
                transcription = ""
                overall_confidence = 0.0
                language_token = None

            if transcription and evaluate_confidence(overall_confidence, language_token, threshold=confidence_threshold, main_language=lang):
                # Update the existing transcription segment
                for t_seg in transcription_segments:
                    if (t_seg['speaker_id'] == segment['speaker_id'] and 
                        t_seg['start'] == original_start and 
                        t_seg['end'] == original_end):
                        t_seg['text'] = transcription if transcription is not None else ""
                        t_seg['confidence'] = overall_confidence
                        t_seg['language'] = language_token
                        t_seg['low_confidence'] = overall_confidence < confidence_threshold or not evaluate_confidence(
                            overall_confidence, language_token, threshold=confidence_threshold, main_language=lang
                        )
                        break
            else:
                logging.warning(f"Retry failed to transcribe segment {idx} with sufficient confidence.")
                retry_failed_segments.append(failure)
        except RetryError:
            logging.error(f"Segment {idx} failed after multiple retry attempts due to timeout.")
            retry_failed_segments.append({'segment_index': idx, 'segment': segment, 'reason': "Timeout after multiple retries"})
            continue
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
    '_find_longest_common_sequence',
]
