import logging
import concurrent.futures
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm  # Import tqdm for progress bars
from transformers import EncoderDecoderCache  # Import EncoderDecoderCache for caching mechanisms
from yawt.config import SAMPLING_RATE  # Import the SAMPLING_RATE constant

class TimeoutException(Exception):
    """
    Custom exception to indicate a timeout during transcription.
    """
    pass

def get_device():
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

def load_and_optimize_model(model_id):
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

        # Load the pre-trained speech-to-text model with specified configurations
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        ).to(device)

        # Convert model to half precision if on CUDA or MPS for performance
        if device.type in ["cuda", "mps"] and model.dtype != torch.float16:
            model = model.half()
            logging.info("Converted model to float16.")

        # Load the corresponding processor for the model
        processor = AutoProcessor.from_pretrained(model_id)
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
            logging.exception(f"Model optimization failed: {e}")  # Capture stack trace for debugging

        # Remove forced_decoder_ids from model configuration if present to avoid unintended behavior
        if hasattr(model.config, 'forced_decoder_ids'):
            model.config.forced_decoder_ids = None
            logging.info("Removed forced_decoder_ids from model config.")

        return model, processor, device, torch_dtype
    except Exception as e:
        logging.exception(f"Failed to load and optimize model '{model_id}': {e}")  # Capture stack trace for debugging
        sys.exit(1)  # Exit the program if model loading fails

def model_generate_with_timeout(model, inputs, generate_kwargs, transcription_timeout):
    """
    Generates output from the model with a timeout.

    Args:
        model: The transcription model.
        inputs (dict): The input tensor dictionary.
        generate_kwargs (dict): Generation keyword arguments.
        transcription_timeout (int): Timeout for transcription in seconds.

    Returns:
        torch.Tensor: Generated token IDs.

    Raises:
        TimeoutException: If transcription exceeds the specified timeout.
    """
    import concurrent.futures  # Import within the function to avoid global namespace clutter

    def generate():
        # Convert past_key_values tuple to EncoderDecoderCache if present for optimized caching
        if 'past_key_values' in generate_kwargs and isinstance(generate_kwargs['past_key_values'], tuple):
            generate_kwargs['past_key_values'] = EncoderDecoderCache.from_legacy_cache(generate_kwargs['past_key_values'])
        
        # Add 'input_features' from inputs to generate_kwargs as required by the model
        generate_kwargs['input_features'] = inputs['input_features']
        
        logging.debug(f"Final generate_kwargs before generation: {generate_kwargs}")  # Debug log for current generation parameters
        
        # Explicitly set use_cache=True to enable caching during generation
        return model.generate(**generate_kwargs, use_cache=True)
    
    # Use ThreadPoolExecutor to handle the generation process with a timeout
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate)
        try:
            return future.result(timeout=transcription_timeout)
        except concurrent.futures.TimeoutError:
            logging.error("Transcription timed out.")  # Log the timeout event
            raise TimeoutException("Transcription timed out.")

def transcribe_single_segment(
    model,
    processor,
    inputs,
    generate_kwargs,
    idx,
    chunk_start,
    chunk_end,
    device,
    torch_dtype,
    transcription_timeout,
    generate_timeout,
    max_target_positions,
    buffer_tokens
):
    """
    Transcribes a single audio segment.

    Args:
        model: The transcription model.
        processor: The processor for the transcription model.
        inputs (dict): The input tensor dictionary.
        generate_kwargs (dict): Generation keyword arguments, including 'decoder_input_ids'.
        idx (int): Index of the current segment.
        chunk_start (int): Start time of the audio chunk in seconds.
        chunk_end (int): End time of the audio chunk in seconds.
        device: The device to run the model on.
        torch_dtype: The data type for torch tensors.
        transcription_timeout (int): Timeout for transcription in seconds.
        generate_timeout (int): Timeout for generation in seconds.
        max_target_positions (int): Maximum target positions for the model.
        buffer_tokens (int): Buffer tokens to reserve.

    Returns:
        str or None: The transcribed text or None if transcription fails.
    """
    logging.debug(f"Transcribe Segment {idx}: generate_kwargs={generate_kwargs}")  # Debug log for current transcription parameters

    try:
        # Dynamically adjust 'max_new_tokens' based on 'decoder_input_ids' length to prevent exceeding model limits
        if 'decoder_input_ids' in generate_kwargs and generate_kwargs['decoder_input_ids'] is not None:
            prompt_length = generate_kwargs['decoder_input_ids'].shape[1]
            # Subtract an additional token to account for any automatic tokens the model might add
            adjusted_max_new_tokens = max_target_positions - prompt_length - 1
            if adjusted_max_new_tokens <= 0:
                logging.error(
                    f"Prompt_length ({prompt_length}) exceeds or meets the max_target_positions ({max_target_positions}). Cannot generate new tokens."
                )
                return None
            adjusted_generate_kwargs = generate_kwargs.copy()
            adjusted_generate_kwargs["max_new_tokens"] = adjusted_max_new_tokens
            logging.debug(f"Adjusted max_new_tokens based on prompt length: {adjusted_generate_kwargs['max_new_tokens']}")
        else:
            adjusted_generate_kwargs = generate_kwargs.copy()
            adjusted_generate_kwargs["max_new_tokens"] = max_target_positions - buffer_tokens
            logging.debug(f"Adjusted max_new_tokens based on buffer_tokens: {adjusted_generate_kwargs['max_new_tokens']}")
        
        # Generate transcription with timeout handling
        generated_ids = model_generate_with_timeout(model, inputs, adjusted_generate_kwargs, transcription_timeout)
        # Decode the generated token IDs to obtain the transcription text
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except TimeoutException:
        logging.error(f"Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")  # Log the timeout event for the specific segment
        return None
    except RuntimeError as e:
        if "MPS" in str(e):
            logging.warning("MPS error encountered. Falling back to CPU.")  # Log the MPS error and fallback
            model_cpu = model.to('cpu')  # Move the model to CPU
            inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}  # Move inputs to CPU
            try:
                # Attempt transcription again on CPU
                generated_ids = model_generate_with_timeout(model_cpu, inputs_cpu, adjusted_generate_kwargs, transcription_timeout)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return transcription.strip()
            except TimeoutException:
                logging.error(f"CPU Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")  # Log CPU timeout
                return None
            except RuntimeError as cpu_e:
                logging.exception(f"Runtime error on CPU during transcription: {cpu_e}")  # Log any CPU-specific errors
                return None
            finally:
                model.to(device)  # Move the model back to the original device
        else:
            logging.exception(f"Runtime error: {e}")  # Log any other runtime errors
            return None

def transcribe_segments(args, diarization_segments, audio_array, model, processor, device, torch_dtype, generate_timeout, max_target_positions, buffer_tokens, transcription_timeout, generate_kwargs):
    """
    Transcribes all segments of the audio based on diarization.

    Args:
        args: Parsed command-line arguments.
        diarization_segments (list): List of diarization segments.
        audio_array (np.array): Array of audio data.
        model: The transcription model.
        processor: The processor for the transcription model.
        device: The device to run the model on.
        torch_dtype: The data type for torch tensors.
        generate_timeout (int): Timeout for generation.
        max_target_positions (int): Maximum target positions.
        buffer_tokens (int): Buffer tokens.
        transcription_timeout (int): Timeout for transcription.
        generate_kwargs (dict): Keyword arguments for model generation, including 'decoder_input_ids'.

    Returns:
        tuple: Transcription segments and any failed segments.
    """
    transcription_segments = []  # List to store successfully transcribed segments
    failed_segments = []         # List to store segments that failed transcription

    # Initialize tqdm progress bar for tracking transcription progress
    for idx, segment in enumerate(tqdm(diarization_segments, desc="Transcribing Segments", unit="segment"), 1):
        try:
            # Extract start and end times in seconds for the current segment
            chunk_start = int(segment['start'])
            chunk_end = int(segment['end'])

            # Extract the audio chunk corresponding to the current diarization segment
            chunk = audio_array[int(chunk_start * SAMPLING_RATE):int(chunk_end * SAMPLING_RATE)]  # Replace 16000 with SAMPLING_RATE
            inputs = processor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt")  # Process the audio chunk
            inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}  # Move inputs to the correct device and dtype
            inputs['attention_mask'] = torch.ones_like(inputs['input_features'])  # Add attention mask

            # Transcribe the audio chunk and obtain the transcription text
            transcription = transcribe_single_segment(
                model=model,
                processor=processor,
                inputs=inputs,
                generate_kwargs=generate_kwargs,
                idx=idx,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                device=device,
                torch_dtype=torch_dtype,
                transcription_timeout=transcription_timeout,
                generate_timeout=generate_timeout,
                max_target_positions=max_target_positions,
                buffer_tokens=buffer_tokens
            )

            # Store transcription details as a dictionary
            transcript = {
                'speaker_id': segment['speaker_id'],
                'start': segment['start'],
                'end': segment['end'],
                'text': transcription if transcription else ""
            }
            transcription_segments.append(transcript)  # Add to the list of successful transcriptions
        except Exception as e:
            # If transcription fails, add the segment to failed_segments with the reason
            failed_segments.append({'segment_index': idx, 'segment': segment, 'reason': str(e)})
            logging.exception(f"Failed to transcribe segment {idx} ({segment}): {e}")  # Log the exception with stack trace

    return transcription_segments, failed_segments  # Return the results

def retry_transcriptions(model, processor, audio_array, diarization_segments, failed_segments, generate_kwargs, device, torch_dtype, base_name, transcription_segments, generate_timeout, max_target_positions, buffer_tokens, transcription_timeout=300, max_retries=3):
    """
    Retries transcription for failed segments up to max_retries times.

    Args:
        model: The transcription model.
        processor: The processor for the transcription model.
        audio_array (np.array): Array of audio data.
        diarization_segments (list): List of diarization segments.
        failed_segments (list): List of failed segments to retry.
        generate_kwargs (dict): Keyword arguments for model generation, including 'decoder_input_ids'.
        device: The device to run the model on.
        torch_dtype: The data type for torch tensors.
        base_name (str): Base name for output files.
        transcription_segments (list): List of successfully transcribed segments.
        generate_timeout (int): Timeout for generation.
        max_target_positions (int): Maximum target positions.
        buffer_tokens (int): Buffer tokens.
        transcription_timeout (int, optional): Timeout for transcription. Defaults to 300.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.

    Returns:
        list: Remaining failed segments after retries.
    """
    for attempt in range(1, max_retries + 1):
        if not failed_segments:
            break  # Exit if there are no more segments to retry
        logging.info(f"Starting retry attempt {attempt} for failed segments.")
        logging.info(f"Number of segments to retry: {len(failed_segments)}")
        
        retry_failed_segments = []  # Initialize list for segments that fail again in this attempt
        for failure in tqdm(failed_segments, desc=f"Retrying Segments (Attempt {attempt})", unit="segment"):
            # Retrieve segment_index and segment details from the failure record
            idx = failure['segment_index']
            seg = diarization_segments[idx - 1]
            start, end = seg['start'], seg['end']
            logging.info(f"Retrying transcription for segment {idx}: {start}-{end}s")

            # Extract the audio chunk based on segment start and end times
            chunk_start = int(start)
            chunk_end = int(end)
            chunk = audio_array[int(chunk_start * SAMPLING_RATE):int(chunk_end * SAMPLING_RATE)]  # Replace 16000 with SAMPLING_RATE
            inputs = processor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt")  # Process the audio chunk
            inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}  # Move inputs to the correct device and dtype
            inputs['attention_mask'] = torch.ones_like(inputs['input_features'])  # Add attention mask

            # Optionally add decoder_input_ids if available in generate_kwargs
            if 'decoder_input_ids' in generate_kwargs and generate_kwargs['decoder_input_ids'] is not None:
                inputs['decoder_input_ids'] = generate_kwargs["decoder_input_ids"]

            # Attempt to transcribe the segment again
            transcription = transcribe_single_segment(
                model=model,
                processor=processor,
                inputs=inputs,
                generate_kwargs=generate_kwargs,
                idx=idx,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                device=device,
                torch_dtype=torch_dtype,
                transcription_timeout=transcription_timeout,
                generate_timeout=generate_timeout,
                max_target_positions=max_target_positions,
                buffer_tokens=buffer_tokens
            )

            if transcription:
                # If transcription is successful, update the corresponding transcription segment
                for t_seg in transcription_segments:
                    if (t_seg['speaker_id'] == seg['speaker_id'] and 
                        t_seg['start'] == start and 
                        t_seg['end'] == end):
                        t_seg['text'] = transcription  # Update the transcribed text
                        break
            else:
                # If transcription fails again, log the failure and add to retry list
                logging.warning(f"Retry {attempt}: Failed to transcribe segment {idx}-{chunk_start}-{chunk_end}s")
                retry_failed_segments.append(failure)
        
        failed_segments = retry_failed_segments  # Update the list of failed segments for the next attempt
    return failed_segments  # Return any segments that still failed after retries