import logging
import concurrent.futures
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm  # {{ edit: import tqdm }}
from transformers import EncoderDecoderCache  # {{ edit: import EncoderDecoderCache }}

class TimeoutException(Exception):
    pass

def get_device():
    """
    Determines the available device for computation: CUDA, MPS, or CPU.

    Returns:
        torch.device: The selected device.
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

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa"
        ).to(device)

        if device.type in ["cuda", "mps"] and model.dtype != torch.float16:
            model = model.half()
            logging.info("Converted model to float16.")

        processor = AutoProcessor.from_pretrained(model_id)
        logging.info(f"Model loaded on {device} with dtype {model.dtype}.")

        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_utils")
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.whisper.modeling_whisper")

        # Optimize model
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logging.info("Model optimized with torch.compile.")
        except Exception as e:
            logging.exception(f"Model optimization failed: {e}")  # {{ edit: Use logging.exception to capture stack trace }}

        # Remove forced_decoder_ids if present
        if hasattr(model.config, 'forced_decoder_ids'):
            model.config.forced_decoder_ids = None
            logging.info("Removed forced_decoder_ids from model config.")

        return model, processor, device, torch_dtype
    except Exception as e:
        logging.exception(f"Failed to load and optimize model '{model_id}': {e}")  # {{ edit: Use logging.exception to capture stack trace }}
        sys.exit(1)

def model_generate_with_timeout(model, inputs, generate_kwargs, transcription_timeout):
    """
    Generates output from the model with a timeout.

    Args:
        model: The transcription model.
        inputs (dict): The input tensor dictionary.
        generate_kwargs (dict): Generation keyword arguments.
        transcription_timeout (int): Timeout for transcription.

    Returns:
        torch.Tensor: Generated token IDs.
    """
    import concurrent.futures

    def generate():
        # Convert past_key_values tuple to EncoderDecoderCache if present
        if 'past_key_values' in generate_kwargs and isinstance(generate_kwargs['past_key_values'], tuple):
            generate_kwargs['past_key_values'] = EncoderDecoderCache.from_legacy_cache(generate_kwargs['past_key_values'])
        
        # Add 'input_features' from inputs to generate_kwargs
        generate_kwargs['input_features'] = inputs['input_features']  # {{ edit: Add input_features from inputs }}
        
        logging.debug(f"Final generate_kwargs before generation: {generate_kwargs}")  # {{ edit: Add DEBUG log for generate_kwargs }}
        
        # {{ edit_4: Explicitly set use_cache=True }}
        return model.generate(**generate_kwargs, use_cache=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(generate)
        try:
            return future.result(timeout=transcription_timeout)
        except concurrent.futures.TimeoutError:
            logging.error("Transcription timed out.")  # Error remains as it's a critical failure
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
    logging.debug(f"Transcribe Segment {idx}: generate_kwargs={generate_kwargs}")  # Existing DEBUG log

    try:
        # Dynamically adjust 'max_new_tokens' based on 'decoder_input_ids' length
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
        
        generated_ids = model_generate_with_timeout(model, inputs, adjusted_generate_kwargs, transcription_timeout)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except TimeoutException:
        logging.error(f"Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")  # Critical failure
        return None
    except RuntimeError as e:
        if "MPS" in str(e):
            logging.warning("MPS error encountered. Falling back to CPU.")
            model_cpu = model.to('cpu')
            inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
            try:
                generated_ids = model_generate_with_timeout(model_cpu, inputs_cpu, adjusted_generate_kwargs, transcription_timeout)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return transcription.strip()
            except TimeoutException:
                logging.error(f"CPU Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")
                return None
            except RuntimeError as cpu_e:
                logging.exception(f"Runtime error on CPU during transcription: {cpu_e}")  # {{ edit: Use logging.exception for stack trace }}
                return None
            finally:
                model.to(device)  # Move the model back to the original device
        else:
            logging.exception(f"Runtime error: {e}")  # Capture stack trace
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
    transcription_segments = []
    failed_segments = []

    # Use the passed `generate_kwargs` instead of extracting from `args`
    # Initialize tqdm progress bar
    for idx, segment in enumerate(tqdm(diarization_segments, desc="Transcribing Segments", unit="segment"), 1):
        try:
            # Extract chunk_start and chunk_end from the segment
            chunk_start = int(segment['start'])
            chunk_end = int(segment['end'])

            # Process audio chunk for the current segment
            chunk = audio_array[int(chunk_start * 16000):int(chunk_end * 16000)]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}
            inputs['attention_mask'] = torch.ones_like(inputs['input_features'])

            transcription = transcribe_single_segment(
                model=model,
                processor=processor,
                inputs=inputs,
                generate_kwargs=generate_kwargs,  # Existing pass
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
            transcription_segments.append(transcript)
        except Exception as e:
            # Include segment index in failed_segments
            failed_segments.append({'segment_index': idx, 'segment': segment, 'reason': str(e)})
            logging.exception(f"Failed to transcribe segment {idx} ({segment}): {e}")  # Capture stack trace

    return transcription_segments, failed_segments

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
            break  # No more segments to retry
        logging.info(f"Starting retry attempt {attempt} for failed segments.")
        logging.info(f"Number of segments to retry: {len(failed_segments)}")
        
        # Initialize tqdm progress bar for retries
        retry_failed_segments = []
        for failure in tqdm(failed_segments, desc=f"Retrying Segments (Attempt {attempt})", unit="segment"):
            # Retrieve segment_index instead of segment dict
            idx = failure['segment_index']
            seg = diarization_segments[idx - 1]
            start, end = seg['start'], seg['end']
            logging.info(f"Retrying transcription for segment {idx}: {start}-{end}s")

            # Extract audio chunk based on segment start and end
            chunk_start = int(start)
            chunk_end = int(end)
            chunk = audio_array[int(chunk_start * 16000):int(chunk_end * 16000)]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}
            inputs['attention_mask'] = torch.ones_like(inputs['input_features'])

            # Optionally add decoder_input_ids if available
            if 'decoder_input_ids' in generate_kwargs and generate_kwargs['decoder_input_ids'] is not None:
                inputs['decoder_input_ids'] = generate_kwargs["decoder_input_ids"]

            transcription = transcribe_single_segment(
                model=model,
                processor=processor,
                inputs=inputs,
                generate_kwargs=generate_kwargs,  # Existing pass
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
                # Update the corresponding transcription segment
                for t_seg in transcription_segments:
                    if (t_seg['speaker_id'] == seg['speaker_id'] and 
                        t_seg['start'] == start and 
                        t_seg['end'] == end):
                        t_seg['text'] = transcription
                        break
            else:
                # If transcription fails again, log it
                logging.warning(f"Retry {attempt}: Failed to transcribe segment {idx}-{chunk_start}-{chunk_end}s")
                retry_failed_segments.append(failure)
        
        failed_segments = retry_failed_segments  # Update the list of failed segments
    return failed_segments