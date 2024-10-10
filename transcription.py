import logging
import concurrent.futures
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from tqdm import tqdm
from config import GENERATE_TIMEOUT, MAX_TARGET_POSITIONS, BUFFER_TOKENS, DEFAULT_MODEL_ID

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

def load_and_optimize_model(model_id=DEFAULT_MODEL_ID):
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
            logging.error(f"Model optimization failed: {e}")

        # Remove forced_decoder_ids if present
        if hasattr(model.config, 'forced_decoder_ids'):
            model.config.forced_decoder_ids = None
            logging.info("Removed forced_decoder_ids from model config.")

        return model, processor, device, torch_dtype
    except Exception as e:
        logging.error(f"Failed to load and optimize model '{model_id}': {e}")
        sys.exit(1)

def model_generate_with_timeout(model, inputs, generate_kwargs, timeout):
    """
    Generates transcription with a timeout using ThreadPoolExecutor.
    """
    def generate():
        with torch.no_grad():
            return model.generate(**inputs, **generate_kwargs)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generate)
        try:
            generated_ids = future.result(timeout=timeout)
            return generated_ids
        except concurrent.futures.TimeoutError:
            raise TimeoutException("Generation timed out")

def transcribe_single_segment(model, processor, inputs, generate_kwargs, idx, chunk_start, chunk_end, device, torch_dtype):
    """
    Transcribes a single audio segment using the provided model and inputs.
    """
    try:
        generated_ids = model_generate_with_timeout(model, inputs, generate_kwargs, GENERATE_TIMEOUT)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except TimeoutException:
        logging.error(f"Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")
        return None
    except RuntimeError as e:
        if "MPS" in str(e):
            logging.warning("MPS error encountered. Falling back to CPU.")
            model_cpu = model.to('cpu')
            inputs_cpu = {k: v.to('cpu') for k, v in inputs.items()}
            try:
                generated_ids = model_generate_with_timeout(model_cpu, inputs_cpu, generate_kwargs, GENERATE_TIMEOUT)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return transcription.strip()
            except TimeoutException:
                logging.error(f"CPU Transcription timed out for segment {idx}-{chunk_start}-{chunk_end}s")
                return None
            except RuntimeError as cpu_e:
                logging.error(f"Runtime error on CPU during transcription: {cpu_e}")
                return None
            finally:
                model.to(device)  # Move the model back to the original device
        else:
            logging.error(f"Runtime error: {e}")
            return None

def retry_transcriptions(model, processor, audio_array, diarization_segments, failed_segments, generate_kwargs, device, torch_dtype, base_name, transcription_segments, MAX_RETRIES=3):
    """
    Retries transcription for failed segments up to MAX_RETRIES times.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        if not failed_segments:
            break  # No more segments to retry
        logging.info(f"Starting retry attempt {attempt} for failed segments.")
        logging.info(f"Number of segments to retry: {len(failed_segments)}")
        retry_failed_segments = []
        for failure in failed_segments:
            idx = failure['segment']
            seg = diarization_segments[idx - 1]
            start, end = seg['start'], seg['end']
            logging.info(f"Retrying transcription for segment {idx}: {start}-{end}s")
            chunk_start = int(start)
            chunk_end = min(chunk_start + GENERATE_TIMEOUT // 10, end)  # Adjusted for chunk duration
            chunk = audio_array[int(chunk_start*16000):int(chunk_end*16000)]
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device).to(torch_dtype) for k, v in inputs.items()}
            inputs['attention_mask'] = torch.ones_like(inputs['input_features'])

            if 'decoder_input_ids' in generate_kwargs and generate_kwargs['decoder_input_ids'] is not None:
                inputs['decoder_input_ids'] = generate_kwargs['decoder_input_ids']

            transcription = transcribe_single_segment(model, processor, inputs, generate_kwargs, idx, chunk_start, chunk_end, device, torch_dtype)
            if transcription:
                # Update the transcription segments with the new transcription
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