import logging
import concurrent.futures
import torch
from tqdm import tqdm
from config import GENERATE_TIMEOUT, MAX_TARGET_POSITIONS, BUFFER_TOKENS

class TimeoutException(Exception):
    pass

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