import requests
import logging
import time
import sys

def submit_diarization_job(pyannote_token, audio_url, num_speakers, diarization_timeout, retries=0, max_retries=3):
    headers = {'Authorization': f'Bearer {pyannote_token}', 'Content-Type': 'application/json'}
    data = {'url': audio_url}
    if num_speakers:
        data['numSpeakers'] = num_speakers
    try:
        response = requests.post('https://api.pyannote.ai/v1/diarize', headers=headers, json=data, timeout=diarization_timeout)
        if response.status_code == 200:
            job_info = response.json()
            logging.info(f"Diarization job submitted: {job_info['jobId']}")
            return job_info['jobId']
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            if retries >= max_retries:
                logging.error("Max retries reached. Unable to submit diarization job.")
                raise Exception("Max retries reached for diarization submission.")
            time.sleep(retry_after)
            return submit_diarization_job(pyannote_token, audio_url, num_speakers, diarization_timeout, retries+1, max_retries)
        else:
            error_msg = f"Diarization submission failed: {response.status_code} {response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
    except requests.exceptions.Timeout:
        error_msg = "Diarization submission timed out."
        logging.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Diarization submission error: {e}"
        logging.error(error_msg)
        raise

def get_job_status(pyannote_token, job_id, job_status_timeout):
    headers = {'Authorization': f'Bearer {pyannote_token}'}
    try:
        response = requests.get(f'https://api.pyannote.ai/v1/jobs/{job_id}', headers=headers, timeout=job_status_timeout)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return get_job_status(pyannote_token, job_id, job_status_timeout)
        else:
            error_msg = f"Failed to get job status: {response.status_code} {response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
    except requests.exceptions.Timeout:
        error_msg = "Getting job status timed out."
        logging.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Error getting job status: {e}"
        logging.error(error_msg)
        raise

def wait_for_diarization(pyannote_token, job_id, audio_url, diarization_timeout, job_status_timeout, num_speakers=None):
    start_time = time.time()
    retries = 0
    max_retries = 3  # Define as needed
    check_interval = 30  # Seconds between status checks

    logging.info("Waiting for speaker recognition to complete...")

    while True:
        try:
            job_info = get_job_status(pyannote_token, job_id, job_status_timeout)
            status = job_info.get('status')
            elapsed = int(time.time() - start_time)

            if status == 'succeeded':
                logging.info(f"Diarization succeeded in {elapsed} seconds.")
                logging.info("Speaker recognition complete.")
                return job_info
            elif status in ['failed', 'cancelled']:
                if status == 'cancelled':
                    retries += 1
                    logging.warning(f"Diarization job cancelled. Retry {retries}/{max_retries}.")
                    if retries > max_retries:
                        logging.error("Max retries reached. Aborting.")
                        raise Exception("Diarization job cancelled multiple times.")
                    job_id = submit_diarization_job(pyannote_token, audio_url, num_speakers, diarization_timeout)
                    continue
                logging.error(f"Diarization failed after {elapsed} seconds.")
                raise Exception("Diarization job failed.")
            else:
                logging.info(f"Diarization status: {status}. Elapsed time: {elapsed}s")
                print(f"\rPerforming speaker recognition... {elapsed}s elapsed.", end='', flush=True)
                time.sleep(check_interval)

            if elapsed > diarization_timeout:
                logging.error("Diarization job timed out.")
                raise Exception("Diarization job timed out.")
        except Exception as e:
            logging.exception(f"Diarization process encountered an error: {e}")
            raise

def perform_diarization(pyannote_token, audio_url, num_speakers, diarization_timeout, job_status_timeout):
    job_id = submit_diarization_job(pyannote_token, audio_url, num_speakers, diarization_timeout)
    job_info = wait_for_diarization(pyannote_token, job_id, audio_url, diarization_timeout, job_status_timeout, num_speakers)

    logging.debug(f"Job Info: {job_info}")  # Debugging statement

    # Extract diarization segments
    diarization_segments = job_info.get('output', {}).get('diarization', [])

    logging.debug(f"Diarization Segments: {diarization_segments}")  # Debugging statement

    formatted_segments = []
    for seg in diarization_segments:
        formatted_segment = {
            'speaker': seg.get('speaker', 'Unknown'),
            'start': seg.get('start', 0),
            'end': seg.get('end', 0)
        }
        logging.debug(f"Formatted Segment: {formatted_segment}")  # Debugging statement
        formatted_segments.append(formatted_segment)

    logging.debug(f"Formatted Segments: {formatted_segments}")  # Additional logging
    return formatted_segments