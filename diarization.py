import requests
import logging
import time
import sys
from config import DIARIZATION_TIMEOUT, JOB_STATUS_TIMEOUT

def submit_diarization_job(api_token, audio_url, num_speakers=None):
    headers = {'Authorization': f'Bearer {api_token}', 'Content-Type': 'application/json'}
    data = {'url': audio_url}
    if num_speakers:
        data['numSpeakers'] = num_speakers
    try:
        response = requests.post('https://api.pyannote.ai/v1/diarize', headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            job_info = response.json()
            logging.info(f"Diarization job submitted: {job_info['jobId']}")
            return job_info['jobId']
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return submit_diarization_job(api_token, audio_url, num_speakers)
        else:
            logging.error(f"Diarization submission failed: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        logging.error("Diarization submission timed out.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Diarization submission error: {e}")
        sys.exit(1)

def get_job_status(api_token, job_id):
    headers = {'Authorization': f'Bearer {api_token}'}
    try:
        response = requests.get(f'https://api.pyannote.ai/v1/jobs/{job_id}', headers=headers, timeout=JOB_STATUS_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logging.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return get_job_status(api_token, job_id)
        else:
            logging.error(f"Failed to get job status: {response.status_code} {response.text}")
            sys.exit(1)
    except requests.exceptions.Timeout:
        logging.error("Getting job status timed out.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error getting job status: {e}")
        sys.exit(1)

def wait_for_diarization(api_token, job_id, audio_url, check_interval=5, max_retries=3, timeout=DIARIZATION_TIMEOUT):
    start_time = time.time()
    retries = 0
    print("Waiting for speaker recognition to complete...")

    while True:
        job_info = get_job_status(api_token, job_id)
        status = job_info['status']
        elapsed = int(time.time() - start_time)

        if status == 'succeeded':
            logging.info(f"Diarization succeeded in {elapsed} seconds.")
            print("\nSpeaker recognition complete.")
            return job_info
        elif status in ['failed', 'cancelled']:
            if status == 'cancelled':
                retries += 1
                logging.warning(f"Diarization job cancelled. Retry {retries}/{max_retries}.")
                if retries > max_retries:
                    logging.error("Max retries reached. Aborting.")
                    raise Exception("Diarization job cancelled multiple times.")
                job_id = submit_diarization_job(api_token, audio_url)
                continue
            logging.error(f"Diarization failed after {elapsed} seconds.")
            raise Exception("Diarization job failed.")
        else:
            logging.info(f"Diarization status: {status}. Elapsed time: {elapsed}s")
            print(f"\rPerforming speaker recognition... {elapsed}s elapsed.", end='', flush=True)
            time.sleep(check_interval)

        if elapsed > timeout:
            logging.error("Diarization job timed out.")
            raise Exception("Diarization job timed out.")