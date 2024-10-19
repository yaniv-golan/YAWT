import requests
import logging
import time
import sys
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

def is_rate_limit_exception(exception):
    """
    Custom predicate to check for rate limit exceptions.

    Args:
        exception (Exception): The exception to check.

    Returns:
        bool: True if it's a rate limit exception, False otherwise.
    """
    return isinstance(exception, Exception) and '429' in str(exception)

@retry(
    retry=retry_if_exception(is_rate_limit_exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def submit_diarization_job(pyannote_token, audio_url, num_speakers, diarization_timeout):
    """
    Submits a diarization job to the Pyannote API.

    Args:
        pyannote_token (str): API token for authenticating with Pyannote.
        audio_url (str): URL of the audio file to be diarized.
        num_speakers (int, optional): Expected number of speakers. Defaults to None.
        diarization_timeout (int): Timeout for the diarization request in seconds.

    Returns:
        str: Job ID of the submitted diarization job.

    Raises:
        Exception: If the job submission fails after maximum retries.
    """
    headers = {'Authorization': f'Bearer {pyannote_token}', 'Content-Type': 'application/json'}
    data = {'url': audio_url}
    if num_speakers:
        data['numSpeakers'] = num_speakers
    try:
        # Submit the diarization job to the Pyannote API
        response = requests.post('https://api.pyannote.ai/v1/diarize', headers=headers, json=data, timeout=diarization_timeout)
        if response.status_code == 200:
            job_info = response.json()
            logging.info(f"Diarization job submitted: {job_info['jobId']}")
            return job_info['jobId']
        elif response.status_code == 429:
            # Raise exception to trigger retry
            error_msg = f"Rate limit exceeded: {response.status_code} {response.text}"
            logging.warning(error_msg)
            raise Exception(error_msg)
        else:
            # Log and raise exception for other HTTP errors
            error_msg = f"Diarization submission failed: {response.status_code} {response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
    except requests.exceptions.Timeout:
        # Handle request timeout
        error_msg = "Diarization submission timed out."
        logging.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        # Handle any other exceptions during submission
        error_msg = f"Diarization submission error: {e}"
        logging.error(error_msg)
        raise

@retry(
    retry=retry_if_exception(is_rate_limit_exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_job_status(pyannote_token, job_id, job_status_timeout):
    """
    Retrieves the status of a submitted diarization job.

    Args:
        pyannote_token (str): API token for authenticating with Pyannote.
        job_id (str): ID of the diarization job.
        job_status_timeout (int): Timeout for the job status request in seconds.

    Returns:
        dict: JSON response containing job status and details.

    Raises:
        Exception: If retrieving job status fails after maximum retries.
    """
    headers = {'Authorization': f'Bearer {pyannote_token}'}
    try:
        # Make a GET request to check the status of the diarization job
        response = requests.get(f'https://api.pyannote.ai/v1/jobs/{job_id}', headers=headers, timeout=job_status_timeout)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            # Raise exception to trigger retry
            error_msg = f"Rate limit exceeded: {response.status_code} {response.text}"
            logging.warning(error_msg)
            raise Exception(error_msg)
        else:
            # Log and raise exception for other HTTP errors
            error_msg = f"Failed to get job status: {response.status_code} {response.text}"
            logging.error(error_msg)
            raise Exception(error_msg)
    except requests.exceptions.Timeout:
        # Handle request timeout
        error_msg = "Getting job status timed out."
        logging.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        # Handle any other exceptions during status retrieval
        error_msg = f"Error getting job status: {e}"
        logging.error(error_msg)
        raise

def wait_for_diarization(pyannote_token, job_id, audio_url, diarization_timeout, job_status_timeout, num_speakers=None):
    """
    Waits for the diarization job to complete by periodically checking its status.

    Args:
        pyannote_token (str): API token for authenticating with Pyannote.
        job_id (str): ID of the diarization job.
        audio_url (str): URL of the audio file being diarized.
        diarization_timeout (int): Total time to wait for diarization in seconds.
        job_status_timeout (int): Timeout for each job status request in seconds.
        num_speakers (int, optional): Expected number of speakers. Defaults to None.

    Returns:
        dict: JSON response containing the final job status and results.

    Raises:
        Exception: If the job fails, is canceled, or times out.
    """
    start_time = time.time()
    retries = 0
    max_retries = 3  # Maximum number of retries for canceled jobs
    check_interval = 30  # Seconds between status checks

    logging.info("Waiting for speaker recognition to complete...")

    while True:
        try:
            # Retrieve the current status of the diarization job
            job_info = get_job_status(pyannote_token, job_id, job_status_timeout)
            status = job_info.get('status')
            elapsed = int(time.time() - start_time)

            if status == 'succeeded':
                # Job completed successfully
                logging.info(f"Diarization succeeded in {elapsed} seconds.")
                logging.info("Speaker recognition complete.")
                return job_info
            elif status in ['failed', 'cancelled']:
                if status == 'cancelled':
                    # Handle canceled jobs by retrying submission
                    retries += 1
                    logging.warning(f"Diarization job cancelled. Retry {retries}/{max_retries}.")
                    if retries > max_retries:
                        logging.error("Max retries reached. Aborting.")
                        raise Exception("Diarization job cancelled multiple times.")
                    # Resubmit the diarization job
                    job_id = submit_diarization_job(pyannote_token, audio_url, num_speakers, diarization_timeout)
                    continue
                # Job failed due to an error
                logging.error(f"Diarization failed after {elapsed} seconds.")
                raise Exception("Diarization job failed.")
            else:
                # Job is still in progress; log and wait before next check
                logging.info(f"Diarization status: {status}. Elapsed time: {elapsed}s")
                print(f"\rPerforming speaker recognition... {elapsed}s elapsed.", end='', flush=True)
                time.sleep(check_interval)

            if elapsed > diarization_timeout:
                # Overall timeout reached without job completion
                logging.error("Diarization job timed out.")
                raise Exception("Diarization job timed out.")
        except Exception as e:
            # Log any exceptions encountered during the waiting process
            logging.exception(f"Diarization process encountered an error: {e}")
            raise

def perform_diarization(pyannote_token, audio_url, num_speakers, diarization_timeout, job_status_timeout):
    """
    Orchestrates the submission and monitoring of a diarization job.

    Args:
        pyannote_token (str): API token for authenticating with Pyannote.
        audio_url (str): URL of the audio file to be diarized.
        num_speakers (int, optional): Expected number of speakers. Defaults to None.
        diarization_timeout (int): Timeout for the diarization process in seconds.
        job_status_timeout (int): Timeout for each job status request in seconds.

    Returns:
        list: List of formatted diarization segments with speaker labels and timestamps.
    """
    # Submit the diarization job and obtain the job ID
    job_id = submit_diarization_job(pyannote_token, audio_url, num_speakers, diarization_timeout)
    # Wait for the diarization job to complete and retrieve job information
    job_info = wait_for_diarization(pyannote_token, job_id, audio_url, diarization_timeout, job_status_timeout, num_speakers)

    logging.debug(f"Job Info: {job_info}")  # Debugging statement

    # Extract diarization segments from the job output
    diarization_segments = job_info.get('output', {}).get('diarization', [])

    logging.debug(f"Diarization Segments: {diarization_segments}")  # Debugging statement

    formatted_segments = []
    for seg in diarization_segments:
        # Format each segment with speaker label and start/end times
        formatted_segment = {
            'speaker': seg.get('speaker', 'Unknown'),
            'start': seg.get('start', 0),
            'end': seg.get('end', 0)
        }
        logging.debug(f"Formatted Segment: {formatted_segment}")  # Debugging statement
        formatted_segments.append(formatted_segment)

    logging.debug(f"Formatted Segments: {formatted_segments}")  # Additional logging
    return formatted_segments
