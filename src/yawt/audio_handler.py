import os
import sys
import numpy as np
import ffmpeg
import tempfile
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
import logging
from yawt.config import SAMPLING_RATE, APP_NAME, APP_VERSION, CONTACT_INFO
import mimetypes
from dataclasses import dataclass

@dataclass
class AudioInput:
    input_url: str
    local_audio_path: str
    should_delete_local_audio_file: bool

def load_audio(input_file, sampling_rate=SAMPLING_RATE, download_timeout=None):
    """
    Loads an audio file and converts it into a NumPy array.
    
    Args:
        input_file (str): Path to the input audio file.
        sampling_rate (int, optional): Desired sampling rate for the audio. Defaults to 16000 Hz.
        download_timeout (int, optional): Timeout for downloading the audio file. Defaults to None.
    
    Returns:
        np.ndarray: Array of audio samples.
    
    Raises:
        ffmpeg._run.Error: If FFmpeg encounters an error during processing.
        Exception: For any unexpected errors.
    """
    try:
        # Determine the file type
        file_type, _ = mimetypes.guess_type(input_file)
        is_video = file_type and file_type.startswith('video')
        
        if is_video:
            logging.info(f"Input file {input_file} is a video. Extracting audio.")
        else:
            logging.info(f"Input file {input_file} is an audio file.")

        # Use FFmpeg to convert audio to a raw floating-point format
        out, _ = (
            ffmpeg.input(input_file, threads=0)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ar=sampling_rate, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        # Convert the raw bytes to a NumPy array of floats
        audio = np.frombuffer(out, np.float32)
        logging.info(f"Audio loaded from {input_file}.")
        return audio
    except ffmpeg._run.Error as e:
        # Log FFmpeg-specific errors
        logging.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except Exception as e:
        # Log any other unexpected errors
        logging.error(f"Unexpected error in load_audio: {str(e)}")
        raise

def upload_file(file_path, service='0x0.st', secret=None, expires=None, supported_upload_services=None, upload_timeout=None):
    """
    Uploads a file to the specified service.
    
    Args:
        file_path (str): Path to the file to upload.
        service (str): The upload service to use. Defaults to '0x0.st'.
        secret (str, optional): Secret for the upload service.
        expires (str, optional): Expiration time for the upload.
        supported_upload_services (set, optional): Set of supported upload services. Defaults to {'0x0.st', 'file.io'}.
        upload_timeout (int, optional): Timeout for the upload in seconds. Defaults to None.
    
    Returns:
        str: URL of the uploaded file.
    
    Raises:
        ValueError: If the specified service is unsupported.
        Exception: If the upload fails.
    """
    if supported_upload_services is None:
        supported_upload_services = {'0x0.st', 'file.io'}
        
    if service not in supported_upload_services:
        logging.error(f"Unsupported service: '{service}'")
        raise ValueError(f"Unsupported service: '{service}'")
    
    try:
        if service == '0x0.st':
            url = 'https://0x0.st'
            user_agent = f"{APP_NAME}/{APP_VERSION} ({CONTACT_INFO})"
            headers = {'User-Agent': user_agent}
            logging.info(f"Uploading '{file_path}' to '{service}'...")
            with open(file_path, 'rb') as f:
                encoder = MultipartEncoder(fields={'file': (os.path.basename(file_path), f)})
                if secret:
                    encoder.fields['secret'] = secret
                if expires:
                    encoder.fields['expires'] = expires

                with tqdm(total=encoder.len, unit='B', unit_scale=True, desc="Uploading") as pbar:
                    def progress_callback(monitor):
                        pbar.update(monitor.bytes_read - pbar.n)
                    monitor = MultipartEncoderMonitor(encoder, progress_callback)
                    response = requests.post(
                        url,
                        data=monitor,
                        headers={'Content-Type': monitor.content_type, 'User-Agent': headers['User-Agent']},
                        timeout=upload_timeout if upload_timeout else 120  # Set default upload_timeout if None
                    )

            if response.status_code == 200:
                file_url = response.text.strip()
                logging.info(f"Uploaded to '{service}': {file_url}")
                return file_url
            else:
                logging.error(f"Upload failed: {response.status_code} {response.text}")
                raise Exception(f"Upload failed: {response.status_code} {response.text}")

        elif service == 'file.io':
            url = 'https://file.io/'
            headers = {'User-Agent': f"{APP_NAME}/{APP_VERSION} ({CONTACT_INFO})"}
            logging.info(f"Uploading '{file_path}' to '{service}'...")
            with open(file_path, 'rb') as f:
                encoder = MultipartEncoder(fields={'file': (os.path.basename(file_path), f)})
                if secret:
                    encoder.fields['secret'] = secret
                if expires:
                    encoder.fields['expires'] = expires

                response = requests.post(
                    url,
                    data=encoder,
                    headers={'Content-Type': encoder.content_type},
                    timeout=upload_timeout if upload_timeout else 120  # Set default upload_timeout if None
                )

            if response.status_code == 200 and response.json().get('success'):
                file_url = response.json().get('link')
                logging.info(f"Uploaded to '{service}': {file_url}")
                return file_url
            else:
                logging.error(f"Upload failed: {response.status_code} {response.text}")
                raise Exception(f"Upload failed: {response.status_code} {response.text}")
    except requests.exceptions.RequestException as e:
        # Log network-related errors during upload
        logging.error(f"Request error during upload: {e}")
        raise
    except Exception as e:
        # Log any other unexpected errors during upload
        logging.error(f"Unexpected error during upload: {e}")
        raise

def download_audio(audio_url, destination_dir=None, download_timeout=None):
    """
    Downloads an audio file from the given URL to the specified directory.
    
    Args:
        audio_url (str): URL of the audio file to download.
        destination_dir (str, optional): Directory to save the downloaded audio. Defaults to None.
        download_timeout (int, optional): Timeout for the download in seconds. Defaults to None.
    
    Returns:
        str: Path to the downloaded audio file.
    
    Raises:
        Exception: If the download fails or the file is empty.
    """
    if destination_dir is None:
        destination_dir = tempfile.gettempdir()  # Use the system's temporary directory
    try:
        logging.info(f"Downloading audio from {audio_url}...")
        with requests.get(audio_url, stream=True, timeout=download_timeout) as r:
            if r.status_code == 200:
                file_path = os.path.join(destination_dir, 'audio.wav')
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Check if the downloaded file is empty and raise an exception if true
                if os.path.getsize(file_path) == 0:
                    logging.error(f"Downloaded file is empty: {file_path}")
                    raise Exception("Downloaded audio file is empty.")
                
                logging.info(f"Audio downloaded successfully and saved to {file_path}.")
                return file_path
            else:
                logging.error(f"Failed to download audio. Status code: {r.status_code}")
                raise Exception(f"Failed to download audio: {r.status_code}")
    except Exception as e:
        # Log any errors that occur during the download process
        logging.error(f"Error during audio download: {e}")
        raise

def extract_audio(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        raise

def handle_audio_input(args, supported_upload_services, upload_timeout):
    """
    Handles the input audio by either downloading it from a URL or uploading a local file.
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        supported_upload_services (set): Set of supported upload services.
        upload_timeout (int): Timeout for the upload in seconds.
    
    Returns:
        AudioInput: Structured input containing input URL, local path, and deletion flag.
    
    Raises:
        SystemExit: If downloading or uploading fails.
    """
    if args.audio_url:
        input_url = args.audio_url
        logging.info(f"Using provided input URL: {input_url}")
        try:
            # Determine if the provided URL is an audio or video file
            file_type, _ = mimetypes.guess_type(input_url)
            is_video = file_type and file_type.startswith('video')

            if is_video:
                logging.info("Provided URL points to a video file. Processing to extract audio.")
                # Create a temporary directory that persists after function returns
                temp_dir = tempfile.mkdtemp()
                downloaded_file_path = download_audio(input_url, destination_dir=temp_dir)
                logging.info(f"Downloaded video/audio to: {downloaded_file_path}")

                # Extract audio from the downloaded video file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                    extract_audio(downloaded_file_path, temp_audio.name)
                    local_audio_path = temp_audio.name
                    # Indicate that the local audio file is temporary and should be deleted
                    should_delete_local_audio_file = True
                logging.info(f"Extracted audio to: {local_audio_path}")

                # Upload the extracted audio to obtain a new input_url
                input_url = upload_file(
                    local_audio_path,
                    service='0x0.st',
                    supported_upload_services=supported_upload_services,
                    upload_timeout=upload_timeout
                )
                logging.info(f"Uploaded extracted audio to '0x0.st': {input_url}")
            else:
                logging.info("Provided URL points to an audio file. Using it directly.")
                # Download the audio file to a local path for processing
                temp_dir = tempfile.mkdtemp()
                local_audio_path = download_audio(input_url, destination_dir=temp_dir)
                # Indicate that the local audio file is temporary and should be deleted
                should_delete_local_audio_file = True
        except Exception as e:
            logging.error(f"Failed to process input URL: {e}")
            sys.exit(1)
    else:
        input_file = args.input_file
        if not os.path.isfile(input_file):
            logging.error(f"Input file {input_file} does not exist.")
            sys.exit(1)
        logging.info(f"Using local file: {input_file}")
        
        # Check if the input is a video file
        file_type, _ = mimetypes.guess_type(input_file)
        is_video = file_type and file_type.startswith('video')
        
        if is_video:
            logging.info("Input is a video file. Extracting audio...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                extract_audio(input_file, temp_audio.name)
                local_audio_path = temp_audio.name
                # Indicate that the local audio file is temporary and should be deleted
                should_delete_local_audio_file = True
            # Upload the extracted audio to obtain input_url
            try:
                input_url = upload_file(
                    local_audio_path,
                    service='0x0.st',
                    supported_upload_services=supported_upload_services,
                    upload_timeout=upload_timeout
                )
                logging.info(f"Uploaded extracted audio to '0x0.st': {input_url}")
            except Exception as e:
                logging.error(f"Failed to upload extracted audio: {e}")
                sys.exit(1)
        else:
            try:
                input_url = upload_file(
                    input_file, 
                    service='0x0.st', 
                    supported_upload_services=supported_upload_services,
                    upload_timeout=upload_timeout
                )
                logging.info(f"Uploaded to '0x0.st': {input_url}")
                local_audio_path = input_file
                # Indicate that the local audio file should not be deleted
                should_delete_local_audio_file = False
            except Exception as e:
                logging.error(f"Failed to upload audio: {e}")
                sys.exit(1)
    
    return AudioInput(input_url, local_audio_path, should_delete_local_audio_file)
