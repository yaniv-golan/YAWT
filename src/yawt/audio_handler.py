import os
import sys
import numpy as np
import ffmpeg
import tempfile
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
import logging

def load_audio(input_file, sampling_rate=16000, download_timeout=None):
    try:
        out, _ = (
            ffmpeg.input(input_file, threads=0)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ar=sampling_rate, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.float32)
        logging.info(f"Audio loaded from {input_file}.")
        return audio
    except ffmpeg._run.Error as e:  # Change this line
        logging.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in load_audio: {str(e)}")
        raise

def upload_file(file_path, service='0x0.st', secret=None, expires=None, supported_upload_services=None, upload_timeout=None):
    """
    Uploads a file to the specified service.
    
    Args:
        file_path (str): Path to the file to upload.
        service (str): The upload service to use.
        secret (str, optional): Secret for the upload service.
        expires (str, optional): Expiration time for the upload.
        supported_upload_services (set): Set of supported upload services.
        upload_timeout (int): Timeout for the upload in seconds.
    
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
            headers = {'User-Agent': 'TranscriberWithContext/1.0 (your_email@example.com)'}
            logging.info(f"Uploading '{file_path}' to '{service}'...")
            with open(file_path, 'rb') as f:
                encoder = MultipartEncoder(fields={'file': (os.path.basename(file_path), f)})
                if secret:
                    encoder.fields['secret'] = secret
                if expires:
                    encoder.fields['expires'] = expires

                def progress_callback(monitor):
                    pbar.update(monitor.bytes_read - pbar.n)

                with tqdm(total=encoder.len, unit='B', unit_scale=True, desc="Uploading") as pbar:
                    monitor = MultipartEncoderMonitor(encoder, progress_callback)
                    response = requests.post(
                        url,
                        data=monitor,
                        headers={'Content-Type': monitor.content_type, 'User-Agent': headers['User-Agent']},
                        timeout=upload_timeout if upload_timeout else 120  # {{ edit: Set default upload_timeout if None }}
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
            headers = {'User-Agent': 'TranscriberWithContext/1.0 (your_email@example.com)'}
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
                    timeout=upload_timeout if upload_timeout else 120  # {{ edit: Set default upload_timeout if None }}
                )

            if response.status_code == 200 and response.json().get('success'):
                file_url = response.json().get('link')
                logging.info(f"Uploaded to '{service}': {file_url}")
                return file_url
            else:
                logging.error(f"Upload failed: {response.status_code} {response.text}")
                raise Exception(f"Upload failed: {response.status_code} {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error during upload: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during upload: {e}")
        raise

def download_audio(audio_url, destination_dir='/desired/path', download_timeout=None):
    try:
        logging.info(f"Downloading audio from {audio_url}...")
        with requests.get(audio_url, stream=True, timeout=download_timeout) as r:
            if r.status_code == 200:
                file_path = os.path.join(destination_dir, 'audio.wav')
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # {{ edit: Check if the downloaded file is empty and raise an exception }}
                if os.path.getsize(file_path) == 0:
                    logging.error(f"Downloaded file is empty: {file_path}")
                    raise Exception("Downloaded audio file is empty.")
                
                logging.info(f"Audio downloaded successfully and saved to {file_path}.")
                return file_path
            else:
                logging.error(f"Failed to download audio. Status code: {r.status_code}")
                raise Exception(f"Failed to download audio: {r.status_code}")
    except Exception as e:
        logging.error(f"Error during audio download: {e}")
        raise

def handle_audio_input(args, supported_upload_services, upload_timeout):
    """
    Handles the input audio by either downloading it from a URL or uploading a local file.
    
    Args:
        args: Parsed command-line arguments.
        supported_upload_services (set): Set of supported upload services.
        upload_timeout (int): Timeout for the upload in seconds.
    
    Returns:
        tuple: (audio_url, local_audio_path)
    """
    if args.audio_url:
        audio_url = args.audio_url
        logging.info(f"Using audio URL: {audio_url}")
        try:
            local_audio_path = download_audio(audio_url, destination_dir=os.path.dirname(args.input_file))  # {{ edit: Set destination_dir appropriately }}
            logging.info(f"Downloaded audio to: {local_audio_path}")
        except Exception as e:
            logging.error(f"Failed to download audio from URL: {e}")
            sys.exit(1)
    else:
        input_file = args.input_file
        if not os.path.isfile(input_file):
            logging.error(f"Input file {input_file} does not exist.")
            sys.exit(1)
        logging.info(f"Using local file: {input_file}")
        try:
            audio_url = upload_file(
                input_file, 
                service='0x0.st', 
                supported_upload_services=supported_upload_services,
                upload_timeout=upload_timeout  # Passed upload_timeout parameter
            )
            logging.info(f"Uploaded to '0x0.st': {audio_url}")
            local_audio_path = input_file
        except Exception as e:
            logging.error(f"File upload failed: {e}")
            sys.exit(1)
    return audio_url, local_audio_path
