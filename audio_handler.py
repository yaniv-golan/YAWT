import os
import sys
import numpy as np
import ffmpeg
import tempfile
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
from tqdm import tqdm
import logging
from config import SUPPORTED_UPLOAD_SERVICES, DOWNLOAD_TIMEOUT, UPLOAD_TIMEOUT

def load_audio(input_file, sampling_rate=16000):
    try:
        out, _ = (
            ffmpeg.input(input_file, threads=0)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ar=sampling_rate, ac=1)
            .run(capture_stdout=True, capture_stderr=True)
        )
        audio = np.frombuffer(out, np.float32)
        logging.info(f"Audio loaded from {input_file}.")
        return audio
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        sys.exit(1)

def upload_file(file_path, service='0x0.st', secret=None, expires=None):
    if service not in SUPPORTED_UPLOAD_SERVICES:
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

                monitor = MultipartEncoderMonitor(encoder, progress_callback)
                with tqdm(total=encoder.len, unit='B', unit_scale=True, desc="Uploading") as pbar:
                    response = requests.post(
                        url,
                        data=monitor,
                        headers={'Content-Type': monitor.content_type, 'User-Agent': headers['User-Agent']},
                        timeout=UPLOAD_TIMEOUT
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
                    timeout=UPLOAD_TIMEOUT
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

def download_audio(audio_url):
    try:
        logging.info(f"Downloading audio from {audio_url}...")
        with requests.get(audio_url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
        logging.info(f"Downloaded to {tmp_file_path}")

        if os.path.getsize(tmp_file_path) == 0:
            logging.error("Downloaded audio file is empty.")
            os.remove(tmp_file_path)
            raise Exception("Empty audio file.")

        load_audio(tmp_file_path)
        logging.info("Audio integrity verified.")
        return tmp_file_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Download error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during audio download: {e}")
        raise