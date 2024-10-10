import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from config import LOG_DIRECTORY

def setup_logging(debug=False, verbose=False):
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIRECTORY, f"transcription_log_{current_time}.log")

    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    handlers = [
        RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ] if debug or verbose else [
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.captureWarnings(True)
    logging.info("Logging setup complete.")