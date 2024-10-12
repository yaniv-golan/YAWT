import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(log_directory, max_log_size, backup_count, debug=False, verbose=False):
    """
    Sets up logging with a rotating file handler and optional console output.

    Args:
        log_directory (str): Directory where log files will be stored.
        max_log_size (int): Maximum size of a log file in bytes before it's rotated.
        backup_count (int): Number of backup log files to keep.
        debug (bool): If True, set logging level to DEBUG.
        verbose (bool): If True, set logging level to INFO.
    """
    os.makedirs(log_directory, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_directory, f"transcription_log_{current_time}.log")

    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    handlers = []
    handlers.append(RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count))
    
    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    handlers.append(console_handler)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logging.captureWarnings(True)
    logging.info("Logging setup complete.")
