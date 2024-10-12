import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from transformers import logging as transformers_logging  # Import Transformers logging utilities
import warnings  # Import warnings module to filter warnings

# Module-level flag to prevent multiple setups
_setup_done = False

class ExcludeTransformersFilter(logging.Filter):
    """
    Custom logging filter to exclude logs originating from the transformers library.
    This includes logs directly from the 'transformers' logger and warnings captured via the warnings module.
    """
    def filter(self, record):
        # Exclude logs from the 'transformers' logger
        if record.name.startswith("transformers"):
            logging.getLogger("FilterDiagnostics").debug(f"Excluded (logger): {record.getMessage()}")
            return False
        # Exclude logs where 'transformers' is in the pathname
        if hasattr(record, 'pathname') and 'transformers' in record.pathname:
            logging.getLogger("FilterDiagnostics").debug(f"Excluded (pathname): {record.getMessage()}")
            return False
        # Exclude logs where 'transformers' is in the message
        if 'transformers' in record.getMessage().lower():
            logging.getLogger("FilterDiagnostics").debug(f"Excluded (message): {record.getMessage()}")
            return False
        return True

def setup_logging(log_directory, max_log_size, backup_count, debug=False, verbose=False):
    """
    Sets up logging with rotating file handlers and console output.
    
    Args:
        log_directory (str): Directory to store log files.
        max_log_size (int): Max size for log files before rotation.
        backup_count (int): Number of backup log files to keep.
        debug (bool): Enable DEBUG level logging.
        verbose (bool): Enable INFO level logging.
    """
    global _setup_done
    if _setup_done:
        return  # Prevent multiple setups

    os.makedirs(log_directory, exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_directory, f"transcription_log_{current_time}.log")

    # Determine the logging level
    if debug:
        log_level = logging.DEBUG
    elif verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all messages, filtering occurs on handlers

    # Remove all existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=max_log_size, backupCount=backup_count)
    file_handler.setLevel(logging.DEBUG)  # All logs go to file
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Create console handler for stdout
    console_stdout_handler = logging.StreamHandler(sys.stdout)
    console_stdout_handler.setLevel(log_level)
    console_stdout_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_stdout_handler.setFormatter(console_stdout_formatter)
    # Add filter to exclude transformers logs
    console_stdout_handler.addFilter(ExcludeTransformersFilter())
    root_logger.addHandler(console_stdout_handler)

    # Create console handler for stderr
    console_stderr_handler = logging.StreamHandler(sys.stderr)
    console_stderr_handler.setLevel(log_level)
    console_stderr_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_stderr_handler.setFormatter(console_stderr_formatter)
    # Add filter to exclude transformers logs
    console_stderr_handler.addFilter(ExcludeTransformersFilter())
    root_logger.addHandler(console_stderr_handler)

    # Capture warnings via the logging system
    logging.captureWarnings(True)

    # Suppress transformers warnings via the warnings module
    warnings.filterwarnings("ignore", message=".*transformers.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*transformers.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*transformers.*", category=FutureWarning)

    # Log that logging has been set up
    logging.debug("Console and file handlers added to root logger.")

    # Configure Transformers' logger
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)  # Suppress warnings, show errors and above
    transformers_logger.propagate = False        # Don't propagate to root logger

    # Add file handler to transformers logger to ensure logs are written to the file
    transformers_logger.addHandler(file_handler)

    # Set Transformers logging to ERROR level using the Transformers logging utility
    transformers_logging.set_verbosity_error()

    logging.debug("Transformers logger configured to ERROR level and propagation disabled.")

    # Create a dedicated logger for filter diagnostics
    filter_logger = logging.getLogger("FilterDiagnostics")
    filter_logger.setLevel(logging.DEBUG)
    # Add handler to filter_logger (same as file_handler)
    filter_logger.addHandler(file_handler)

    # Set the setup flag
    _setup_done = True