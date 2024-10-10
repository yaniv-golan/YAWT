import pytest
from unittest.mock import patch, MagicMock
from yawt.logging_setup import setup_logging
import logging

def test_setup_logging_default(mocker):
    mock_makedirs = mocker.patch('yawt.logging_setup.os.makedirs')
    mock_RotatingFileHandler = mocker.patch('yawt.logging_setup.RotatingFileHandler')
    mock_StreamHandler = mocker.patch('yawt.logging_setup.logging.StreamHandler')

    setup_logging()

    mock_makedirs.assert_called_once_with("logs", exist_ok=True)
    assert logging.getLogger().level == logging.WARNING
    mock_StreamHandler.assert_called_once_with()

def test_setup_logging_debug(mocker):
    mock_makedirs = mocker.patch('yawt.logging_setup.os.makedirs')
    mock_RotatingFileHandler = mocker.patch('yawt.logging_setup.RotatingFileHandler')
    mock_StreamHandler = mocker.patch('yawt.logging_setup.logging.StreamHandler')
    mock_basicConfig = mocker.patch('yawt.logging_setup.logging.basicConfig')

    setup_logging(debug=True)

    mock_basicConfig.assert_called_once()
    args, kwargs = mock_basicConfig.call_args
    assert kwargs['level'] == logging.DEBUG

def test_setup_logging_verbose(mocker):
    mock_makedirs = mocker.patch('yawt.logging_setup.os.makedirs')
    mock_RotatingFileHandler = mocker.patch('yawt.logging_setup.RotatingFileHandler')
    mock_StreamHandler = mocker.patch('yawt.logging_setup.logging.StreamHandler')
    mock_basicConfig = mocker.patch('yawt.logging_setup.logging.basicConfig')

    setup_logging(verbose=True)

    mock_basicConfig.assert_called_once()
    args, kwargs = mock_basicConfig.call_args
    assert kwargs['level'] == logging.INFO