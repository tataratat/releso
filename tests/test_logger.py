import logging

import pytest

from releso.util.logger import VerbosityLevel, get_parser_logger, set_up_logger


def test_get_parser_logger():
    parser_log = get_parser_logger()
    assert parser_log.name == "ReLeSO_parser"

    # check factory method
    parser_log2 = get_parser_logger()
    assert parser_log is parser_log2


def test_verbosity_level_enum():
    assert VerbosityLevel.ERROR == logging.ERROR
    assert VerbosityLevel.WARNING == logging.WARNING
    assert VerbosityLevel.INFO == logging.INFO
    assert VerbosityLevel.DEBUG == logging.DEBUG


@pytest.mark.parametrize(
    "logger_name, log_file_location, verbosity, console_logging, logger",
    [
        (None, None, None, None, None),
        (None, None, None, None, logging.getLogger("test")),
    ],
)
def test_set_up_logger(
    logger_name,
    log_file_location,
    verbosity,
    console_logging,
    logger,
    dir_save_location,
    clean_up_provider,
):
    if logger is not None:
        original_name = logger.name
    calling_dict = {
        "logger": logger,
    }
    if logger_name is not None:
        calling_dict["logger_name"] = logger_name
    if log_file_location is not None:
        calling_dict["log_file_location"] = log_file_location
    if verbosity is not None:
        calling_dict["verbosity"] = verbosity
    if console_logging is not None:
        calling_dict["console_logging"] = console_logging

    ret_logger = set_up_logger(**calling_dict)
    if logger is not None:
        assert ret_logger is logger
        assert ret_logger.name != original_name
    else:
        if logger_name is not None:
            assert ret_logger.name == logger_name

    n_handlers = 1
    if console_logging:
        n_handlers += 1
    if verbosity:
        if verbosity <= VerbosityLevel.WARNING:
            assert (dir_save_location / f"{logger_name}.log").exists()
            n_handlers += 1
    else:
        n_handlers += 1
    if console_logging:
        n_handlers += 1
    assert len(ret_logger.handlers) == n_handlers
    if log_file_location is not None:
        assert ret_logger.handlers[0].baseFilename == log_file_location
    if verbosity is not None:
        assert ret_logger.level == verbosity
    else:
        assert ret_logger.level == VerbosityLevel.INFO

    clean_up_provider(dir_save_location)
