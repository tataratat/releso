import datetime
import pathlib

import pytest

from releso.util.logger import logging
from releso.verbosity import Verbosity, VerbosityLevel


def test_clean_dir(dir_save_location, clean_up_provider):
    clean_up_provider(dir_save_location)


@pytest.mark.parametrize(
    [
        "parser",
        "wanted_parser",
        "environment",
        "wanted_environment",
        "logfile_location",
        "console_logging",
        "base_logger_name",
        "environment_extension",
        "error",
    ],
    [
        (
            None,
            VerbosityLevel.INFO,
            None,
            VerbosityLevel.INFO,
            None,
            None,
            None,
            None,
            False,
        ),
        (
            "DEBUG",
            VerbosityLevel.DEBUG,
            "WARNING",
            VerbosityLevel.WARNING,
            None,
            None,
            None,
            None,
            False,
        ),
        (
            "ERROR",
            VerbosityLevel.ERROR,
            "INFO",
            VerbosityLevel.INFO,
            None,
            None,
            None,
            None,
            False,
        ),
        # ("Error", "Error", False, False, False, False),
    ],
)
def test_verbosity_default(
    parser,
    wanted_parser,
    environment,
    wanted_environment,
    logfile_location,
    console_logging,
    base_logger_name,
    environment_extension,
    error,
    dir_save_location,
    caplog,
    clean_up_provider,
):
    # check folder empty
    files_in_folder = list(dir_save_location.glob("*"))
    assert len(list(files_in_folder)) == 0

    calling_dict = {
        "save_location": dir_save_location,
    }
    if parser:
        calling_dict["parser"] = parser
    if environment:
        calling_dict["environment"] = environment
    if logfile_location:
        calling_dict["logfile_location"] = logfile_location
    if console_logging:
        calling_dict["console_logging"] = console_logging
    if base_logger_name:
        calling_dict["base_logger_name"] = base_logger_name
    if environment_extension:
        calling_dict["environment_extension"] = environment_extension
    if wanted_parser <= VerbosityLevel.DEBUG:
        with caplog.at_level(VerbosityLevel.DEBUG):
            verbosity = Verbosity(**calling_dict)
            assert (
                f"Parser logger has logging level: {str(wanted_parser.value)}"
                in caplog.text
            )
            assert (
                f"Environment logger has logging level: "
                f"{str(wanted_environment.value)}"
            ) in caplog.text
    else:
        verbosity = Verbosity(**calling_dict)

    assert verbosity.parser is wanted_parser
    assert verbosity.environment is wanted_environment

    logging.getLogger(verbosity._environment_validation_logger).warning("test")
    logging.getLogger(verbosity._environment_logger).warning("test")
    logging.getLogger("releso_parser").warning("test")
    # verbosity._parser_logger.debug("test")
    # objects_in_folder = list((dir_save_location / "logging").glob("*"))
    # # try:
    # #     assert len(objects_in_folder) == 4
    # # except AssertionError:
    # #     raise RuntimeError(f"{objects_in_folder}")
    clean_up_provider(dir_save_location)


@pytest.mark.parametrize(
    "add_save_path, log_path",
    [(True, "test"), (False, "test"), (False, "test_{}"), (True, "test_{}")],
)
def test_verbosity_make_logfile_location_absolute(
    add_save_path, log_path, clean_up_provider, dir_save_location
):
    v = log_path

    values = {}
    if add_save_path:
        values["save_location"] = dir_save_location
    ret_path = Verbosity.make_logfile_location_absolute(v, values)
    if add_save_path:
        assert ret_path == dir_save_location / v.format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    else:
        if "{}" in v:
            assert (
                ret_path
                == pathlib.Path(
                    v.format(
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    )
                ).resolve()
            )
        else:
            assert ret_path == pathlib.Path(v).resolve()
    clean_up_provider(ret_path)
