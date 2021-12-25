import logging, pathlib
import enum, os

class VerbosityLevel(enum.IntEnum):
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG

def set_up_logger(loggerName: str = '', log_file_location: pathlib.Path = pathlib.Path("."), verbosity: VerbosityLevel = VerbosityLevel.INFO, console_logging: bool = False) -> logging.Logger:
    """Create a logger instance with a specified name

    Author: 
        Daniel Wolff (wolff@cats.rwth-aachen.de),
        Clemens Fricke (clemens.fricke@rwth-aachen.de)

    Args:
        loggerName (str, optional): Name of the logging instance. Defaults to ''.
        log_file_location (Path): Path to the directory into which the log file(s) will be placed into. Defaults to ".".
        verbosity (VerbosityLevel): Enum value for the verbosity of the given logger. Defaults to VerbosityLevel.INFO 
        console_logging (bool): Toggle whether to also log into the console. Defaults to False.

    Returns:
        logging.Logger: Configured logger instance for simultaneously writing to file and console
    """
    
    logger = logging.getLogger(loggerName)
    # log everything which is debug or above
    logger.setLevel(verbosity.value)
    # create formatter for file output and add it to the handlers
    fileFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_file_location.mkdir(parents=True, exist_ok=True)

    if verbosity <= VerbosityLevel.WARNING:
        logFileName = log_file_location / '{}.log'.format(loggerName)
        # create log handler which logs even debug messages
        lh = logging.FileHandler(logFileName)
        # create console handler with a higher log level
        lh.setLevel(verbosity.value)
        lh.setFormatter(fileFormatter)
        logger.addHandler(lh)
    
    if console_logging:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter for console output and add it to the handlers
        consoleFormatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(consoleFormatter)
        logger.addHandler(ch)

    # create error handler which logs only error messages
    errFileName = log_file_location / '{}.err'.format(loggerName)
    eh = logging.FileHandler(errFileName)
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fileFormatter)
    # add the handlers to logger
    logger.addHandler(eh)
    
    return logger