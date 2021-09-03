import logging

def set_up_logger(loggerName: str = '') -> logging.Logger:
    """Create a logger instance with a specified name

    Author: 
        Daniel Wolff (wolff@cats.rwth-aachen.de)

    Args:
        loggerName (str, optional): Name of the logging instance. Defaults to ''.

    Returns:
        logging.Logger: Configured logger instance for simultaneously writing to file and console
    """
    
    logFileName = '{}.log'.format(loggerName)
    errFileName = '{}.err'.format(loggerName)
    
    logger = logging.getLogger(loggerName)
    # log everything which is debug or above
    logger.setLevel(logging.DEBUG)
    # create log handler which logs even debug messages
    lh = logging.FileHandler(logFileName)
    lh.setLevel(logging.DEBUG)
    # create error handler which logs only error messages
    eh = logging.FileHandler(errFileName)
    eh.setLevel(logging.ERROR)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter for file output and add it to the handlers
    fileFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    lh.setFormatter(fileFormatter)
    eh.setFormatter(fileFormatter)
    # create formatter for console output and add it to the handlers
    consoleFormatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(consoleFormatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(lh)
    logger.addHandler(eh)
    
    return logger




parser_logger = set_up_logger("SbSOvRL_parser")
environment_logger = set_up_logger("SbSOvRL_environment")