import logging
import sys, os, platform, enum
from typing import List

class Stylings(enum.Enum):
    """Possible stylings supported by this Tool.
    """
    Bold = 1
    Faint = 2
    Italic = 3
    Underline = 4
    Black = 30
    Red = 31
    Green = 32
    Yellow = 33
    Blue = 34
    Magenta = 35
    Cyan = 36
    White = 37

    def __str__(self):
        return '%s' % self.value

def check_if_color_is_supported_in_console() -> bool:
    """ Checks if the connected consol has color support. Original from this gist https://gist.github.com/ssbarnea/1316877.

    Returns:
        bool: Terminal supports color.
    """
    for handle in [sys.stdout, sys.stderr]:
        if (hasattr(handle, "isatty") and handle.isatty()) or \
            ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
            if platform.system()=='Windows' and not ('TERM' in os.environ and os.environ['TERM']=='ANSI'):
                return False
            else:
                return True
        else:
            return False

color_supported = check_if_color_is_supported_in_console()

def output_styling(message: str, stylings: List[Stylings]):
    """
    If ANSI styling is available the message is styled by an the defined styling. Only the first 3 Stylings are actually used the others will be ignored.
    Valid stylings can be found in the enum: SbSOvRL.exceptions.Stylings
    """
    return f"\033[{';'.join(map(str, stylings[:3]))}m{message}\033[0m" if color_supported else message

def red(message):
    """
    If ANSI styling is available the message is colored red and bold. After the message the styling is removed again.
    For more information please see SbSOvRL.exceptions.output_styling().
    """
    return output_styling(message=message, stylings=[Stylings.Red, Stylings.Bold])

def underline(message):
    """
    If ANSI styling is available the message is styled by an underscore and bold. After the message the styling is removed again.
    For more information please see SbSOvRL.exceptions.output_styling().
    """
    return output_styling(message=message, stylings=[Stylings.Underline, Stylings.Bold])

class SbSOvRLParserException(Exception):
    """Parser Exception for the SbSOvRL package. Shows the context of the error. Also colors the output if colouring is available.
    """
    def __init__(self, parent: str, item: str, message: str) -> None:
        logging.getLogger("SbSOVRL_parser").exception(f"In {parent} object while parsing {item} the following error has occurred: {message}")
        super().__init__(f"In {underline(parent)} object while parsing {underline(item)} the following error has occurred: {red(message)}")

class SbSOvRLAgentUnknownException(Exception):
    """Parser Exception for the SbSOvRL package. Thrown when an agent is unknown. Configurable logger.

    Args:
        agent (str): Name of the unknown agent.
        logger (str): Name of the logger where the exception should be logged over. (Defaults to: 'SbSOvRL_environment')
    """
    def __init__(self, agent: str, logger: str = "SbSOvRL_environment") -> None:
        mes_str: str = f"The {agent} is unknown. Check the spelling or put in a request to add this agent."
        logging.getLogger(logger).exception(mes_str)
        super().__init__(mes_str)

class SbSOvRLValidationNotSet(Exception):
    """Parser Exception for the SbSOvRL package. Thrown when a validation environment is needed but can not be created due to uncofigured validation settings. Configurable logger.

    Args:
        logger (str): Name of the logger where the exception should be logged over. (Defaults to: 'SbSOvRL_environment')
    """
    def __init__(self, logger: str = "SbSOvRL_validation_environment") -> None:
        mes_str: str = f"Could not create a validation environment due to inavailability of validation parameters. Please add the validation parameters to the json file."
        logging.getLogger(logger).exception(type(self).__name__ + ": " + mes_str)
        super().__init__(mes_str)