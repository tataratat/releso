"""File defines the exceptions of the ReLeSO toolbox/library."""

import enum
import logging
import os
import platform
import sys
from typing import List

from releso.util.logger import get_parser_logger


class Stylings(enum.Enum):
    """Possible stylings supported by this Tool."""

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

    def __str__(self) -> str:
        """Encode enum value into a string.

        Returns:
            str: encoded enum value.
        """
        return "%s" % self.value


def check_if_color_is_supported_in_console() -> bool:
    """Checks if the connected consol has color support.

    Original from this gist https://gist.github.com/ssbarnea/1316877.

    Returns:
        bool: Terminal supports color.
    """
    for handle in [sys.stdout, sys.stderr]:
        if (hasattr(handle, "isatty") and handle.isatty()) or (
            "TERM" in os.environ and os.environ["TERM"] == "ANSI"
        ):
            if platform.system() == "Windows" and not (
                "TERM" in os.environ and os.environ["TERM"] == "ANSI"
            ):
                return False
            else:
                return True
        else:
            return False
    return False


#: boolean value on whether or not coloring is supported
color_supported = check_if_color_is_supported_in_console()


def output_styling(message: str, stylings: List[Stylings]) -> str:
    """Style the string according to the given styles.

    If ANSI styling is available the message is styled by an the defined
    styling. Only the first 3 Stylings are actually used the others will be
    ignored.

    Valid stylings can be found in the enum: ReLeSO.exceptions.Stylings

    Args:
        message (str): string to style
        stylings (List[Stylings]): Stylings to apply to the message.

    Returns:
        (str): ANSI encoded styled message.
    """
    return (
        f"\033[{';'.join(map(str, stylings[:3]))}m{message}\033[0m"
        if color_supported
        else message
    )


def red(message: str) -> str:
    """Color string red.

    If ANSI styling is available the message is colored red and bold. After the
    message the styling is removed again.

    For more information please see ReLeSO.exceptions.output_styling().

    Args:
        message (str): string to color

    Returns:
        (str): ANSI encoded colored message.
    """
    return output_styling(
        message=message, stylings=[Stylings.Red, Stylings.Bold]
    )


def underline(message: str) -> str:
    """Underline the string.

    If ANSI styling is available the message is styled by an underscore and
    bold. After the message the styling is removed again.

    For more information please see ReLeSO.exceptions.output_styling().

    Args:
        message (str): string to strike through

    Returns:
        (str): ANSI encoded struck through message.
    """
    return output_styling(
        message=message, stylings=[Stylings.Underline, Stylings.Bold]
    )


class ParserException(ValueError):
    """Parser Exception for the ReLeSO package.

    Shows the context of the error. Also colors the output if coloring is
    available.

    Uses the ValueError base class for compatibility with the pydantic
    validation engine.

    Args:
        parent (str): Parent items names where the error occurred.
        item (str): Name of the item which caused the error.
        message (str): Custom message to display in the Exception.
    """

    def __init__(self, parent: str, item: str, message: str) -> None:
        """Parser Exception constructor.

        Args:
            parent (str): Parent items names where the error occurred.
            item (str): Name of the item which caused the error.
            message (str): Custom message to display in the Exception.
        """
        get_parser_logger().exception(
            f"In {parent} object while parsing {item} "
            f"the following error has occurred: {message}"
        )
        super().__init__(
            f"In {underline(parent)} object while parsing {underline(item)} "
            f"the following error has occurred: {red(message)}"
        )


class AgentUnknownException(Exception):
    """Agent unknown exception for the ReLeSO package.

    Thrown when an agent is unknown. Configurable logger.

    Args:
        agent (str): Name of the agent which is not set.
        logger (str, optional): Name of the logger.
        Defaults to "ReLeSO_environment".
    """

    def __init__(self, agent: str, logger: str = "ReLeSO_environment") -> None:
        """Constructor AgentUnknownException.

        Args:
            agent (str): Name of the agent which is not set.
            logger (str, optional): Name of the logger.
            Defaults to "ReLeSO_environment".
        """
        mes_str: str = f"The {agent} is unknown. Check the spelling or"
        " put in a request to add this agent."
        logging.getLogger(logger).exception(mes_str)
        super().__init__(mes_str)


class ValidationNotSet(Exception):
    """Validation not set exception.

    Thrown when a validation
    environment is needed but can not be created due to un-configured
    validation settings. Configurable logger.

    Args:
        logger (str):
            Name of the logger where the exception should be logged over.
            Defaults to: 'ReLeSO_environment'
    """

    def __init__(self, logger: str = "ReLeSO_validation_environment") -> None:
        """Constructor ValidationNotSetException.

        Args:
            logger (str, optional): Name of the logger.
            Defaults to "ReLeSO_validation_environment".
        """
        mes_str: str = (
            "Could not create a validation environment due to"
            "unavailability of validation parameters. Please add the "
            "validation parameters to the json file."
        )
        logging.getLogger(logger).exception(
            type(self).__name__ + ": " + mes_str
        )
        super().__init__(mes_str)
