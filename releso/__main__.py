#!/usr/bin/env python
"""ReLeSO main file and framework starting point.

File defines the main entry point of the framework if it is called via the
command line. (via $python -m ReLeSO; or $releso)
"""
import argparse
import datetime
import pathlib
import pprint
import shutil

import gymnasium
import hjson
import stable_baselines3
import torch

from releso.__version__ import version
from releso.base_parser import BaseParser

try:
    import splinepy
except ImportError:
    splinepy = None


def main(args) -> None:
    """Calling function of framework.

    Functions control how the framework works when called from the command
    line.

    Args:
        args ([type]): Command line arguments

    Raises:
        ValueError: Thrown if the json file could not be found.
    """
    ###########################
    #                         #
    #   Loading and parsing   #
    #     the json file       #
    #                         #
    ###########################
    file_path = pathlib.Path(args.input_file).expanduser().resolve()
    if file_path.exists() and file_path.is_file():
        with open(file_path) as file:
            json_content = hjson.load(file)
    else:
        raise ValueError(f"Could not find the given file: {file_path}.")

    optimization_object = BaseParser(**json_content)

    ###########################
    #                         #
    #  Save json definition   #
    #                         #
    ###########################

    shutil.copy(file_path, optimization_object.save_location / file_path.name)
    versions = [
        f"releso version: {version}; ",
        f"stable-baselines3 version: {stable_baselines3.__version__}; ",
        f"torch version: {torch.__version__}; ",
        f"gymnasium version: {gymnasium.__version__} ",
    ]
    if splinepy:
        versions.append(f"splinepy version: {splinepy.__version__}")
    optimization_object.get_logger().info("".join(versions))
    ###########################
    #                         #
    #   Validate json file    #
    #                         #
    ###########################

    if args.json_validate:
        pprint.pprint(optimization_object.dict())
        return

    ###########################
    #                         #
    #    Only validation      #
    #                         #
    ###########################
    if args.validate_only:
        optimization_object.evaluate_model(throw_error_if_none=True)
        return

    ###########################
    #                         #
    #  Training the RL Agent  #
    #                         #
    ###########################
    optimization_object.learn()


def entry():
    """Entry point if this package is called directly from the command line."""
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning based Shape Optimization (releso) "
        "Toolbox. This python program loads a problem "
        "definition and trains the resulting problem. Further the "
        "model can be evaluated"
        f"The package version is: {version}."
    )
    parser.add_argument(
        "-i",
        "--input_file",
        action="store",
        help="Path to the json file storing the optimization definition.",
    )
    parser.add_argument(
        "-v",
        "--validate_only",
        action="store_true",
        help="If this is set only validation on this configuration is run. "
        "Please configure the validation object in the json file so that this "
        "option can be correctly executed.",
    )
    parser.add_argument(
        "-j",
        "--json_only",
        dest="json_validate",
        action="store_true",
        help="If this is set only the json validation is performed, nothing "
        "else.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        action="store_true",
        help="Returns the version of the package.",
    )
    args = parser.parse_args()
    if args.version:
        print(f"releso: {version}")
        return
    if args.input_file is None:
        print(
            "The command option for the input_file is required.\n",
            "An input file can be added via the -i option.\n",
            "Please use the -h option to see the help.",
        )
        return
    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    main(args)


if __name__ == "__main__":  # pragma: no cover
    entry()
