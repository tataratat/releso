#!/usr/bin/env python
"""
    File defines the main entry point of the framework if it is called via the
    command line. (via python -m SbSOvRL)
"""
import argparse
import datetime
import pathlib
import pprint
import shutil

import hjson

from SbSOvRL.base_parser import BaseParser


def main(args) -> None:
    """
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
        optimization_object.evaluate_model(throw_error_if_None=True)
        return

    ###########################
    #                         #
    #  Training the RL Agent  #
    #                         #
    ###########################
    optimization_object.learn()


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Spline base Shape Optimization via Reinforcement "
        "Learning Toolbox. This python program loads a problem "
        "definition and trains the resulting problem. Further the "
        "model can be evaluated")
    parser.add_argument(
        "-i",
        "--input_file",
        action="store",
        required=True,
        help="Path to the json file storing the optimization definition.")
    parser.add_argument(
        "-v",
        "--validate_only",
        action="store_true",
        help="If this is set only validation on this configuration is run. "
        "Please configure the validation object in the json file so that this "
        "option can be correctly executed.")
    parser.add_argument(
        "-j",
        "--json_only",
        dest="json_validate",
        action="store_true",
        help="If this is set only the json validation is performed, nothing "
        "else.")
    args = parser.parse_args()
    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    main(args)
