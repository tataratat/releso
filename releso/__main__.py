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
from typing import Literal, Union

import gymnasium
import hjson
import stable_baselines3
import torch

from releso.__version__ import __version__
from releso.base_parser import BaseParser
from releso.util.visualization import plot_episode_log

try:
    import splinepy
except ImportError:
    splinepy = None


def check_positive(value) -> int:
    """Checks if the provided value is positive.

    Note:
        Adapted from https://stackoverflow.com/a/14117511.

    Args:
        value (Any): Value to be cast to integer.

    Raises:
        argparse.ArgumentTypeError: Throws error if the value is not positive.

    Returns:
        int: checked value
    """
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Provided value is not positive.")
    return value


def check_window_size(value) -> Union[tuple[int, int], Literal["auto"]]:
    """Check window check for error.

    Raises:
        argparse.ArgumentTypeError: If value is not correct.

    Returns:
        Union[tuple[int, int], Literal["auto"]]: Validated value.
    """
    if str(value) == "auto":
        return "auto"
    elif value := int(value):
        return value
    raise argparse.ArgumentTypeError(
        "Provided values are not 'auto' or two integer values."
    )


def main(args) -> pathlib.Path:
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
        f"releso version: {__version__}; ",
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

    return optimization_object.save_location


def entry():
    """Entry point if this package is called directly from the command line."""
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning based Shape Optimization (releso) "
        "Toolbox. This python program loads a problem "
        "definition and trains the resulting problem. Further the "
        "model can be evaluated"
        f"The package version is: {__version__}."
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
    sub_parser = parser.add_subparsers()
    parser_visualize = sub_parser.add_parser(
        "visualize",
        help=(
            "Visualize training progress. List all folders you want to "
            "include. If run a training during this call as well the visualization"
            " will be exported after the training has finished and will also "
            "include the new results automatically. The legend name of each "
            "experiment is the folder name. If you need to display custom"
            " names please use the call the function from your own script. "
            "You can find the function in "
            "'releso.util.visualization.plot_episode_log'. "
        ),
    )
    parser_visualize.add_argument(
        "-f",
        "--folders",
        type=pathlib.Path,
        nargs="*",
        help="Visualize given training episode logs."
        "If training was performed the results will be exported after training"
        " is finished and will also include the new results. You can also use"
        " wildcard arguments '*' or '?' at least on some systems.",
    )
    parser_visualize.add_argument(
        "-e",
        "--export_path",
        type=pathlib.Path,
        default="./",
        help="Path where the visualization should be saved to. If folder it "
        "will be saved to 'path/episode_log.html'. If it is a path with a "
        "suffix it will be exported in the suffix if it is possible. Available"
        " suffixes are [png, jpg, webp, svg, pdf, html]. Defaults to './'.",
    )
    parser_visualize.add_argument(
        "-w",
        "--window",
        type=check_positive,
        default=5,
        help="Episode visualization uses windowing to smooth the graph. Set "
        "the window length. Defaults to 5.",
    )
    parser_visualize.add_argument(
        "-s",
        "--figure-size",
        type=check_window_size,
        default="auto",
        nargs="+",
        help="Figure",
    )
    args = parser.parse_args()
    if args.version:
        print(f"releso: {__version__}")
        return

    run_folder = None
    if args.input_file is None:
        if not hasattr(args, "folders"):
            print(
                "The command option for the input_file is required.\n",
                "An input file can be added via the -i option.\n",
                "Please use the -h option to see the help.",
            )
            return
    else:
        print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_folder = main(args)

    if hasattr(args, "folders"):
        folders_to_process: list = [
            folder.resolve() for folder in args.folders
        ]
        if run_folder is not None:
            folders_to_process.append(run_folder)
        folder_dict = {folder.stem: folder for folder in folders_to_process}
        plot_episode_log(
            folder_dict,
            args.export_path,
            args.window,
            window_size=args.figure_size,
        )


if __name__ == "__main__":  # pragma: no cover
    entry()
