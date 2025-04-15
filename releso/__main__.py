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
import numpy as np
import stable_baselines3
import torch

from releso.__version__ import __version__
from releso.base_parser import BaseParser
from releso.util.visualization import export_figure, plot_episode_log, \
    plot_step_log

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
        description=(
            "Reinforcement Learning based Shape Optimization (releso) Toolbox. "
            "This python program loads a problem definition and trains an RL "
            "agent to solve the resulting problem. Furthermore a trained "
            "agent can be evaluated or data gathered during a training "
            f"visualized. The package version is: {__version__}."
        )
    )
    parser.add_argument(
        "--version",
        dest="version",
        action="store_true",
        help="Returns the version of the package.",
    )
    sub_parser = parser.add_subparsers(
        title="execution modes",
        dest="execution_mode"
    )
    parser_run = sub_parser.add_parser(
        "run",
        help=(
            "Start a releso run with a given json file. Choose this option if "
            "you want to train a model or validate a given input file."
        ),
    )
    parser_run.add_argument(
        "-i",
        "--input-file",
        action="store",
        required=True,
        help="Path to the json file storing the optimization definition.",
    )
    parser_run.add_argument(
        "-v",
        "--validate-only",
        action="store_true",
        help=(
            "If this is set only validation on this configuration is run. "
            "Please configure the validation object in the json file so that "
            "this option can be correctly executed.",
        )
    )
    parser_run.add_argument(
        "-j",
        "--json-only",
        dest="json_validate",
        action="store_true",
        help=(
            "If this is set only the json validation is performed, nothing "
            "else."
        ),
    )
    parser_visualize = sub_parser.add_parser(
        "visualize",
        help=(
            "Visualize training data. Choose between visualizing the data "
            "contained in the episode log (global training information) and "
            "the data contained in the step log (more in-depth training "
            "specific information)."
        ),
    )
    visualize_shared_args = argparse.ArgumentParser(add_help=False)
    visualize_shared_args.add_argument(
        "-e",
        "--export-path",
        type=pathlib.Path,
        default="./",
        help=(
            "Path where the visualization should be saved to. If a directory "
            "is provided the created figure will be saved in that directory "
            "as html file under a default filename which depends on the type "
            "of visualization that has been chosen. If the path contains a "
            "filename with a suffix, it will be exported as the file type "
            "indicated by the suffix if that is possible. Available suffixes "
            "are [png, jpg, webp, svg, pdf, html]. Defaults to './'."
        ),
    )
    visualize_shared_args.add_argument(
        "-s",
        "--figure-size",
        type=check_window_size,
        default="auto",
        nargs="+",
        help="Size of the figure.",
    )
    sub_parser_visualize = parser_visualize.add_subparsers(
        title="visualization modes",
        dest="visualization_mode",
        required=True
    )
    parser_visualize_episodelog = sub_parser_visualize.add_parser(
        "episode-log",
        parents=[visualize_shared_args],
        help=(
            "Visualize training progress. List all folders you want to "
            "include. If run a training during this call as well the visualization "
            "will be exported after the training has finished and will also "
            "include the new results automatically. The legend name of each "
            "experiment is the folder name. If you need to display custom "
            "names please use the call the function from your own script. "
            "You can find the function in 'releso.util.visualization.plot_episode_log'. "
        ),
    )
    parser_visualize_episodelog.add_argument(
        "-f",
        "--folders",
        type=pathlib.Path,
        nargs="*",
        required=True,
        help=(
            "Visualize given training episode logs. You can also use wildcard "
            "arguments '*' or '?' at least on some systems."
        )
        ,
    )
    parser_visualize_episodelog.add_argument(
        "-w",
        "--window",
        type=check_positive,
        default=5,
        help=(
            "Episode visualization uses windowing to smooth the graph. Set "
            "the window length. Defaults to 5."
        ),
    )
    parser_visualize_episodelog.add_argument(
        "-c",
        "--cut-off-point",
        type=check_positive,
        default=np.iinfo(int).max,
        help=(
            "Cut off point for the visualization (in timesteps). If set, the "
            "visualization will exclude all timesteps after the specified "
            "cutoff point. Defaults to np.iinfo(int).max, which means that all "
            "timesteps will be included."
        )
    )
    parser_visualize_steplog = sub_parser_visualize.add_parser(
        "step-log",
        parents=[visualize_shared_args],
        help=(
            "Visualize the strategy of the agent employed in each episode. "
            "If you want to customize the visualization please use the call "
            "the corresponding function from your own script. "
            "You can find the function in 'releso.util.visualization.plot_step_log'. "
        ),
    )
    parser_visualize_steplog.add_argument(
        "-l",
        "--logfile",
        type=pathlib.Path,
        required=True,
        help=(
            "Visualize a given training step log to analyze the strategy "
            "learned by an agent in a specific run."
        )
    )
    parser_visualize_steplog.add_argument(
        "-i",
        "--episode-id",
        type=check_positive,
        default=0,
        help=(
            "ID of the environment whose data is supposed to be visualized "
            "(only relevant for multi-environment trainings). Defaults to 0."
        )
    )
    parser_visualize_steplog.add_argument(
        "-f",
        "--from-episode",
        type=check_positive,
        default=0,
        help=(
            "Starting episode of the interactive visualization. Defaults to 0."
        ),
    )
    parser_visualize_steplog.add_argument(
        "-n",
        "--n-episodes",
        type=check_positive,
        default=1,
        help=(
            "Select the number of episodes to be included in each visualization. "
            "Defaults to 1 which means that every episode between from_episode "
            "and until_episode will be visualized. Set to a larger value to "
            "include less episodes in the visualization."
        ),
    )
    parser_visualize_steplog.add_argument(
        "-u",
        "--until-episode",
        type=check_positive,
        default=None,
        help=(
            "Final episode of the interactive visualization. Defaults to None, "
            "which means that all episodes after the chosen starting one will "
            "be visualized."
        ),
    )
    args = parser.parse_args()
    if args.version:
        print(f"releso: {__version__}")
        return

    run_folder = None

    # We want to run releso
    if args.execution_mode == "run":
        print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        run_folder = main(args)
    # We want to use releso for visualization
    else: # args.execution_mode == "visualize"
        # Determine the figure size
        figure_size = args.figure_size
        if len(figure_size) == 1:
            figure_size = figure_size[0]
        # Visualize contents of episode_log.csv
        if args.visualization_mode == "episode_log":
            folders_to_process: list = [
                folder.resolve() for folder in args.folders
            ]
            if run_folder is not None:
                folders_to_process.append(run_folder)
            folder_dict = {folder.stem: folder for folder in folders_to_process}

            fig = plot_episode_log(
                folder_dict,
                args.window,
                window_size=figure_size,
                cut_off_point=args.cut_off_point,
            )
            export_figure(fig, args.export_path, "episode_log.html")
        # Visualize contents of step_log.jsonl
        else: # args.visualization_mode == "step_log"
            fig = plot_step_log(
                args.logfile.resolve(),
                args.episode_id,
                episode_start=args.from_episode,
                episode_end=args.until_episode,
                episode_step=args.n_episodes,
                figure_size=figure_size,
            )
            # Export the plot as the suffix or as "steplog_plot.html"
            export_figure(fig, args.export_path, "steplog_plot.html")


if __name__ == "__main__":  # pragma: no cover
    entry()
