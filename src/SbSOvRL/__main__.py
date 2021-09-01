import sys
from SbSOvRL.base_parser import BaseParser
import argparse
from argparse import Action
import pathlib
import json

def main(args) -> None:
    
    ###########################
    #                         #
    #   Loading and parsing   #
    #     the json file       #
    #                         #
    ###########################
    file_path = pathlib.Path(args.input_file).expanduser().resolve()
    if file_path.exists() and file_path.is_file():
        with open(file_path) as file:
            json_content = json.load(file)
    else:
        raise ValueError(f"Could not find the given file: {file_path}.")

    optimization_object = BaseParser(**json_content)

    ###########################
    #                         #
    #  Training the RL Agent  #
    #                         #
    ###########################
    optimization_object.train()

    ###########################
    #                         #
    #    exporting spline     #
    #                         #
    ###########################
    if args.export_spline is not None:
        optimization_object.export_spline(args.export_spline)

    ###########################
    #                         #
    #     exporting mesh      #
    #                         #
    ###########################
    if args.export_mesh is not None:
        optimization_object.export_mesh(args.export_mesh)

if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description="Spline base Shape Optimization via Reinforcement Learning Toolbox. This is the basic script that can load a json file and run the resulting optimization problem.")
    parser.add_argument("-i", "--input_file", action="store", required=True, help="Path to the json file storing the optimization definition.")
    parser.add_argument("-s", "--export_spline", action="store", help="Path to the file location where the resulting spline should be exported to. Suffix can be .iges | .xml | .itd")
    parser.add_argument("-m", "--export_mesh", action="store", help="Path to the file location where the resulting spline should be exported to.  Suffix can be `.grd` | `.xns` | `.campiga` | `.h5`")
    args = parser.parse_args()
    main(args)
