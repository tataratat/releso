#!/usr/bin/env python

from SbSOvRL.base_parser import BaseParser
import argparse
import pathlib
import hjson
import datetime


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
            json_content = hjson.load(file)
    else:
        raise ValueError(f"Could not find the given file: {file_path}.")

    optimization_object = BaseParser(**json_content)
    
    ###########################
    #                         #
    #   Only json validation  #
    #                         #
    ###########################
    
    if args.json_validate:
        print(optimization_object.dict())
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

    # ###########################
    # #                         #
    # #    exporting spline     #
    # #                         #
    # ###########################
    # if args.export_spline is not None:
    #     optimization_object.export_spline(args.export_spline)

    # ###########################
    # #                         #
    # #     exporting mesh      #
    # #                         #
    # ###########################
    # if args.export_mesh is not None:
    #     optimization_object.export_mesh(args.export_mesh)

    # ###########################
    # #                         #
    # #      saving model       #
    # #                         #
    # ###########################
    # if args.save_model is not None:
    #     optimization_object.export_mesh(args.export_mesh)

if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser(description="Spline base Shape Optimization via Reinforcement Learning Toolbox. This python program loads a problem definition and trains the resulting problem. Futher the model can be evaluated")
    parser.add_argument("-i", "--input_file", action="store", required=True, help="Path to the json file storing the optimization definition.")
    # # parser.add_argument("--export_spline", action="store", help="Path to the file location where the resulting spline should be exported to. Suffix can be [.iges | .xml | .itd]")
    # # parser.add_argument("--export_mesh", action="store", help="Path to the file location where the resulting spline should be exported to.  Suffix can be [`.grd` | `.xns` | `.campiga` | `.h5`]")
    # # parser.add_argument("--save_model", action="store", help="Path to the file location where the resulting model should be saved to.")
    # # parser.add_argument("--evaluate_values", action="extend", nargs="+", type=float, help="Will either be run after training or for the model defined in model_path.")
    # # parser.add_argument("-p", "--path", action="store", help="Base path for the results of the evaluation")
    # parser.add_argument("-m", "--model_load_path", action="store", help="Path to the location a previously trained model. If this is given no training will be performed. Currently there is no way to continue training of a model.")
    parser.add_argument("-v", "--validate_only", action="store_true", help="If this is set only validation on this configuration is run. Please configure the validation object in the json file so that this option can be correctly executed.")
    parser.add_argument("-j", "--json_only", dest="json_validate", action="store_true", help="If this is set only the json validation is performed, nothing else.")
    args = parser.parse_args()
    print(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    main(args)
