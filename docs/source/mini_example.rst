======================
Mini Example
======================

This minimal example goes through the definition of all required files for the following task:


The mini example consists of 5 points in space which are movable only in x direction. The goal is to move the points to the maximal x direction. The reward given is dense and consists of the sum of all x coordinates of the points.

In general everything is defined in the *hjson* file. It defines:
1. **Environment**: Geometry (definition of the actions) and the steps to compute the resulting observations and rewards.
2. **Agent**: ReLeSO uses agents provided by Stable Baselines3.
3. **General**: Verbosity, number of episodes, number of steps, etc.

As a separate part of the **Environment** the **SPOR** system needs to be discussed. It is the way the user of ReLeSO defines the custom steps to generate the observations and the reward.

Environment
-----------

You can define the environment by defining the geometry, the SPOR system and environment parameters.

In the following the environment definition is shown. The geometry and SPOR part are shortened since they will be expanded later on.

.. code-block::

    "environment": {
        "geometry": {...},
        "spor": {...},
        "max_timesteps_in_episode": 50,
        "reward_on_episode_exceeds_max_timesteps": 0,
        "multi_processing": {
            "number_of_cores": 1
        }
    }


The max_timesteps_in_episode defines the maximum number of steps in an episode. If this number is exceeded, the episode is terminated and the reward is set to the value defined in *reward_on_episode_exceeds_max_timesteps*. The default reward value is 0.0.
multi_processing defines the number of cores that should be used for the environment, this is mostly for MPI applications. Only define something else if you really know what you are doing. The default value is 1 core.


Geometry
~~~~~~~~

The first step in the definition of the environment is the definition of the geometry. The geometry defines the actions and some of the observations.

For the example given above the geometry consists of 5 control points, where each can be moved in the x direction between [-3,3]. This can be defined in the *hjson* file as:


.. code-block::

    "geometry": {
        "shape_definition":{
                "control_points": [
                    [
                        {
                            "current_position": 0.0,
                            "min_value": -3.0,
                            "max_value": 3.0,
                        },
                        {
                            "current_position": 0.0
                        }
                    ],
                    [
                        {
                            "current_position": 0.0,
                            "min_value": -3.0,
                            "max_value": 3.0,
                        },
                        {
                            "current_position": 1.0
                        }
                    ],
                    [
                        {
                            "current_position": 0.0,
                            "min_value": -3.0,
                            "max_value": 3.0,
                        },
                        {
                            "current_position": 2.0
                        }
                    ],
                    [
                        {
                            "current_position": 0.0,
                            "min_value": -3.0,
                            "max_value": 3.0,
                        },
                        {
                            "current_position": 3.0
                        }
                    ],
                    [
                        {
                            "current_position": 0.0,
                            "min_value": -3.0,
                            "max_value": 3.0,
                        },
                        {
                            "current_position": 4.0
                        }
                    ],
                ]
            },
        },
        "discrete_actions": true,
        "action_based_observation": true,
    }

The geometry is defined by a shape and by the actions that arise out of this shape. In this case the most basic shape definition ReLeSO offers is used. The shape is only defined by a list of control points.
In this example each control point exists in two dimensions, where the second dimension is fixed in place. For the first dimension each control point is bounded by [-3,3] with a starting value of 0.0.

The actions that arise out of this shape can either be discrete or continuous actions. In this case discrete actions are chosen. This means that for each movable dimension, for each control point, two actions are defined. One to increase the value and one to decrease the value. This means that his example has 10 actions.
The step length of the actions are not directly defined here, so it uses default value which means the step length is 1/10 of the range of the parameter, in this case 3/5.

The last item *actions_based_observation*, defines that the observations should also include the current location of the movable control points.

SPOR
~~~~

Solver, Postprocessing, Observation and, Reward is where the most customization is needed. Here is where additional observations and the reward are calculated. As this is custom for each use case, ReLeSO provides the user with tools to implement them themselves. The SPOR system is the way to define these steps. The SPOR system is a list of steps which are executed in order. Each step can be a solver, postprocessing step, observation or reward step or a combination of them. The reward of all steps is combined into one final reward.

For this simple example the SPOR definition could look like this:

.. code-block::

    "spor": {
        "steps": [
            {
                "name": "control_point_sum",
                "stop_after_error": false,
                "reward_on_error": -10,
                "run_on_reset": true,
                "working_directory": "./",
                "python_file_path": "mini_example.py",
                "use_communication_interface": true,
                "add_step_information": true
            }
        ],
        "reward_aggregation": "sum"
    },


In this example an external python script is used to calculate the reward.

Only a single step is defined. The definition can be categorized into 3 parts:
**General** The name of the step/task, whether or not to stop and terminate the episode if an error is thrown in this step, if a reward should be applied if the step fails and, whether or not the run the task during a reset step. There are more potential options, but they are not used here since the default values for those are sufficient.
**User Defined Python Function** First the location where the python script should be run is defined, then that it is a python script that is called (Attention: The script is not called with this command, but first releso tries to load a specific *main()* function from the script (see below) and run it "internalized"), the path to the script, and lastly there are two options that define if the communication interface should be used and if the step information should be added to the communication.

**Reward Aggregation** The last part is the aggregation of the reward. In this case the sum of all rewards is used. This means that if multiple steps are defined, the reward of all steps is summed up. If only one step is defined, the reward of that step is used.

As mentioned before if a python file is defined as shown above ReLeSO first tries to load the script and run the function *main()* from it. For this to work this function needs the following signature:

.. code-block:: python

    def main(args, logger, func_data):
        """
        Main function of the external task.
        :param args: A named tuple of the arguments in the communication interface.
        :param logger: A logger object specifically for this step, defined by releso.
        :param func_data: An empty object can be used to store persistent data the step needs.
        :return: The return object includes the reward, observations, done and, info.
        """
        # your code
        return {
            "reward": reward,
            "observations": observations,
            "done": done,
            "info": info
        }, func_data


In this example the *python* script could look this this:

.. code-block:: python

    from collections import namedtuple
    from releso.util.reward_helpers import spor_com_parse_arguments, write_json, load_json
    import numpy as np
    from logging import Logger
    from typing import Optional, Any
    from pathlib import Path
    import os

    def main(args: namedtuple, logger: Logger, func_data: Optional[Any]):
        done = False
        info = {}

        # if add_step_information is not true, the json_object is None
        # but it is needed to calculate the reward
        if not args.json_object:
            print("No additional payload, please provide the needed payload.")

        # setup the func_data object, it is not used in this example
        if func_data is None:
            func_data = dict()

        # calculate the reward
        reward = sum(np.array(args.json_object["info"]["geometry_information"]))[0]

        # if reward is very close to the maximum of 15 it is considered as done
        if reward >= (15-1e-7):
            logger.warning(f"This is triggered why? : {reward}")
            reward = 30
            done = True
            info["reset_reason"] = "goal_reached"

        logger.info(
            f"{args.json_object['info']['geometry_information']}, Sum: {sum(np.array(args.json_object['info']['geometry_information']))[0]}, Reward: {reward}"
        )

        return {
                "reward": reward,
                "done": done,
                "info": info,
                "observations": []
        }, func_data

    # Add option of running the script manually, or with original command line
    # spor step in ReLeSO.
    if __name__ == "__main__":
        args = spor_com_parse_arguments()
        if not args.json_object:
            print("No additional payload, please provide the needed payload.")

        # create path to a folder usable as variable store
        path = Path(f"{os.getcwd()}/{args.run_id}")
        path.mkdir(exist_ok=True, parents=True)

        local_variable_store_path = path/"local_variable_store.json"
        # create the local variable store if it does not exist
        if not path.exists():
            func_data = {
                "last_error": 0
            }
            write_json(local_variable_store_path, func_data)
        func_data = load_json(local_variable_store_path)

        # run the main function
        step_data, func_data = main(args, False, func_data)

        write_json(path, func_data)

        print(step_data)




In general you won't need the last 25 lines of code since ReLeSO will directly call the *main* function. but it can be used to trouble shoot the script manually (and also is an old way to use the SPOR system).


Agent
~~~~~

The agent is defined in the *hjson* file. The agent is a standard agent from Stable Baselines3. In this case the PPO agent is used. The only thing that needs to be defined is the type of agent and the parameters for the agent. The parameters are passed directly to the constructor of the agent.

The *tensorboard_log* parameter is used to define the path where the tensorboard logs should be saved. This is not needed, but it is recommended to use it since it helps to debug the agent and the training process.

.. code-block::

    "agent": {
        "type": "PPO",
        "policy": "MlpPolicy",
        "tensorboard_log": "tensorboard",
    },


General
~~~~~~~
The general part of the *hjson* file defines the number of episodes, the number of steps, the verbosity and the path to the log file. The verbosity is used to define how much information should be printed to the console. The log file is used to save the logs of the training process.

.. code-block::

    {
        "agent": {...},
        "environment": {...},
        "number_of_timesteps": 100000,
        "number_of_episodes": 100000,
        "save_location": "mini_example_{}/",
        "verbosity": {
            "environment": "INFO",
            "parser": "INFO"
        }
    }


The *number_of_timesteps* defines the number of timesteps that should be used for training. The *number_of_episodes* defines the number of episodes that should be used for training. The *save_location* defines the location where the logs should be saved. The *verbosity* defines the verbosity of the logs.

In the full example can be found in the *examples* folder of the ReLeSO repository.
