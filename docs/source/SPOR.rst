Solver Postprocessing Observation Reward
========================================

The definition of the solver for this framework is complicated, since we wanted to be able to incorporate multiple solvers and the solver for the first use-case is does not have a Python interface/library.

To incorporate the use-case solver there are two possible methods, 1. write a custom python interface 2. call the solver via an existing command line interface. The first option would limit the framework to only this single solver and if another solver were to be used in a different use-case the framework would need to be extended to also interface with this new solver. The second option leaves the ecosystem of Python and gives the user more responsibility and needs some additional hardening to external errors, but is able to interface with any solver which has a command line interface.

The same problem is present when we look at the next step in the process the definition of the reward. The reward is the metric/measure the agent uses to optimize its policy. And is therefore very use-case specific it encodes what the goal of the problem is. This means that this needs to be changed for every use-case. The question again is how can the framework be designed to accommodate this. The answer is again is to use the command line to call user supplied/generated reward scripts.

These two different problems have the same answer. The use-case had only these two steps after computing the FFD, but after analyzing further potential use-cases the decision was made that the framework has the SPOR field in which FFD Postprocessing steps are defined. This can be portioning of the resulting mesh, running the solver on the mesh, reading in the solver output for additional observations, generating the reward and many more.

In future the framework might also include SPOR object which are directly part of the framework and are not needed to be called from the command line.

SPOR
----
Since all SPOR object are able to generate reward values, a variable defining the math operation to perform on all collected rewards needs to be chosen in the SPOR list object. The possible operations are:

1. mean: The mean of all given rewards is returned as the ultimate reward.
2. max: The maximal reward is returned.
3. min: The minimal reward is returned.
4. minmax: The reward with the biggest absolute value is returned (with sign).
5. sum: The sum of all rewards is returned. *Default*

Additionally a global option on wether to stop after a step returned an error code can be set. This will overrule SPOR object which say to continue but will not overwrite the SPOR object if it is set to stop on error.

Example:

  * The global option is set to stop after a returned error. Three steps are defined and all have the option to continue after an error is returned. Step two fails. Since the global option is set to stop the third step is **not** executed.

  * The global option on wether to stop on an error is not set. Three steps are defined and all have the option to continue after an error is returned. Step two fails. The third step **is** executed.

  * The global option on wether to stop on an error is not set. Three steps are defined and all have the option to stop after an error is returned. Step two fails. The third step is **not** executed, since step two says to stop after an error is returned.

SPOR Object
-----------

Each SPOR object has the 3 options which must be set:

1. Name of the spor object.
2. Stop after error.
3. Reward on error.

and the 2 options which can be set:

4. Reward if the step succeeded. *Default: None*
5. Additional Observations, defining observations that are produced in this step.

Additional options need/can to be set depending on the specific SPOR object.
See :doc:`_autosummary/releso.spor` for a list of all available SPOR objects.

.. _sporcominterface:

SPOR Communication Interface
----------------------------

The communications interface is currently only available for command line calls. The communication interface adds at least one command line option and one configurable command line option if other information from the framework needs to be passed to the script/program. These are:

::

  --run_id uuid4                  # The uuid is custom to this specific task and environment. So if you are in a multi environment setup the same task in different environments can be easily distinguished.
  --validation_value              # To distinguish between training and validation this parameter is send during validation with the id of the current validation step. If no value is given, default training should be assumed.
        [validation_id]
  --reset                         # This option is only send if the current episode needs to be reset.
  --json_object                   # Current step information. Need to be activated in spor definition with 'add_step_information'
  --base_save_location            # This value is the base path to the directory where the records of the current jobs are stored. To keep all information together it is advised to save the logs, memory and persistent data here.
  --environment_id                # This value will provide a unique id for the current environment. This is needed to be able to distinguish between different environments in a multi environment setup.


During the configuration if the validation the number of different episodes each validation should include. Each of these validation episodes gets validation id, which is transmitted to the external program.

To help the user in the development of these external task/steps the framework has helper functions in the :doc:`_autosummary/releso.util.reward_helpers` module. These are able to setup the working space of the task with "persistent" memory and the parsing of the communication interface command line options can also be handled via these functions.

Also look at the examples and tests to see how you can use internalized python functions to speed up execution.

Custom Python based SPOR object
-------------------------------

The framework is able to use custom python based SPOR objects. In most examples ReLeSO is currently used like this. It is the main way to integrate the external solver and reward calculation.


Use the definition for the :doc:`_autosummary/releso.spor.SPORObjectExternalPythonFunction` to use this functionality. A small example is given :doc:`mini_example`.
