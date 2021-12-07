Usage
=====

This framework can be used in two modes:
1. Command line based program 
2. Python library

The first mode is the intended use case. The usage of this mode is shown below.

Command line based program
--------------------------

The package can be called via command line, after installing it (:doc:`installation`), with the following command.

.. code-block:: console

    (SbSOvRL) $ python -m SbSOvRL [-h] -i INPUT_FILE [-v]

The arguments have to following meaning:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Path to the json file storing the optimization definition.
  -v, --validate_only   If this is set only validation on this configuration is run. Please configure the validation
                        object in the json file so that this option can be correctly executed.

The INPUT_FILE should be the path to a json file defining the problem. The corresponding json-schema is shown in :doc:`json_schema`.

The base json object has the following attributes:

.. list-table:: base json object
    :widths: 25 75
    :header-rows: 0
    :stub-columns: 1

    * - agent
      - Defining the agent that is supposed to be used. The python class definition of the agents can be found at :doc:`_autosummary/SbSOvRL.agent`
    * - environment
      - Defining the environment of the task. This is the most complicated definition the class definition of the object can be found at :doc:`_autosummary/SbSOvRL.parser_environment`
    * - verbosity
      - Defining the verbosity of the training process and environment loading. The object can be found at :doc:`_autosummary/SbSOvRL.verbosity.Verbosity`
    * - validation
      - Definition of the validation parameters. The object can be found at :doc:`_autosummary/SbSOvRL.validation.Validation`
    * - number_of_timesteps
      - Number of time steps the agent should be trained for is superseded by the number of episodes the agent is trained for. Must be of type (int).
    * - number_of_episodes
      - Number of episodes the agent should be trained it supersedes by the number of time steps the agent is trained for. Must be of type (int).
    * - save_location
      - Path to the directory where the log and validation results are to be stored. If {} is inside the string a timestamp is added to distinguish different training runs. Must be of type (str).
  

The package uses the Python library ``pydantic`` to parse the json file into Python classes. It tries to match the json attributes in each object to the Python attributes of the corresponding class. So when looking at the Python class :doc:`_autosummary/SbSOvRL.base_parser.BaseParser` the listed attributes of this class are also the attributes of the base json object. This also holds recursively for all referenced types.