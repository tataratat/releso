Usage
=====

This framework can be used in two modes:
1. Command line based program 
2. Python library

The first mode is the intended use case. The usage of this mode is shown below.

Command line based program
==========================

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
    :header-rows: 1
    :stub-columns: 1

    * - agent
      - Defining the agent that is supposed to be used. The python class definition of the agents can be found at :doc:`_autosummary/SbSOvRL.agent`
    * - environment
      - Defining the environment of the task. This is the most complicated definition the class definition of the object can be found at
  