import datetime
import json
import logging

import numpy as np
import pytest
from conftest import dir_save_location_path

from releso.util.util_funcs import (
    JSONEncoder,
    call_commandline,
    get_path_extension,
    join_infos,
)
from releso.verbosity import VerbosityLevel


@pytest.mark.parametrize(
    "dictionary, wanted_value",
    [
        ({"test": np.ones(12)}, np.ones(12).tolist()),
        ({"test": np.int64(123)}, 123),
        ({"test": bytes("test", "utf-8")}, "test"),
        ({"test": np.longdouble(13.1)}, False),
    ],
)
def test_json_encode(dictionary, wanted_value, capsys):
    with pytest.raises(TypeError):
        json.dumps(dictionary)
    if not wanted_value:
        with pytest.raises(TypeError):
            json.dumps(dictionary, cls=JSONEncoder)
        assert "class 'numpy.longdouble'" in capsys.readouterr().out
        return
    json_string = json.dumps(dictionary, cls=JSONEncoder)
    py_dict = json.loads(json_string)
    assert py_dict["test"] == wanted_value


def test_get_path_extension():
    ret_str = get_path_extension()
    assert datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") in ret_str


def test_join_infos():
    starting_dict = {"test": 1, "test1": 1}
    new_dict = {"test2": 2, "test": 3}
    join_infos(starting_dict, new_dict, "test")
    assert starting_dict == {"test": 3, "test1": 1, "test2": 2}


def test_call_command_line(dir_save_location, clean_up_provider, caplog):
    dir_save_location_path(True)
    log = logging.getLogger("test")
    log.setLevel(logging.DEBUG)
    (dir_save_location / "test.txt").touch()
    with caplog.at_level(VerbosityLevel.DEBUG):
        exit_code, output = call_commandline("ls", dir_save_location)
        exit_code1, _ = call_commandline(
            "ls",
            ".",
            logging.getLogger("test"),
        )
        assert "Executing command ls in ." in caplog.text
        exit_code2, _ = call_commandline("lsl", ".", logging.getLogger("test"))
        assert "Execution failed with return code" in caplog.text
    assert exit_code == 0
    assert exit_code1 == 0
    assert exit_code2 == 127
    assert output == b"test.txt\n"
    clean_up_provider(dir_save_location)
