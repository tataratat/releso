"""Testing BaseModel class"""

import datetime
import os
import pathlib
import sys
import uuid
from collections import OrderedDict
from copy import deepcopy

import pytest

from releso.base_model import BaseModel, add_save_location_if_elem_is_o_dict

base_path = str(pathlib.Path(__file__).parent)

dir_save_location_local = str(
    (
        pathlib.Path(__file__).parent / "test_save_location_please_delete"
    ).resolve()
)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10")
@pytest.mark.parametrize(
    "test_input, changes",
    [
        (OrderedDict(), True),
        (
            OrderedDict([("save_location", "something")]),
            False,
        ),
        (
            OrderedDict([
                ("test", OrderedDict()),
                ("save_location", "something"),
            ]),
            False,
        ),
        (["test", "asdas", 12, 12.0], False),
        ({"test": "asdas", "test2": 12, "test3": 12.0}, True),
        (
            {"test": "asdas", "test2": 12, "test3": 12.0, "save_location": 12},
            False,
        ),
        ([OrderedDict(), OrderedDict()], True),
        ([OrderedDict(), "testing"], True),
    ],
)
def test_base_model_add_save_location_if_elem_is_o_dict(test_input, changes):
    input_copy = deepcopy(test_input)
    save_location_str = "asdasdedascsadwasdvril324"
    add_save_location_if_elem_is_o_dict(test_input, save_location_str)
    assert (test_input == input_copy) is not changes
    if changes:
        if isinstance(input_copy, list):
            for idx in range(len(input_copy)):
                if isinstance(test_input[idx], OrderedDict | dict):
                    assert (
                        test_input[idx]["save_location"] == save_location_str
                    )
        else:
            assert test_input["save_location"] == save_location_str
    else:
        if isinstance(input_copy, OrderedDict | dict):
            assert test_input["save_location"] != save_location_str


@pytest.mark.parametrize(
    "test_input",
    [
        (dir_save_location_local),
        (f"please_delete_{uuid.uuid4()}"),
        ("please_delete_{}"),  # this test might fail if the time is exactly
    ],
)
def test_base_model_convert_to_pathlib_add_datetime(test_input):
    simple_path = None
    if "{}" not in test_input:
        if (simple_path := pathlib.Path(test_input)).is_dir():
            os.rmdir(simple_path)
    else:
        simple_path = pathlib.Path(
            test_input.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        ).resolve()
    result = BaseModel.convert_to_pathlib_add_datetime(test_input)
    if simple_path:
        assert result == simple_path.resolve()
    assert isinstance(result, pathlib.Path)
    assert result.is_dir()
    os.rmdir(result)


@pytest.mark.parametrize(
    "logger_name",
    [
        (None),
        ("this_is_a_logger_name"),
    ],
)
def test_base_model_initiate(logger_name, dir_save_location):
    model = BaseModel(save_location=dir_save_location, logger_name=logger_name)

    assert model.save_location == pathlib.Path(dir_save_location)
    assert model.logger_name == logger_name


@pytest.mark.parametrize("original_logger_name", [None, "original_name"])
def test_base_model_set_logger_name_recursively(
    original_logger_name, dir_save_location
):
    model = BaseModel(
        save_location=dir_save_location, logger_name=original_logger_name
    )

    model.__dict__["test"] = deepcopy(model)
    model.__dict__["test2"] = [
        deepcopy(model),
        123,
        [deepcopy(model), 123, 123.0, "test"],
    ]

    model.set_logger_name_recursively("new_logger_name")

    assert model.logger_name == "new_logger_name"


def test_base_model_multiprocessing_logger(dir_save_location):
    model = BaseModel(
        save_location=dir_save_location_local,
        logger_name="original_logger_name_multiprocessing",
    )

    logger = model.get_logger()

    assert logger.name == "multiprocessing"
