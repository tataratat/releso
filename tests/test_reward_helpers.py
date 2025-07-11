import json
import uuid
from pathlib import Path

import pytest
from conftest import dir_save_location_path

from releso.util.reward_helpers import (
    load_json,
    spor_com_additional_information,
    spor_com_parse_arguments,
    write_json,
)


def test_load_json(dir_save_location: Path, clean_up_provider):
    file_loc: Path = dir_save_location / "test.json"
    assert not file_loc.exists()
    dir_save_location.mkdir(parents=True, exist_ok=True)
    res = load_json(file_loc)
    assert res == {}
    assert file_loc.exists()
    test_dict = {"test": 12, "other": "test"}
    with open(file_loc, "w") as f_open:
        json.dump(test_dict, f_open)
    res = load_json(file_loc)
    assert res == test_dict
    clean_up_provider(file_loc)


def test_load_json_folder_does_not_exist(dir_save_location, clean_up_provider):
    file_loc: Path = dir_save_location / "test" / "test.json"
    with pytest.raises(RuntimeError) as err:
        load_json(file_loc)
    assert "You have to create the folder for the json first" in str(err.value)
    clean_up_provider(file_loc)


def test_write_json(dir_save_location: Path, clean_up_provider):
    dir_save_location.mkdir(parents=True, exist_ok=True)
    file_loc: Path = dir_save_location / "test.json"
    test_dict = {"test": 12, "other": "test"}
    write_json(file_loc, test_dict)

    with open(file_loc) as f_open:
        res = json.load(f_open)
    clean_up_provider(file_loc)
    assert res == test_dict


def test_additional_information():
    # the tested function removes the quotation marks around the string.
    # These appear due to it being processed by argparse
    test_dict = {"test": 12, "other": "test"}
    res = spor_com_additional_information(f"'{json.dumps(test_dict)}'")
    assert res == test_dict


@pytest.mark.parametrize(
    ["initialize", "i", "reset"],
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
@pytest.mark.parametrize(
    ["r", "run", "run_id", "error_r"],
    [
        (str(uuid.uuid4()), False, False, False),
        # (False, str(uuid.uuid4()), False, False),
        # (False, False, str(uuid.uuid4()), False),
        (
            False,
            False,
            False,
            "-r/--run/--run_id",
        ),
    ],
)
@pytest.mark.parametrize(
    ["v", "validation_value", "error_v"],
    [
        (
            "True",
            False,
            True,
        ),
        (
            False,
            False,
            False,
        ),
        (
            str(1.2),
            False,
            False,
        ),
        (
            False,
            "True",
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    ["j", "additional_values", "json_object", "error_j"],
    [
        (
            True,
            False,
            False,
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    ["l_parameter", "base_save_location", "error_l"],
    [
        (
            False,
            str(dir_save_location_path(False)),
            False,
        ),
        (
            False,
            False,
            "-l/--base_save_location",
        ),
    ],
)
@pytest.mark.parametrize(
    ["e", "environment_id", "error_e"],
    [
        (
            "test",
            False,
            False,
        ),
        (
            False,
            False,
            "-e/--environment_id",
        ),
    ],
)
def test_parse_communication_interface(
    initialize,
    i,
    reset,
    r,
    run,
    run_id,
    error_r,
    v,
    validation_value,
    error_v,
    j,
    additional_values,
    json_object,
    error_j,
    l_parameter,
    base_save_location,
    error_l,
    e,
    environment_id,
    error_e,
    dir_save_location,
    capsys,
):
    args = []
    if initialize:
        args.append("--initialize")
    elif i:
        args.append("-i")
    elif reset:
        args.append("--reset")
    if r:
        args.extend(["-r", r])
    elif run:
        args.extend(["--run", run])
    elif run_id:
        args.extend(["--run_id", run_id])
    if v:
        args.extend(["-v", v])
    elif validation_value:
        args.extend(["--validation_value", validation_value])
    if l_parameter:
        args.extend(["-l", l_parameter])
    elif base_save_location:
        args.extend(["--base_save_location", base_save_location])
    if e:
        args.extend(["-e", e])
    elif environment_id:
        args.extend(["--environment_id", environment_id])

    if any([error_r, error_v, error_j, error_l, error_e]):
        try:
            spor_com_parse_arguments(args)
        except SystemExit:
            pass
        err = capsys.readouterr().err
        errors = []
        for error in [error_r, error_v, error_j, error_l, error_e]:
            if error:
                if isinstance(error, str):
                    errors.append(error in err)
                else:
                    errors.append(True)
        assert any(errors)
        return
    argparse_ret = spor_com_parse_arguments(args)
    # raise RuntimeError(f"{argparse_ret}")
    assert argparse_ret.reset == any([initialize, i, reset])
    if run:
        assert argparse_ret.run_id == uuid.UUID(run)
    elif run_id:
        assert argparse_ret.run_id == uuid.UUID(run_id)
    elif r:
        assert argparse_ret.run_id == uuid.UUID(r)
    if validation_value:
        assert argparse_ret.validation_value == float(validation_value)
    elif v:
        assert argparse_ret.validation_value == float(v)
    if base_save_location:
        assert argparse_ret.base_save_location == base_save_location
    elif l_parameter:
        assert argparse_ret.base_save_location == l_parameter
    if environment_id:
        assert argparse_ret.environment_id == environment_id
    elif e:
        assert argparse_ret.environment_id == e


@pytest.mark.parametrize(
    ["j", "additional_values", "json_object", "error_j"],
    [
        (
            False,
            False,
            False,
            False,
        ),
        (
            False,
            f'"{json.dumps({"test": 12, "other": "test"})}"',
            False,
            False,
        ),
        (
            False,
            False,
            f'"{json.dumps({"test": 12, "other": "test"})}"',
            False,
        ),
        (
            f'"{json.dumps({"test": 12, "other": "test"})}"',
            False,
            False,
            False,
        ),
    ],
)
def test_parse_communication_interface_json(
    j, additional_values, json_object, error_j, dir_save_location, capsys
):
    args = []
    args.extend([
        f"-r{str(uuid.uuid4())}",
        f"-e{str(uuid.uuid4())}",
        f"-l{str(dir_save_location)}",
    ])
    if j:
        args.extend(["-j", j])
    elif additional_values:
        args.extend(["--additional_values", additional_values])
    elif json_object:
        args.extend(["--json_object", json_object])
    if error_j:
        try:
            spor_com_parse_arguments(args)
        except SystemExit:
            pass
        err = capsys.readouterr().err
        assert error_j in err
        return
    argparse_ret = spor_com_parse_arguments(args)
    if json_object:
        assert argparse_ret.json_object == {"test": 12, "other": "test"}
    elif additional_values:
        assert argparse_ret.json_object == {"test": 12, "other": "test"}
    elif j:
        assert argparse_ret.json_object == {"test": 12, "other": "test"}
    else:
        assert argparse_ret.json_object is None
