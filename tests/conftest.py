import builtins
import os
import pathlib

import pytest


@pytest.fixture
def hide_available_import(monkeypatch):
    """Hide the available import from the user.

    This is used to test the import of the available module.

    Author:
        https://stackoverflow.com/a/60229056
    """
    import_orig = builtins.__import__

    def mock_import_available(name, *args, **kwargs):
        # raise RuntimeError(name)
        with open("import.txt", "a") as file:
            file.write(f"{name}\n")
        if name == "splinepy.helpme.ffd":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_available)


@pytest.fixture
def dir_save_location():
    dir_save_location = str(
        (
            pathlib.Path(__file__).parent / "test_save_location_please_delete"
        ).resolve()
    )
    yield dir_save_location
    if os.path.isdir(dir_save_location):
        os.rmdir(dir_save_location)
