import builtins
import os
import pathlib

import pytest
import requests


@pytest.fixture
def clean_up_provider():
    def recursive_file_remove(path):
        """Remove a file or directory and its contents.

        Author: GitHubCopilot (14.08.2023)
        """
        if not path.exists():
            return
        if path.is_file():
            path.unlink()
            return
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                recursive_file_remove(child)
        path.rmdir()

    return recursive_file_remove


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


def dir_save_location_path():
    dir_save_location = (
        pathlib.Path(__file__).parent / "test_save_location_please_delete"
    ).resolve()
    return dir_save_location


@pytest.fixture
def dir_save_location():
    path = dir_save_location_path()
    yield path
    if os.path.isdir(path):
        os.rmdir(path)


@pytest.fixture
def load_sample_file(request):
    file_name = request.param
    base_url = "https://raw.githubusercontent.com/tataratat/samples/main/"
    local_path = pathlib.Path(__file__).parent / "samples/"
    local_file = local_path / file_name
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if not local_file.is_file():
        url = base_url + file_name
        # print(f"Downloading {url} to {local_file}")
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Could not download {url}")
        with open(local_file, "wb") as file:
            file.write(response.content)
    return local_file
