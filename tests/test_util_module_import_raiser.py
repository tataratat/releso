import pytest

from releso.util.module_import_raiser import ModuleImportRaiser


def test_module_import_raiser_additional_error_message():
    raiser = ModuleImportRaiser("module_name", "additional error message")
    with pytest.raises(ImportError) as excinfo:
        raiser()
    assert "additional error message" in str(excinfo.value)


def test_module_import_raiser():
    raiser = ModuleImportRaiser("module_name")
    # test call functionality
    with pytest.raises(ImportError) as excinfo:
        raiser()
    assert "module_name" in str(excinfo.value)
    # changed call
    with pytest.raises(ImportError):
        raiser("additional error message")
    # getattr
    with pytest.raises(ImportError):
        raiser.test  # noqa: B018
    # setattr
    with pytest.raises(ImportError):
        raiser.test = "hallo"
    # setattr
    with pytest.raises(ImportError):
        raiser["hallo"] = "hallo"
    with pytest.raises(ImportError):
        raiser["hallo"]
