"""Module Raiser optional dependency for modules."""

from typing import Any, Optional


class ModuleImportRaiser:
    """Import error deferrer until it is actually called.

    Class used to have better import error handling in the case that a package
    package is not installed. This is necessary due to that some packages are
    not a dependency of `splinepy`, but some parts require them to function.
    Examples are `splinepy`, `torchvision`, and `imageio`.
    """

    def __init__(
        self, lib_name: str, error_mesg: Optional[str] = None
    ) -> None:
        """Constructor of object of class ModuleImportRaiser.

        Args:
            lib_name (str): Name of the library which can not be loaded. Will
                be inserted into the error message of the deferred import
                Error. Is not checked for correctness.
            error_mesg (Optional[str], optional): Original error msg. Defaults
                to None.
        """
        self._message = str(
            "Parts of the requested functionality in ReLeSO depend on the "
            f"external `{lib_name}` package which could not be found on "
            "your system. Please refer to the installation instructions "
            "for more information."
            f"{f'Original error message {error_mesg}' if error_mesg else ''}"
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Dummy method for object(args, kwargs).

        Is called when the object is called by object(). Will notify the user,
        that the functionality is not accessible and how to proceed to access
        the functionality.
        """
        raise ImportError(self._message)

    def __getattr__(self, __name: str) -> Any:
        """Dummy method for object.__name.

        Is called when any attribute of the object is accessed by object.attr.
        Will notify the user, that the functionality is not accessible and how
        to proceed to access the functionality.
        """
        # if __name == "_ModuleImportRaiser__message":
        #     return object.__getattr__(self, __name[-8:])
        # else:
        raise ImportError(self._message)

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Dummy method for object.__name = __value.

        Is called when any attribute of the object is set by object.attr = new.
        Will notify the user, that the functionality is not accessible and how
        to proceed to access the functionality.
        """
        if __name == "_message":
            object.__setattr__(self, __name, __value)
        else:
            raise ImportError(self._message)

    def __getitem__(self, key):
        """Dummy method for object[key].

        Is called when the object is subscripted object[x]. Will notify the
        user, that the functionality is not accessible and how to proceed to
        access the functionality.
        """
        raise ImportError(self._message)

    def __setitem__(self, key, value):
        """Dummy method for object[key] = value.

        Is called when the object is subscripted object[x]. Will notify the
        user, that the functionality is not accessible and how to proceed to
        access the functionality.
        """
        raise ImportError(self._message)
