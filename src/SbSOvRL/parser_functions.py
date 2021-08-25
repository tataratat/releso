# from SbSOvRL.exceptions import SbSOvRLParserException
# import pathlib


# def validate_file_path_to_absolute(value: str, parent: str, field: str) -> pathlib.Path:
#     """Validates that the given value points to an existing file. Throws

#     Args:
#         value (str): Value that is to be checked.
#         parent (str): Parent to the current field.
#         field (str): Current field for which the value is validated.
#     Raises:
#         SbSOvRLParserException: Error if the path is not valid/path points to directory.

#     Returns:
#         [pathlib.Path]: Absolute path to the given file.
#     """
#     path = pathlib.Path(value)
#     if not path.exists():
#         raise SbSOvRLParserException(parent=parent, item=field, message="The given path does not point to a valid location.")
#     if path.is_dir():
#         raise SbSOvRLParserException(parent=parent, item=field, message="The given path should point to a file and not a directory.")
#     return path.resolve()