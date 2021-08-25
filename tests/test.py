import pathlib
import json
from SbSOvRL.base_parser import BaseParser
from pprint import pprint


example_file_path = pathlib.Path("../../04-Data/examples/input.json")

example_input = None
print(example_file_path.resolve())
# print(example_file_path.read_text())

if example_file_path.exists() and example_file_path.is_file():
    example_input = json.loads(example_file_path.read_text())
else:
    print(example_file_path.resolve())

# print(example_input)

base = BaseParser(**example_input)

pprint(base.dict(), indent=1)