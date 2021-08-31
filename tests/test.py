import pathlib
import json
from SbSOvRL.base_parser import BaseParser
from pprint import pprint


example_file_path = pathlib.Path("/home/clemensfricke/repos/sa072021-rl-shape-opti-framework/04-Data/examples/input.json")

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

# testing mesh loading
print(base.environment.mesh.get_mesh())

print(base.environment.spline.get_actions())

print(base.environment.spline.get_spline())

base.environment.step()
print(base.environment._FFD.deformed_mesh)