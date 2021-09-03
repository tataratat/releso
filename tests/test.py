import pathlib
import json
from SbSOvRL.base_parser import BaseParser
from pprint import pprint
from stable_baselines3.common.env_checker import check_env


example_file_path = pathlib.Path("../../04-Data/examples/input_c.json")

example_input = None
print(example_file_path.resolve())
# print(example_file_path.read_text())

if example_file_path.exists() and example_file_path.is_file():
    example_input = json.loads(example_file_path.read_text())
else:
    print(example_file_path.resolve())

# print(example_input)

base = BaseParser(**example_input)

# base.environment.step(action = "asd")

env = base.environment.get_gym_environment()

print(env.reset())

check_env(env)

# pprint(base.dict(), indent=1)

# # testing mesh loading
# print(base.environment.mesh.get_mesh())

# print(base.environment.spline.get_actions())

# print(base.environment.spline.get_spline())

# # base.environment.step()
# print(base.environment._FFD.deformed_mesh)
