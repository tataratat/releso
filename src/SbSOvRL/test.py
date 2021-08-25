from solver import Solver
from pydantic import BaseModel
from typing import List


class TestModel(BaseModel):
    test_variable: List[Solver]


example = {
    "test_variable": [
        1.0,
        {
            "current_position": 0.5,
            "min_value": 0.4,
            "max_value": 0.6,
            "discret": True
        }
    ]
}

if __name__ == "__main__":
    t = TestModel(**example)

    for item in t.test_variable:
        print(type(item))

