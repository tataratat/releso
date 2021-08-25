from typing import Union, List
from pydantic import BaseModel
from pydantic.class_validators import validator
from pydantic.types import confloat

class B(BaseModel):
    number: float


class Blub(BaseModel):
    a: List[Union[B, confloat(ge=-12)]]

    @validator("a", each_item=True)
    @classmethod
    def checkstuffout(cls, v):
        print(type(v))
        return v if type(v) is not float else B(number=v)

    @validator("a", each_item=True)
    @classmethod
    def checkstuffout_v(cls, v):
        print("second", type(v))
        return v



example = {
    "a": [
        {"number": 12},
        0.,
        {"number": 12},
        12,
        {"number": 12}
    ]
}

asd = Blub(**example)