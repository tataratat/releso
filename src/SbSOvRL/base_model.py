from pydantic import BaseModel, Extra

class SbSOvRL_BaseModel(BaseModel):

    class Config:
        extra = Extra.forbid