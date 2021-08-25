from SbSOvRL.agent import AgentTypeDefinition
from SbSOvRL.parser_environment import Environment

from pydantic import BaseModel
from typing import Union

class BaseParser(BaseModel):
    agent: AgentTypeDefinition
    environment: Union[Environment]
