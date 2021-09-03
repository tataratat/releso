from SbSOvRL.agent import AgentTypeDefinition
from SbSOvRL.parser_environment import Environment
from pydantic import BaseModel
from typing import Union
from pydantic.types import conint
import pathlib

class BaseParser(BaseModel):
    agent: AgentTypeDefinition
    environment: Union[Environment]
    number_of_episodes: conint(ge=1)

    def learn(self) -> None:
        agent = self.agent.get_agent(self.environment.get_gym_environment())
        agent.learn(self.number_of_episodes)

    def export_spline(self, file_name: str) -> None:
        self.environment.export_spline(file_name)

    def export_mesh(self, file_name:str) -> None:
        self.environment.export_mesh(file_name)

    def save_model(self, file_name:str) -> None:
        path = pathlib.Path(file_name)
        path.parent.mkdir(parent=True, exists=True)
        self.agent.save(path)
            

