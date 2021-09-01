from SbSOvRL.agent import AgentTypeDefinition
from SbSOvRL.parser_environment import Environment
from pydantic import BaseModel
from typing import Union
from pydantic.types import conint

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
