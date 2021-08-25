from pydantic import BaseModel
from typing import Literal, Union


class Agent(BaseModel):
    pass

class PPOAgent(Agent):
    type: Literal["PPO"]

class DQNAgent(Agent):
    type: Literal["DQN"]

class baseC(BaseModel):
    agent: Union[PPOAgent, DQNAgent]

example = {
    "agent":{
        "type": "DQN"
    }
}

ba = baseC(**example)

print(type(ba.agent))

example1 = {
    "agent":{
        "type": "PPO"
    }
}

ba1 = baseC(**example1)

print(type(ba1.agent))

