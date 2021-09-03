from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel
from pydantic.class_validators import validator
from pydantic.fields import Field
from pydantic.types import DirectoryPath, FilePath
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.util.util_funcs import call_commandline, which
import sys
from ast import literal_eval
from SbSOvRL.util.logger import parser_logger, environment_logger



class PythonRewardFunction(BaseModel):
    working_directory: DirectoryPath = Field(description="Directory in which the reward function script should be called.")
    reward_script: FilePath
    calling_function: Optional[str] = "python"

    @validator("reward_script")
    @classmethod
    def check_if_reward_script_is_python(cls, v) -> FilePath:
        if not v.suffix == ".py":
            raise SbSOvRLParserException("PythonRewardFunction", "reward_script", "The script should have the python suffix which the current one does not have.") 
        return v

    @validator("calling_function", always=True)
    @classmethod
    def check_if_calling_function_exists(cls, v) -> Optional[str]:
        if which(v) is None:
            if v is "python":
                if which("python3") is not None:
                    parser_logger.warning("Exchanging calling function \"python\" with \"python3\"")
                    return "python3"
            if v is "python3":
                if which("python") is not None:
                    parser_logger.warning("Exchanging calling function \"python3\" with \"python\"")
                    return "python"
            raise SbSOvRLParserException("PythonRewardFunction", "calling_function", "The calling_function could not be found, please validate that it is a real program.")
        return v
    
    def get_reward(self, additional_parameter: List[str] = [], *kwargs) -> Dict[str, Any]:
        additional_function_parameter = " " + " ".join(additional_parameter)
        exit_code, output = call_commandline(self.calling_function + " " + str(self.reward_script) + additional_function_parameter, self.working_directory, environment_logger)
        return literal_eval(output.decode(sys.getdefaultencoding()))

RewardFunctionTypes = Union[PythonRewardFunction]