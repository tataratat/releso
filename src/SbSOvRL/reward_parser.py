from typing import Optional, Union, List, Dict, Any
import uuid
from SbSOvRL.base_model import SbSOvRL_BaseModel
from pydantic.class_validators import validator
from pydantic.fields import Field, PrivateAttr
from pydantic.types import UUID4, DirectoryPath, FilePath
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.util.util_funcs import call_commandline, which
import sys
from ast import literal_eval
import logging



class PythonRewardFunction(SbSOvRL_BaseModel):
    working_directory: DirectoryPath = Field(description="Directory in which the reward function script should be called.")
    reward_script: FilePath
    calling_function: Optional[str] = "python"

    _reward_communication_id: UUID4 = PrivateAttr(default=None)

    @validator("reward_script")
    @classmethod
    def check_if_reward_script_is_python(cls, v) -> FilePath:
        if not v.suffix == ".py":
            raise SbSOvRLParserException("PythonRewardFunction", "reward_script", "The script should have the python suffix which the current one does not have.") 
        return v

    @validator("calling_function", always=True)
    @classmethod
    def check_if_calling_function_exists(cls, v) -> Optional[str]:
        """Validates if an existing calling function is used. If python is given but only python3 exists it will use it. Vice versa.

        Args:
            v (string): Calling function
        Raises:
            SbSOvRLParserException: If the calling function is not found and also exchaning python<>python3 can not be used.

        Returns:
            Optional[str]: string of the calling function
        """
        if which(v) is None:
            if v == "python":
                if which("python3") is not None:
                    logging.getLogger("SbSOvRL_parser").warning("Exchanging calling function \"python\" with \"python3\"")
                    return "python3"
            if v == "python3":
                if which("python") is not None:
                    logging.getLogger("SbSOvRL_parser").warning("Exchanging calling function \"python3\" with \"python\"")
                    return "python"
            raise SbSOvRLParserException("PythonRewardFunction", "calling_function", "The calling_function could not be found, please validate that it is a real program.")
        return v
    
    def get_reward(self, additional_parameter: List[str] = [], *kwargs) -> Dict[str, Any]:
        """Calls the reward/output script and reads in the reward and the solver observations.

        Args:
            additional_parameter (List[str], optional): Additional parameters the reward script can use to perform its duty. Defaults to [].

        Returns:
            Dict[str, Any]: Dict in this format: 
            {
                "reward": float,
                "observations": List[float]
            }
        """
        if not self._reward_communication_id:
            self._reward_communication_id = uuid.uuid4()
        additional_function_parameter = " " + " ".join(additional_parameter) + f" --run_id {self._reward_communication_id} "
        exit_code, output = call_commandline(self.calling_function + " " + str(self.reward_script) + additional_function_parameter, self.working_directory, logging.getLogger("SbSOvRL_environment"))
        return literal_eval(output.decode(sys.getdefaultencoding()))


    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)

        logging.getLogger("environment_logger").debug(f"The uuid of the reward function is {__pydantic_self__._reward_communication_id}.")

RewardFunctionTypes = PythonRewardFunction