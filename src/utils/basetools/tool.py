from typing import Callable, Optional, Type
from pydantic import BaseModel


class Tool(BaseModel):
    """
    A unified wrapper for AI agent tools.
    Provides a standard interface with input/output models and a callable runner.
    """

    name: str
    func: Callable
    description: str
    input_model: Optional[Type[BaseModel]] = None
    output_model: Optional[Type[BaseModel]] = None

    def run(self, input_data: BaseModel):
        """
        Call the tool function using a Pydantic input model.
        """
        if self.input_model:
            if not isinstance(input_data, self.input_model):
                raise ValueError(f"Input must be of type {self.input_model.__name__}")
        return self.func(**input_data.dict())
