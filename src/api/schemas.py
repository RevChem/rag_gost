from typing import Literal
from pydantic import BaseModel


class AskWithAIResponse(BaseModel):
    response: str
    provider: Literal["deepseek", "gigachat"] = "deepseek"
