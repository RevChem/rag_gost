from typing import Literal
from pydantic import BaseModel


class AskWithAIResponse(BaseModel):
    response: str
    category: str | None = None
    provider: Literal["deepseek", "gigachat"] = "deepseek"
