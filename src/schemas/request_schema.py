from typing import Dict, List

from pydantic import BaseModel, field_validator


class HelloWorldBodySchema(BaseModel):
    text: str
