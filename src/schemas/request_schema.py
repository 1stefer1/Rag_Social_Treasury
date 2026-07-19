from pydantic import BaseModel


class HelloWorldBodySchema(BaseModel):
    text: str
