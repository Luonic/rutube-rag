from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field


class Request(BaseModel):
    question: str = Field(..., title='Question')

class Response(BaseModel):
    answer: str = Field(..., title='Answer')
    class_1: str = Field(..., title='Classifier 1')
    class_2: str = Field(..., title='Classifier 2')

class ResponseWithContext(BaseModel):
    answer: str = Field(..., title='Answer')
    class_1: str = Field(..., title='Classifier 1')
    class_2: str = Field(..., title='Classifier 2')
    contexts: List[str] = Field(..., title='Contexts')

class ValidationError(BaseModel):
    loc: List[Union[str, int]] = Field(..., title='Location')
    msg: str = Field(..., title='Message')
    type: str = Field(..., title='Error Type')

class HTTPValidationError(BaseModel):
    detail: Optional[List[ValidationError]] = Field(None, title='Detail')
