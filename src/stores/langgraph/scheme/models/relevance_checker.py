from pydantic import BaseModel, Field
from typing import Optional

class RelevanceCheckerInput(BaseModel):
    """Input model for relevance checking."""
    accumulated_symptoms: str = Field(description="Accumulated symptoms to check for relevance")

class RelevanceCheckerOutput(BaseModel):
    """Output model for relevance checking."""
    is_relevant: bool = Field(description="Whether the symptoms are medically relevant") 