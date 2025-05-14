from pydantic import BaseModel, Field
from typing import Optional

class FinalOutputPreparerInput(BaseModel):
    """Input model for final output preparation."""
    final_explanation: Optional[str] = Field(description="Final explanation if any", default=None)
    final_response: Optional[str] = Field(description="Existing final response if any", default=None)

class FinalOutputPreparerOutput(BaseModel):
    """Output model for final output preparation."""
    final_response: str = Field(description="Final response to be returned to user") 