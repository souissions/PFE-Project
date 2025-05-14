from pydantic import BaseModel, Field
from typing import Optional

class SymptomGathererInput(BaseModel):
    """Input model for symptom gathering."""
    accumulated_symptoms: str = Field(description="Previously accumulated symptoms")
    user_query: str = Field(description="Current user query")
    uploaded_image_bytes: Optional[bytes] = Field(description="Optional image data if provided", default=None)

class SymptomGathererOutput(BaseModel):
    """Output model for symptom gathering."""
    accumulated_symptoms: str = Field(description="Updated accumulated symptoms")
    final_response: Optional[str] = Field(description="Follow-up question or response if any", default=None)
    uploaded_image_bytes: Optional[bytes] = Field(description="Processed image data", default=None) 