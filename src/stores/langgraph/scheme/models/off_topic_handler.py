from pydantic import BaseModel, Field
from typing import Optional

class OffTopicHandlerInput(BaseModel):
    """Input model for off-topic handling."""
    uploaded_image_bytes: Optional[bytes] = Field(description="Optional image data if provided", default=None)

class OffTopicHandlerOutput(BaseModel):
    """Output model for off-topic handling."""
    final_response: str = Field(description="Response for off-topic query")
    uploaded_image_bytes: Optional[bytes] = Field(description="Processed image data", default=None) 