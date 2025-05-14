from pydantic import BaseModel, Field
from typing import Optional

class InfoRequestHandlerInput(BaseModel):
    """Input model for information request handling."""
    user_query: str = Field(description="The user's information request query")
    uploaded_image_bytes: Optional[bytes] = Field(description="Optional image data if provided", default=None)

class InfoRequestHandlerOutput(BaseModel):
    """Output model for information request handling."""
    final_response: str = Field(description="Response to the information request")
    rag_context: Optional[str] = Field(description="Retrieved RAG context if any", default=None)
    uploaded_image_bytes: Optional[bytes] = Field(description="Processed image data", default=None) 