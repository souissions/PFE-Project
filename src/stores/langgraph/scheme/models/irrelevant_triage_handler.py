from pydantic import BaseModel, Field
from typing import Optional

class IrrelevantTriageHandlerInput(BaseModel):
    """Input model for irrelevant triage handling."""
    uploaded_image_bytes: Optional[bytes] = Field(description="Optional image data if provided", default=None)

class IrrelevantTriageHandlerOutput(BaseModel):
    """Output model for irrelevant triage handling."""
    final_response: str = Field(description="Response for irrelevant triage")
    rag_context: str = Field(description="Placeholder for RAG context")
    matched_icd_codes: str = Field(description="Placeholder for ICD codes")
    uploaded_image_bytes: Optional[bytes] = Field(description="Processed image data", default=None) 