from pydantic import BaseModel, Field
from typing import Optional

class FinalAnalysisInput(BaseModel):
    """Input model for final analysis."""
    accumulated_symptoms: str = Field(description="Accumulated symptoms text")
    uploaded_image_bytes: Optional[bytes] = Field(description="Optional image data if provided", default=None)

class FinalAnalysisOutput(BaseModel):
    """Output model for final analysis."""
    initial_explanation: str = Field(description="Initial explanation from analysis")
    rag_context: Optional[str] = Field(description="Retrieved RAG context", default=None)
    evaluator_critique: Optional[str] = Field(description="Evaluator critique if any", default=None)
    loop_count: int = Field(description="Refinement loop counter", default=0)
    uploaded_image_bytes: Optional[bytes] = Field(description="Processed image data", default=None)