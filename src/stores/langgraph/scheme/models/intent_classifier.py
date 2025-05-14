from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class IntentClassifierInput(BaseModel):
    """Input model for intent classification."""
    user_query: str = Field(description="The user's query text")
    conversation_history: List[Dict[str, str]] = Field(description="Previous conversation messages")
    uploaded_image_bytes: Optional[bytes] = Field(description="Optional image data if provided", default=None)

class IntentClassifierOutput(BaseModel):
    """Output model for intent classification."""
    user_intent: str = Field(description="Classified intent: SYMPTOM_TRIAGE, MEDICAL_INFORMATION_REQUEST, or OFF_TOPIC")
    uploaded_image_bytes: Optional[bytes] = Field(description="Processed image data", default=None)
    is_relevant: Optional[bool] = Field(description="Relevance flag", default=None)
    rag_context: Optional[str] = Field(description="RAG context if any", default=None)
    matched_icd_codes: Optional[str] = Field(description="Matched ICD codes if any", default=None)
    initial_explanation: Optional[str] = Field(description="Initial explanation if any", default=None)
    evaluator_critique: Optional[str] = Field(description="Evaluator critique if any", default=None)
    loop_count: int = Field(description="Refinement loop counter", default=0)
    final_explanation: Optional[str] = Field(description="Final explanation if any", default=None)
    recommended_specialist: Optional[str] = Field(description="Recommended specialist if any", default=None)
    doctor_recommendations: Optional[Any] = Field(description="Doctor recommendations if any", default=None)
    no_doctors_found_specialist: Optional[str] = Field(description="Specialist with no doctors found", default=None)
    final_response: Optional[str] = Field(description="Final response if any", default=None) 