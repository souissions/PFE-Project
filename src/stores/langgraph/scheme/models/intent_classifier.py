from pydantic import BaseModel, Field
from typing import Literal

class IntentQuery(BaseModel):
    """Model to classify a user query based on the user's intent."""
    intent: Literal['SYMPTOM_TRIAGE', 'MEDICAL_INFORMATION_REQUEST', 'OFF_TOPIC'] = Field(
        ...,
        description="Classifies the user's intent based on the latest text and/or image input."
    )