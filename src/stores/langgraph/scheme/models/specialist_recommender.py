from pydantic import BaseModel, Field
from typing import Optional, Any

class SpecialistRecommenderInput(BaseModel):
    """Input model for specialist and doctor recommendation."""
    initial_explanation: str = Field(description="Final explanation containing specialist information")
    accumulated_symptoms: str = Field(description="Accumulated symptoms for doctor matching")

class SpecialistRecommenderOutput(BaseModel):
    """Output model for specialist and doctor recommendation."""
    final_explanation: str = Field(description="Final explanation")
    recommended_specialist: Optional[str] = Field(description="Extracted specialist recommendation", default=None)
    doctor_recommendations: Optional[Any] = Field(description="Recommended doctors DataFrame", default=None)
    no_doctors_found_specialist: Optional[str] = Field(description="Specialist with no doctors found", default=None) 