from pydantic import BaseModel, Field
from typing import Optional

class ExplanationEvaluatorInput(BaseModel):
    """Input model for explanation evaluation."""
    initial_explanation: str = Field(description="The explanation to evaluate")

class ExplanationEvaluatorOutput(BaseModel):
    """Output model for explanation evaluation."""
    evaluator_critique: str = Field(description="Evaluation critique of the explanation")

class ExplanationRefinerInput(BaseModel):
    """Input model for explanation refinement."""
    initial_explanation: str = Field(description="The explanation to refine")
    evaluator_critique: str = Field(description="Critique to guide refinement")

class ExplanationRefinerOutput(BaseModel):
    """Output model for explanation refinement."""
    initial_explanation: str = Field(description="Refined explanation")
    loop_count: int = Field(description="Updated refinement loop counter") 