from langchain_core.prompts import PromptTemplate

symptom_sufficiency_template = """
Analyze the accumulated symptom description provided by a user. Does it contain enough specific detail (e.g., nature of symptom, location, severity, duration, onset, triggers, associated factors) to make a *preliminary assessment* for suggesting a potential next step?

Focus on whether you have concrete details to work with, not on making a diagnosis. Consider that an image might have been provided separately, but evaluate based primarily on the text description's detail level.

Accumulated Text Symptoms:
---
{accumulated_symptoms}
---

Is there enough specific symptom information in the text description to proceed with analysis? Respond with only the word 'YES' or 'NO'.
Sufficient Information:"""
symptom_sufficiency_prompt = PromptTemplate(
    template=symptom_sufficiency_template,
    input_variables=["accumulated_symptoms"]
)
