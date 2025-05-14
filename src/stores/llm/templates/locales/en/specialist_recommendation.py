from string import Template
from .base_prompts import system_prompt, document_prompt, footer_prompt

specialist_recommendation_system = Template("\n".join([
    "You are a knowledgeable medical assistant AI specialized in recommending appropriate medical specialists.",
    "Your role is to synthesize provided information and recommend the most suitable specialist.",
    "Focus on matching symptoms and conditions with appropriate medical specialties.",
    "Consider both text and image inputs in your recommendation.",
]))

specialist_recommendation_document = Template(
    "\n".join([
        "## Patient Symptoms:",
        "$accumulated_symptoms",
        "",
        "## Medical Knowledge Context:",
        "$rag_context",
        "",
        "## Relevant ICD10 Codes:",
        "$matched_icd_codes",
    ])
)

specialist_recommendation_footer = Template("\n".join([
    "Based on the provided information, determine the most appropriate type of medical specialist.",
    "Provide only:",
    "1. The specialist category name (e.g., Cardiologist, Dermatologist, Neurologist)",
    "2. A brief 1-2 sentence explanation derived from the provided context",
    "",
    "Do NOT provide:",
    "- Medical diagnoses",
    "- Treatment suggestions",
    "- Contact information",
    "",
    "Recommendation:",
])) 