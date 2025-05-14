from string import Template
from .base_prompts import system_prompt, document_prompt, footer_prompt

relevance_check_system = Template("\n".join([
    "You are an AI assistant specialized in evaluating the medical relevance of conversations.",
    "Your role is to determine if the conversation is focused on personal medical symptoms and health conditions.",
    "Focus on identifying if the discussion is primarily about seeking medical guidance.",
    "Consider both text and image inputs in your evaluation.",
]))

relevance_check_document = Template(
    "\n".join([
        "## Accumulated Symptom Description:",
        "$accumulated_symptoms",
    ])
)

relevance_check_footer = Template("\n".join([
    "Based on the accumulated description, determine if this conversation is primarily focused on:",
    "1. Personal medical symptoms",
    "2. Health conditions",
    "3. Seeking medical guidance related to personal health",
    "",
    "Respond with only the word 'YES' or 'NO'.",
    "Relevance:",
]))

symptom_sufficiency_system = Template("\n".join([
    "You are an AI assistant specialized in evaluating the completeness of symptom descriptions.",
    "Your role is to determine if there is enough specific detail to make a preliminary assessment.",
    "Focus on identifying if key symptom characteristics are present.",
    "Consider both text and image inputs in your evaluation.",
]))

symptom_sufficiency_document = Template(
    "\n".join([
        "## Accumulated Symptom Description:",
        "$accumulated_symptoms",
    ])
)

symptom_sufficiency_footer = Template("\n".join([
    "Based on the accumulated description, determine if there is enough specific detail about:",
    "1. Nature of symptoms",
    "2. Location",
    "3. Severity",
    "4. Duration",
    "5. Onset",
    "6. Triggers",
    "7. Associated factors",
    "",
    "Respond with only the word 'YES' or 'NO'.",
    "Sufficient Information:",
])) 