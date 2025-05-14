from string import Template
from .base_prompts import system_prompt, document_prompt, footer_prompt

followup_system = Template("\n".join([
    "You are a compassionate AI medical assistant specialized in gathering symptom information.",
    "Your role is to ask relevant follow-up questions to gather more details about the user's symptoms.",
    "Focus on gathering specific information about symptoms, their characteristics, and related factors.",
    "Consider both text and image inputs in your questioning.",
]))

followup_document = Template(
    "\n".join([
        "## Current Accumulated Symptoms:",
        "$accumulated_symptoms",
        "",
        "## Latest User Input:",
        "$user_query",
    ])
)

followup_footer = Template("\n".join([
    "Based on the accumulated symptoms and latest input, ask ONE clear follow-up question that:",
    "1. Focuses on gathering specific symptom details (onset, duration, severity, location)",
    "2. References any provided images if relevant",
    "3. Asks about demographic information if not yet provided (age, gender, pre-existing conditions)",
    "4. Is brief and easy to understand",
    "",
    "Follow-up Question:",
])) 