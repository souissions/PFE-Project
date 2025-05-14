from string import Template
from .base_prompts import system_prompt, document_prompt, footer_prompt

intent_classifier_system = Template("\n".join([
    "You are an AI assistant specialized in classifying user intents in medical conversations.",
    "You will analyze user input and conversation history to determine the primary intent.",
    "Focus on identifying if the user is seeking medical symptom triage or general medical information.",
    "Consider both text and image inputs in your analysis.",
]))

intent_classifier_document = Template(
    "\n".join([
        "## Conversation History:",
        "$conversation_history",
        "",
        "## Latest User Input:",
        "$user_query",
    ])
)

intent_classifier_footer = Template("\n".join([
    "Based on the conversation history and latest input, classify the intent into one of these categories:",
    "- SYMPTOM_TRIAGE: User is describing medical symptoms or health complaints",
    "- MEDICAL_INFORMATION_REQUEST: User is asking for general medical information",
    "- OFF_TOPIC: User query is unrelated to medical topics",
    "",
    "Respond with only ONE of the intent labels:",
    "Intent:",
])) 