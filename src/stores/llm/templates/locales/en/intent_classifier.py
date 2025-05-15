from string import Template
import re

#### System Prompt ####
system_prompt = Template("\n".join([
    "You are a classification assistant.",
    "Your task is to determine the primary intent of the user's latest input.",
    "The input may include both text and an image.",
    "Use the conversation history for context, if needed.",
    "Only respond with one of the predefined intent labels.",
    "Prioritize SYMPTOM_TRIAGE if the user input (text or image) indicates a medical concern.",
]))

#### Document Prompt ####
chat_history_prompt = Template("\n".join([
    "### Conversation History:",
    "$conversation_history",
    "",
]))

#### Footer Prompt ####
footer_prompt = Template("\n".join([
    "### Latest User Input Text:",
    "$user_query",
    "",
    "(Note: An image may also have been provided along with the text.)",
    "",
    "### Possible Intents:",
    "- SYMPTOM_TRIAGE: User is describing medical symptoms or visual issues (e.g., rash, injury).",
    "- MEDICAL_INFORMATION_REQUEST: User is seeking factual medical information or advice not tied to personal symptoms.",
    "- OFF_TOPIC: User's query/image is unrelated to medical concerns (e.g., general knowledge, greetings).",
    "",
    "Classify the *intent of the latest user input* using ONLY one of the following labels:",
    "SYMPTOM_TRIAGE",
    "MEDICAL_INFORMATION_REQUEST",
    "OFF_TOPIC",
    "",
    "Intent:"
]))

# For compatibility: alias for legacy code expecting 'intent_classifier_system', 'intent_classifier_document', 'intent_classifier_footer'
intent_classifier_system = system_prompt
intent_classifier_document = chat_history_prompt
intent_classifier_footer = footer_prompt

def normalize_intent_output(raw: str) -> str:
    """Robustly extract the intent label from LLM output, inspired by the working logic in your friend's graph_builder.py."""
    raw = raw.strip().upper()
    # Accept only exact valid labels if present
    valid_intents = ["SYMPTOM_TRIAGE", "MEDICAL_INFORMATION_REQUEST", "OFF_TOPIC"]
    for intent in valid_intents:
        # Match as a whole word (not substring)
        if re.search(rf"\b{intent}\b", raw):
            return intent
    # Fallback: if LLM output is not a valid label, default to OFF_TOPIC
    return "OFF_TOPIC"

