# Triage Relevance Check Prompt
# Purpose: Determines if the user's input is about personal medical symptoms or health concerns (i.e., is the conversation relevant for medical triage?).
# Output: "YES" or "NO" (if "NO", the workflow can deflect or ask for more relevant info).

from langchain_core.prompts import PromptTemplate

# --- Triage Relevance Check (Updated for potential image, with few-shot examples) ---
relevance_check_template = """
Analyze the following accumulated symptom description from a user chat. The user might have also provided images during the conversation, although only the text is summarized here.
---
{accumulated_symptoms}
---
Based on the accumulated TEXT description, is this conversation primarily focused on discussing personal medical symptoms, health conditions, or seeking medical guidance related to personal health?
If the text is about pets, hobbies, sports, greetings, general feelings, or anything not related to personal health, answer 'NO'.

IMPORTANT: Only answer 'YES' if the text clearly describes a personal medical symptom, health problem, or a request for medical advice about the user's own health. Otherwise, answer 'NO'.

Examples:
Input: "I have a headache and fever." → YES
Input: "Can you tell me about the history of the stethoscope?" → NO
Input: "I have cats." → NO
Input: "My chest hurts and I feel dizzy." → YES
Input: "I like to play sport." → NO
Input: "I enjoy running." → NO
Input: "I feel happy today." → NO
Input: "What is the capital of France?" → NO
Input: "My stomach hurts after eating." → YES

Respond with only the word 'YES' or 'NO'.
Relevance:"""
relevance_check_prompt = PromptTemplate(
    template=relevance_check_template,
    input_variables=["accumulated_symptoms"]
)