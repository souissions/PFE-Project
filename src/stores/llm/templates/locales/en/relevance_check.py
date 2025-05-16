from langchain_core.prompts import PromptTemplate

# --- Triage Relevance Check (Updated for potential image, with few-shot examples) ---
relevance_check_template = """
Analyze the following accumulated symptom description from a user chat. The user might have also provided images during the conversation, although only the text is summarized here.
---
{accumulated_symptoms}
---
Based on the accumulated TEXT description, is this conversation primarily focused on discussing personal medical symptoms, health conditions, or seeking medical guidance related to personal health?
If the text is about pets, hobbies, greetings, or anything not related to personal health, answer 'NO'.
Examples:
Input: "I have a headache and fever." → YES
Input: "Can you tell me about the history of the stethoscope?" → NO
Input: "I have cats." → NO
Input: "My chest hurts and I feel dizzy." → YES

Respond with only the word 'YES' or 'NO'.
Relevance:"""
relevance_check_prompt = PromptTemplate(
    template=relevance_check_template,
    input_variables=["accumulated_symptoms"]
)