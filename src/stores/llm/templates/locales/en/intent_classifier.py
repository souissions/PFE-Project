from string import Template
from langchain_core.prompts import PromptTemplate

# --- Intent Classifier Prompt Template (LangChain compatible) ---
intent_classifier_template = """
Analyze the latest user input, which may include text and/or an image, in the context of the conversation history. Classify the primary intent of the *latest user input*.

Conversation History (may include text and image references):
{conversation_history}

Latest User Input Text: {user_query}
(An image may also have been provided simultaneously with this text query).

Possible Intents:
- SYMPTOM_TRIAGE: User is describing medical symptoms, health complaints (textually or visually in an image), or asking for help identifying a potential issue. Prioritize this if an image shows a clear medical concern (rash, injury, etc.) even if text is minimal. Also use SYMPTOM_TRIAGE if the user is providing demographic or follow-up information (such as age, gender, duration, severity, or answers to previous follow-up questions) in response to a previous symptom or follow-up prompt.
- MEDICAL_INFORMATION_REQUEST: User is asking for general medical information, not about their own symptoms.
- OFF_TOPIC: User query/image is unrelated to medical symptoms or information (e.g., greetings, politics, general knowledge, non-medical images).

# Examples:
# Q: "I feel bad" → SYMPTOM_TRIAGE
# Q: "20 female" (after being asked for age/gender) → SYMPTOM_TRIAGE
# Q: "Since yesterday, mild headache" (after a follow-up) → SYMPTOM_TRIAGE
# Q: "What is diabetes?" → MEDICAL_INFORMATION_REQUEST
# Q: "Hello!" → OFF_TOPIC

Classify the intent of the *latest user input (text and/or image)*. Respond with only ONE of the intent labels:
SYMPTOM_TRIAGE
MEDICAL_INFORMATION_REQUEST
OFF_TOPIC
Intent:"""

intent_classifier_prompt = PromptTemplate(
    template=intent_classifier_template,
    input_variables=["conversation_history", "user_query"]
)

# --- PATCH: If the user provides only demographic/follow-up info, ask for a concrete symptom in the next follow-up, but do NOT loop forever. If the user has already provided demographics, next follow-up should be a concrete symptom prompt, and after one more turn, accept sufficiency or escalate to a human/exit gracefully.
# (This logic is implemented in the symptom gathering node, not the intent classifier prompt.)
# The intent classifier prompt is already robust for multi-turn, so no change needed here.

