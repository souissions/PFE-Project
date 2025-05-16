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
- SYMPTOM_TRIAGE: User is describing medical symptoms, health complaints (textually or visually in an image), or asking for help identifying a potential issue. Prioritize this if an image shows a clear medical concern (rash, injury, etc.) even if text is minimal.
- OFF_TOPIC: User query/image is unrelated to medical symptoms or information (e.g., greetings, politics, general knowledge, non-medical images).

Classify the intent of the *latest user input (text and/or image)*. Respond with only ONE of the intent labels:
SYMPTOM_TRIAGE
MEDICAL_INFORMATION_REQUEST
OFF_TOPIC
Intent:"""

intent_classifier_prompt = PromptTemplate(
    template=intent_classifier_template,
    input_variables=["conversation_history", "user_query"]
)

