from langchain_core.prompts import PromptTemplate

# --- Basic Deflection Messages ---
# Defined as constants for direct use in nodes
# These are the canonical definitions for use throughout the project

off_topic_response = "Sorry, I am designed to assist with preliminary medical information and symptom discussion. I can't help with topics outside of that scope. Do you have any other symptoms you'd like to discuss?"
irrelevant_triage_response = "Based on our conversation, it doesn't seem like we focused on specific medical symptoms for triage. If you have health concerns you'd like assistance understanding or finding potential specialist types for, please describe them clearly. For any medical advice or diagnosis, please consult a qualified healthcare professional."

rag_relevance_evaluator_template = """
Evaluate whether the following retrieved context is likely sufficient and relevant to directly answer the user's specific question. Focus on direct answerability, not just topic overlap.

Retrieved Context:
---
{rag_context}
---

User Question: {user_query}

Is the context directly relevant and likely sufficient to answer the specific question? Respond with only the word 'YES' or 'NO'.
Relevant & Sufficient:"""

rag_relevance_evaluator_prompt = PromptTemplate(
    template=rag_relevance_evaluator_template,
    input_variables=["rag_context", "user_query"]
)