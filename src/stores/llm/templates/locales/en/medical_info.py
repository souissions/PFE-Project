from langchain_core.prompts import PromptTemplate

medical_info_template = """
You are an AI assistant providing information based *only* on the following context documents retrieved from a medical knowledge base. Do NOT provide medical advice, opinions, or diagnoses. Do not use any external knowledge beyond the provided context.
Answer the user's question factually based *only* on the text within the context. If the context does not contain the answer, clearly state that the information is not available in the provided documents. Keep the answer concise.

Context Documents:
---
{rag_context}
---

User Question: {user_query}

Answer:"""
medical_info_prompt = PromptTemplate(
    template=medical_info_template,
    input_variables=["rag_context", "user_query"]
)