from langchain_core.prompts import PromptTemplate

rag_relevance_evaluator_template = """
Évaluez si le contexte récupéré ci-dessous est pertinent et probablement suffisant pour répondre directement à la question spécifique de l'utilisateur. Concentrez-vous sur la possibilité de répondre directement, pas seulement sur le chevauchement de sujet.

Contexte récupéré :
---
{rag_context}
---

Question de l'utilisateur : {user_query}

Le contexte est-il directement pertinent et probablement suffisant pour répondre à la question spécifique ? Répondez uniquement par 'OUI' ou 'NON'.
Pertinent & Suffisant :"""

rag_relevance_evaluator_prompt = PromptTemplate(
    template=rag_relevance_evaluator_template,
    input_variables=["rag_context", "user_query"]
)
