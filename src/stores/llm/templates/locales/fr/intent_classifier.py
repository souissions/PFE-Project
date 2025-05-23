from string import Template
from langchain_core.prompts import PromptTemplate

intent_classifier_template = """
Analysez la dernière entrée de l'utilisateur, qui peut inclure du texte et/ou une image, dans le contexte de l'historique de la conversation. Classez l'intention principale de la *dernière entrée utilisateur*.

Historique de la conversation (peut inclure du texte et des références d'image) :
{conversation_history}

Dernière entrée utilisateur (texte) : {user_query}
(Une image peut également avoir été fournie simultanément avec cette requête texte).

Intentions possibles :
- SYMPTOM_TRIAGE : L'utilisateur décrit des symptômes médicaux, des plaintes de santé (par texte ou image), ou demande de l'aide pour identifier un problème potentiel. Priorisez ceci si une image montre un problème médical évident (éruption, blessure, etc.) même si le texte est minimal.
- OFF_TOPIC : La requête/l'image de l'utilisateur n'est pas liée à des symptômes ou informations médicales (ex : salutations, politique, culture générale, images non médicales).

Classez l'intention de la *dernière entrée utilisateur (texte et/ou image)*. Répondez avec UN SEUL des labels d'intention :
SYMPTOM_TRIAGE
MEDICAL_INFORMATION_REQUEST
OFF_TOPIC
Intention :"""

intent_classifier_prompt = PromptTemplate(
    template=intent_classifier_template,
    input_variables=["conversation_history", "user_query"]
)
