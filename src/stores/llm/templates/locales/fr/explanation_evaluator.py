from string import Template
from langchain_core.prompts import PromptTemplate

explanation_evaluator_template = """
Vous êtes une IA qui évalue l'explication d'une autre IA. Votre objectif est de juger si l'explication est claire, simple et évite le jargon inexpliqué pour un public général (non spécialiste).

Explication à évaluer :
---
{initial_explanation}
---

Critères d'évaluation :
1.  **Clarté & Simplicité :** Le langage est-il facile à comprendre ? Les phrases sont-elles raisonnablement courtes et directes ?
2.  **Jargon :** Y a-t-il des termes médicaux qu'un non spécialiste ne comprendrait probablement pas ? (ex : 'idiopathique', 'bradycardie', 'sténose', noms de tests spécifiques sans contexte).

Tâche d'évaluation :
Examinez l'explication.
- Si elle répond aux critères (claire, simple, sans jargon inexpliqué), répondez uniquement par 'OK'.
- Si elle doit être améliorée, répondez par 'REVOIR' suivi d'une brève critique spécifique indiquant le jargon ou les parties peu claires. Concentrez-vous uniquement sur la clarté et le jargon, pas sur l'exactitude médicale.

Exemples de sortie :
OK
REVOIR - Utilise le terme 'paresthésie' sans explication.
REVOIR - Structure de phrase complexe. Le terme 'étiologie' est du jargon.

Votre évaluation :"""

explanation_evaluator_prompt = PromptTemplate(
    template=explanation_evaluator_template,
    input_variables=["initial_explanation"]
)
