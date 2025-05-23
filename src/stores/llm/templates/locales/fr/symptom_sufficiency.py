from langchain_core.prompts import PromptTemplate

symptom_sufficiency_template = """
Analysez la description des symptômes accumulés fournie par un utilisateur. Contient-elle suffisamment de détails spécifiques (nature du symptôme, localisation, gravité, durée, apparition, déclencheurs, facteurs associés) pour permettre une *évaluation préliminaire* et suggérer une prochaine étape potentielle ?

Concentrez-vous sur la présence de détails concrets, sans poser de diagnostic. Considérez qu'une image peut avoir été fournie séparément, mais évaluez principalement le niveau de détail du texte.

Symptômes textuels accumulés :
---
{accumulated_symptoms}
---

Y a-t-il suffisamment d'informations spécifiques dans la description textuelle pour procéder à l'analyse ? Répondez uniquement par 'OUI' ou 'NON'.
Information suffisante :"""
symptom_sufficiency_prompt = PromptTemplate(
    template=symptom_sufficiency_template,
    input_variables=["accumulated_symptoms"]
)
