from langchain_core.prompts import PromptTemplate

explanation_refiner_template = """
Vous êtes un assistant IA qui réécrit une explication médicale pour améliorer sa clarté pour un non spécialiste, sur la base de retours spécifiques.

Explication originale :
---
{initial_explanation}
---

Critique / Points à améliorer :
---
{evaluator_critique}
---

Réécrivez l'"Explication originale" pour répondre directement à la "Critique". Vos objectifs :
1.  Remplacer ou fournir des explications simples pour tout terme de jargon mentionné dans la critique.
2.  Simplifier les structures de phrases identifiées comme complexes.
3.  Veiller à ce que le sens principal reste exact et cohérent avec l'original.
4.  N'ajoutez PAS de nouvelles informations médicales, conseils ou opinions.

Ne produisez en sortie que l'explication réécrite.

Explication réécrite :"""
explanation_refiner_prompt = PromptTemplate(
    template=explanation_refiner_template,
    input_variables=["initial_explanation", "evaluator_critique"]
)
