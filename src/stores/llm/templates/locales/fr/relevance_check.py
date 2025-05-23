from langchain_core.prompts import PromptTemplate

relevance_check_template = """
Analysez la description des symptômes accumulés d'un utilisateur. L'utilisateur a peut-être aussi fourni des images, mais seul le texte est résumé ici.
---
{accumulated_symptoms}
---
Sur la base de la description TEXTUELLE accumulée, cette conversation porte-t-elle principalement sur des symptômes médicaux personnels, des problèmes de santé ou une demande de conseils médicaux personnels ?
Si le texte concerne des animaux, des loisirs, du sport, des salutations, des sentiments généraux ou tout ce qui n'est pas lié à la santé personnelle, répondez 'NON'.

IMPORTANT : Répondez 'OUI' uniquement si le texte décrit clairement un symptôme médical personnel, un problème de santé ou une demande de conseil médical sur la santé de l'utilisateur. Sinon, répondez 'NON'.

Exemples :
Entrée : "J'ai mal à la tête et de la fièvre." → OUI
Entrée : "Pouvez-vous me parler de l'histoire du stéthoscope ?" → NON
Entrée : "J'ai des chats." → NON
Entrée : "J'ai mal à la poitrine et je me sens étourdi." → OUI
Entrée : "J'aime faire du sport." → NON
Entrée : "J'aime courir." → NON
Entrée : "Je me sens heureux aujourd'hui." → NON
Entrée : "Quelle est la capitale de la France ?" → NON
Entrée : "J'ai mal au ventre après avoir mangé." → OUI

Répondez uniquement par 'OUI' ou 'NON'.
Pertinence :"""
relevance_check_prompt = PromptTemplate(
    template=relevance_check_template,
    input_variables=["accumulated_symptoms"]
)
