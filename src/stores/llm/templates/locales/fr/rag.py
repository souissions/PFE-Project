from string import Template

#### RAG PROMPTS ####

#### System ####
system_prompt = Template("\n".join([
    "Vous êtes un assistant chargé de générer une réponse pour l'utilisateur.",
    "Vous recevrez un ensemble de documents associés à la requête de l'utilisateur.",
    "Vous devez générer une réponse basée sur les documents fournis.",
    "Ignorez les documents qui ne sont pas pertinents pour la requête de l'utilisateur.",
    "Vous pouvez présenter des excuses à l'utilisateur si vous n'êtes pas en mesure de générer une réponse.",
    "Vous devez générer la réponse dans la même langue que la requête de l'utilisateur.",
    "Soyez poli et respectueux envers l'utilisateur.",
    "Soyez précis et concis dans votre réponse. Évitez les informations inutiles.",
]))

#### Document ####
document_prompt = Template(
    "\n".join([
        "## Document n°: $doc_num",
        "### Contenu: $chunk_text",
    ])
)

#### Footer ####
footer_prompt = Template("\n".join([
    "En vous basant uniquement sur les documents ci-dessus, veuillez générer une réponse pour l'utilisateur.",
    "## Question:",
    "$query",
    "",
    "## Réponse:",
]))
