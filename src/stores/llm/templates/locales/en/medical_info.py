from string import Template

medical_info_system = Template("\n".join([
    "You are an AI assistant providing information based only on the provided medical knowledge base.",
    "Your role is to answer medical questions factually based on the given context.",
    "Do not provide medical advice, opinions, or diagnoses.",
    "Do not use any external knowledge beyond the provided context.",
]))

medical_info_document = Template(
    "\n".join([
        "## Context Documents:",
        "$rag_context",
    ])
)

medical_info_footer = Template("\n".join([
    "Based only on the above context documents, answer the following question:",
    "## Question:",
    "$user_query",
    "",
    "## Answer:",
])) 