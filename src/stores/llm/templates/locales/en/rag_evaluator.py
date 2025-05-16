from string import Template

rag_relevance_system = Template("\n".join([
    "You are an AI specialized in evaluating the relevance of retrieved context for answering user questions.",
    "Your role is to determine if the provided context is sufficient and directly relevant to answer the specific question.",
    "Focus on direct answerability, not just topic overlap.",
    "Consider both the specificity of the question and the completeness of the context.",
]))

rag_relevance_document = Template(
    "\n".join([
        "## Retrieved Context:",
        "$rag_context",
        "",
        "## User Question:",
        "$user_query",
    ])
)

rag_relevance_footer = Template("\n".join([
    "Evaluate if the context is:",
    "1. Directly relevant to the specific question",
    "2. Sufficient to provide a complete answer",
    "",
    "Respond with only the word 'YES' or 'NO'.",
    "Relevant & Sufficient:",
])) 