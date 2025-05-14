from string import Template
from .base_prompts import system_prompt, document_prompt, footer_prompt

explanation_evaluator_system = Template("\n".join([
    "You are an AI specialized in evaluating the clarity of medical explanations.",
    "Your role is to assess if explanations are clear and accessible to laypeople.",
    "Focus on identifying unclear language and unexplained medical jargon.",
    "Consider the needs of a general audience in your evaluation.",
]))

explanation_evaluator_document = Template(
    "\n".join([
        "## Explanation to Evaluate:",
        "$initial_explanation",
    ])
)

explanation_evaluator_footer = Template("\n".join([
    "Evaluate the explanation based on:",
    "1. Clarity & Simplicity: Is the language easy to understand? Are sentences reasonably short and direct?",
    "2. Jargon: Are there medical terms used that a layperson likely wouldn't know?",
    "",
    "Respond with:",
    "- 'OK' if the explanation meets the criteria",
    "- 'REVISE' followed by a brief, specific critique if improvements are needed",
    "",
    "Your Evaluation:",
])) 