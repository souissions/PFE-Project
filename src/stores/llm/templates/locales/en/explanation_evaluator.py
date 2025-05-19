from string import Template
from langchain_core.prompts import PromptTemplate

explanation_evaluator_template = """
You are an AI evaluating another AI's explanation. Your goal is to assess if the explanation is clear, simple, and avoids unexplained jargon for a general audience (layperson).

Explanation to Evaluate:
---
{initial_explanation}
---

Evaluation Criteria:
1.  **Clarity & Simplicity:** Is the language easy to understand? Are sentences reasonably short and direct?
2.  **Jargon:** Are there medical terms used that a layperson likely wouldn't know? (e.g., 'idiopathic', 'bradycardia', 'stenosis', specific test names without context).

Evaluation Task:
Review the explanation.
- If it meets the criteria (clear, simple, no unexplained jargon), respond with only the word 'OK'.
- If it needs improvement, respond with 'REVISE' followed by a brief, specific critique pointing out the jargon or unclear parts. Focus only on clarity and jargon, not medical accuracy.

Examples of Output:
OK
REVISE - Uses the term 'paresthesia' without explanation.
REVISE - Sentence structure is complex. The term 'etiology' is jargon.

Your Evaluation:"""

explanation_evaluator_prompt = PromptTemplate(
    template=explanation_evaluator_template,
    input_variables=["initial_explanation"]
)