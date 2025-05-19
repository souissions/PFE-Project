from langchain_core.prompts import PromptTemplate

explanation_refiner_template = """
You are an AI assistant rewriting a medical explanation to improve its clarity for a layperson, based on specific feedback.

Original Explanation:
---
{initial_explanation}
---

Critique / Areas to Improve:
---
{evaluator_critique}
---

Rewrite the 'Original Explanation' to directly address the 'Critique'. Your goals are:
1.  Replace or provide simple explanations for any specific jargon terms mentioned in the critique.
2.  Simplify sentence structures identified as complex.
3.  Ensure the core meaning remains accurate and consistent with the original.
4.  Do NOT add any new medical information, advice, or opinions.

Output only the rewritten explanation.

Rewritten Explanation:"""
explanation_refiner_prompt = PromptTemplate(
    template=explanation_refiner_template,
    input_variables=["initial_explanation", "evaluator_critique"]
)