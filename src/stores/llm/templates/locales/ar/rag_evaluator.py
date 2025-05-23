from langchain_core.prompts import PromptTemplate

rag_relevance_evaluator_template = """
قيّم ما إذا كان السياق المسترجع أدناه ذا صلة ومحتمل أن يكون كافيًا للإجابة مباشرة على سؤال المستخدم المحدد. ركز على إمكانية الإجابة المباشرة، وليس فقط على تداخل الموضوع.

السياق المسترجع:
---
{rag_context}
---

سؤال المستخدم: {user_query}

هل السياق ذو صلة مباشرة ومحتمل أن يكون كافيًا للإجابة على السؤال المحدد؟ أجب فقط بكلمة 'نعم' أو 'لا'.
ذو صلة وكافٍ:"""

rag_relevance_evaluator_prompt = PromptTemplate(
    template=rag_relevance_evaluator_template,
    input_variables=["rag_context", "user_query"]
)
