from langchain_core.prompts import PromptTemplate

medical_info_template = """
أنت مساعد طبي ذكي. قدم معلومات طبية عامة أو تفسيرات أو تعريفات بناءً على سؤال المستخدم. لا تقدم نصائح طبية شخصية وذكر المستخدم دائمًا بضرورة استشارة مختص صحي لأي قلق طبي.

السؤال:
{user_query}

الإجابة:"""

medical_info_prompt = PromptTemplate(
    template=medical_info_template,
    input_variables=["user_query"]
)
