from langchain_core.prompts import PromptTemplate

symptom_sufficiency_template = """
حلل وصف الأعراض المتراكمة الذي قدمه المستخدم. هل يحتوي على تفاصيل كافية (مثل طبيعة العرض، الموقع، الشدة، المدة، البداية، المحفزات، العوامل المرتبطة) لإجراء *تقييم أولي* واقتراح الخطوة التالية؟

ركز على وجود تفاصيل ملموسة، دون تقديم تشخيص. ضع في اعتبارك أنه قد تم تقديم صورة بشكل منفصل، لكن قيم بشكل أساسي مستوى التفاصيل في النص.

الأعراض النصية المتراكمة:
---
{accumulated_symptoms}
---

هل هناك معلومات كافية في الوصف النصي للمتابعة بالتحليل؟ أجب فقط بكلمة 'نعم' أو 'لا'.
معلومات كافية:"""
symptom_sufficiency_prompt = PromptTemplate(
    template=symptom_sufficiency_template,
    input_variables=["accumulated_symptoms"]
)
