from langchain_core.prompts import PromptTemplate

explanation_refiner_template = """
أنت مساعد ذكاء اصطناعي يعيد صياغة شرح طبي لتحسين وضوحه لغير المتخصصين، بناءً على ملاحظات محددة.

الشرح الأصلي:
---
{initial_explanation}
---

النقد / نقاط التحسين:
---
{evaluator_critique}
---

أعد كتابة "الشرح الأصلي" لمعالجة "النقد" مباشرة. أهدافك:
1.  استبدل أو قدم تفسيرات بسيطة لأي مصطلحات طبية مذكورة في النقد.
2.  بسط تراكيب الجمل المعقدة.
3.  تأكد من أن المعنى الأساسي يبقى دقيقًا ومتسقًا مع الأصل.
4.  لا تضف أي معلومات طبية جديدة أو نصائح أو آراء.

أنتج فقط الشرح المعاد صياغته.

الشرح المعاد صياغته:"""
explanation_refiner_prompt = PromptTemplate(
    template=explanation_refiner_template,
    input_variables=["initial_explanation", "evaluator_critique"]
)
