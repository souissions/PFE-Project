from string import Template
from langchain_core.prompts import PromptTemplate

explanation_evaluator_template = """
أنت ذكاء اصطناعي يقيم شرح ذكاء اصطناعي آخر. هدفك هو تقييم ما إذا كان الشرح واضحًا وبسيطًا ويتجنب المصطلحات الطبية غير المفسرة للجمهور العام (غير المتخصص).

الشرح المراد تقييمه:
---
{initial_explanation}
---

معايير التقييم:
1.  **الوضوح والبساطة:** هل اللغة سهلة الفهم؟ هل الجمل قصيرة ومباشرة؟
2.  **المصطلحات:** هل هناك مصطلحات طبية قد لا يعرفها غير المتخصص (مثل 'مجهول السبب'، 'بطء القلب'، 'تضيق'، أسماء اختبارات محددة بدون شرح)؟

مهمة التقييم:
راجع الشرح.
- إذا استوفى المعايير (واضح، بسيط، بدون مصطلحات غير مفسرة)، أجب فقط بـ 'موافق'.
- إذا كان يحتاج إلى تحسين، أجب بـ 'راجع' متبوعة بنقد موجز يوضح المصطلحات أو الأجزاء غير الواضحة. ركز فقط على الوضوح والمصطلحات، وليس الدقة الطبية.

أمثلة للإخراج:
موافق
راجع - استخدم مصطلح 'تنميل' بدون شرح.
راجع - بنية الجملة معقدة. مصطلح 'مسببات المرض' هو مصطلح طبي.

تقييمك:"""

explanation_evaluator_prompt = PromptTemplate(
    template=explanation_evaluator_template,
    input_variables=["initial_explanation"]
)
