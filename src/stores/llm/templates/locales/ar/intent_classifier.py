from string import Template
from langchain_core.prompts import PromptTemplate

intent_classifier_template = """
حلل أحدث إدخال للمستخدم، والذي قد يتضمن نصًا و/أو صورة، في سياق سجل المحادثة. صنف النية الأساسية لـ *أحدث إدخال للمستخدم*.

سجل المحادثة (قد يتضمن نصًا وإشارات إلى صور):
{conversation_history}

أحدث إدخال نصي للمستخدم: {user_query}
(قد يكون قد تم تقديم صورة مع هذا النص).

النيات الممكنة:
- SYMPTOM_TRIAGE: يصف المستخدم أعراضًا طبية أو شكاوى صحية (نصًا أو صورة) أو يطلب المساعدة في تحديد مشكلة محتملة. أعطِ الأولوية إذا أظهرت الصورة مشكلة طبية واضحة (طفح جلدي، إصابة، إلخ) حتى لو كان النص قليلًا.
- OFF_TOPIC: الاستفسار/الصورة غير مرتبطين بأعراض أو معلومات طبية (مثل التحيات، السياسة، المعرفة العامة، الصور غير الطبية).

صنف نية *أحدث إدخال للمستخدم (نص و/أو صورة)*. أجب بواحد فقط من التصنيفات التالية:
SYMPTOM_TRIAGE
MEDICAL_INFORMATION_REQUEST
OFF_TOPIC
النية:"""

intent_classifier_prompt = PromptTemplate(
    template=intent_classifier_template,
    input_variables=["conversation_history", "user_query"]
)
