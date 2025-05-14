from string import Template

off_topic_response = Template("""
Sorry, I am designed to assist with preliminary medical information and symptom discussion. I can't help with topics outside of that scope. Do you have any other symptoms you'd like to discuss?
""")

irrelevant_triage_response = Template("""
Based on our conversation, it doesn't seem like we focused on specific medical symptoms for triage. If you have health concerns you'd like assistance understanding or finding potential specialist types for, please describe them clearly. For any medical advice or diagnosis, please consult a qualified healthcare professional.
""") 