
import streamlit as st
import streamlit.components.v1 as components

# ------------------------
# Define symptom mapping
# ------------------------
symptom_map = {
    "head": ["Headache", "Dizziness", "Blurred vision", "Fever"],
    "neck": ["Neck pain", "Stiffness", "Swollen lymph nodes"],
    "chest": ["Chest pain", "Shortness of breath", "Palpitations"],
    "abdomen": ["Abdominal pain", "Bloating", "Nausea"],
    "pelvis": ["Pelvic pain", "Urinary issues"],
    "left_arm": ["Arm pain", "Numbness", "Weakness"],
    "right_arm": ["Arm pain", "Numbness", "Tingling"],
    "left_leg": ["Leg pain", "Swelling", "Cramps"],
    "right_leg": ["Leg pain", "Swelling", "Numbness"],
    "back": ["Back pain", "Muscle stiffness", "Spasms"]
}

# ------------------------
# Load modified SVG
# ------------------------
svg_path = "bodymap_with_ids.svg"
with open(svg_path, "r", encoding="utf-8") as file:
    svg_data = file.read()

# ------------------------
# Inject JS to detect clicks and update a div
# ------------------------
html_code = f"""
<div id='selected_part'>none</div>
{svg_data}
<script>
  const paths = document.querySelectorAll("svg path");
  paths.forEach(path => {{
    path.style.cursor = "pointer";
    path.addEventListener("click", () => {{
      const part = path.id;
      const div = document.getElementById('selected_part');
      div.innerText = part;
    }});
  }});
</script>
"""

# ------------------------
# Render HTML with SVG
# ------------------------
st.title("🧍 Clickable Body Map - Symptom Selector")
selected_part = st.empty()
components.html(html_code, height=600)

# Manual input for now (Streamlit can't read innerText from JS directly)
selected = st.text_input("Enter clicked body part ID (from DevTools or JS):")

# ------------------------
# Show dropdown of symptoms
# ------------------------
if selected and selected in symptom_map:
    st.subheader(f"Symptoms for: {selected.replace('_', ' ').title()}")
    selected_symptoms = st.multiselect(
        "Select related symptoms:",
        symptom_map[selected],
        key="symptoms"
    )
    st.write("✅ Selected symptoms:", selected_symptoms)
else:
    st.info("Click a region on the body map and type the ID above to continue.")
