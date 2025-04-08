import streamlit as st
import requests

# --- Config ---
API_URL = "http://localhost:5000/api/v1/nlp/index/answer/all"
st.set_page_config(page_title="Mini-RAG Medical Chatbot", page_icon="🩺", layout="centered")

# --- Page Title & Info ---
st.title("🤖 RAG Medical Chatbot")
st.markdown("Ask health-related questions and get answers from multiple medical projects. "
            "_Note: This assistant does not replace professional medical advice._")
st.divider()

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    limit = st.slider("🔎 Number of documents to retrieve", 1, 10, 5)

# --- Initialize Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Display Area ---
for speaker, message in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(message)

# --- Chat Input at Bottom ---
if user_input := st.chat_input("Type your medical question here..."):
    # Show user message
    st.chat_message("user").markdown(user_input)

    # Prepare API payload
    payload = {
        "text": user_input,
        "limit": limit
    }

    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()
        results = data.get("results", [])

        def is_not_found(text):
            return "do not contain" in text.lower() or "not contain any information" in text.lower()

        # Filter valid responses
        valid_answers = [r for r in results if not is_not_found(r["answer"])]

        # Select best one
        best = None
        if valid_answers:
            keyword_hits = [r for r in valid_answers if "symptom" in r["answer"].lower() or "include" in r["answer"].lower()]
            best = keyword_hits[0] if keyword_hits else valid_answers[0]

        # Generate response
        if best:
            bot_message = f"**Project {best['project_id']}**\n\n{best['answer']}"
        else:
            bot_message = "❌ Sorry, none of the medical projects contain information related to your question."

    except Exception as e:
        bot_message = f"⚠️ Error: {e}"

    # Show bot message
    st.chat_message("assistant").markdown(bot_message)

    # Update chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", bot_message))
