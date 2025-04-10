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

# --- Initialize Chat History & Cache ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cached_queries" not in st.session_state:
    st.session_state.cached_queries = {}

# --- Display Previous Messages ---
for speaker, message in st.session_state.chat_history:
    with st.chat_message(speaker):
        st.markdown(message)

# --- Helper: Check for irrelevant answers ---
def is_not_found(text):
    return "do not contain" in text.lower() or "not contain any information" in text.lower()

# --- Chat Input ---
if user_input := st.chat_input("Type your medical question here..."):
    st.chat_message("user").markdown(user_input)

    # If query was asked before, reuse the cached response
    if user_input in st.session_state.cached_queries:
        bot_message = st.session_state.cached_queries[user_input]
    else:
        with st.spinner("🔍 Searching medical knowledge base..."):
            try:
                payload = {"text": user_input, "limit": limit}
                response = requests.post(API_URL, json=payload, timeout=15)
                data = response.json()
                results = data.get("results", [])

                valid_answers = [r for r in results if not is_not_found(r["answer"])]

                # Pick best answer (keyword-based priority)
                best = None
                if valid_answers:
                    keyword_hits = [r for r in valid_answers if "symptom" in r["answer"].lower() or "include" in r["answer"].lower()]
                    best = keyword_hits[0] if keyword_hits else valid_answers[0]

                bot_message = (
                    f"**Project {best['project_id']}**\n\n{best['answer']}"
                    if best else
                    "❌ Sorry, none of the medical projects contain information related to your question."
                )

            except Exception as e:
                bot_message = f"⚠️ Error while contacting the RAG API: `{e}`"

        # Cache the result
        st.session_state.cached_queries[user_input] = bot_message

    # Show bot response
    st.chat_message("assistant").markdown(bot_message)

    # Update chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", bot_message))
