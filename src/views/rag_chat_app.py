import streamlit as st
import requests

# --- Config ---
API_URL_TEMPLATE = "http://localhost:5000/api/v1/nlp/index/answer/1"  # Default project_id is 1
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

# --- Chat Input ---
if user_input := st.chat_input("Type your medical question here..."):
    st.chat_message("user").markdown(user_input)

    # If query was asked before, reuse the cached response
    if user_input in st.session_state.cached_queries:
        bot_message = st.session_state.cached_queries[user_input]
    else:
        with st.spinner("🔍 Searching medical knowledge base..."):
            try:
                # Construct the API URL (using default project_id=1 for all files)
                api_url = API_URL_TEMPLATE  # Always use the default project_id (1)

                # Prepare the payload for the API request
                payload = {"text": user_input, "limit": limit}
                response = requests.post(api_url, json=payload, timeout=15)

                # Check if the response is valid (status code 200)
                if response.status_code == 200:
                    data = response.json()

                    # Debugging: Show full API response in Streamlit
                    st.write("Full API Response:", data)  # Display the full response to understand structure

                    # Extract the answer directly from the response
                    answer = data.get("answer", "No answer found.")
                    bot_message = f"**Answer:** {answer}"

                else:
                    # Handle API response error
                    bot_message = f"⚠️ Error: Received unexpected status code {response.status_code} from the RAG API."

            except Exception as e:
                # Handle exception during API call
                bot_message = f"⚠️ Error while contacting the RAG API: `{e}`"

        # Cache the result for future use
        st.session_state.cached_queries[user_input] = bot_message

    # Show bot response
    st.chat_message("assistant").markdown(bot_message)

    # Update chat history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("assistant", bot_message))

# --- Reset Button (Optional) ---
if st.button("Clear Chat"):
    st.session_state.chat_history.clear()
    st.session_state.cached_queries.clear()
    st.experimental_rerun()  # Reload the page
