# app_agentic.py

import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import nltk
import pandas as pd
from dotenv import load_dotenv
import traceback
from typing import Dict, List, Optional, Any # For type hinting
from PIL import Image # Needed to check image type/validity
import io # Needed to read image bytes
import requests
import json
from stores.langgraph.scheme.state import AgentState
from stores.langgraph.utils import (
    load_embedding_model, load_llm,
    load_dataframes
)
from stores.langgraph.graph import Graph

# --- DB, Vector, LLM, Embedding, and Template Client Setup for In-Process RAG ---
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser

def get_db_client():
    settings = get_settings()
    db_url = getattr(settings, 'DATABASE_URL', None) or os.getenv("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL not set in environment or config.")
        return None
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return async_session

def get_clients():
    settings = get_settings()
    db_client = get_db_client()
    llm_provider_factory = LLMProviderFactory(config=settings)
    vectordb_provider_factory = VectorDBProviderFactory(config=settings, db_client=db_client)
    generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    vectordb_client = vectordb_provider_factory.create(provider=settings.VECTOR_DB_BACKEND)
    template_parser = TemplateParser(language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG)
    return {
        "db_client": db_client,
        "generation_client": generation_client,
        "embedding_client": embedding_client,
        "vectordb_client": vectordb_client,
        "template_parser": template_parser,
    }

async def load_project(project_id, db_client):
    from models.ProjectModel import ProjectModel
    project_model = await ProjectModel.create_instance(db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    return project

# --- Must be first Streamlit command ---
st.set_page_config(page_title="AI Symptom checker", layout="wide")

# --- Load Environment Variables ---
load_dotenv()
BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:5000")

# --- Now safe to import custom modules & utils ---


# --- NLTK Download ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' tokenizer...")
    nltk.download('punkt', quiet=True)
except Exception as e:
     print(f"Could not check/download NLTK data: {e}")

# --- Load Global Components ---
print("Loading application components...")
embedding_model = load_embedding_model()
llm = load_llm()

doctor_df_g, cases_df_g = load_dataframes()
print("Finished loading application components.")

# --- Build LangGraph Agent ---
@st.cache_resource
def get_compiled_graph():
    """Builds or retrieves the compiled LangGraph agent."""
    print("Attempting to build/retrieve compiled LangGraph agent...")
    if llm and embedding_model:
        try:
            # Instantiate GraphFlow and Graph, then build
            from stores.langgraph.graphFlow import GraphFlow
            graph_flow = GraphFlow(AgentState())
            graph = Graph(graph_flow)
            compiled_app = graph.build()
            if compiled_app:
                print("LangGraph agent compiled successfully.")
                return compiled_app
            else:
                st.error("Graph building function returned None.")
                return None
        except Exception as e:
            st.error(f"Failed to build the agent workflow graph: {e}")
            print(f"CRITICAL: Graph build failed: {e}")
            traceback.print_exc()
            return None
    else:
        missing_comps = [name for comp, name in zip(
            [llm, embedding_model],
            ['LLM', 'Embeddings']
        ) if comp is None]
        error_msg = f"Core components ({', '.join(missing_comps)}) failed to load earlier. Agent cannot be built."
        st.error(error_msg)
        print(error_msg)
        return None

graph_app = get_compiled_graph()

# --- Streamlit UI Elements ---
st.title("🩺 AI Symptom Checker")

#st.warning("Disclaimer: This is a prototype AI for informational purposes only. It cannot provide diagnosis or medical advice. Visual analysis is experimental. Always consult with a qualified healthcare professional.")

# --- Initialize Streamlit Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
    initial_greeting = "Hi! I'm your pre-consultation assistant, here to help your doctor better understand your condition!"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

if 'agent_final_state' not in st.session_state:
    st.session_state.agent_final_state = None

if 'agent_input_history' not in st.session_state:
    st.session_state.agent_input_history = []
    if st.session_state.messages and st.session_state.messages[0]['role'] == 'assistant':
        st.session_state.agent_input_history.append(st.session_state.messages[0])

# --- Display Chat Interface ---
chat_container = st.container(height=450, border=True)
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display text content
            if isinstance(message["content"], str):
                 st.markdown(message["content"])
            # Display image if message content indicates one (e.g., user uploaded)
            #elif isinstance(message["content"], dict) and message["content"].get("type") == "image_display":
                 #st.image(message["content"]["image_bytes"], caption="Uploaded Image", width=200)
                 #if message["content"].get("text"): # Display accompanying text if any
                     #st.markdown(message["content"]["text"])

# --- Separate Display Area for Results (Table, Expander) ---
results_container = st.container()

# --- User Input Area ---
# Use columns for better layout of text input and file uploader
input_col, upload_col = st.columns([4, 1])

with input_col:
    prompt = st.chat_input("Describe symptoms or ask a question...")

#with upload_col:
    # Add file uploader for images
    #uploaded_file = st.file_uploader(
        #"Upload Image (Optional)",
        #type=["png", "jpg", "jpeg", "webp"],
        #key="file_uploader" # Assign a key
    #)

# --- Agent Execution Logic (Triggered by User Text Input OR File Upload + Text) ---
# We trigger when the text prompt is submitted
if prompt:
    user_text_input = prompt
    user_image_bytes = None
    image_caption = None

    # Process uploaded file if it exists when prompt is submitted
    #if uploaded_file is not None:
        #try:
            # Read image bytes
            #user_image_bytes = uploaded_file.getvalue()
            # Optional: Validate image format/size
            #img = Image.open(io.BytesIO(user_image_bytes))
            #print(f"Image uploaded: {uploaded_file.name}, size: {len(user_image_bytes)} bytes, format: {img.format}")
            #image_caption = f"Image: {uploaded_file.name}" # For display history
            # Clear the uploader after processing by resetting its key or form
            # This requires more complex state management or using st.form
            # Simple approach: User needs to manually clear it if needed.
        #except Exception as e_img:
            #st.error(f"Error processing uploaded image: {e_img}")
            #user_image_bytes = None # Don't proceed with corrupted image

    # 1. Add user input to display history (handle text + optional image)
    if user_image_bytes:
        # Store as a dict for special display handling
        display_content = {
            "type": "image_display",
            "text": user_text_input,
            "image_bytes": user_image_bytes
        }
        st.session_state.messages.append({"role": "user", "content": display_content})
        # For agent history, represent image existence conceptually or pass bytes later
        # Let's add text + image marker to agent history
        #st.session_state.agent_input_history.append({"role": "user", "content": f"{user_text_input} [Image Uploaded: {uploaded_file.name}]"})
        st.session_state.agent_input_history.append({"role": "user", "content": f"{user_text_input} "})

    
    else:
        # Just add text
        st.session_state.messages.append({"role": "user", "content": user_text_input})
        st.session_state.agent_input_history.append({"role": "user", "content": user_text_input})

    # 2. Display user message immediately (handled by chat history display loop)

    # 3. Check if the agent graph is ready
    if not graph_app:
        error_msg = "Sorry, the AI agent is currently unavailable."
        with chat_container:
             with st.chat_message("assistant"): st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.agent_final_state = None
        st.rerun()
    else:
        # 4. Prepare the input state for the graph
        current_agent_history = st.session_state.agent_input_history
        accumulated_symptoms_str = "\n".join([ str(msg.get('content','')) for msg in current_agent_history ]) # Simple concat

        initial_graph_input: AgentState = {
            "conversation_history": current_agent_history,
            "user_query": user_text_input,
            "uploaded_image_bytes": user_image_bytes, # Pass image bytes for this turn
            "image_prompt_text": user_text_input if user_image_bytes else None, # Text associated with image
            "accumulated_symptoms": accumulated_symptoms_str,
            "loop_count": 0,
            "user_intent": None, "is_relevant": None, "rag_context": None,
            "matched_icd_codes": None, "initial_explanation": None,
            "evaluator_critique": None, "final_explanation": None,
            "recommended_specialist": None, "doctor_recommendations": None,
            "no_doctors_found_specialist": None, "final_response": None
        }

        # --- Build or get project and NLPService for in-process RAG ---
        from services.NLPService import NLPService
        clients = get_clients()
        project_id = 1  # or user-selected
        db_client = clients["db_client"]
        project = None
        if db_client is not None:
            try:
                import sys
                if sys.platform == "win32":
                    project = asyncio.run(load_project(project_id, db_client))
                else:
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                        loop = asyncio.get_event_loop()
                        project = loop.run_until_complete(load_project(project_id, db_client))
                    except Exception:
                        project = asyncio.run(load_project(project_id, db_client))
            except Exception as e:
                print(f"Error loading project: {e}")
                st.error(f"Error loading project: {e}")
        else:
            st.warning("DB client not available. RAG will not use real project context.")
        nlp_service = NLPService(
            vectordb_client=clients["vectordb_client"],
            generation_client=clients["generation_client"],
            embedding_client=clients["embedding_client"],
            template_parser=clients["template_parser"]
        )
        initial_graph_input["project"] = project
        initial_graph_input["nlp_service"] = nlp_service

        # 5. Execute the agent graph (in-process, not HTTP)
        final_state: Optional[Dict[str, Any]] = None
        with st.spinner("Processing your request..."):
            try:
                # Use the compiled LangGraph app directly
                if graph_app:
                    import asyncio
                    # Run the graph async function
                    if sys.platform == "win32":
                        final_state = asyncio.run(graph_app.ainvoke(initial_graph_input))
                    else:
                        try:
                            import nest_asyncio
                            nest_asyncio.apply()
                            loop = asyncio.get_event_loop()
                            final_state = loop.run_until_complete(graph_app.ainvoke(initial_graph_input))
                        except Exception:
                            final_state = asyncio.run(graph_app.ainvoke(initial_graph_input))
                    agent_response = final_state.get("final_response") if final_state else None
                else:
                    error_msg = "Sorry, the AI agent is currently unavailable."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.agent_final_state = None
                    st.rerun()

                st.session_state.agent_final_state = final_state
                if agent_response:
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})
                    st.session_state.agent_input_history.append({"role": "assistant", "content": agent_response})
                else:
                    error_msg = "No response received from the AI."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                print(f"Error processing request: {e}")
                traceback.print_exc()
                error_msg = f"An error occurred while processing your request: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.agent_final_state = None

        # 6. Extract primary text response for chat history
        agent_response = final_state.get("final_response") if final_state else None

        # --- NEW: Handle symptom sufficiency follow-up ---
        needs_more_symptom_detail = final_state.get("needs_more_symptom_detail") if final_state else False
        followup_message = final_state.get("followup_message") if final_state else None
        if needs_more_symptom_detail and followup_message:
            st.session_state.messages.append({"role": "assistant", "content": followup_message})
            st.session_state.agent_input_history.append({"role": "assistant", "content": followup_message})

        # Add agent's response (or error) to display history
        if agent_response:
            # Only append if not a duplicate of the last message
            if not (st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] == agent_response):
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                st.session_state.agent_input_history.append({"role": "assistant", "content": agent_response})
        elif graph_app and final_state is None: # Error during invoke
             error_msg = f"An error occurred during processing: Check logs." # Already printed traceback
             st.session_state.messages.append({"role": "assistant", "content": error_msg})
             # Don't add error to agent_input_history
        elif graph_app and final_state is not None and not agent_response: # Graph ran but no response text
            fallback_response = "Processing complete. Please let me know if there's anything else."
            print("Warning: Agent finished but no 'final_response' in state. Using fallback.")
            st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            st.session_state.agent_input_history.append({"role": "assistant", "content": fallback_response})
        # Else: Error case where graph_app itself was None already handled

        # 7. Rerun Streamlit to update chat display and show results below
        st.rerun()


# --- Display Results Section (Reads from stored final state) ---
if st.session_state.agent_final_state:
    final_state_to_display = st.session_state.agent_final_state
    with results_container:
        # Display Doctor Recommendations Table if available
        

        # Display "No Doctors Found" Message
        

        # Display Context Expander
        rag_context = final_state_to_display.get("rag_context")
        accumulated_symptoms_display = final_state_to_display.get("accumulated_symptoms")

        # Determine if there's relevant context/codes to show
        show_expander = False
        # Check if rag_context is valid string and not an error/NA message from the tool itself
        if rag_context and isinstance(rag_context, str) and not rag_context.startswith(("N/A", "Error:")) and rag_context != "No relevant documents found.":
            show_expander = True

        if show_expander:
            with st.expander("🔍 Show Analysis Details (Context)", expanded=False):
                st.markdown("**Symptoms Used for Analysis (Accumulated Text):**")
                st.text(accumulated_symptoms_display.replace("[User provided an image]", "").strip() if accumulated_symptoms_display else "N/A")
                st.markdown("**Retrieved Context from Knowledge Base:**")
                if rag_context and isinstance(rag_context, str) and not rag_context.startswith(("N/A", "Error:")) and rag_context != "No relevant documents found.":
                    context_blocks = rag_context.split("\n=====\n")
                    for i, block in enumerate(context_blocks):
                        source_line = ""
                        content_text = block
                        if block.startswith("Source:") and "\n---\n" in block:
                            parts = block.split("\n---\n", 1)
                            source_line = parts[0].strip()
                            content_text = parts[1].strip() if len(parts) > 1 else ""
                        st.markdown(f"--- **Chunk {i+1}** ({source_line}) ---")
                        st.text_area(f"chunk_content_{i}", value=content_text, height=150, disabled=True, key=f"expander_rag_chunk_{i}", label_visibility="collapsed")
                else:
                    st.text(rag_context if rag_context else "N/A")

# --- Sidebar Elements ---
st.sidebar.header("About")
st.sidebar.info("AI Symptom Checker using LangGraph and FastAPI")
st.sidebar.header("Data Sources")
st.sidebar.markdown("- Medical knowledge base")

# --- Restart Button ---
if st.sidebar.button("🔄 Restart Conversation"):
    print("Restarting conversation...")
    keys_to_clear = ['messages', 'agent_final_state', 'agent_input_history']
    for key in keys_to_clear:
        if key in st.session_state: del st.session_state[key]
    # Re-initialize empty lists and add greeting
    st.session_state.messages = []
    st.session_state.agent_input_history = []
    initial_greeting = "Hi! I'm your pre-consultation assistant, here to help your doctor better understand your condition!"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
    st.session_state.agent_input_history.append({"role": "assistant", "content": initial_greeting})
    st.rerun()

