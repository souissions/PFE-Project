# streamlit_app.py

import sys
import os
from pathlib import Path

# Get absolute paths
current_file = Path(__file__).resolve()
current_dir = current_file.parent
src_dir = current_dir.parent
project_root = src_dir.parent

# Add src directory to Python path if not already there
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Set PYTHONPATH
os.environ['PYTHONPATH'] = str(src_dir)

# Debug prints
print(f"Current file: {current_file}")
print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")

print(f"Project Structure:")
print(f"Current directory: {current_dir}")
print(f"Source directory: {src_dir}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

import logging
import streamlit as st
import nltk
from dotenv import load_dotenv
import traceback
from typing import Dict, List, Optional, Any
from PIL import Image
import io
import requests
import json
import asyncio
import nest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from stores.langgraph.scheme.state import AgentState
from stores.langgraph.utils import load_embedding_model, load_llm
from stores.langgraph.graph import Graph
from helpers.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('streamlit_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Path is already configured at the top of the file

# --- DB, Vector, LLM, Embedding, and Template Client Setup for In-Process RAG ---
def get_db_client():
    settings = get_settings()
    # Try to get DATABASE_URL from environment first
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # If not in environment, try settings
        db_url = getattr(settings, 'DATABASE_URL', None)
    
    if not db_url:
        logger.error("DATABASE_URL not set in environment or config.")
        st.error("DATABASE_URL not set. Please check your .env file in the src directory.")
        return None
    
    try:
        logger.info(f"Initializing database connection with URL: {db_url[:db_url.find('@') + 1]}***")
        engine = create_async_engine(db_url, echo=False)
        async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        return async_session
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {str(e)}")
        st.error(f"Database connection failed: {str(e)}")
        return None

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
    try:
        from models.ProjectModel import ProjectModel
        logger.info(f"Creating ProjectModel instance with db_client type: {type(db_client)}")
        project_model = await ProjectModel.create_instance(db_client)
        if not project_model:
            logger.error("Failed to create ProjectModel instance")
            return None
        
        logger.info(f"Attempting to get/create project with ID: {project_id}")
        project = await project_model.get_project_or_create_one(project_id=project_id)
        if not project:
            logger.error("Project was not found and could not be created")
        else:
            logger.info(f"Successfully loaded/created project with ID: {getattr(project, 'id', None)}")
        return project
    except Exception as e:
        logger.error(f"Error in load_project: {str(e)}", exc_info=True)
        return None

# --- Must be first Streamlit command ---
st.set_page_config(page_title="AI Symptom checker", layout="wide")

# --- Initialize Async Runtime ---
def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Initialize event loop
loop = get_or_create_event_loop()
nest_asyncio.apply(loop)

# --- Load Environment Variables ---
# Get the src directory path and load .env from there
src_env_path = os.path.join(src_dir, '.env')
load_dotenv(dotenv_path=src_env_path)
logger.info(f"Loading environment variables from: {src_env_path}")
BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:5000")

# Verify database URL is loaded
db_url = os.getenv("DATABASE_URL")
if db_url:
    # Mask sensitive information in logs
    masked_url = db_url[:db_url.find('@') + 1] + '***'
    logger.info(f"Database URL loaded successfully: {masked_url}")
else:
    logger.error("DATABASE_URL not found in environment variables")

# --- Test Database Connection ---
async def test_db_connection():
    try:
        db_client = get_db_client()
        if not db_client:
            return False, "DATABASE_URL not set in environment"
        
        async with db_client() as session:
            # Try a simple query with proper text declaration
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
            await session.commit()
            return True, "Database connection successful"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"

# Execute the connection test
try:
    loop = get_or_create_event_loop()
    is_connected, message = loop.run_until_complete(test_db_connection())
    if not is_connected:
        st.error(f"Database Connection Error: {message}")
        logger.error(f"Database Connection Error: {message}")
    else:
        logger.info(message)
except Exception as e:
    logger.error(f"Error testing database connection: {str(e)}", exc_info=True)
    st.error(f"Error testing database connection: {str(e)}")

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
embedding_model = None
llm = None

try:
    embedding_model = load_embedding_model()
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Error loading embedding model: {str(e)}", exc_info=True)
    st.error("Failed to load embedding model. Some features may be limited.")

try:
    llm = load_llm()
    logger.info("LLM loaded successfully")
except Exception as e:
    logger.error(f"Error loading LLM: {str(e)}", exc_info=True)
    st.error("Failed to load LLM. Some features may be limited.")

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
    logger.info('Initializing session state: messages')
    st.session_state.messages = []
    initial_greeting = "Hi! I'm your pre-consultation assistant, here to help your doctor better understand your condition!"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

if 'agent_final_state' not in st.session_state:
    logger.info('Initializing session state: agent_final_state')
    st.session_state.agent_final_state = None

if 'agent_input_history' not in st.session_state:
    logger.info('Initializing session state: agent_input_history')
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
    logger.info(f'User submitted prompt: {prompt}')
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
        logger.info('User uploaded an image with the prompt.')
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
        logger.info('User submitted text only.')
        # Just add text
        st.session_state.messages.append({"role": "user", "content": user_text_input})
        st.session_state.agent_input_history.append({"role": "user", "content": user_text_input})

    # 2. Display user message immediately (handled by chat history display loop)

    # 3. Check if the agent graph is ready
    if not graph_app:
        logger.error('AI agent is unavailable.')
        error_msg = "Sorry, the AI agent is currently unavailable."
        with chat_container:
             with st.chat_message("assistant"): st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.session_state.agent_final_state = None
        st.rerun()
    else:
        logger.info('Preparing input state for agent graph.')
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
        project_id = 1  # Default to project 1
        db_client = clients["db_client"]
        project = None        
        if db_client is not None:
            try:
                logger.info('Loading project for RAG context.')
                # Check if DATABASE_URL is properly set
                settings = get_settings()
                db_url = getattr(settings, 'DATABASE_URL', None) or os.getenv("DATABASE_URL")
                if not db_url:
                    logger.error("DATABASE_URL is not set in either settings or environment")
                    st.error("Database connection settings are missing. Please check your configuration.")
                    project = None
                else:
                    logger.info(f"Attempting to connect to database: {db_url[:db_url.find('@') + 1]}***")
                    loop = asyncio.get_event_loop()
                    
                    async def load_project_with_context():
                        try:
                            project = await load_project(project_id, db_client)
                            if project:
                                logger.info(f'Project loaded successfully with ID: {getattr(project, "id", None)}')
                            else:
                                logger.warning('Project could not be loaded (None returned). Check database connection and project table.')
                            return project
                        except Exception as e:
                            logger.error(f"Error in project loading: {e}", exc_info=True)
                            raise

                    project = loop.run_until_complete(load_project_with_context())

            except Exception as e:
                logger.error(f'Error loading project: {str(e)}', exc_info=True)
                st.error("Failed to load project. Check the logs for details.")
                project = None
        else:
            logger.error('DB client initialization failed. Check database connection settings.')
            st.warning('Database connection failed. The system will operate with limited functionality.')# NLP service will be initialized in the async context
        if project:
         initial_graph_input["project"] = project
         initial_graph_input["project_id"] = getattr(project, "project_id", None)
         logger.info(f"✅ Injected project with ID: {getattr(project, 'project_id', None)}")
         logger.info(f"📂 Injected project type: {type(project)}")

         logger.info(f"✅ Injected project with ID: {initial_graph_input['project_id']}")
        else:
         logger.warning("⚠️ Project loading failed — RAG may not work.")

        # 5. Execute the agent graph (in-process, not HTTP)
        final_state: Optional[Dict[str, Any]] = None
        with st.spinner("Processing your request..."):
            try:
                logger.info('Invoking agent graph.')
                if graph_app:
                    # Use the global event loop
                    loop = asyncio.get_event_loop()
                    
                    # Define single async context for all operations
                    async def execute_full_pipeline():
                        try:
                            # Initialize NLP service
                            nlp_service = NLPService(
                                vectordb_client=clients["vectordb_client"],
                                generation_client=clients["generation_client"],
                                embedding_client=clients["embedding_client"],
                                template_parser=clients["template_parser"]
                            )
                            logger.info('NLPService initialized within async context.')
                            initial_graph_input["nlp_service"] = nlp_service
                            
                            # Execute graph
                            result = await graph_app.ainvoke(initial_graph_input)
                            logger.info('Graph execution completed.')
                            return result
                        except Exception as e:
                            logger.error(f"Error in pipeline execution: {e}")
                            raise

                    # Execute everything in a single async context
                    final_state = loop.run_until_complete(execute_full_pipeline())
                    agent_response = final_state.get("final_response") if final_state else None
                    logger.info('Agent graph executed successfully.')
                else:
                    logger.error('AI agent is unavailable after check.')
                    error_msg = "Sorry, the AI agent is currently unavailable."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.agent_final_state = None
                    st.rerun()

                st.session_state.agent_final_state = final_state
                if agent_response:
                    logger.info('Agent response received.')
                    st.session_state.messages.append({"role": "assistant", "content": agent_response})
                    st.session_state.agent_input_history.append({"role": "assistant", "content": agent_response})
                else:
                    logger.warning('No response received from the AI.')
                    error_msg = "No response received from the AI."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except Exception as e:
                logger.error(f'Error processing request: {e}')
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
            logger.info('Agent requested more symptom detail.')
            st.session_state.messages.append({"role": "assistant", "content": followup_message})
            st.session_state.agent_input_history.append({"role": "assistant", "content": followup_message})

        # Add agent's response (or error) to display history
        if agent_response:
            logger.info('Appending agent response to chat history.')
            # Only append if not a duplicate of the last message
            if not (st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant" and st.session_state.messages[-1]["content"] == agent_response):
                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                st.session_state.agent_input_history.append({"role": "assistant", "content": agent_response})
        elif graph_app and final_state is None:
            logger.error('Error during agent graph invoke.')
            error_msg = f"An error occurred during processing: Check logs." # Already printed traceback
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            # Don't add error to agent_input_history
        elif graph_app and final_state is not None and not agent_response:
            logger.warning('Agent finished but no final_response in state.')
            fallback_response = "Processing complete. Please let me know if there's anything else."
            print("Warning: Agent finished but no 'final_response' in state. Using fallback.")
            st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            st.session_state.agent_input_history.append({"role": "assistant", "content": fallback_response})
        # Else: Error case where graph_app itself was None already handled

        # 7. Rerun Streamlit to update chat display and show results below
        st.rerun()


# --- Display Results Section (Reads from stored final state) ---
if st.session_state.agent_final_state:
    logger.info('Displaying results section for final agent state.')
    final_state_to_display = st.session_state.agent_final_state
    with results_container:
        # Show warning if project context is missing
        if not final_state_to_display.get('project'):
            st.warning('Project context is not available. Some RAG features may be limited.')
        # Optionally show project info if available
        project = final_state_to_display.get('project')
        if project:
            st.info(f"Project ID: {getattr(project, 'id', 'N/A')}")
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
            logger.info('Showing context expander with RAG context.')
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
    logger.info('Restarting conversation via sidebar button.')
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

