import os
import requests
from langchain_google_community import GoogleSearchAPIWrapper  # Corrected Import
from langchain_core.tools import tool
from dotenv import load_dotenv  # Ensure load_dotenv is imported
import numpy as np
import pandas as pd
import re
import traceback
import streamlit as st  # Import for potential access to cached resources if needed
from sklearn.metrics.pairwise import cosine_similarity  # Ensure this import is included


# --- Tool Configuration ---
RAG_K = 5  # Number of chunks to retrieve for RAG context
ICD_TOP_N = 5  # Number of top ICD codes to consider matching
ICD_SIM_THRESHOLD = 0.25  # Similarity threshold for ICD matching

# --- Accessing Global Resources ---
# Load components needed by tools using the utility functions.
print("tools.py: Loading shared resources via utils...")
try:
    # Assuming utils.py correctly loads these using @st.cache_resource
    from utils import load_embedding_model, load_icd_data_and_embeddings
    embedding_model_t = load_embedding_model()
    icd_codes_t, icd_embeddings_t = load_icd_data_and_embeddings(embedding_model_t)
    print("tools.py: Shared resources loaded.")
except ImportError:
    print("tools.py: Error importing from utils. Make sure utils.py is in the same directory.")
    embedding_model_t, icd_codes_t, icd_embeddings_t = None, [], None
except Exception as e_load:
    print(f"tools.py: Error loading components via utils: {e_load}")
    embedding_model_t, icd_codes_t, icd_embeddings_t = None, [], None

# --- Load Google CSE Credentials ---
# load_dotenv()  # Ensure .env is loaded
# GOOGLE_API_KEY_CSE = os.getenv("GOOGLE_API_KEY")  # Key enabled for Custom Search API
# GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GOOGLE_API_KEY_CSE = None
GOOGLE_CSE_ID = None
if hasattr(st, 'secrets'):
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY_CSE = st.secrets["GOOGLE_API_KEY"]
        print("tools.py: Loaded GOOGLE_API_KEY from st.secrets.")
    else:
        print("tools.py: Warning - GOOGLE_API_KEY not found in st.secrets.")

    if "GOOGLE_CSE_ID" in st.secrets:
        GOOGLE_CSE_ID = st.secrets["GOOGLE_CSE_ID"]
        print("tools.py: Loaded GOOGLE_CSE_ID from st.secrets.")
    else:
        print("tools.py: Warning - GOOGLE_CSE_ID not found in st.secrets.")
else:
    print("tools.py: Warning - st.secrets not available (running locally without Streamlit context?)")


# --- Tool Definitions ---

@tool
def retrieve_relevant_documents(user_symptoms: str) -> str:
    """
    Calls the custom FastAPI RAG backend to retrieve the final answer for a given user query.
    This is directly integrated with your RAG system to fetch relevant documents based on user symptoms.
    """
    tool_name = "retrieve_relevant_documents (via backend RAG API)"
    print(f"\n--- Tool: {tool_name} ---")
    print(f"User input: '{user_symptoms[:100]}...'")

    BASE_URL = "http://localhost:5000"  # Your FastAPI backend URL
    PROJECT_ID = 1  # Replace with your actual project_id

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/nlp/index/answer/{PROJECT_ID}",
            json={"text": user_symptoms, "limit": 5},  # Sending the user's symptoms for document retrieval
            timeout=10
        )
        response.raise_for_status()  # Raise error for non-200 responses
        data = response.json()

        # You can choose to modify the returned data based on your RAG output structure.
        return data.get("answer", "No answer returned from backend.")  # Ensure the answer key exists in your RAG response
    except Exception as e:
        error_msg = f"Error calling backend RAG: {e}"
        print(f"{tool_name}: {error_msg}")
        return error_msg


@tool
def match_relevant_icd_codes(user_symptoms: str) -> str:
    """
    Matches potentially relevant ICD-10 medical codes to the user's described text symptoms
    using semantic similarity search against a database of ICD code descriptions.
    This tool can match ICD codes to the text from the RAG output.
    """
    tool_name = "match_relevant_icd_codes"
    print(f"\n--- Tool: {tool_name} ---")
    print(f"Input symptoms (first 100 chars): '{user_symptoms[:100]}...'")

    # Ensure necessary components (ICD embeddings and model) are available
    if icd_embeddings_t is None or icd_codes_t is None or not embedding_model_t:
        error_msg = "ICD code embeddings or model are missing."
        print(f"{tool_name}: {error_msg}")
        return error_msg

    try:
        # Embed the user symptoms
        query_emb_list = embedding_model_t.embed_query(user_symptoms)
        query_emb = np.array([query_emb_list]).astype('float32')

        # Perform cosine similarity search with the ICD embeddings
        sims = cosine_similarity(icd_embeddings_t, query_emb).flatten()
        sorted_indices = np.argsort(sims)[::-1]

        # Collect matched ICD codes above the threshold
        matched = []
        for idx in sorted_indices[:ICD_TOP_N]:
            if sims[idx] >= ICD_SIM_THRESHOLD:
                code = icd_codes_t[idx]
                score = sims[idx]
                matched.append(f"{code} (Similarity: {score:.2f})")

        if not matched:
            return "No relevant ICD codes found with sufficient similarity."
        else:
            return "; ".join(matched)

    except Exception as e:
        error_msg = f"Error during ICD code matching: {e}"
        print(f"{tool_name}: {error_msg}")
        traceback.print_exc()
        return f"Error matching ICD codes: {e}"


# --- NEW: Google Search Tool using Custom Search API ---
@tool
def google_search(query: str) -> str:
    """
    Performs a Google search when the internal knowledge base or RAG fails.
    This tool leverages Google Search API for external information retrieval.
    """
    tool_name = "google_search"
    print(f"\n--- Tool: {tool_name} ---")
    print(f"Input query: '{query[:100]}...'")

    if not GOOGLE_API_KEY_CSE or not GOOGLE_CSE_ID:
        error_msg = "Google API credentials are missing."
        print(f"{tool_name}: {error_msg}")
        return error_msg

    try:
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY_CSE,
            google_cse_id=GOOGLE_CSE_ID,
            k=3  # Limit the search results
        )
        results = search_wrapper.run(query)
        return f"Web search results:\n---\n{results}\n---" if results else "No relevant results found."
    except Exception as e:
        error_msg = f"Error during Google Search: {e}"
        print(f"{tool_name}: {error_msg}")
        traceback.print_exc()
        return f"Error during web search: {e}"
