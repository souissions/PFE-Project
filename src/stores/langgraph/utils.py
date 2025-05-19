import os
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import traceback
from typing import List, Dict, Any, Optional, Tuple
from helpers.config import get_settings
import logging

logger = logging.getLogger("uvicorn")

# Load environment variables from .env file
load_dotenv()

# --- Configuration Constants ---
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
PATIENT_CASES_PATH = os.getenv("PATIENT_CASES_PATH", "merged_reviews_new.csv")
SPECIALIST_LIST_PATH = os.getenv("SPECIALIST_LIST_PATH", "specialist_categories_list.txt")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")

# --- Loading Functions ---

def load_embedding_model():
    """Loads the HuggingFace embedding model."""
    logger.info("Attempting to load embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"FATAL ERROR loading embedding model: {e}")
        traceback.print_exc()
        return None

def load_llm():
    """Loads the Google Gemini LLM."""
    logger.info(f"Attempting to load LLM: {LLM_MODEL_NAME}...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Fatal Error: GOOGLE_API_KEY not found in environment.")
        return None
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=api_key
        )
        try:
            test_result = llm.invoke("Hello!")
            logger.info(f"LLM '{LLM_MODEL_NAME}' loaded and text test invocation successful.")
        except Exception as test_e:
            logger.warning(f"Warning: LLM loaded but test invocation failed: {test_e}")
            traceback.print_exc()
        return llm
    except Exception as e:
        logger.error(f"FATAL ERROR loading LLM '{LLM_MODEL_NAME}': {e}")
        traceback.print_exc()
        return None

def load_dataframes():
    """Loads Doctor profiles and Patient cases CSVs."""
    logger.info("Attempting to load dataframes (Doctors, Cases)...")
    doctor_df, cases_df = None, None
    try:
        if os.path.exists(PATIENT_CASES_PATH):
            cases_df = pd.read_csv(PATIENT_CASES_PATH)
            logger.info(f"Loaded {len(cases_df)} patient cases from '{PATIENT_CASES_PATH}'.")
        else:
            logger.warning(f"Warning: Patient cases file not found at '{PATIENT_CASES_PATH}'.")
    except Exception as e:
        logger.error(f"Error loading dataframes: {e}")
        traceback.print_exc()
        doctor_df, cases_df = None, None

    return doctor_df, cases_df

def load_specialist_list():
    """Loads the list of specialist names."""
    logger.info("Attempting to load specialist list...")
    specialist_list = []
    if not os.path.exists(SPECIALIST_LIST_PATH):
        logger.warning(f"Warning: Specialist list file not found at '{SPECIALIST_LIST_PATH}'.")
        return specialist_list  # Return empty list

    try:
        with open(SPECIALIST_LIST_PATH, "r", encoding='utf-8') as f:
            specialist_list = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(specialist_list)} specialists from '{SPECIALIST_LIST_PATH}'.")
    except Exception as e:
        logger.error(f"Error loading specialist list: {e}")
        traceback.print_exc()
        specialist_list = []  # Return empty list on error

    return specialist_list
