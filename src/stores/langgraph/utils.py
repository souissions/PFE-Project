import os
import pickle
import pandas as pd
import numpy as np
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
ICD_CSV_PATH = os.getenv("ICD_CSV_PATH", "ICD10-Disease-Mapping.csv")
ICD_CACHE_PATH = os.getenv("ICD_CACHE_PATH", "icd_embeddings.pkl")
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

def load_icd_data_and_embeddings(_embeddings):
    """Loads ICD codes/descriptions and their embeddings."""
    logger.info("Attempting to load ICD data and embeddings...")
    icd_codes_list, icd_embeddings_array = [], None

    if not _embeddings:
        logger.error("Error: Cannot load/compute ICD embeddings because the embedding model is not loaded.")
        return icd_codes_list, icd_embeddings_array  # Return empty/None

    if not os.path.exists(ICD_CSV_PATH):
        logger.warning(f"Warning: ICD mapping CSV file not found at '{ICD_CSV_PATH}'.")
        return icd_codes_list, icd_embeddings_array  # Return empty/None

    try:
        icd_df = pd.read_csv(ICD_CSV_PATH, dtype=str)
        icd_df.columns = [col.strip(" '\"") for col in icd_df.columns]
        required_cols = ["ICD-10-CM CODE", "ICD-10-CM CODE DESCRIPTION"]
        if not all(col in icd_df.columns for col in required_cols):
            logger.error(f"Error: Required columns missing in '{ICD_CSV_PATH}'. Need {required_cols}")
            return [], None

        icd_df.dropna(subset=required_cols, inplace=True)
        icd_df["ICD-10-CM CODE"] = icd_df["ICD-10-CM CODE"].str.strip(" '\"")
        icd_df["ICD-10-CM CODE DESCRIPTION"] = icd_df["ICD-10-CM CODE DESCRIPTION"].str.strip(" '\"").str.lower()
        icd_codes_list = icd_df["ICD-10-CM CODE"].tolist()
        icd_descriptions = icd_df["ICD-10-CM CODE DESCRIPTION"].tolist()
        logger.info(f"Loaded {len(icd_codes_list)} ICD codes/descriptions from CSV.")

    except Exception as e:
        logger.error(f"Error loading or processing ICD CSV '{ICD_CSV_PATH}': {e}")
        traceback.print_exc()
        return [], None  # Return empty/None

    cached_embeddings = None
    if os.path.exists(ICD_CACHE_PATH) and os.path.getsize(ICD_CACHE_PATH) > 0:
        logger.info(f"Cache file found at '{ICD_CACHE_PATH}'. Attempting to load...")
        try:
            with open(ICD_CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, dict) and 'embeddings' in cached_data and 'descriptions' in cached_data:
                cached_embeddings = cached_data['embeddings']
                logger.info(f"Successfully loaded valid ICD embeddings from cache ({cached_embeddings.shape}).")
        except Exception as e:
            logger.error(f"Error loading ICD cache from '{ICD_CACHE_PATH}': {e}. Recomputing.")
            traceback.print_exc()

    if cached_embeddings is None:
        logger.info(f"Computing ICD embeddings for {len(icd_descriptions)} descriptions (this may take a while)...")
        try:
            if not callable(getattr(_embeddings, "embed_documents", None)):
                raise TypeError("Provided embedding model cannot embed documents.")

            icd_embeddings_list = _embeddings.embed_documents(icd_descriptions)
            icd_embeddings_array = np.array(icd_embeddings_list).astype('float32')
            logger.info(f"ICD embeddings computed ({icd_embeddings_array.shape}).")
            try:
                with open(ICD_CACHE_PATH, "wb") as f:
                    pickle.dump({'embeddings': icd_embeddings_array, 'descriptions': icd_descriptions}, f)
                logger.info(f"New ICD embeddings cached to '{ICD_CACHE_PATH}'.")
            except Exception as e_cache:
                logger.warning(f"Warning: Could not save ICD embeddings cache to '{ICD_CACHE_PATH}': {e_cache}")
        except Exception as e_compute:
            logger.error(f"Error computing ICD embeddings: {e_compute}")
            traceback.print_exc()
            return icd_codes_list, None  # Return codes but None embeddings
    else:
        icd_embeddings_array = cached_embeddings

    return icd_codes_list, icd_embeddings_array
