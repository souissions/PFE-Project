import os
import requests
import numpy as np
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import tool
import logging
from typing import Optional, Tuple, List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from helpers.config import get_settings

logger = logging.getLogger("uvicorn")

class Tools:
    def __init__(self):
        """Initialize the Tools class with required resources and configurations."""
        logger.info("🛠️ Initializing Tools...")
        
        # Load settings
        self.settings = get_settings()
        
        # Initialize resources
        self._initialize_resources()
        self._validate_credentials()
        
        logger.info("✅ Tools initialized successfully")

    def _initialize_resources(self) -> None:
        """Initialize required resources using utility functions."""
        try:
            from stores.langgraph.utils import load_embedding_model, load_icd_data_and_embeddings
            self.embedding_model = load_embedding_model()
            self.icd_codes, self.icd_embeddings = load_icd_data_and_embeddings(self.embedding_model)
            logger.info("✅ Resources loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Error importing from utils: {e}")
            self.embedding_model = None
            self.icd_codes = []
            self.icd_embeddings = None
        except Exception as e:
            logger.error(f"❌ Error loading components: {e}")
            self.embedding_model = None
            self.icd_codes = []
            self.icd_embeddings = None

    def _validate_credentials(self) -> None:
        """Validate API credentials and configurations."""
        if not self.settings.GOOGLE_API_KEY or not self.settings.GOOGLE_CSE_ID:
            logger.warning("⚠️ Google API credentials not found in .env")
        else:
            logger.info("✅ Google API credentials loaded")

    @tool
    def retrieve_relevant_documents(self, user_symptoms: str) -> str:
        """Retrieves relevant documents from the RAG backend based on user symptoms."""
        logger.info("📚 Retrieving relevant documents...")
        logger.debug(f"Input: '{user_symptoms[:100]}...'")

        try:
            response = requests.post(
                f"{self.settings.FASTAPI_URL}/api/v1/nlp/index/answer/{self.settings.PROJECT_ID}",
                json={"text": user_symptoms, "limit": self.settings.RAG_K},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            result = data.get("answer", "No answer returned from backend.")
            logger.info("✅ Documents retrieved successfully")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling backend RAG: {e}"
            logger.error(f"❌ {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error in document retrieval: {e}"
            logger.error(f"❌ {error_msg}")
            return error_msg

    @tool
    def match_relevant_icd_codes(self, text: str) -> List[str]:
        """Match text against ICD codes using embeddings."""
        try:
            response = requests.post(
                f"{self.settings.FASTAPI_URL}/api/v1/icd/match",
                json={"text": text, "top_n": self.settings.ICD_TOP_N}
            )
            if response.status_code == 200:
                return response.json()
            logger.error(f"Failed to match ICD codes: {response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Error matching ICD codes: {e}")
            return []

    @tool
    def google_search(self, query: str) -> str:
        """Search Google for additional information."""
        logger.info("🔍 Performing Google search...")
        logger.debug(f"Query: '{query[:100]}...'")

        if not self._validate_google_credentials():
            return "Google API credentials are missing."

        try:
            search_wrapper = GoogleSearchAPIWrapper(
                google_api_key=self.settings.GOOGLE_API_KEY,
                google_cse_id=self.settings.GOOGLE_CSE_ID,
                k=3
            )
            results = search_wrapper.run(query)
            
            if not results:
                logger.info("ℹ️ No search results found")
                return "No relevant results found."
                
            formatted_results = f"Web search results:\n---\n{results}\n---"
            logger.info("✅ Search completed successfully")
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error during Google Search: {e}"
            logger.error(f"❌ {error_msg}")
            return error_msg

    def _validate_icd_components(self) -> bool:
        """Validate that ICD matching components are properly initialized."""
        if self.icd_embeddings is None or not self.icd_codes or not self.embedding_model:
            logger.error("❌ ICD code matching components are missing")
            return False
        return True

    def _validate_google_credentials(self) -> bool:
        """Validate that Google API credentials are available."""
        if not self.settings.GOOGLE_API_KEY or not self.settings.GOOGLE_CSE_ID:
            logger.error("❌ Google API credentials are missing")
            return False
        return True

    def _collect_matches(self, sorted_indices: np.ndarray, sims: np.ndarray) -> List[str]:
        """Collect ICD code matches above the similarity threshold."""
        matched = []
        for idx in sorted_indices[:self.settings.ICD_TOP_N]:
            if sims[idx] >= self.settings.ICD_SIM_THRESHOLD:
                code = self.icd_codes[idx]
                score = sims[idx]
                matched.append(f"{code} (Similarity: {score:.2f})")
        return matched

# Create a singleton instance
tools = Tools()

# Export the tools for use in other modules
__all__ = ['tools']
