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
        self.settings = get_settings()
        self._initialize_resources()
        self._validate_credentials()
        logger.info("✅ Tools initialized successfully")

    def _initialize_resources(self) -> None:
        """Initialize required resources using utility functions."""
        try:
            from stores.langgraph.utils import load_embedding_model
            self.embedding_model = load_embedding_model()
            logger.info("✅ Resources loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Error importing from utils: {e}")
            self.embedding_model = None
        except Exception as e:
            logger.error(f"❌ Error loading components: {e}")
            self.embedding_model = None

    def _validate_credentials(self) -> None:
        """Validate API credentials and configurations."""
        if not self.settings.GOOGLE_API_KEY or not self.settings.GOOGLE_CSE_ID:
            logger.warning("⚠️ Google API credentials not found in .env")
        else:
            logger.info("✅ Google API credentials loaded")

    def retrieve_relevant_documents(self, user_symptoms: str, nlp_service=None, project=None, project_id=None) -> dict:

        """Retrieves relevant documents from the RAG backend based on user symptoms, and evaluates context sufficiency."""
        logger.info("📚 Retrieving relevant documents (direct call)...")
        logger.debug(f"Input: '{user_symptoms[:100]}...'")
        logger.info(f"nlp_service type: {type(nlp_service)}, project type: {type(project)}")
        context = None
        is_sufficient = False

        # If nlp_service and project are provided, use direct function call
        if nlp_service is not None and project is not None:
            import asyncio
            try:
                # Try project object first, fallback to project_id param
                project_id = (
                 getattr(project, "project_id", None)
                 or getattr(project, "id", None)
                 or project_id
          )

                if not project_id:
                    logger.warning("⚠️ Project object is missing a valid ID. Skipping document retrieval.")
                    return {"context": "N/A (No project ID)", "is_sufficient": False}

                logger.info(f"📁 Using project_id={project_id} for in-process RAG")
                logger.info("🔄 Forcing direct in-process RAG call")

                answer, full_prompt, chat_history = asyncio.run(
                    nlp_service.answer_rag_question(project, user_symptoms, limit=3)
                )

                logger.info("✅ Documents retrieved successfully (direct)")
                context = answer or "No relevant documents found."

                if context and len(context) > 100 and "No relevant documents found" not in context:
                    is_sufficient = True

                return {"context": context, "is_sufficient": is_sufficient}
            except Exception as e:
                import traceback
                error_msg = f"Error in direct RAG retrieval: {e}\n{traceback.format_exc()}"
                logger.error(f"❌ {error_msg}")
                return {"context": error_msg, "is_sufficient": False}

        # Fallback: old HTTP call (for legacy use)
        try:
            project_id = (
             getattr(project, "project_id", None)
             or getattr(project, "id", None)
             or project_id
             or 1
            )

            logger.info(f"🌐 Falling back to HTTP RAG request using project_id={project_id}")

            response = requests.post(
                f"{self.settings.FASTAPI_URL}/api/v1/nlp/index/answer/{project_id}",
                json={"text": user_symptoms, "limit": 3},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            result = data.get("answer", "No answer returned from backend.")
            context = result

            if context and len(context) > 100 and "No answer returned from backend" not in context:
                is_sufficient = True

            logger.info("✅ Documents retrieved successfully (HTTP fallback)")
            return {"context": context, "is_sufficient": is_sufficient}
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling backend RAG: {e}"
            logger.error(f"❌ {error_msg}")
            return {"context": error_msg, "is_sufficient": False}
        except Exception as e:
            error_msg = f"Unexpected error in document retrieval: {e}"
            logger.error(f"❌ {error_msg}")
            return {"context": error_msg, "is_sufficient": False}

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

    def _validate_google_credentials(self) -> bool:
        """Validate that Google API credentials are available."""
        if not self.settings.GOOGLE_API_KEY or not self.settings.GOOGLE_CSE_ID:
            logger.error("❌ Google API credentials are missing")
            return False
        return True

# Instantiate and expose a module-level Tools object for import convenience
tools = Tools()

# Export the tools for use in other modules
__all__ = ['tools']
