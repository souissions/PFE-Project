from .BaseService import BaseService
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Dict, Any, Optional
import json
import logging
from stores.langgraph.graph import Graph
from stores.langgraph.scheme.state import AgentState
from string import Template

logger = logging.getLogger('uvicorn.error')

class NLPService(BaseService):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser, graph_flow=None, graph=None):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.graph_flow = graph_flow
        self.graph = graph

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)
    
    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        vectors = self.embedding_client.embed_text(text=texts, 
                                                  document_type=DocumentTypeEnum.DOCUMENT.value)

        # step3: create collection if not exists
        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        # step1: get collection name
        query_vector = None
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: get text embedding vector
        vectors = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vectors or len(vectors) == 0:
            return False
        
        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if not query_vector:
            return False    

        # step3: do semantic search
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )

        if not results:
            return False

        return results
    
    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        import logging
        logger = logging.getLogger('uvicorn.error')
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        # Debug: Log retrieved documents and their text fields
        logger.info(f"[RAG DEBUG] Retrieved {len(retrieved_documents) if retrieved_documents else 0} documents.")
        if retrieved_documents:
            for idx, doc in enumerate(retrieved_documents):
                logger.info(f"[RAG DEBUG] Doc {idx+1}: text='" + str(getattr(doc, 'text', None))[:200] + "...'")

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            Template(self.template_parser.get("rag", "document_prompt")).substitute(
                doc_num=idx + 1,
                chunk_text=self.generation_client.process_text(doc.text),
            )
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = Template(self.template_parser.get("rag", "footer_prompt")).substitute(
            query=query
        )

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history
    
    @staticmethod
    def normalize_intent_output(output: str) -> str:
        """Robustly extract the intent label from LLM output."""
        import re
        allowed_intents = [
            "SYMPTOM_TRIAGE",
            "MEDICAL_INFORMATION_REQUEST",
            "OFF_TOPIC"
        ]
        # Search for allowed intent as a whole word (case-insensitive)
        for intent in allowed_intents:
            pattern = r"\\b" + re.escape(intent) + r"\\b"
            if re.search(pattern, output, re.IGNORECASE):
                return intent
        # Fallback: try to extract a single word matching any allowed intent (case-insensitive)
        output_upper = output.upper()
        for intent in allowed_intents:
            if intent in output_upper:
                return intent
        # Fallback: return OFF_TOPIC
        return "OFF_TOPIC"

    async def classify_intent(self, state: AgentState) -> Dict[str, Any]:
        """Classify user intent using LangGraph (state-based)."""
        try:
            result = await self.graph.classify_intent(state)
            raw_intent = getattr(result.intent_classification, "intent", "")
            normalized_intent = self.normalize_intent_output(str(raw_intent))
            return {
                "intent": normalized_intent,
                "confidence": getattr(result.intent_classification, "confidence", None)
            }
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            raise

    async def check_relevance(self, state: AgentState) -> Dict[str, Any]:
        """Check if the case is relevant for triage (state-based)."""
        try:
            result = await self.graph.check_relevance(state)
            return {
                "is_relevant": result.relevance_check.is_relevant,
                "confidence": result.relevance_check.confidence
            }
        except Exception as e:
            logger.error(f"Error in relevance check: {e}")
            raise

    async def process_information(self, state: AgentState) -> str:
        """Process information requests (state-based)."""
        try:
            result = await self.graph_flow.handle_info_request(state)
            return result.final_output
        except Exception as e:
            logger.error(f"Error in information processing: {e}")
            raise

    async def perform_analysis(self, state: AgentState) -> Dict[str, Any]:
        """Perform final analysis (state-based)."""
        try:
            result = await self.graph_flow.perform_final_analysis(state)
            return {
                "analysis": result.final_analysis.analysis,
                "relevant_docs": result.final_analysis.relevant_docs
            }
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise

    async def evaluate_explanation(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate explanation quality (state-based)."""
        try:
            result = await self.graph_flow.evaluate_explanation(state)
            return {
                "needs_refinement": result.get("needs_refinement"),
                "confidence": result.get("confidence")
            }
        except Exception as e:
            logger.error(f"Error in explanation evaluation: {e}")
            raise

    async def refine_explanation(self, state: AgentState) -> str:
        """Refine explanation if needed (state-based)."""
        try:
            result = await self.graph_flow.refine_explanation(state)
            return result.final_analysis.analysis
        except Exception as e:
            logger.error(f"Error in explanation refinement: {e}")
            raise

    async def prepare_output(self, state: AgentState) -> str:
        """Prepare final output (state-based)."""
        try:
            result = await self.graph_flow.prepare_final_output(state)
            return result.final_output
        except Exception as e:
            logger.error(f"Error in output preparation: {e}")
            raise

    async def run_langgraph_flow(self, query: str, image_bytes: Optional[bytes] = None, project=None, nlp_service=None) -> Dict[str, Any]:
        """Run the complete LangGraph flow."""
        try:
            logger.info("Starting LangGraph flow...")
            # Build a full AgentState dict as required by the graph
            state = {
                "conversation_history": [],
                "user_query": query,
                "uploaded_image_bytes": image_bytes,
                "image_prompt_text": None,
                "user_intent": None,
                "accumulated_symptoms": "",
                "is_relevant": None,
                "loop_count": 0,
                "rag_context": None,
                "matched_icd_codes": None,
                "initial_explanation": None,
                "evaluator_critique": None,
                "final_explanation": None,
                "recommended_specialist": None,
                "doctor_recommendations": None,
                "no_doctors_found_specialist": None,
                "final_response": None,
                # Pass project and nlp_service for in-process RAG
                "project": project,
                "nlp_service": nlp_service
            }
            result = await self.graph_flow.prepare_final_output(state)
            logger.info("LangGraph flow completed successfully")
            logger.info(f"LangGraph flow result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in LangGraph flow: {e}")
            return {"error": str(e)}

