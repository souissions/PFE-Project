from .BaseService import BaseService
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Dict, Any, Optional
import json
import logging
from stores.langgraph.graph import graph, compiled_graph
from stores.langgraph.scheme.state import AgentState

logger = logging.getLogger('uvicorn.error')

class NLPService(BaseService):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

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
        
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": self.generation_client.process_text(doc.text),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

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

    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify user intent using LangGraph."""
        try:
            state = AgentState(user_input=text)
            result = await graph.classify_intent(state)
            raw_intent = getattr(result.intent_classification, "intent", "")
            normalized_intent = self.normalize_intent_output(str(raw_intent))
            return {
                "intent": normalized_intent,
                "confidence": getattr(result.intent_classification, "confidence", None)
            }
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            raise

    async def check_relevance(self, text: str) -> Dict[str, Any]:
        """Check if the case is relevant for triage."""
        try:
            state = AgentState(user_input=text)
            result = await graph.check_relevance(state)
            return {
                "is_relevant": result.relevance_check.is_relevant,
                "confidence": result.relevance_check.confidence
            }
        except Exception as e:
            logger.error(f"Error in relevance check: {e}")
            raise

    async def process_information(self, query: str, docs: List[Dict], web_results: List[Dict]) -> str:
        """Process information requests."""
        try:
            state = AgentState(user_input=query)
            state.relevant_docs = docs
            state.web_results = web_results
            result = await graph._handle_information_request(state)
            return result.final_output
        except Exception as e:
            logger.error(f"Error in information processing: {e}")
            raise

    async def perform_analysis(self, symptoms: str, docs: List[Dict], icd_codes: List[Dict]) -> Dict[str, Any]:
        """Perform final analysis."""
        try:
            state = AgentState(user_input=symptoms)
            state.relevant_docs = docs
            state.matched_icd_codes = icd_codes
            result = await graph._final_analysis(state)
            return {
                "analysis": result.final_analysis.analysis,
                "relevant_docs": result.final_analysis.relevant_docs,
                "matched_icd_codes": result.final_analysis.matched_icd_codes
            }
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise

    async def evaluate_explanation(self, explanation: str) -> Dict[str, Any]:
        """Evaluate explanation quality."""
        try:
            state = AgentState()
            state["initial_explanation"] = explanation
            result = await graph._evaluate_explanation(state)
            return {
                "needs_refinement": result.get("needs_refinement"),
                "confidence": result.get("confidence")
            }
        except Exception as e:
            logger.error(f"Error in explanation evaluation: {e}")
            raise

    async def refine_explanation(self, explanation: str, critique: str) -> str:
        """Refine explanation if needed."""
        try:
            state = AgentState()
            state.final_analysis.analysis = explanation
            state.explanation_evaluation = {"critique": critique}
            result = await graph._refine_explanation(state)
            return result.final_analysis.analysis
        except Exception as e:
            logger.error(f"Error in explanation refinement: {e}")
            raise

    async def prepare_output(self, analysis: str, icd_codes: List[Dict]) -> str:
        """Prepare final output."""
        try:
            state = AgentState()
            state.final_analysis.analysis = analysis
            state.final_analysis.matched_icd_codes = icd_codes
            result = await graph._prepare_final_output(state)
            return result.final_output
        except Exception as e:
            logger.error(f"Error in output preparation: {e}")
            raise

    async def run_langgraph_flow(self, query: str, image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
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
                "final_response": None
            }
            result = await compiled_graph.ainvoke(state)
            logger.info("LangGraph flow completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in LangGraph flow: {e}")
            return {"error": str(e)}

