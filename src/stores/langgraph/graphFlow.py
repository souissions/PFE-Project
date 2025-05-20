import logging
import re
from typing import Dict, Any
from stores.langgraph.tools import tools
from stores.llm.templates.template_parser import TemplateParser
from stores.langgraph.utils import load_llm
from stores.langgraph.scheme.state import AgentState
from stores.langgraph.scheme.models import IntentQuery
from stores.langgraph.scheme.models.relevance_checker import RelevanceCheckerInput, RelevanceCheckerOutput
from stores.llm.templates.locales.en.relevance_check import relevance_check_prompt
from stores.langgraph.scheme.models.final_analysis import FinalAnalysisInput, FinalAnalysisOutput
from stores.langgraph.scheme.models.info_request_handler import InfoRequestHandlerInput, InfoRequestHandlerOutput
from stores.langgraph.scheme.models.explanation_evaluator import ExplanationEvaluatorInput, ExplanationEvaluatorOutput, ExplanationRefinerInput, ExplanationRefinerOutput
from stores.langgraph.scheme.models.off_topic_handler import OffTopicHandlerInput, OffTopicHandlerOutput
from stores.langgraph.scheme.models.irrelevant_triage_handler import IrrelevantTriageHandlerInput, IrrelevantTriageHandlerOutput
from stores.langgraph.scheme.models.final_output_preparer import FinalOutputPreparerInput, FinalOutputPreparerOutput
from stores.llm.templates.locales.en.rag_evaluator import off_topic_response, irrelevant_triage_response

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import base64
import json
from langchain.schema import HumanMessage

logger = logging.getLogger("uvicorn")

class GraphFlow:
    def __init__(self, state: AgentState):
        self.state = state
        self.parser = TemplateParser(language='en')
        self.llm = load_llm()
        
        # Initialize prompts
        self.intent_prompt_str = self.parser.get("intent_classifier", "intent_classifier_prompt")
        self.followup_prompt_str = self.parser.get("followup", "followup_system")
        self.relevance_check_prompt = relevance_check_prompt
        
        # Initialize LangChain components
        self.intent_prompt = PromptTemplate(
            template=self.intent_prompt_str,
            input_variables=["conversation_history", "user_query"]
        )
        self.followup_prompt = PromptTemplate(
            template=self.followup_prompt_str,
            input_variables=["accumulated_symptoms", "user_query"]
        )
        
        # Create runnable sequences
        self.intent_chain = self.intent_prompt | self.llm
        self.followup_chain = self.followup_prompt | self.llm

    async def classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent based on conversation history and latest query."""
        try:
            logger.info(f"[DEBUG] classify_intent input: conversation_history={state['conversation_history']}, user_query={state['user_query']}")
            input_data = {
                "conversation_history": state["conversation_history"],
                "user_query": state["user_query"]
            }
            # Run intent classification
            result = await self.intent_chain.ainvoke(input_data)
            logger.info(f"[DEBUG] classify_intent raw result: {result}")
            # Use robust normalization for intent extraction
            intent_label = result.content.strip().upper()
            logger.info(f"[DEBUG] classify_intent extracted intent: {intent_label}")
            state["intent"] = intent_label
            return state
        except Exception as e:
            logger.error(f"Error in classify_intent: {e}")
            state["intent"] = "UNKNOWN"
            return state

    async def gather_symptoms(self, state: AgentState) -> AgentState:
        """Gather and analyze symptoms from user input, then check sufficiency before relevance."""
        try:
            # Prepare input for symptom analysis
            input_data = {
                "accumulated_symptoms": state["accumulated_symptoms"],
                "user_query": state["user_query"]
            }
            # Run symptom analysis
            result = await self.followup_chain.ainvoke(input_data)
            # Robustly extract string content
            if hasattr(result, 'content'):
                symptoms = result.content.strip()
            elif hasattr(result, 'text'):
                symptoms = result.text.strip()
            elif hasattr(result, 'to_string'):
                symptoms = result.to_string().strip()
            else:
                symptoms = str(result).strip()
            state["accumulated_symptoms"] = symptoms

            # --- Symptom sufficiency check ---
            from stores.llm.templates.locales.en.symptom_sufficiency import symptom_sufficiency_prompt
            sufficiency_result = await symptom_sufficiency_prompt.ainvoke({"accumulated_symptoms": symptoms})
            if hasattr(sufficiency_result, 'text'):
                sufficiency_answer = sufficiency_result.text.strip().upper()
            elif hasattr(sufficiency_result, 'to_string'):
                sufficiency_answer = sufficiency_result.to_string().strip().upper()
            else:
                sufficiency_answer = str(sufficiency_result).strip().upper()
            state["symptom_sufficiency"] = sufficiency_answer
            if sufficiency_answer.startswith("NO"):
                # Not enough detail, ask for more and halt flow for now
                state["needs_more_symptom_detail"] = True
                state["followup_message"] = "Can you provide more details about your symptoms? For example, describe the nature, location, severity, duration, onset, triggers, or any associated factors."
                return state
            # If YES, proceed as normal
            state["needs_more_symptom_detail"] = False
            return state
        except Exception as e:
            logger.error(f"Error in gather_symptoms: {e}")
            return state

    async def check_triage_relevance(self, state: AgentState) -> AgentState:
        """Checks if the accumulated symptoms are medically relevant using a LangChain prompt."""
        logger.info("🔍 Checking triage relevance...")
        accumulated_symptoms = state.get('accumulated_symptoms', '')
        # Remove image placeholder text if present
        if accumulated_symptoms and isinstance(accumulated_symptoms, str):
            accumulated_symptoms = accumulated_symptoms.replace("[IMAGE UPLOADED]", "").strip()
        # If no text symptoms, assume irrelevant
        if not accumulated_symptoms:
            logger.info("No accumulated symptoms text found. Marking as irrelevant.")
            state["is_relevant"] = False
            return state
        # Format the prompt string correctly
        prompt_str = None
        if hasattr(self.relevance_check_prompt, 'format'):
            prompt_val = self.relevance_check_prompt.format(accumulated_symptoms=accumulated_symptoms)
            # If the result is a function, call it until it's not
            while callable(prompt_val):
                prompt_val = prompt_val()
            if hasattr(prompt_val, 'to_string'):
                prompt_str = prompt_val.to_string()
            elif hasattr(prompt_val, 'text'):
                prompt_str = prompt_val.text
            elif hasattr(prompt_val, 'content'):
                prompt_str = prompt_val.content
            else:
                prompt_str = str(prompt_val)
        elif callable(self.relevance_check_prompt):
            prompt_val = self.relevance_check_prompt(accumulated_symptoms=accumulated_symptoms)
            while callable(prompt_val):
                prompt_val = prompt_val()
            if hasattr(prompt_val, 'to_string'):
                prompt_str = prompt_val.to_string()
            elif hasattr(prompt_val, 'text'):
                prompt_str = prompt_val.text
            elif hasattr(prompt_val, 'content'):
                prompt_str = prompt_val.content
            else:
                prompt_str = str(prompt_val)
        else:
            prompt_str = str(self.relevance_check_prompt)
        logger.debug(f"[DEBUG] triage relevance prompt_str type: {type(prompt_str)}, value: {prompt_str!r}")
        try:
            # Ensure prompt_str is a string
            if not isinstance(prompt_str, str):
                prompt_str = str(prompt_str)
            llm_result = await self.llm.ainvoke(prompt_str)
            logger.debug(f"[DEBUG] triage relevance llm_result type: {type(llm_result)}, value: {llm_result!r}")
            # If the result is a function, call it until it's not
            while callable(llm_result):
                llm_result = llm_result()
            # Robustly extract string from LLM result
            if hasattr(llm_result, 'content'):
                answer = llm_result.content.strip().upper()
            elif hasattr(llm_result, 'text'):
                answer = llm_result.text.strip().upper()
            elif hasattr(llm_result, 'to_string'):
                answer = llm_result.to_string().strip().upper()
            else:
                answer = str(llm_result).strip().upper()
            logger.debug(f"Parsed triage relevance answer: {answer}")
            if answer == 'YES':
                is_relevant = True
            elif answer == 'NO':
                is_relevant = False
            else:
                logger.warning(f"⚠️ Unexpected LLM answer for triage relevance: {answer}. Defaulting to relevant.")
                is_relevant = True  # Fail-safe: default to relevant
        except Exception as e:
            logger.error(f"❌ Error in triage relevance LLM chain: {e}. Defaulting to relevant.")
            is_relevant = True  # Fail-safe: default to relevant
        logger.info(f"🔍 Triage relevance: {is_relevant}")
        state["is_relevant"] = is_relevant
        return state

    async def handle_info_request(self, state: AgentState) -> AgentState:
        """Handles general medical information requests."""
        logger.info("📚 Handling information request...")
        input_data = InfoRequestHandlerInput(
            user_query=state.get('user_query', ''),
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
        if not input_data.user_query:
            logger.warning("⚠️ No query provided for information request")
            state["final_response"] = "No query provided."
            state["rag_context"] = None
            state["uploaded_image_bytes"] = None
            return state
        # Use project and nlp_service from state for in-process RAG
        project = state.get('project')
        nlp_service = state.get('nlp_service')
        context = tools.retrieve_relevant_documents(input_data.user_query, nlp_service=nlp_service, project=project)
        response_content = f"Context: {context}"
        logger.info("✅ Information request handled")
        state["final_response"] = response_content
        state["rag_context"] = context
        state["uploaded_image_bytes"] = None
        return state

    async def perform_final_analysis(self, state: AgentState) -> AgentState:
        """Performs final analysis of symptoms and generates recommendations."""
        logger.info("🔬 Performing final analysis...")
        input_data = FinalAnalysisInput(
            accumulated_symptoms=state.get('accumulated_symptoms', ''),
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
                # Get project_id safely from state or fallback
        project = state.get("project")
        project_id = state.get("project_id") or (getattr(project, "project_id", None) if project else None)
        nlp_service = state.get("nlp_service")

        logger.info(f"📁 Using project ID in analysis: {project_id}")
        nlp_service = state.get('nlp_service')
        logger.info(f"📁 Using project ID in analysis: {getattr(project, 'project_id', None)}")

        rag_result = tools.retrieve_relevant_documents(
         input_data.accumulated_symptoms,
         nlp_service=nlp_service,
         project_id=project_id  # ← changed to pass id directly
       )
        context = rag_result.get("context")
        is_sufficient = rag_result.get("is_sufficient", False)
        matched_icd_codes = state.get('matched_icd_codes', None)
        logger.info(f"✅ Final analysis completed. Context sufficiency: {is_sufficient}")
        state["rag_context"] = context
        state["context_is_sufficient"] = is_sufficient
        if not is_sufficient:
            state["initial_explanation"] = "Sorry, the information in our knowledge base is not sufficient to answer your question. Please provide more details or try again later."
        else:
            state["initial_explanation"] = "Analysis complete"
        state["matched_icd_codes"] = matched_icd_codes
        state["evaluator_critique"] = None
        state["loop_count"] = 0
        state["uploaded_image_bytes"] = None
        return state

    async def evaluate_explanation(self, state: AgentState) -> AgentState:
        """Evaluates the clarity and quality of the explanation."""
        logger.info("📊 Evaluating explanation...")
        
        input_data = ExplanationEvaluatorInput(
            initial_explanation=state.get('initial_explanation', '')
        )
        
        # TODO: Implement proper evaluation
        critique = "OK"
        
        logger.info(f"📊 Evaluation result: {critique}")
        state["evaluator_critique"] = critique
        return state

    async def refine_explanation(self, state: AgentState) -> AgentState:
        """Refines the explanation based on evaluation feedback."""
        logger.info("🔄 Refining explanation...")
        
        input_data = ExplanationRefinerInput(
            initial_explanation=state.get('initial_explanation', ''),
            evaluator_critique=state.get('evaluator_critique', 'OK')
        )
        
        if input_data.evaluator_critique == "OK":
            logger.info("✅ No refinement needed")
            state["initial_explanation"] = input_data.initial_explanation
            state["loop_count"] = state.get('loop_count', 0) + 1
            return state

        # TODO: Implement proper refinement
        refined_explanation = "Refined explanation"
        
        logger.info("✅ Explanation refined")
        state["initial_explanation"] = refined_explanation
        state["loop_count"] = state.get('loop_count', 0) + 1
        return state

    async def handle_off_topic(self, state: AgentState) -> AgentState:
        """Handles off-topic queries."""
        logger.info("🚫 Handling off-topic query...")
        
        input_data = OffTopicHandlerInput(
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
        
        response = "Sorry, this topic is outside my medical assistant capabilities. Please describe a health-related concern."
        logger.info("✅ Off-topic response generated")
        
        state["final_response"] = response
        state["uploaded_image_bytes"] = None
        return state

    async def handle_irrelevant_triage(self, state: AgentState) -> AgentState:
        """Handles irrelevant triage cases."""
        logger.info("⚠️ Handling irrelevant triage...")
        
        input_data = IrrelevantTriageHandlerInput(
            uploaded_image_bytes=state.get('uploaded_image_bytes')
        )
        
        response = "Your symptoms don't appear to require immediate medical attention. However, if you're concerned, please consult a healthcare professional."
        logger.info("✅ Irrelevant triage response generated")
        
        state["final_response"] = response
        state["rag_context"] = "N/A (Triage irrelevant)"
        state["uploaded_image_bytes"] = None
        return state

    async def prepare_final_output(self, state: AgentState) -> AgentState:
        """Prepares the final output for the user."""
        logger.info(f"📤 Preparing final output... Incoming state: {state}")
        input_data = FinalOutputPreparerInput(
            final_explanation=state.get('final_explanation'),
            final_response=state.get('final_response')
        )
        final_response = input_data.final_response or input_data.final_explanation or "Processing complete."
        logger.info("✅ Final output prepared")
        state["final_response"] = final_response
        state["final_output"] = final_response  # Ensure final_output is set for downstream consumers
        return state
